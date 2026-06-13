"""
GraphView — UMAP scatter plot with lasso selection, thumbnail strip,
batch status management, and visual status tags.
"""

import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QCheckBox, QButtonGroup,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QColor

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.widgets import LassoSelector
    from matplotlib.path import Path as MplPath
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Status tag config ─────────────────────────────────────────────────
STATUS_CFG = {
    "pending":   ("Pendiente", "#808080"),
    "reviewed":  ("Revisada",  "#2ECC71"),
    "discarded": ("Descartada","#E74C3C"),
}


# ── SelectableThumb ───────────────────────────────────────────────────

class SelectableThumb(QFrame):
    """Thumbnail card with checkbox, status tag, and optional class chip."""
    clicked       = Signal(int)        # image_id
    check_changed = Signal(int, bool)  # image_id, checked

    W, H = 130, 110

    def __init__(self, image_id: int, thumbnail_path: str,
                 filename: str, status: str,
                 class_label: str = "", class_color: str = "",
                 parent=None):
        super().__init__(parent)
        self.image_id = image_id
        self.status   = status
        self.setFixedSize(self.W + 4, self.H + 56)
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)
        self._apply_style(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # checkbox + status tag row
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self._cb = QCheckBox()
        self._cb.setFixedSize(16, 16)
        self._cb.toggled.connect(lambda v: self.check_changed.emit(self.image_id, v))
        top.addWidget(self._cb)
        top.addStretch()

        label_text, label_color = STATUS_CFG.get(status, ("?", "#555"))
        self._status_tag = QLabel(f"● {label_text}")
        self._status_tag.setStyleSheet(
            f"color:{label_color}; font-size:9px; font-weight:bold;"
        )
        top.addWidget(self._status_tag)
        layout.addLayout(top)

        # thumbnail
        thumb = QLabel()
        thumb.setFixedSize(self.W, self.H)
        thumb.setAlignment(Qt.AlignCenter)
        if thumbnail_path and Path(thumbnail_path).exists():
            pix = QPixmap(thumbnail_path).scaled(
                self.W, self.H, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            thumb.setPixmap(pix)
        else:
            thumb.setText("?")
            thumb.setStyleSheet("color:#555;")
        layout.addWidget(thumb)

        # class chip
        self._class_chip = QLabel()
        self._class_chip.setAlignment(Qt.AlignCenter)
        self._class_chip.setFixedHeight(16)
        layout.addWidget(self._class_chip)
        self.update_label(class_label, class_color)

        # filename
        name = filename if len(filename) <= 18 else filename[:8] + "…" + filename[-8:]
        fname_lbl = QLabel(name)
        fname_lbl.setAlignment(Qt.AlignCenter)
        fname_lbl.setStyleSheet("color:#aaa; font-size:9px;")
        layout.addWidget(fname_lbl)

    def _apply_style(self, checked: bool):
        _, color = STATUS_CFG.get(self.status, ("?", "#555"))
        border = "2px solid #4fc3f7" if checked else f"1px solid {color}"
        self.setStyleSheet(
            f"QFrame{{background:#1e1e1e; border:{border}; border-radius:4px;}}"
        )

    def set_checked(self, v: bool):
        self._cb.setChecked(v)
        self._apply_style(v)

    def is_checked(self) -> bool:
        return self._cb.isChecked()

    def update_status(self, status: str):
        self.status = status
        label_text, label_color = STATUS_CFG.get(status, ("?", "#555"))
        self._status_tag.setText(f"● {label_text}")
        self._status_tag.setStyleSheet(
            f"color:{label_color}; font-size:9px; font-weight:bold;"
        )
        self._apply_style(self._cb.isChecked())

    def update_label(self, class_label: str, class_color: str):
        if class_label:
            self._class_chip.setText(class_label)
            self._class_chip.setStyleSheet(
                f"background:{class_color}; color:white; font-size:9px; "
                f"font-weight:bold; border-radius:3px; padding:0 4px;"
            )
            self._class_chip.setVisible(True)
        else:
            self._class_chip.setVisible(False)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._cb.underMouse():
            self.clicked.emit(self.image_id)
        super().mousePressEvent(event)


# ── GraphView ─────────────────────────────────────────────────────────

class GraphView(QWidget):
    image_clicked            = Signal(int)        # image_id (click single point)
    lasso_selection_changed  = Signal(list)       # [image_id,...] ordered; empty=cleared
    images_status_changed    = Signal(list, str)  # [image_id,...], status
    images_label_changed     = Signal(list, int)  # [image_id,...], class_id
    images_delete_requested  = Signal(list)       # [image_id,...]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proj_rows: list     = []
        self._cluster_colors: dict[int, str] = {}
        self._image_ids: list[int]= []
        self._xs = np.array([])
        self._ys = np.array([])
        self._statuses: list[str] = []
        self._scatter             = None
        self._status_filter: str  = "todas"

        # lasso state
        self._lasso               = None
        self._lasso_active        = False
        self._selected_mask       = np.array([], dtype=bool)

        # strip state
        self._strip_thumbs: list[SelectableThumb] = []
        self._checked_ids: set[int] = set()

        # pan state
        self._is_panning = False
        self._pan_start  = None
        self._pan_xlim   = None
        self._pan_ylim   = None

        # classification state
        self._classes: list[tuple] = []          # [(id, name, color), ...]
        self._ann_cache: dict[int, list] = {}    # image_id -> [(x,y,w,h,color,name),...]

        self._setup_ui()

    # ─── UI ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        if not HAS_MPL:
            outer.addWidget(QLabel("matplotlib no disponible"))
            return

        # ── Toolbar ──
        top = QHBoxLayout()

        self._lasso_btn = QPushButton("Lazo")
        self._lasso_btn.setCheckable(True)
        self._lasso_btn.setFixedWidth(70)
        self._lasso_btn.setToolTip(
            "Activar lazo: dibujá una selección libre sobre el mapa.\n"
            "Click derecho + drag = pan. Rueda = zoom."
        )
        self._lasso_btn.toggled.connect(self._toggle_lasso)
        self._lasso_btn.setStyleSheet("""
            QPushButton { border:1px solid #888; border-radius:4px; padding:3px 8px; }
            QPushButton:checked { background:#1a5c8c; border-color:#4fc3f7;
                                  color:#fff; font-weight:bold; }
        """)
        top.addWidget(self._lasso_btn)

        clear_btn = QPushButton("Limpiar selección")
        clear_btn.setFixedWidth(130)
        clear_btn.clicked.connect(self._clear_lasso)
        top.addWidget(clear_btn)

        reset_btn = QPushButton("Reset vista")
        reset_btn.setFixedWidth(90)
        reset_btn.clicked.connect(self._reset_view)
        top.addWidget(reset_btn)

        top.addStretch()
        self._info_label = QLabel("Sin datos")
        self._info_label.setStyleSheet("color:#aaa; font-size:11px;")
        top.addWidget(self._info_label)
        outer.addLayout(top)

        # ── Scatter ──
        self._fig    = Figure(facecolor="#1e1e1e")
        self._canvas = FigureCanvas(self._fig)
        self._ax     = self._fig.add_subplot(111)
        self._style_axes()
        outer.addWidget(self._canvas, stretch=4)

        self._canvas.mpl_connect("button_press_event",   self._on_mouse_press)
        self._canvas.mpl_connect("motion_notify_event",  self._pan_move)
        self._canvas.mpl_connect("button_release_event", self._pan_release)
        self._canvas.mpl_connect("scroll_event",         self._on_scroll)

        # ── Strip header ──
        strip_hdr = QHBoxLayout()
        self._strip_lbl = QLabel("Seleccioná con el lazo para ver imágenes aquí")
        self._strip_lbl.setStyleSheet("color:#aaa; font-size:11px; font-style:italic;")
        strip_hdr.addWidget(self._strip_lbl)
        strip_hdr.addStretch()

        self._sel_all_btn = QPushButton("Seleccionar todo")
        self._sel_all_btn.setFixedHeight(24)
        self._sel_all_btn.setEnabled(False)
        self._sel_all_btn.clicked.connect(self._select_all_strip)
        strip_hdr.addWidget(self._sel_all_btn)

        self._desel_btn = QPushButton("Deseleccionar")
        self._desel_btn.setFixedHeight(24)
        self._desel_btn.setEnabled(False)
        self._desel_btn.clicked.connect(self._deselect_all_strip)
        strip_hdr.addWidget(self._desel_btn)

        self._review_btn = QPushButton("Revisar marcadas")
        self._review_btn.setFixedHeight(24)
        self._review_btn.setEnabled(False)
        self._review_btn.setStyleSheet(
            "QPushButton:enabled{background:#1a5c2e;color:white;font-weight:bold;"
            "border-radius:3px;padding:0 10px;}"
        )
        self._review_btn.clicked.connect(lambda: self._batch_status("reviewed"))
        strip_hdr.addWidget(self._review_btn)

        self._discard_btn = QPushButton("Descartar marcadas")
        self._discard_btn.setFixedHeight(24)
        self._discard_btn.setEnabled(False)
        self._discard_btn.setStyleSheet(
            "QPushButton:enabled{background:#8B1A1A;color:white;font-weight:bold;"
            "border-radius:3px;padding:0 10px;}"
        )
        self._discard_btn.clicked.connect(lambda: self._batch_status("discarded"))
        strip_hdr.addWidget(self._discard_btn)

        self._label_btn = QPushButton("Etiquetar ▾")
        self._label_btn.setFixedHeight(24)
        self._label_btn.setEnabled(False)
        self._label_btn.setStyleSheet(
            "QPushButton:enabled{background:#2a4a7a;color:white;font-weight:bold;"
            "border-radius:3px;padding:0 10px;}"
        )
        self._label_btn.clicked.connect(self._show_label_menu)
        strip_hdr.addWidget(self._label_btn)

        self._delete_btn = QPushButton("Eliminar marcadas")
        self._delete_btn.setFixedHeight(24)
        self._delete_btn.setEnabled(False)
        self._delete_btn.setStyleSheet(
            "QPushButton:enabled{background:#5a1a1a;color:white;font-weight:bold;"
            "border-radius:3px;padding:0 10px;}"
        )
        self._delete_btn.clicked.connect(self._batch_delete)
        strip_hdr.addWidget(self._delete_btn)

        outer.addLayout(strip_hdr)

        # ── Strip scroll ──
        self._strip_scroll = QScrollArea()
        self._strip_scroll.setFixedHeight(170)
        self._strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._strip_scroll.setWidgetResizable(True)
        self._strip_container = QWidget()
        self._strip_layout    = QHBoxLayout(self._strip_container)
        self._strip_layout.setContentsMargins(4, 4, 4, 4)
        self._strip_layout.setSpacing(4)
        self._strip_layout.addStretch()
        self._strip_scroll.setWidget(self._strip_container)
        outer.addWidget(self._strip_scroll)

    def _style_axes(self):
        self._ax.set_facecolor("#1a1a1a")
        self._ax.tick_params(colors="#555")
        for sp in self._ax.spines.values():
            sp.set_color("#333")
        self._ax.set_title(
            "Vista UMAP — clusters por similitud  |  Lazo para seleccionar",
            color="#ccc", fontsize=11,
        )

    # ─── Public API ──────────────────────────────────────────────────

    def load_projections(self, proj_rows: list, cluster_colors: dict[int, str]):
        """proj_rows: (image_id, x, y, filename, thumbnail_path, cluster_id, status)"""
        self._proj_rows      = proj_rows
        self._cluster_colors = cluster_colors
        if not proj_rows or not HAS_MPL:
            return
        self._image_ids      = [r[0] for r in proj_rows]
        self._xs             = np.array([r[1] for r in proj_rows])
        self._ys             = np.array([r[2] for r in proj_rows])
        self._statuses       = [r[6] for r in proj_rows]
        self._selected_mask  = np.zeros(len(proj_rows), dtype=bool)
        self._redraw()

    def apply_status_filter(self, status: str):
        self._status_filter = status
        if HAS_MPL and self._proj_rows:
            self._redraw()

    def highlight_point(self, image_id: int):
        if not self._proj_rows or not HAS_MPL:
            return
        try:
            idx = self._image_ids.index(image_id)
        except ValueError:
            return
        self._ax.plot(
            self._xs[idx], self._ys[idx], "o",
            markersize=14, markerfacecolor="none",
            markeredgecolor="#FFD700", markeredgewidth=2,
        )
        self._canvas.draw()

    def update_image_status(self, image_id: int, status: str):
        """Sync status tag on strip thumbnail after external change."""
        try:
            idx = self._image_ids.index(image_id)
            self._statuses[idx] = status
        except ValueError:
            pass
        for thumb in self._strip_thumbs:
            if thumb.image_id == image_id:
                thumb.update_status(status)

    # ─── Scatter ─────────────────────────────────────────────────────

    def _redraw(self):
        self._ax.clear()
        self._style_axes()
        self._scatter = None

        if not self._proj_rows:
            self._canvas.draw()
            return

        colors = [self._cluster_colors.get(r[5], "#808080") for r in self._proj_rows]

        if self._status_filter == "todas":
            match = np.ones(len(self._xs), dtype=bool)
        else:
            match = np.array([s == self._status_filter for s in self._statuses])

        # Dim non-matching
        if (~match).any():
            self._ax.scatter(
                self._xs[~match], self._ys[~match],
                c=[colors[i] for i, m in enumerate(match) if not m],
                s=10, alpha=0.07, linewidths=0, zorder=1,
            )

        # Draw matching
        if match.any():
            edge_clrs = ["#4fc3f7" if self._selected_mask[i] else "#333"
                         for i, m in enumerate(match) if m]
            edge_w    = [1.8 if self._selected_mask[i] else 0.5
                         for i, m in enumerate(match) if m]
            self._scatter = self._ax.scatter(
                self._xs[match], self._ys[match],
                c=[colors[i] for i, m in enumerate(match) if m],
                s=45, alpha=0.85, linewidths=edge_w, edgecolors=edge_clrs,
                zorder=2,
            )

        n_sel   = int(self._selected_mask.sum())
        n_match = int(match.sum())
        title   = f"Vista UMAP — {n_match} imagen(es)"
        if self._status_filter != "todas":
            title += f"  ({self._status_filter})"
        if n_sel:
            title += f"  |  {n_sel} en lazo"
        self._ax.set_title(title, color="#ccc", fontsize=11)
        self._canvas.draw()

    # ─── Lasso ───────────────────────────────────────────────────────

    def _toggle_lasso(self, active: bool):
        self._lasso_active = active
        if active:
            self._lasso = LassoSelector(
                self._ax, self._on_lasso_complete,
                props={"color": "#4fc3f7", "linewidth": 1.5},
                useblit=True,
            )
        else:
            if self._lasso:
                self._lasso.disconnect_events()
                self._lasso = None

    def _on_lasso_complete(self, verts):
        if len(verts) < 3 or len(self._xs) == 0:
            return
        path = MplPath(verts)
        points = np.column_stack([self._xs, self._ys])
        in_lasso = path.contains_points(points)

        if self._status_filter != "todas":
            status_ok = np.array([s == self._status_filter for s in self._statuses])
            in_lasso  = in_lasso & status_ok

        self._selected_mask = in_lasso
        self._redraw()
        self._populate_strip(np.where(in_lasso)[0])
        self.lasso_selection_changed.emit(self._lasso_image_ids())

    def _lasso_image_ids(self) -> list[int]:
        indices = np.where(self._selected_mask)[0]
        seen: set[int] = set()
        ids: list[int] = []
        for i in indices:
            img_id = self._image_ids[i]
            if img_id not in seen:
                seen.add(img_id)
                ids.append(img_id)
        return ids

    def _clear_lasso(self):
        self._selected_mask = np.zeros(len(self._xs), dtype=bool)
        self._redraw()
        self._clear_strip()
        self.lasso_selection_changed.emit([])

    # ─── Strip ───────────────────────────────────────────────────────

    def _populate_strip(self, indices):
        self._clear_strip()
        seen: set[int] = set()
        for idx in indices:
            img_id   = self._image_ids[idx]
            if img_id in seen:
                continue
            seen.add(img_id)
            row      = self._proj_rows[idx]
            fname    = row[3]
            thumb_p  = row[4] or ""
            status   = self._statuses[idx]
            chip_label, chip_color = self._get_chip(img_id)
            thumb = SelectableThumb(img_id, thumb_p, fname, status,
                                    chip_label, chip_color)
            thumb.clicked.connect(self._on_thumb_click)
            thumb.check_changed.connect(self._on_thumb_check)
            self._strip_layout.insertWidget(
                self._strip_layout.count() - 1, thumb
            )
            self._strip_thumbs.append(thumb)

        n = len(self._strip_thumbs)
        self._strip_lbl.setText(
            f"{n} imagen{'es' if n != 1 else ''} en lazo"
            if n else "Seleccioná con el lazo para ver imágenes aquí"
        )
        has = n > 0
        self._sel_all_btn.setEnabled(has)
        self._desel_btn.setEnabled(has)
        self._update_action_btns()

    def _clear_strip(self):
        self._strip_thumbs.clear()
        self._checked_ids.clear()
        while self._strip_layout.count() > 1:
            item = self._strip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._sel_all_btn.setEnabled(False)
        self._desel_btn.setEnabled(False)
        self._label_btn.setEnabled(False)
        self._delete_btn.setEnabled(False)
        self._update_action_btns()

    def _on_thumb_click(self, image_id: int):
        self.image_clicked.emit(image_id)

    def _on_thumb_check(self, image_id: int, checked: bool):
        if checked:
            self._checked_ids.add(image_id)
        else:
            self._checked_ids.discard(image_id)
        self._update_action_btns()

    def _select_all_strip(self):
        for t in self._strip_thumbs:
            t.set_checked(True)

    def _deselect_all_strip(self):
        for t in self._strip_thumbs:
            t.set_checked(False)

    def _update_action_btns(self):
        n = len(self._checked_ids)
        lbl = f"{n}" if n else ""
        self._review_btn.setEnabled(n > 0)
        self._review_btn.setText(f"Revisar {lbl}" if lbl else "Revisar marcadas")
        self._discard_btn.setEnabled(n > 0)
        self._discard_btn.setText(f"Descartar {lbl}" if lbl else "Descartar marcadas")
        self._label_btn.setEnabled(n > 0 and bool(self._classes))
        self._label_btn.setText(f"Etiquetar {lbl} ▾" if lbl else "Etiquetar ▾")
        self._delete_btn.setEnabled(n > 0)
        self._delete_btn.setText(f"Eliminar {lbl}" if lbl else "Eliminar marcadas")

    def set_classes(self, classes: list):
        """Receive [(id, name, color), ...] from main window."""
        self._classes = classes

    def set_ann_cache(self, ann_cache: dict):
        """Receive {image_id: [(x,y,w,h,color,name),...]} from main window."""
        self._ann_cache = ann_cache

    def _get_chip(self, image_id: int) -> tuple[str, str]:
        """Return (class_name, class_color) for classification chip, or ('','')."""
        anns = self._ann_cache.get(image_id, [])
        for ann in anns:
            if len(ann) >= 6 and ann[2] >= 0.99 and ann[3] >= 0.99:
                return ann[5], ann[4]
        return "", ""

    def _batch_status(self, status: str):
        if not self._checked_ids:
            return
        ids = list(self._checked_ids)
        for thumb in self._strip_thumbs:
            if thumb.image_id in self._checked_ids:
                thumb.update_status(status)
        for i, img_id in enumerate(self._image_ids):
            if img_id in self._checked_ids:
                self._statuses[i] = status
        self._checked_ids.clear()
        self._update_action_btns()
        self._deselect_all_strip()
        self._redraw()
        self.images_status_changed.emit(ids, status)

    def _show_label_menu(self):
        from PySide6.QtWidgets import QMenu
        if not self._classes or not self._checked_ids:
            return
        menu = QMenu(self)
        for class_id, name, color in self._classes:
            action = menu.addAction(f"● {name}")
            action.setData((class_id, name, color))
        chosen = menu.exec(self._label_btn.mapToGlobal(
            self._label_btn.rect().bottomLeft()
        ))
        if chosen and chosen.data() is not None:
            class_id, name, color = chosen.data()
            ids = list(self._checked_ids)
            # Update chip in strip thumbs
            for thumb in self._strip_thumbs:
                if thumb.image_id in self._checked_ids:
                    thumb.update_label(name, color)
            # Update local ann_cache
            for img_id in ids:
                self._ann_cache[img_id] = [(0.0, 0.0, 1.0, 1.0, color, name)]
            self._checked_ids.clear()
            self._update_action_btns()
            self._deselect_all_strip()
            self.images_label_changed.emit(ids, class_id)

    def _batch_delete(self):
        if not self._checked_ids:
            return
        ids = list(self._checked_ids)
        self._checked_ids.clear()
        self._update_action_btns()
        self.images_delete_requested.emit(ids)

    # ─── Click / pan / zoom ──────────────────────────────────────────

    def _on_mouse_press(self, event):
        if event.button == 3 and event.xdata is not None and not self._lasso_active:
            self._is_panning = True
            self._pan_start  = (event.xdata, event.ydata)
            self._pan_xlim   = list(self._ax.get_xlim())
            self._pan_ylim   = list(self._ax.get_ylim())
        if event.button == 1 and not self._lasso_active:
            self._on_click(event)

    def _on_click(self, event):
        if event.xdata is None or len(self._xs) == 0:
            return
        ax_range = self._ax.get_xlim()
        radius   = (ax_range[1] - ax_range[0]) * 0.025
        dists    = np.sqrt((self._xs - event.xdata)**2 + (self._ys - event.ydata)**2)
        for nearest in np.argsort(dists):
            if dists[nearest] >= radius:
                break
            if (self._status_filter != "todas"
                    and self._statuses[nearest] != self._status_filter):
                continue
            img_id = self._image_ids[nearest]
            self._info_label.setText(
                f"{self._proj_rows[nearest][3]}  |  {self._statuses[nearest]}"
            )
            self.highlight_point(img_id)
            self.image_clicked.emit(img_id)
            break

    def _pan_move(self, event):
        if self._is_panning and event.xdata is not None and self._pan_start:
            dx = self._pan_start[0] - event.xdata
            dy = self._pan_start[1] - event.ydata
            self._ax.set_xlim([x + dx for x in self._pan_xlim])
            self._ax.set_ylim([y + dy for y in self._pan_ylim])
            self._canvas.draw()

    def _pan_release(self, event):
        if event.button == 3:
            self._is_panning = False

    def _on_scroll(self, event):
        if event.xdata is None:
            return
        f  = 0.85 if event.button == "up" else 1.15
        cx, cy = event.xdata, event.ydata
        self._ax.set_xlim([cx + (x - cx)*f for x in self._ax.get_xlim()])
        self._ax.set_ylim([cy + (y - cy)*f for y in self._ax.get_ylim()])
        self._canvas.draw()

    def _reset_view(self):
        if HAS_MPL:
            self._ax.autoscale()
            self._canvas.draw()
