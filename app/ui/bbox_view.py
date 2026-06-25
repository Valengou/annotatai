"""
BBoxOutlierView — scatter plot of bbox crop embeddings with:
- Lasso selection tool to box-select outlier points
- Crop thumbnail strip with checkboxes for batch operations
- Batch discard of selected images
"""

import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QCheckBox, QSizePolicy, QComboBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QColor

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.cm as cm
    from matplotlib.widgets import LassoSelector
    from matplotlib.path import Path as MplPath
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from ..core.bbox_embeddings import crop_bbox


def _pil_to_qpixmap(pil_img, w: int, h: int) -> QPixmap:
    pil_img = pil_img.convert("RGB").resize((w, h))
    data = pil_img.tobytes("raw", "RGB")
    qi = QImage(data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qi)


# ─────────────────────────── CropThumb ───────────────────────────────

class CropThumb(QFrame):
    clicked       = Signal(int)        # annotation_id
    check_changed = Signal(int, bool)  # annotation_id, checked

    SIZE = 90

    STATUS_CFG = {
        "pending":   ("Pendiente", "#808080"),
        "reviewed":  ("Revisada",  "#2ECC71"),
        "discarded": ("Descartada","#E74C3C"),
    }

    def __init__(self, annotation_id: int, outlier_score: float,
                 image_id: int, img_status: str, pil_crop, parent=None):
        super().__init__(parent)
        self.annotation_id = annotation_id
        self.image_id      = image_id
        self.outlier_score = outlier_score
        self.img_status    = img_status
        self.setFixedSize(self.SIZE + 4, self.SIZE + 48)
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)

        r = int(min(255, outlier_score * 510))
        g = int(min(255, (1 - outlier_score) * 510))
        self._border_color = f"rgb({r},{g},50)"
        self._apply_style(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        # checkbox + score row
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self._cb = QCheckBox()
        self._cb.setFixedSize(16, 16)
        self._cb.toggled.connect(lambda v: self.check_changed.emit(self.annotation_id, v))
        top.addWidget(self._cb)
        top.addStretch()
        score_lbl = QLabel(f"{outlier_score:.2f}")
        score_lbl.setStyleSheet(f"color:{self._border_color}; font-size:9px;")
        top.addWidget(score_lbl)
        layout.addLayout(top)

        thumb = QLabel()
        thumb.setFixedSize(self.SIZE, self.SIZE - 4)
        thumb.setAlignment(Qt.AlignCenter)
        if pil_crop:
            pix = _pil_to_qpixmap(pil_crop, self.SIZE, self.SIZE - 4)
            thumb.setPixmap(pix)
        else:
            thumb.setText("?")
            thumb.setStyleSheet("color:#555;")
        layout.addWidget(thumb)

        # status tag
        tag_text, tag_color = self.STATUS_CFG.get(img_status, ("?", "#555"))
        self._status_tag = QLabel(f"● {tag_text}")
        self._status_tag.setAlignment(Qt.AlignCenter)
        self._status_tag.setStyleSheet(
            f"color:{tag_color}; font-size:8px; font-weight:bold;"
        )
        layout.addWidget(self._status_tag)

    def _apply_style(self, checked: bool):
        border = f"2px solid {'#4fc3f7' if checked else self._border_color}"
        self.setStyleSheet(
            f"QFrame{{background:#1e1e1e; border:{border}; border-radius:4px;}}"
        )

    def set_checked(self, v: bool):
        self._cb.setChecked(v)
        self._apply_style(v)

    def is_checked(self) -> bool:
        return self._cb.isChecked()

    def update_status(self, status: str):
        self.img_status = status
        tag_text, tag_color = self.STATUS_CFG.get(status, ("?", "#555"))
        self._status_tag.setText(f"● {tag_text}")
        self._status_tag.setStyleSheet(
            f"color:{tag_color}; font-size:8px; font-weight:bold;"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._cb.underMouse():
            self.clicked.emit(self.annotation_id)
        super().mousePressEvent(event)


# ─────────────────────────── BBoxOutlierView ─────────────────────────

class BBoxOutlierView(QWidget):
    open_image_requested  = Signal(int, int)   # image_id, annotation_id
    images_status_changed = Signal(list, str)  # [image_id,...], status
    lasso_selection_changed = Signal(list)     # [image_id,...] ordered; empty = cleared

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proj_rows: list = []
        self._ann_ids  = np.array([], dtype=int)
        self._xs       = np.array([])
        self._ys       = np.array([])
        self._scores   = np.array([])
        self._statuses: list[str] = []
        self._status_filter: str = "todas"
        self._classes_per_point: list[str] = []
        self._class_filter: str | None = None
        self._discarded_ann_ids: set[int] = set()

        self._scatter       = None
        self._highlight_art = None
        self._lasso         = None
        self._lasso_active  = False
        self._selected_mask = np.array([], dtype=bool)  # points selected by lasso

        self._strip_thumbs: list[CropThumb] = []
        self._checked_ann_ids: set[int] = set()
        self._colorbar = None

        self._setup_ui()

    # ─── UI ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        if not HAS_MPL:
            outer.addWidget(QLabel("matplotlib no disponible"))
            return

        # ── Top toolbar ──
        top = QHBoxLayout()

        self._lasso_btn = QPushButton("Lazo")
        self._lasso_btn.setCheckable(True)
        self._lasso_btn.setFixedWidth(70)
        self._lasso_btn.setToolTip(
            "Activar lazo: dibujá una selección libre sobre el scatter.\n"
            "Desactivar para volver al modo click-para-abrir."
        )
        self._lasso_btn.toggled.connect(self._toggle_lasso)
        self._lasso_btn.setStyleSheet("""
            QPushButton { border: 1px solid #888; border-radius:4px; padding:3px 8px; }
            QPushButton:checked { background:#1a5c8c; border-color:#4fc3f7; color:#fff; font-weight:bold; }
        """)
        top.addWidget(self._lasso_btn)

        clear_sel_btn = QPushButton("Limpiar selección")
        clear_sel_btn.setFixedWidth(130)
        clear_sel_btn.clicked.connect(self._clear_lasso_selection)
        top.addWidget(clear_sel_btn)

        cls_lbl = QLabel("Clase:")
        cls_lbl.setStyleSheet("color:#999; font-size:11px; margin-left:10px;")
        top.addWidget(cls_lbl)
        self._class_combo = QComboBox()
        self._class_combo.addItem("Todas", None)
        self._class_combo.currentIndexChanged.connect(self._on_class_filter_changed)
        top.addWidget(self._class_combo)

        top.addStretch()
        self._info_lbl = QLabel("Sin datos — generá embeddings de bboxes primero")
        self._info_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        top.addWidget(self._info_lbl)
        outer.addLayout(top)

        # ── Scatter ──
        self._fig = Figure(facecolor="#1e1e1e")
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._style_ax()
        outer.addWidget(self._canvas, stretch=4)

        self._canvas.mpl_connect("button_press_event",   self._on_click)
        self._canvas.mpl_connect("scroll_event",         self._on_scroll)
        self._canvas.mpl_connect("button_press_event",   self._pan_press)
        self._canvas.mpl_connect("motion_notify_event",  self._pan_move)
        self._canvas.mpl_connect("button_release_event", self._pan_release)
        self._is_panning  = False
        self._pan_start   = None
        self._pan_xlim    = None
        self._pan_ylim    = None

        # ── Strip header ──
        strip_hdr = QHBoxLayout()
        self._strip_lbl = QLabel("Tira de crops:")
        self._strip_lbl.setStyleSheet("color:#e74c3c; font-weight:bold; font-size:11px;")
        strip_hdr.addWidget(self._strip_lbl)
        strip_hdr.addStretch()

        sel_all_btn = QPushButton("Seleccionar todo")
        sel_all_btn.setFixedHeight(24)
        sel_all_btn.clicked.connect(self._select_all_strip)
        strip_hdr.addWidget(sel_all_btn)

        desel_btn = QPushButton("Deseleccionar")
        desel_btn.setFixedHeight(24)
        desel_btn.clicked.connect(self._deselect_all_strip)
        strip_hdr.addWidget(desel_btn)

        self._review_btn = QPushButton("Revisar marcadas")
        self._review_btn.setFixedHeight(24)
        self._review_btn.setEnabled(False)
        self._review_btn.setStyleSheet(
            "QPushButton:enabled{background:#1a5c2e;color:white;font-weight:bold;"
            "border-radius:3px;padding:0 10px;}"
        )
        self._review_btn.clicked.connect(self._review_checked)
        strip_hdr.addWidget(self._review_btn)

        self._discard_btn = QPushButton("Descartar marcadas")
        self._discard_btn.setFixedHeight(24)
        self._discard_btn.setEnabled(False)
        self._discard_btn.setStyleSheet(
            "QPushButton:enabled{background:#8B1A1A;color:white;font-weight:bold;"
            "border-radius:3px;padding:0 10px;}"
        )
        self._discard_btn.clicked.connect(self._discard_checked)
        strip_hdr.addWidget(self._discard_btn)

        outer.addLayout(strip_hdr)

        # ── Strip scroll ──
        self._strip_scroll = QScrollArea()
        self._strip_scroll.setFixedHeight(140)
        self._strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._strip_scroll.setWidgetResizable(True)
        self._strip_container = QWidget()
        self._strip_layout = QHBoxLayout(self._strip_container)
        self._strip_layout.setContentsMargins(4, 4, 4, 4)
        self._strip_layout.setSpacing(4)
        self._strip_layout.addStretch()
        self._strip_scroll.setWidget(self._strip_container)
        outer.addWidget(self._strip_scroll)

    def _style_ax(self):
        self._ax.set_facecolor("#1a1a1a")
        self._ax.tick_params(colors="#555")
        for sp in self._ax.spines.values():
            sp.set_color("#333")
        self._ax.set_title(
            "UMAP crops de bboxes — rojo=outlier  |  Lazo para seleccionar",
            color="#ccc", fontsize=11,
        )

    # ─── Public API ──────────────────────────────────────────────────

    def load_projections(self, proj_rows: list):
        self._proj_rows = proj_rows
        if not proj_rows or not HAS_MPL:
            return
        self._ann_ids       = np.array([r[0] for r in proj_rows], dtype=int)
        self._xs            = np.array([r[1] for r in proj_rows])
        self._ys            = np.array([r[2] for r in proj_rows])
        self._scores        = np.array([r[3] for r in proj_rows])
        self._statuses      = [r[11] for r in proj_rows]   # image status
        self._classes_per_point = [r[12] for r in proj_rows]  # class name
        self._selected_mask = np.zeros(len(proj_rows), dtype=bool)
        self._discarded_ann_ids.clear()
        self._redraw_scatter()
        self._populate_strip(self._top_visible_indices(50))

    def apply_status_filter(self, status: str):
        self._status_filter = status
        if self._proj_rows and HAS_MPL:
            self._redraw_scatter()
            self._populate_strip(self._top_visible_indices(50))

    def set_classes(self, classes: list):
        """Llena el combo de clases (id, name, color)."""
        if not HAS_MPL:
            return
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.addItem("Todas", None)
        for _cid, name, _color in classes:
            self._class_combo.addItem(name, name)
        self._class_combo.blockSignals(False)
        self._class_filter = None

    def _on_class_filter_changed(self):
        self._class_filter = self._class_combo.currentData()
        if self._proj_rows and HAS_MPL:
            self._redraw_scatter()
            self._populate_strip(self._top_visible_indices(50))

    def _visible_mask(self) -> np.ndarray:
        n = len(self._xs)
        if n == 0:
            return np.array([], dtype=bool)
        discarded = np.isin(self._ann_ids, list(self._discarded_ann_ids))
        status_ok = (np.ones(n, dtype=bool)
                     if self._status_filter == "todas" or not self._statuses
                     else np.array([s == self._status_filter for s in self._statuses]))
        class_ok = (np.ones(n, dtype=bool)
                    if self._class_filter is None or not self._classes_per_point
                    else np.array([c == self._class_filter for c in self._classes_per_point]))
        return status_ok & class_ok & ~discarded

    def _top_visible_indices(self, n: int) -> np.ndarray:
        vis = np.where(self._visible_mask())[0]
        return vis[np.argsort(self._scores[vis])[::-1]][:n]

    def set_db(self, db):
        self._db = db

    # ─── Scatter ─────────────────────────────────────────────────────

    def _redraw_scatter(self):
        self._ax.clear()
        self._style_ax()
        self._scatter = None
        self._highlight_art = None

        if len(self._xs) == 0:
            self._canvas.draw()
            return

        colors_rgba = cm.RdYlGn_r(Normalize(0, 1)(self._scores))

        # visible = matches status + class filters AND not discarded
        visible = self._visible_mask()
        dim     = ~visible

        # Draw dim background first
        if dim.any():
            self._ax.scatter(
                self._xs[dim], self._ys[dim],
                c=colors_rgba[dim], s=8, alpha=0.07, linewidths=0,
            )

        # Draw visible points on top
        if visible.any():
            edge_clrs = ["#4fc3f7" if self._selected_mask[i] else "#222"
                         for i in range(len(self._xs)) if visible[i]]
            edge_w    = [1.5 if self._selected_mask[i] else 0.3
                         for i in range(len(self._xs)) if visible[i]]
            self._scatter = self._ax.scatter(
                self._xs[visible], self._ys[visible],
                c=colors_rgba[visible], s=25, alpha=0.85,
                linewidths=edge_w, edgecolors=edge_clrs,
            )

        n_sel = int(self._selected_mask.sum())
        n_out = int(np.sum(self._scores >= 0.5))
        n_vis = int(visible.sum())
        info  = f"{len(self._ann_ids)} crops  —  {n_out} outliers (score≥0.5)"
        if self._status_filter != "todas":
            info += f"  —  {n_vis} visibles ({self._status_filter})"
        if n_sel:
            info += f"  —  {n_sel} en lazo"
        self._info_lbl.setText(info)

        if self._colorbar is None:
            sm = ScalarMappable(cmap=cm.RdYlGn_r, norm=Normalize(0, 1))
            sm.set_array([])
            try:
                self._colorbar = self._fig.colorbar(
                    sm, ax=self._ax, label="Outlier score", fraction=0.03
                )
            except Exception:
                pass

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

        # Restrict to currently visible (status + class filters)
        in_lasso = in_lasso & self._visible_mask()

        self._selected_mask = in_lasso
        n = int(self._selected_mask.sum())
        self._redraw_scatter()
        indices = np.where(self._selected_mask)[0]
        indices = indices[np.argsort(self._scores[indices])[::-1]]
        self._populate_strip(indices)
        self._strip_lbl.setText(
            f"Lazo: {n} crop{'s' if n != 1 else ''} seleccionada{'s' if n != 1 else ''}"
        )
        self.lasso_selection_changed.emit(self._get_lasso_image_ids(indices))

    def _get_lasso_image_ids(self, sorted_indices=None) -> list[int]:
        """Unique image_ids of selected crops in outlier-score order (highest first)."""
        if sorted_indices is None:
            idx = np.where(self._selected_mask)[0]
            sorted_indices = idx[np.argsort(self._scores[idx])[::-1]]
        seen: set[int] = set()
        ids: list[int] = []
        for i in sorted_indices:
            img_id = self._proj_rows[i][4]
            if img_id not in seen:
                seen.add(img_id)
                ids.append(img_id)
        return ids

    def _clear_lasso_selection(self):
        self._selected_mask = np.zeros(len(self._xs), dtype=bool)
        self._redraw_scatter()
        self._populate_strip(self._top_visible_indices(50))
        self._strip_lbl.setText("Tira de crops (top-50 por score):")
        self.lasso_selection_changed.emit([])   # clear nav override

    # ─── Strip ───────────────────────────────────────────────────────

    def _populate_strip(self, indices):
        self._strip_thumbs.clear()
        self._checked_ann_ids.clear()

        while self._strip_layout.count() > 1:
            item = self._strip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for idx in indices:
            row = self._proj_rows[idx]
            ann_id, px, py, score, img_id, ax, ay, aw, ah, img_path, fname, img_status, cls_name, cls_color = row
            if ann_id in self._discarded_ann_ids:
                continue
            crop = crop_bbox(Path(img_path), ax, ay, aw, ah)
            thumb = CropThumb(ann_id, score, img_id, img_status, crop)
            thumb.clicked.connect(self._on_thumb_click)
            thumb.check_changed.connect(self._on_thumb_check)
            self._strip_layout.insertWidget(self._strip_layout.count() - 1, thumb)
            self._strip_thumbs.append(thumb)

        self._update_discard_btn()

    def _on_thumb_check(self, ann_id: int, checked: bool):
        if checked:
            self._checked_ann_ids.add(ann_id)
        else:
            self._checked_ann_ids.discard(ann_id)
        self._update_discard_btn()

    def _update_discard_btn(self):
        n = len(self._checked_ann_ids)
        label = f"{n} imagen{'es' if n != 1 else ''}" if n > 0 else ""
        self._review_btn.setEnabled(n > 0)
        self._review_btn.setText(f"Revisar {label}" if label else "Revisar marcadas")
        self._discard_btn.setEnabled(n > 0)
        self._discard_btn.setText(f"Descartar {label}" if label else "Descartar marcadas")

    def _select_all_strip(self):
        for t in self._strip_thumbs:
            t.set_checked(True)

    def _deselect_all_strip(self):
        for t in self._strip_thumbs:
            t.set_checked(False)

    # ─── Discard ─────────────────────────────────────────────────────

    def _review_checked(self):
        self._batch_set_status("reviewed")

    def _discard_checked(self):
        self._batch_set_status("discarded")

    def _batch_set_status(self, status: str):
        if not self._checked_ann_ids:
            return

        ann_to_img = {r[0]: r[4] for r in self._proj_rows}
        image_ids = list({ann_to_img[a] for a in self._checked_ann_ids
                          if a in ann_to_img})

        if status == "discarded":
            self._discarded_ann_ids.update(self._checked_ann_ids)

        # Update status tags before removing
        for thumb in self._strip_thumbs:
            if thumb.annotation_id in self._checked_ann_ids:
                thumb.update_status(status)

        # Remove processed thumbs from strip
        for thumb in list(self._strip_thumbs):
            if thumb.annotation_id in self._checked_ann_ids:
                self._strip_thumbs.remove(thumb)
                thumb.deleteLater()

        self._checked_ann_ids.clear()
        self._update_discard_btn()
        self._redraw_scatter()
        self.images_status_changed.emit(image_ids, status)

    # ─── Click (non-lasso mode) ───────────────────────────────────────

    def _on_click(self, event):
        if self._lasso_active or event.button != 1 or event.xdata is None:
            return
        if len(self._xs) == 0 or not self._scatter:
            return
        ax_range = self._ax.get_xlim()
        radius = (ax_range[1] - ax_range[0]) * 0.025
        dists = np.sqrt((self._xs - event.xdata)**2 + (self._ys - event.ydata)**2)
        nearest = int(np.argmin(dists))
        if dists[nearest] > radius:
            return
        row = self._proj_rows[nearest]
        ann_id, px, py, score, img_id = row[0], row[1], row[2], row[3], row[4]
        self._highlight(px, py)
        self._info_lbl.setText(
            f"{row[10]}  |  {row[12]}  |  score: {score:.3f}  |  {row[11]}"
        )
        self.open_image_requested.emit(img_id, ann_id)

    def _on_thumb_click(self, ann_id: int):
        row = next((r for r in self._proj_rows if r[0] == ann_id), None)
        if row:
            self._highlight(row[1], row[2])
            self.open_image_requested.emit(row[4], ann_id)

    def _highlight(self, x: float, y: float):
        if self._highlight_art:
            try:
                self._highlight_art.remove()
            except Exception:
                pass
        self._highlight_art = self._ax.plot(
            x, y, "o", markersize=14,
            markerfacecolor="none",
            markeredgecolor="#FFD700", markeredgewidth=2,
        )[0]
        self._canvas.draw()

    # ─── Zoom / pan ──────────────────────────────────────────────────

    def _on_scroll(self, event):
        if event.xdata is None:
            return
        f = 0.85 if event.button == "up" else 1.15
        cx, cy = event.xdata, event.ydata
        self._ax.set_xlim([cx + (x - cx)*f for x in self._ax.get_xlim()])
        self._ax.set_ylim([cy + (y - cy)*f for y in self._ax.get_ylim()])
        self._canvas.draw()

    def _pan_press(self, event):
        if event.button == 3 and event.xdata is not None and not self._lasso_active:
            self._is_panning = True
            self._pan_start  = (event.xdata, event.ydata)
            self._pan_xlim   = list(self._ax.get_xlim())
            self._pan_ylim   = list(self._ax.get_ylim())

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
