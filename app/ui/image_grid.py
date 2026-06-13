from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QScrollArea, QVBoxLayout, QHBoxLayout, QLabel,
    QGridLayout, QFrame, QMenu, QCheckBox, QPushButton, QButtonGroup,
    QComboBox, QDoubleSpinBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QColor, QPainter, QPen, QBrush, QFont

from ..models.image_item import ImageItem
from ..utils.config import STATUS_COLORS


class ThumbnailCard(QFrame):
    double_clicked = Signal(int)    # image_id
    status_changed = Signal(int, str)
    check_changed  = Signal(int, bool)  # image_id, checked

    THUMB_W = 160
    THUMB_H = 120

    def __init__(self, image: ImageItem, parent=None):
        super().__init__(parent)
        self.image = image
        self.setFixedSize(self.THUMB_W + 8, self.THUMB_H + 62)
        self.setFrameShape(QFrame.StyledPanel)
        self._checked     = False
        self._highlighted = False
        self._base_pix    = None
        self._annotations: list = []
        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)

        # ── Top row: checkbox + status dot ──
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self._checkbox = QCheckBox()
        self._checkbox.setFixedSize(18, 18)
        self._checkbox.setToolTip("Seleccionar imagen")
        self._checkbox.toggled.connect(self._on_check)
        top.addWidget(self._checkbox)
        top.addStretch()
        self._status_dot = QLabel("●")
        self._status_dot.setFixedWidth(14)
        top.addWidget(self._status_dot)
        layout.addLayout(top)

        # ── Thumbnail ──
        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(self.THUMB_W, self.THUMB_H)
        self._thumb_label.setAlignment(Qt.AlignCenter)
        self._load_thumbnail()
        layout.addWidget(self._thumb_label)

        # ── Filename ──
        name = self.image.filename
        if len(name) > 22:
            name = name[:10] + "…" + name[-10:]
        self._name_label = QLabel(name)
        self._name_label.setAlignment(Qt.AlignCenter)
        self._name_label.setFixedWidth(self.THUMB_W)
        font = self._name_label.font()
        font.setPointSize(8)
        self._name_label.setFont(font)
        layout.addWidget(self._name_label)

        metrics = self._metrics_text()
        self._metrics_label = QLabel(metrics)
        self._metrics_label.setAlignment(Qt.AlignCenter)
        self._metrics_label.setFixedWidth(self.THUMB_W)
        metric_font = self._metrics_label.font()
        metric_font.setPointSize(7)
        self._metrics_label.setFont(metric_font)
        self._metrics_label.setStyleSheet("color:#9cc7ff;")
        self._metrics_label.setToolTip(self._metrics_tooltip())
        layout.addWidget(self._metrics_label)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def _load_thumbnail(self):
        if self.image.thumbnail_path and Path(self.image.thumbnail_path).exists():
            self._base_pix = QPixmap(self.image.thumbnail_path).scaled(
                self.THUMB_W, self.THUMB_H,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._thumb_label.setPixmap(self._base_pix)
        else:
            self._base_pix = None
            self._thumb_label.setText("Sin thumbnail")
            self._thumb_label.setStyleSheet("background:#2a2a2a; color:#666;")

    def set_annotations(self, annotations: list):
        """annotations: [(x, y, w, h, color_hex, class_name), ...] in normalized coords."""
        self._annotations = annotations
        self._draw_bboxes()

    def _draw_bboxes(self):
        if not self._base_pix:
            return
        if not self._annotations:
            self._thumb_label.setPixmap(self._base_pix)
            return
        pix = QPixmap(self._base_pix)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w, h = pix.width(), pix.height()
        for ann in self._annotations:
            x, y, bw, bh, color_hex = ann[0], ann[1], ann[2], ann[3], ann[4]
            label = ann[5] if len(ann) > 5 else ""
            color = QColor(color_hex)
            is_classification = (bw >= 0.99 and bh >= 0.99)

            if is_classification:
                # Draw colored label chip at bottom of thumbnail
                chip_h = 18
                chip_y = h - chip_h
                bg = QColor(color_hex)
                bg.setAlpha(210)
                painter.fillRect(0, chip_y, w, chip_h, bg)
                painter.setPen(QPen(Qt.white))
                font = QFont("Arial", 8, QFont.Bold)
                painter.setFont(font)
                conf = ann[6] if len(ann) > 6 else None
                if conf is not None:
                    try:
                        label = f"{label} {float(conf):.0%}"
                    except Exception:
                        pass
                painter.drawText(0, chip_y, w, chip_h, Qt.AlignCenter, label)
                # Colored border
                painter.setPen(QPen(color, 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(1, 1, w - 2, h - 2)
            else:
                fill = QColor(color_hex)
                fill.setAlpha(35)
                painter.fillRect(int(x * w), int(y * h), int(bw * w), int(bh * h), fill)
                painter.setPen(QPen(color, 1.5))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(int(x * w), int(y * h), int(bw * w), int(bh * h))
        painter.end()
        self._thumb_label.setPixmap(pix)

    def _apply_style(self):
        status_color = STATUS_COLORS.get(self.image.status, "#555")
        self._status_dot.setStyleSheet(f"color: {status_color}; font-size: 10px;")

        if self._checked:
            border = "2px solid #4fc3f7"
            bg = "#1a2a35"
        elif self._highlighted:
            border = "2px solid #FFD700"
            bg = "#2a2a1a"
        else:
            border = f"1px solid {status_color}"
            bg = "#252525"

        self.setStyleSheet(f"""
            ThumbnailCard {{
                background: {bg};
                border: {border};
                border-radius: 5px;
            }}
            ThumbnailCard:hover {{
                border: 2px solid #5a9fd4;
                background: #2d2d2d;
            }}
            QLabel {{ color: #ccc; }}
        """)

    def _on_check(self, checked: bool):
        self._checked = checked
        self._apply_style()
        self.check_changed.emit(self.image.id, checked)

    def set_checked(self, checked: bool):
        self._checkbox.setChecked(checked)  # triggers _on_check via signal

    def set_highlighted(self, highlighted: bool):
        self._highlighted = highlighted
        self._apply_style()

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit(self.image.id)

    def mousePressEvent(self, event):
        # Single click toggles checkbox
        if event.button() == Qt.LeftButton and not self._checkbox.underMouse():
            self._checkbox.setChecked(not self._checkbox.isChecked())
        super().mousePressEvent(event)

    def _context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("Abrir editor", lambda: self.double_clicked.emit(self.image.id))
        menu.addSeparator()
        menu.addAction("Marcar como revisada", lambda: self._set_status("reviewed"))
        menu.addAction("Marcar como pendiente", lambda: self._set_status("pending"))
        menu.addAction("Descartar",             lambda: self._set_status("discarded"))
        menu.exec(self.mapToGlobal(pos))

    def _set_status(self, status: str):
        self.image.status = status
        self._apply_style()
        self.status_changed.emit(self.image.id, status)

    def _fmt_conf(self, value):
        if value is None:
            return "--"
        try:
            return f"{float(value):.2f}"
        except Exception:
            return "--"

    def _metrics_text(self) -> str:
        return (
            f"det {self._fmt_conf(self.image.detection_min_confidence)}/"
            f"{self._fmt_conf(self.image.detection_avg_confidence)} "
            f"cls {self._fmt_conf(self.image.classifier_confidence)}"
        )

    def _metrics_tooltip(self) -> str:
        return "\n".join(
            [
                f"Deteccion min: {self._fmt_conf(self.image.detection_min_confidence)}",
                f"Deteccion avg: {self._fmt_conf(self.image.detection_avg_confidence)}",
                f"Confianza clasificador: {self._fmt_conf(self.image.classifier_confidence)}",
                f"Prioridad revision: {self._fmt_conf(self.image.review_sort_confidence)}",
                f"Razones: {self.image.review_reasons or '-'}",
            ]
        )


_STATUS_FILTERS = ["todas", "pending", "reviewed", "discarded"]
_FILTER_LABELS  = ["Todas", "Pendientes", "Revisadas", "Descartadas"]
_FILTER_COLORS  = ["#444", "#808080", "#2ECC71", "#E74C3C"]


class ImageGrid(QWidget):
    image_opened           = Signal(int)         # image_id
    status_changed         = Signal(int, str)    # image_id, status
    delete_requested       = Signal(list)        # list[int] image_ids
    batch_status_requested = Signal(list, str)   # list[int] image_ids, status
    batch_label_requested  = Signal(list, int)   # list[int] image_ids, class_id

    COLS = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._images: list[ImageItem] = []
        self._cards: dict[int, ThumbnailCard] = {}
        self._checked_ids: set[int] = set()
        self._visible_ids: set[int] = set()
        self._cluster_filter: int = -1
        self._status_filter: str = "todas"
        self._ann_cache: dict[int, list] = {}
        self._classes: list[tuple] = []          # [(id, name, color), ...]
        self._class_filter: str | None = None   # None = todas las clases
        self._sort_mode = "filename"
        self._confidence_filter = "all"
        self._confidence_threshold = 0.50
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Count label ──
        self._count_label = QLabel("")
        self._count_label.setStyleSheet(
            "color:#666; font-size:11px; padding:2px 8px;"
            "background:#1e1e1e; border-bottom:1px solid #333;"
        )
        outer.addWidget(self._count_label)

        sort_bar = QWidget()
        sort_bar.setStyleSheet(
            "background:#1e1e1e; border-bottom:1px solid #333;"
        )
        sort_layout = QHBoxLayout(sort_bar)
        sort_layout.setContentsMargins(8, 3, 8, 3)
        sort_layout.setSpacing(6)
        sort_label = QLabel("Orden:")
        sort_label.setStyleSheet("color:#999; font-size:11px;")
        sort_layout.addWidget(sort_label)
        self._sort_combo = QComboBox()
        self._sort_combo.addItem("Nombre", "filename")
        self._sort_combo.addItem("Prioridad baja conf", "priority")
        self._sort_combo.addItem("Det min baja", "det_min")
        self._sort_combo.addItem("Det avg baja", "det_avg")
        self._sort_combo.addItem("Clasificacion baja", "cls")
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        sort_layout.addWidget(self._sort_combo)
        conf_label = QLabel("Filtro conf:")
        conf_label.setStyleSheet("color:#999; font-size:11px; margin-left:12px;")
        sort_layout.addWidget(conf_label)
        self._confidence_combo = QComboBox()
        self._confidence_combo.addItem("Todas", "all")
        self._confidence_combo.addItem("Prioridad <= umbral", "priority")
        self._confidence_combo.addItem("Det min <= umbral", "det_min")
        self._confidence_combo.addItem("Det avg <= umbral", "det_avg")
        self._confidence_combo.addItem("Clasif <= umbral", "cls")
        self._confidence_combo.currentIndexChanged.connect(self._on_confidence_filter_changed)
        sort_layout.addWidget(self._confidence_combo)
        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.0, 1.0)
        self._confidence_spin.setSingleStep(0.05)
        self._confidence_spin.setDecimals(2)
        self._confidence_spin.setValue(self._confidence_threshold)
        self._confidence_spin.setToolTip("Umbral de confianza para filtrar imagenes dudosas")
        self._confidence_spin.valueChanged.connect(self._on_confidence_threshold_changed)
        sort_layout.addWidget(self._confidence_spin)
        sort_layout.addStretch()
        outer.addWidget(sort_bar)

        # ── Class filter bar (hidden until set_classes() is called) ──
        self._class_filter_bar = QWidget()
        self._class_filter_bar.setStyleSheet(
            "background:#1e1e1e; border-bottom:1px solid #333;"
        )
        self._class_filter_layout = QHBoxLayout(self._class_filter_bar)
        self._class_filter_layout.setContentsMargins(8, 3, 8, 3)
        self._class_filter_layout.setSpacing(4)
        self._class_filter_bar.setVisible(False)
        outer.addWidget(self._class_filter_bar)

        # ── Action bar (visible only when items are checked) ──
        self._action_bar = QWidget()
        self._action_bar.setStyleSheet("background: #1a3a4a; border-bottom: 1px solid #333;")
        bar = QHBoxLayout(self._action_bar)
        bar.setContentsMargins(8, 4, 8, 4)

        self._sel_label = QLabel("0 seleccionadas")
        self._sel_label.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        bar.addWidget(self._sel_label)
        bar.addStretch()

        sel_all_btn = QPushButton("Seleccionar visibles")
        sel_all_btn.setFixedHeight(26)
        sel_all_btn.clicked.connect(self._select_all_visible)
        bar.addWidget(sel_all_btn)

        desel_btn = QPushButton("Deseleccionar todo")
        desel_btn.setFixedHeight(26)
        desel_btn.clicked.connect(self._deselect_all)
        bar.addWidget(desel_btn)

        self._review_btn = QPushButton("Revisar seleccionadas")
        self._review_btn.setFixedHeight(26)
        self._review_btn.setStyleSheet(
            "background: #1a5c2e; color: white; font-weight: bold;"
            "border-radius: 4px; padding: 0 10px;"
        )
        self._review_btn.clicked.connect(
            lambda: self.batch_status_requested.emit(list(self._checked_ids), "reviewed")
        )
        bar.addWidget(self._review_btn)

        self._discard_batch_btn = QPushButton("Descartar seleccionadas")
        self._discard_batch_btn.setFixedHeight(26)
        self._discard_batch_btn.setStyleSheet(
            "background: #5c2e1a; color: white; font-weight: bold;"
            "border-radius: 4px; padding: 0 10px;"
        )
        self._discard_batch_btn.clicked.connect(
            lambda: self.batch_status_requested.emit(list(self._checked_ids), "discarded")
        )
        bar.addWidget(self._discard_batch_btn)

        self._label_btn = QPushButton("Etiquetar ▾")
        self._label_btn.setFixedHeight(26)
        self._label_btn.setStyleSheet(
            "background: #2a4a7a; color: white; font-weight: bold;"
            "border-radius: 4px; padding: 0 10px;"
        )
        self._label_btn.clicked.connect(self._show_label_menu)
        bar.addWidget(self._label_btn)

        self._del_btn = QPushButton("Eliminar del proyecto")
        self._del_btn.setFixedHeight(26)
        self._del_btn.setStyleSheet(
            "background: #8B1A1A; color: white; font-weight: bold;"
            "border-radius: 4px; padding: 0 10px;"
        )
        self._del_btn.clicked.connect(self._emit_delete)
        bar.addWidget(self._del_btn)

        self._action_bar.setVisible(False)
        outer.addWidget(self._action_bar)

        # ── Scrollable grid ──
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self._scroll)

        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(6)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._scroll.setWidget(self._container)

    # ── Load / rebuild ──

    def load_images(self, images: list[ImageItem],
                    ann_cache: dict | None = None):
        self._images = images
        self._cards.clear()
        self._checked_ids.clear()
        if ann_cache is not None:
            self._ann_cache = ann_cache
        self._rebuild_grid()
        self._apply_filters()

    def _sorted_images(self) -> list[ImageItem]:
        def conf_key(value):
            return float(value) if value is not None else 2.0

        if self._sort_mode == "priority":
            return sorted(self._images, key=lambda img: (conf_key(img.review_sort_confidence), img.filename))
        if self._sort_mode == "det_min":
            return sorted(self._images, key=lambda img: (conf_key(img.detection_min_confidence), img.filename))
        if self._sort_mode == "det_avg":
            return sorted(self._images, key=lambda img: (conf_key(img.detection_avg_confidence), img.filename))
        if self._sort_mode == "cls":
            return sorted(self._images, key=lambda img: (conf_key(img.classifier_confidence), img.filename))
        return sorted(self._images, key=lambda img: img.filename)

    def _rebuild_grid(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for idx, img in enumerate(self._sorted_images()):
            card = ThumbnailCard(img)
            card.double_clicked.connect(self.image_opened)
            card.status_changed.connect(self._on_card_status_changed)
            card.check_changed.connect(self._on_card_checked)
            self._cards[img.id] = card
            if img.id in self._ann_cache:
                card.set_annotations(self._ann_cache[img.id])
            row, col = divmod(idx, self.COLS)
            self._grid.addWidget(card, row, col)

    def _on_sort_changed(self):
        self._sort_mode = self._sort_combo.currentData() or "filename"
        self._rebuild_grid()
        self._apply_filters()

    def _on_confidence_filter_changed(self):
        self._confidence_filter = self._confidence_combo.currentData() or "all"
        self._apply_filters()

    def _on_confidence_threshold_changed(self, value: float):
        self._confidence_threshold = float(value)
        self._apply_filters()

    def _confidence_filter_ok(self, image: ImageItem) -> bool:
        if self._confidence_filter == "all":
            return True

        value = None
        if self._confidence_filter == "priority":
            value = image.review_sort_confidence
        elif self._confidence_filter == "det_min":
            value = image.detection_min_confidence
        elif self._confidence_filter == "det_avg":
            value = image.detection_avg_confidence
        elif self._confidence_filter == "cls":
            value = image.classifier_confidence

        if value is None:
            return False
        try:
            return float(value) <= self._confidence_threshold
        except Exception:
            return False

    # ── Combined filter logic ──

    def _apply_filters(self):
        self._visible_ids.clear()
        for img_id, card in self._cards.items():
            cluster_ok = (self._cluster_filter == -1 or
                          card.image.cluster_id == self._cluster_filter)
            status_ok  = (self._status_filter == "todas" or
                          card.image.status == self._status_filter)
            if self._class_filter is None:
                class_ok = True
            else:
                anns = self._ann_cache.get(img_id, [])
                class_ok = any(
                    (ann[5] if len(ann) > 5 else "") == self._class_filter
                    for ann in anns
                )
            confidence_ok = self._confidence_filter_ok(card.image)
            visible = cluster_ok and status_ok and class_ok and confidence_ok
            card.setVisible(visible)
            if visible:
                self._visible_ids.add(img_id)

        # uncheck hidden cards
        for img_id in list(self._checked_ids):
            if img_id not in self._visible_ids:
                self._cards[img_id].set_checked(False)

        self._update_action_bar()
        n = len(self._visible_ids)
        total = len(self._cards)
        self._count_label.setText(f"{n} / {total} imágenes")

    def filter_by_cluster(self, cluster_id: int):
        self._cluster_filter = cluster_id
        self._apply_filters()

    def _set_status_filter(self, status: str):
        self._status_filter = status
        self._apply_filters()

    # ── Card status changed (from context menu inside card) ──

    def _on_card_status_changed(self, image_id: int, status: str):
        self.status_changed.emit(image_id, status)
        # Re-apply filters — card may now disappear from current filter
        self._apply_filters()

    # ── Selection ──

    def _on_card_checked(self, image_id: int, checked: bool):
        if checked:
            self._checked_ids.add(image_id)
        else:
            self._checked_ids.discard(image_id)
        self._update_action_bar()

    def _select_all_visible(self):
        for img_id in self._visible_ids:
            if img_id in self._cards:
                self._cards[img_id].set_checked(True)

    def _deselect_all(self):
        for card in self._cards.values():
            card.set_checked(False)

    def _update_action_bar(self):
        n = len(self._checked_ids)
        self._action_bar.setVisible(n > 0)
        s = f"{n} imagen{'es' if n != 1 else ''}"
        self._sel_label.setText(f"{s} seleccionada{'s' if n != 1 else ''}")
        self._review_btn.setText(f"Revisar {s}")
        self._discard_batch_btn.setText(f"Descartar {s}")
        self._del_btn.setText(f"Eliminar {s}")

    def _emit_delete(self):
        if self._checked_ids:
            self.delete_requested.emit(list(self._checked_ids))

    def refresh_card_annotations(self, image_id: int, annotations: list):
        """Update bbox overlay on a single card (called after save in editor)."""
        self._ann_cache[image_id] = annotations
        if image_id in self._cards:
            self._cards[image_id].set_annotations(annotations)

    def get_visible_image_ids(self) -> list[int]:
        """Visible image IDs in grid order (respects both cluster + status filters)."""
        return [img.id for img in self._images if img.id in self._visible_ids]

    # ── Highlight / status update from outside ──

    def highlight_image(self, image_id: int):
        for iid, card in self._cards.items():
            card.set_highlighted(iid == image_id)

    def update_card_status(self, image_id: int, status: str):
        if image_id in self._cards:
            self._cards[image_id].image.status = status
            self._cards[image_id]._apply_style()
            self._apply_filters()

    def set_classes(self, classes: list):
        """Receive [(id, name, color), ...] from main window. Rebuilds class filter bar."""
        self._classes = classes
        self._class_filter = None

        # Clear old buttons
        while self._class_filter_layout.count():
            item = self._class_filter_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not classes:
            self._class_filter_bar.setVisible(False)
            return

        self._class_filter_bar.setVisible(True)
        self._class_btn_group = QButtonGroup(self)
        self._class_btn_group.setExclusive(True)

        lbl = QLabel("Clase:")
        lbl.setStyleSheet("color:#888; font-size:11px;")
        self._class_filter_layout.addWidget(lbl)

        # "Todas" button
        all_btn = QPushButton("Todas")
        all_btn.setCheckable(True)
        all_btn.setChecked(True)
        all_btn.setFixedHeight(22)
        all_btn.setStyleSheet(self._class_btn_style("#555", checked=True))
        all_btn.clicked.connect(lambda: self._set_class_filter(None))
        self._class_btn_group.addButton(all_btn, -1)
        self._class_filter_layout.addWidget(all_btn)
        self._all_class_btn = all_btn

        for class_id, name, color in classes:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            btn.setStyleSheet(self._class_btn_style(color, checked=False))
            btn.setProperty("class_color", color)
            btn.clicked.connect(lambda checked, n=name, b=btn, c=color: self._on_class_btn(n, b, c))
            self._class_btn_group.addButton(btn)
            self._class_filter_layout.addWidget(btn)

        self._class_filter_layout.addStretch()

    def _class_btn_style(self, color: str, checked: bool) -> str:
        if checked:
            return (f"QPushButton {{ background:{color}; color:white; font-weight:bold; "
                    f"border-radius:3px; padding:0 8px; font-size:11px; }}")
        return (f"QPushButton {{ background:#2a2a2a; color:#aaa; "
                f"border:1px solid {color}; border-radius:3px; padding:0 8px; font-size:11px; }}"
                f"QPushButton:hover {{ background:#333; color:white; }}")

    def _on_class_btn(self, name: str, btn: QPushButton, color: str):
        self._all_class_btn.setStyleSheet(self._class_btn_style("#555", checked=False))
        btn.setStyleSheet(self._class_btn_style(color, checked=True))
        self._set_class_filter(name)

    def _set_class_filter(self, class_name: str | None):
        self._class_filter = class_name
        if class_name is None:
            self._all_class_btn.setStyleSheet(self._class_btn_style("#555", checked=True))
            for i in range(1, self._class_filter_layout.count()):
                w = self._class_filter_layout.itemAt(i).widget()
                if isinstance(w, QPushButton) and w is not self._all_class_btn:
                    c = w.property("class_color") or "#555"
                    w.setStyleSheet(self._class_btn_style(c, checked=False))
        self._apply_filters()

    def _show_label_menu(self):
        if not self._classes or not self._checked_ids:
            return
        menu = QMenu(self)
        for class_id, name, color in self._classes:
            action = menu.addAction(name)
            dot = "● "
            action.setText(f"{dot}{name}")
            action.setData(class_id)
        chosen = menu.exec(self._label_btn.mapToGlobal(
            self._label_btn.rect().bottomLeft()
        ))
        if chosen and chosen.data() is not None:
            self.batch_label_requested.emit(list(self._checked_ids), chosen.data())
