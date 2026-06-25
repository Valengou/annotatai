"""Diálogo (modeless) de outliers geométricos de bboxes por clase.

Lista las cajas raras como crops clickeables (abren la imagen+caja en el editor),
con su score (|z|) y el motivo. Filtrable por clase y por umbral.
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QScrollArea, QWidget, QGridLayout,
)
from PySide6.QtCore import Qt, Signal

from ..core.bbox_geometry import find_geometric_outliers
from ..core.bbox_embeddings import crop_bbox
from .bbox_view import CropThumb


class GeometryOutliersDialog(QDialog):
    open_image_requested = Signal(int, int)   # image_id, annotation_id

    COLS = 6
    MAX_THUMBS = 240

    def __init__(self, db, classes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bboxes raras por geometría (por clase)")
        self.setMinimumSize(820, 560)
        self._db = db
        self._classes = classes or []
        self._ann_to_img: dict[int, int] = {}
        self._setup_ui()
        self._run()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(QLabel("Clase:"))
        self._class_combo = QComboBox()
        self._class_combo.addItem("Todas", None)
        for cid, name, color in self._classes:
            self._class_combo.addItem(name, name)
        self._class_combo.currentIndexChanged.connect(self._run)
        top.addWidget(self._class_combo)

        top.addWidget(QLabel("Umbral z:"))
        self._z_spin = QDoubleSpinBox()
        self._z_spin.setRange(1.5, 8.0)
        self._z_spin.setSingleStep(0.5)
        self._z_spin.setDecimals(1)
        self._z_spin.setValue(3.5)
        self._z_spin.setToolTip("Más bajo = más cajas marcadas como raras")
        self._z_spin.valueChanged.connect(self._run)
        top.addWidget(self._z_spin)

        refresh = QPushButton("Analizar")
        refresh.clicked.connect(self._run)
        top.addWidget(refresh)

        top.addStretch()
        self._info = QLabel("")
        self._info.setStyleSheet("color:#aaa; font-size:11px;")
        top.addWidget(self._info)
        layout.addLayout(top)

        hint = QLabel("Click en una caja para abrirla en el editor. El color/score "
                      "indica cuán atípica es respecto a su clase.")
        hint.setStyleSheet("color:#777; font-size:11px;")
        layout.addWidget(hint)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setSpacing(6)
        self._grid.setContentsMargins(6, 6, 6, 6)
        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll, 1)

    def _clear_grid(self):
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _run(self):
        self._clear_grid()
        self._ann_to_img.clear()
        class_name = self._class_combo.currentData()
        z = self._z_spin.value()
        try:
            outliers = find_geometric_outliers(self._db, z_thresh=z, class_name=class_name)
        except Exception as exc:  # noqa: BLE001
            self._info.setText(f"Error: {exc}")
            return

        total = len(outliers)
        shown = outliers[:self.MAX_THUMBS]
        for idx, o in enumerate(shown):
            self._ann_to_img[o["ann_id"]] = o["image_id"]
            crop = None
            try:
                crop = crop_bbox(Path(o["path"]), o["x"], o["y"], o["w"], o["h"])
            except Exception:
                crop = None
            # score normalizado 0..1 para el color del borde (|z|/6, tope 1)
            norm = max(0.0, min(1.0, o["score"] / 6.0))
            thumb = CropThumb(o["ann_id"], norm, o["image_id"], o["status"], crop)
            thumb.setToolTip(
                f"{o['class_name']} — {o['reason']}\n"
                f"z={o['score']:.1f}  ·  {o['filename']}")
            thumb.clicked.connect(self._on_thumb_click)
            self._grid.addWidget(thumb, idx // self.COLS, idx % self.COLS)

        msg = f"{total} bboxes raras (z ≥ {z:.1f})"
        if total > self.MAX_THUMBS:
            msg += f" — mostrando las {self.MAX_THUMBS} más atípicas"
        self._info.setText(msg)

    def _on_thumb_click(self, ann_id: int):
        img_id = self._ann_to_img.get(ann_id)
        if img_id is not None:
            self.open_image_requested.emit(img_id, ann_id)
