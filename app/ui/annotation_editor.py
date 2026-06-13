from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsItem, QListWidget, QListWidgetItem, QComboBox,
    QSplitter, QMenu, QCheckBox,
)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QSizeF, QEvent
from PySide6.QtGui import (
    QPixmap, QPen, QBrush, QColor, QFont, QPainter, QKeySequence,
)

from ..models.image_item import ImageItem
from ..models.annotation import Annotation


H = 8   # handle size in pixels

HANDLE_CURSORS = {
    "tl": Qt.SizeFDiagCursor, "t": Qt.SizeVerCursor,  "tr": Qt.SizeBDiagCursor,
    "r":  Qt.SizeHorCursor,   "br": Qt.SizeFDiagCursor, "b": Qt.SizeVerCursor,
    "bl": Qt.SizeBDiagCursor, "l":  Qt.SizeHorCursor,
}


# ─────────────────────────── ResizeHandle ───────────────────────────

class ResizeHandle(QGraphicsRectItem):
    """A small square handle that is a child of BoundingBoxItem.

    Handles its own mouse events so dragging it resizes the parent box.
    Becomes visible when the parent box is selected.
    """

    def __init__(self, pos_key: str, parent_box: "BoundingBoxItem"):
        super().__init__(-H / 2, -H / 2, H, H, parent_box)
        self.pos_key = pos_key
        self.parent_box = parent_box
        self._dragging = False
        self._start_scene: QPointF | None = None
        self._start_rect: QRectF | None = None

        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.ItemIsFocusable, False)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setCursor(HANDLE_CURSORS[pos_key])
        self.setZValue(20)
        self.setBrush(QBrush(QColor("#FFFFFF")))
        self.setPen(QPen(QColor("#222222"), 1))
        self.setVisible(False)

    def mousePressEvent(self, event):
        self._dragging = True
        self._start_scene = event.scenePos()
        self._start_rect = QRectF(self.parent_box.rect())
        event.accept()

    def mouseMoveEvent(self, event):
        if not self._dragging or not self._start_rect:
            return
        dx = event.scenePos().x() - self._start_scene.x()
        dy = event.scenePos().y() - self._start_scene.y()
        r = QRectF(self._start_rect)
        k = self.pos_key

        if "l" in k:
            r.setLeft(min(r.left() + dx, r.right() - 4))
        if "r" in k:
            r.setRight(max(r.right() + dx, r.left() + 4))
        if "t" in k:
            r.setTop(min(r.top() + dy, r.bottom() - 4))
        if "b" in k:
            r.setBottom(max(r.bottom() + dy, r.top() + 4))

        # clamp to image bounds
        if self.scene():
            bounds = self.scene().sceneRect()
            r.setLeft(max(bounds.left(), r.left()))
            r.setTop(max(bounds.top(), r.top()))
            r.setRight(min(bounds.right(), r.right()))
            r.setBottom(min(bounds.bottom(), r.bottom()))

        self.parent_box.setRect(r)
        self.parent_box._update_handle_positions()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self.parent_box._emit_changed()
        event.accept()


# ─────────────────────────── BoundingBoxItem ───────────────────────────

class BoundingBoxItem(QGraphicsRectItem):
    """Bounding box with 8 resize handles.

    - Single click  → select → handles appear
    - Drag on body  → move
    - Drag on handle → resize
    - Delete key    → delete (handled by scene)
    """

    def __init__(self, rect: QRectF, annotation: Annotation, color: QColor):
        super().__init__(rect)
        self.annotation = annotation
        self.box_color = color
        self._handles: dict[str, ResizeHandle] = {}
        self._moving = False
        self._move_start: QPointF | None = None
        self._move_start_rect: QRectF | None = None

        self._apply_style()
        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setZValue(5)
        self.setCursor(Qt.SizeAllCursor)

        self._label = QGraphicsTextItem(self._label_text(), self)
        self._label.setDefaultTextColor(color)
        self._label.setFont(QFont("Arial", 9, QFont.Bold))
        self._label.setPos(2, 2)
        self._label.setZValue(6)

        for key in ["tl", "t", "tr", "r", "br", "b", "bl", "l"]:
            self._handles[key] = ResizeHandle(key, self)
        self._update_handle_positions()

    def _is_yolo(self) -> bool:
        return self.annotation.source == "yolo"

    def _label_text(self) -> str:
        if self._is_yolo():
            return f"{self.annotation.class_name} [YOLO {self.annotation.confidence:.0%}]"
        return self.annotation.class_name

    def _apply_style(self):
        if self._is_yolo():
            pen = QPen(self.box_color, 2, Qt.DashLine)
        else:
            pen = QPen(self.box_color, 2)
        self.setPen(pen)
        fill = QColor(self.box_color)
        fill.setAlpha(20 if self._is_yolo() else 30)
        self.setBrush(QBrush(fill))

    def _update_handle_positions(self):
        r = self.rect()
        cx, cy = r.center().x(), r.center().y()
        for key, (x, y) in {
            "tl": (r.left(), r.top()), "t": (cx, r.top()), "tr": (r.right(), r.top()),
            "r":  (r.right(), cy),     "br": (r.right(), r.bottom()), "b": (cx, r.bottom()),
            "bl": (r.left(), r.bottom()), "l": (r.left(), cy),
        }.items():
            self._handles[key].setPos(x, y)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            for h in self._handles.values():
                h.setVisible(bool(value))
            style = Qt.DashLine if self._is_yolo() else Qt.SolidLine
            pen = QPen(self.box_color, 3 if value else 2, style)
            self.setPen(pen)
        return super().itemChange(change, value)

    # ── Move logic (body drag) ──

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._moving = True
            self._move_start = event.pos()
            self._move_start_rect = QRectF(self.rect())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._moving and self._move_start and self._move_start_rect:
            delta = event.pos() - self._move_start
            new_rect = self._move_start_rect.translated(delta.x(), delta.y())
            if self.scene():
                b = self.scene().sceneRect()
                new_rect.moveLeft(max(b.left(), min(new_rect.left(), b.right() - new_rect.width())))
                new_rect.moveTop(max(b.top(), min(new_rect.top(), b.bottom() - new_rect.height())))
            self.setRect(new_rect)
            self._update_handle_positions()
        # intentionally not calling super() — avoids Qt's built-in move

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._moving:
            self._moving = False
            self._emit_changed()
        super().mouseReleaseEvent(event)

    def _emit_changed(self):
        if self._is_yolo():
            self.annotation.source = "human"
            self.annotation.confidence = 1.0
            self._label.setPlainText(self._label_text())
            self._apply_style()
        if self.scene() and hasattr(self.scene(), "box_changed"):
            self.scene().box_changed(self)

    def update_class(self, name: str, color: QColor):
        self.annotation.class_name = name
        self.box_color = color
        self._apply_style()
        self._label.setPlainText(self._label_text())
        self._label.setDefaultTextColor(color)


# ─────────────────────────── AnnotationScene ───────────────────────────

class AnnotationScene(QGraphicsScene):
    box_updated = Signal(object)          # Annotation
    box_deleted = Signal(object)          # Annotation
    class_change_requested = Signal(object)  # BoundingBoxItem

    def __init__(self, parent=None):
        super().__init__(parent)
        self._draw_mode = False
        self._drawing = False
        self._draw_start: QPointF | None = None
        self._temp_rect: QGraphicsRectItem | None = None
        self._current_class_id: int = 0
        self._current_class_name: str = "object"
        self._current_class_color: QColor = QColor("#FF0000")
        self._boxes: list[BoundingBoxItem] = []

    def set_draw_mode(self, enabled: bool):
        self._draw_mode = enabled
        if enabled:
            self.clearSelection()

    def set_current_class(self, class_id: int, name: str, color: str):
        self._current_class_id = class_id
        self._current_class_name = name
        self._current_class_color = QColor(color)

    def set_image(self, pixmap: QPixmap):
        self.clear()
        self._boxes.clear()
        self.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

    def load_annotations(self, annotations: list[Annotation],
                         classes: dict[int, tuple[str, str]]):
        w, h = self.sceneRect().width(), self.sceneRect().height()
        for ann in annotations:
            name, color_str = classes.get(ann.class_id, (ann.class_name, "#FF0000"))
            rect = QRectF(ann.x * w, ann.y * h, ann.width * w, ann.height * h)
            self._add_box(rect, ann, QColor(color_str))

    def _add_box(self, rect: QRectF, annotation: Annotation,
                 color: QColor) -> BoundingBoxItem:
        box = BoundingBoxItem(rect, annotation, color)
        self.addItem(box)
        self._boxes.append(box)
        return box

    # ── Mouse events ──

    def mousePressEvent(self, event):
        if self._draw_mode and event.button() == Qt.LeftButton:
            pos = event.scenePos()
            if self.sceneRect().contains(pos):
                self._drawing = True
                self._draw_start = pos
                self._temp_rect = self.addRect(
                    QRectF(pos, QSizeF(0, 0)),
                    QPen(self._current_class_color, 1, Qt.DashLine),
                    QBrush(Qt.NoBrush),
                )
                event.accept()
                return
        if event.button() == Qt.RightButton:
            items = self.items(event.scenePos())
            box = next(
                (i if isinstance(i, BoundingBoxItem) else
                 (i.parentItem() if isinstance(i.parentItem(), BoundingBoxItem) else None)
                 for i in items), None
            )
            if box:
                self.class_change_requested.emit(box)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing and self._temp_rect and self._draw_start:
            pos = event.scenePos()
            x = min(pos.x(), self._draw_start.x())
            y = min(pos.y(), self._draw_start.y())
            w = abs(pos.x() - self._draw_start.x())
            h = abs(pos.y() - self._draw_start.y())
            self._temp_rect.setRect(QRectF(x, y, w, h))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drawing and self._temp_rect and event.button() == Qt.LeftButton:
            self._drawing = False
            rect = self._temp_rect.rect()
            self.removeItem(self._temp_rect)
            self._temp_rect = None

            if rect.width() > 5 and rect.height() > 5:
                w, h = self.sceneRect().width(), self.sceneRect().height()
                ann = Annotation(
                    class_id=self._current_class_id,
                    class_name=self._current_class_name,
                    x=max(0.0, rect.x() / w),
                    y=max(0.0, rect.y() / h),
                    width=min(1.0, rect.width() / w),
                    height=min(1.0, rect.height() / h),
                    source="human",
                )
                box = self._add_box(rect, ann, self._current_class_color)
                box.setSelected(True)
                self.box_updated.emit(ann)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_boxes()
        super().keyPressEvent(event)

    # ── Box management ──

    def box_changed(self, box: BoundingBoxItem):
        w, h = self.sceneRect().width(), self.sceneRect().height()
        r = box.rect()
        box.annotation.x = max(0.0, r.x() / w)
        box.annotation.y = max(0.0, r.y() / h)
        box.annotation.width = min(1.0, r.width() / w)
        box.annotation.height = min(1.0, r.height() / h)
        self.box_updated.emit(box.annotation)

    def delete_selected_boxes(self):
        for item in list(self.selectedItems()):
            if isinstance(item, BoundingBoxItem):
                self._boxes.remove(item)
                self.box_deleted.emit(item.annotation)
                self.removeItem(item)


# ─────────────────────────── AnnotationView ───────────────────────────

class AnnotationView(QGraphicsView):
    def __init__(self, scene: AnnotationScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setStyleSheet("background: #111; border: none;")
        # RubberBandDrag: drag over empty space to select multiple boxes
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPointF()

    def fit_in_view(self):
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self._zoom = 1.0

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 0.87
        self._zoom = max(0.05, min(self._zoom * factor, 20.0))
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x())
            )
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y())
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


# ─────────────────────────── AnnotationEditor ───────────────────────────

class AnnotationEditor(QWidget):
    annotation_saved     = Signal(int)
    navigate_request     = Signal(int)
    image_status_changed = Signal(int, str)  # image_id, new_status

    _PIXMAP_CACHE_SIZE = 8
    _MAX_UNDO = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = None
        self._image: ImageItem | None = None
        self._classes: dict[int, tuple[str, str]] = {}
        self._pending_annotations: list[Annotation] = []
        self._pixmap_cache: dict[str, QPixmap] = {}   # path -> QPixmap
        self._pixmap_cache_order: list[str] = []       # LRU order
        self._undo_stack: list[list] = []   # list of annotation snapshots
        self._redo_stack: list[list] = []

        # Create scene and view first — _build_toolbar references self._view
        self._scene = AnnotationScene()
        self._scene.box_updated.connect(self._on_box_updated)
        self._scene.box_deleted.connect(self._on_box_deleted)
        self._scene.class_change_requested.connect(self._show_class_menu_for_box)
        self._view = AnnotationView(self._scene)
        self._view.installEventFilter(self)   # intercept arrow keys

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ── Left: toolbar + canvas ──
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        left_layout.addLayout(self._build_toolbar())
        left_layout.addWidget(self._view)

        nav_row = QHBoxLayout()
        prev_btn = QPushButton("← Anterior")
        prev_btn.clicked.connect(lambda: self.navigate_request.emit(-1))
        next_btn = QPushButton("Siguiente →")
        next_btn.clicked.connect(lambda: self.navigate_request.emit(1))
        self._status_label = QLabel("Sin imagen")
        self._status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        nav_row.addWidget(prev_btn)
        nav_row.addStretch()
        nav_row.addWidget(self._status_label)
        nav_row.addStretch()
        nav_row.addWidget(next_btn)
        left_layout.addLayout(nav_row)

        splitter.addWidget(left_widget)

        # ── Right: class selector + annotation list ──
        right_widget = QWidget()
        right_widget.setFixedWidth(220)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        right_layout.addWidget(QLabel("Clase activa:"))
        self._class_combo = QComboBox()
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)
        right_layout.addWidget(self._class_combo)

        hint = QLabel(
            "← / → navegar imágenes\n"
            "R → revisada   X → descartar\n"
            "D → modo dibujo\n"
            "F → ajustar vista\n"
            "Del → borrar bbox seleccionado\n"
            "Ctrl+S → guardar manual\n"
            "Ctrl+Z → deshacer\n"
            "Ctrl+Y → rehacer\n"
            "Click → seleccionar bbox\n"
            "Arrastrar vértice → redimensionar\n"
            "Rueda → zoom   Medio → pan"
        )
        hint.setStyleSheet("color: #666; font-size: 10px;")
        hint.setWordWrap(True)
        right_layout.addWidget(hint)

        right_layout.addWidget(QLabel("Anotaciones:"))
        self._ann_list = QListWidget()
        self._ann_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._ann_list.customContextMenuRequested.connect(self._ann_context_menu)
        self._ann_list.itemClicked.connect(self._on_ann_list_clicked)
        right_layout.addWidget(self._ann_list)

        del_btn = QPushButton("Borrar seleccionado  [Del]")
        del_btn.clicked.connect(self._delete_selected_annotation)
        right_layout.addWidget(del_btn)

        save_btn = QPushButton("Guardar  [Ctrl+S]")
        save_btn.setStyleSheet("background: #1a6e2e; font-weight: bold;")
        save_btn.clicked.connect(self.save_annotations)
        right_layout.addWidget(save_btn)

        status_row = QHBoxLayout()
        mark_btn = QPushButton("Revisada [R]")
        mark_btn.setStyleSheet("background: #1a5c2e; font-weight: bold;")
        mark_btn.clicked.connect(self._mark_reviewed)
        status_row.addWidget(mark_btn)

        discard_btn = QPushButton("Descartar [X]")
        discard_btn.setStyleSheet("background: #5c1a1a; font-weight: bold;")
        discard_btn.clicked.connect(lambda: self._set_image_status("discarded"))
        status_row.addWidget(discard_btn)
        right_layout.addLayout(status_row)

        splitter.addWidget(right_widget)
        splitter.setSizes([700, 220])

    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)

        self._draw_btn = QPushButton("Dibujar BBox [D]")
        self._draw_btn.setCheckable(True)
        self._draw_btn.setToolTip("Activar modo dibujo: click+drag para crear bbox")
        self._draw_btn.toggled.connect(self._toggle_draw_mode)
        row.addWidget(self._draw_btn)

        fit_btn = QPushButton("Ajustar [F]")
        fit_btn.clicked.connect(self._view.fit_in_view)
        row.addWidget(fit_btn)

        self._undo_btn = QPushButton("↩ Deshacer")
        self._undo_btn.setToolTip("Deshacer último cambio  [Ctrl+Z]")
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self.undo)
        row.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("↪ Rehacer")
        self._redo_btn.setToolTip("Rehacer  [Ctrl+Y / Ctrl+Shift+Z]")
        self._redo_btn.setEnabled(False)
        self._redo_btn.clicked.connect(self.redo)
        row.addWidget(self._redo_btn)

        self._autosave_cb = QCheckBox("Autoguardar")
        self._autosave_cb.setChecked(True)
        self._autosave_cb.setToolTip("Guardar automáticamente al crear, mover o borrar un bbox")
        row.addWidget(self._autosave_cb)

        row.addStretch()

        self._img_label = QLabel("Sin imagen")
        self._img_label.setStyleSheet("color: #aaa; font-size: 11px;")
        row.addWidget(self._img_label)

        return row

    # ── DB / class setup ──

    def set_db(self, db):
        self._db = db
        self._refresh_classes()

    def _refresh_classes(self):
        if not self._db:
            return
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._classes.clear()
        for row in self._db.get_all_classes():
            class_id, name, color = row
            self._classes[class_id] = (name, color)
            self._class_combo.addItem(name, (class_id, name, color))
        self._class_combo.blockSignals(False)
        self._update_scene_class()

    def _update_scene_class(self):
        idx = self._class_combo.currentIndex()
        if idx >= 0:
            data = self._class_combo.itemData(idx)
            if data:
                self._scene.set_current_class(*data)

    def _on_class_changed(self, _):
        self._update_scene_class()

    # ── Undo / Redo ──

    def _snapshot(self) -> list:
        return [
            (a.class_id, a.class_name, a.x, a.y, a.width, a.height, a.source, a.confidence)
            for a in self._pending_annotations
        ]

    def _push_undo(self):
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_undo_buttons()

    def _restore_snapshot(self, snapshot: list):
        self._pending_annotations = [
            Annotation(class_id=s[0], class_name=s[1], x=s[2], y=s[3],
                       width=s[4], height=s[5], source=s[6], confidence=s[7])
            for s in snapshot
        ]
        if self._image:
            pix = self._get_pixmap(self._image.path)
            self._scene.set_image(pix)
            self._scene.load_annotations(self._pending_annotations, self._classes)
        self._refresh_ann_list()
        if self._autosave_cb.isChecked():
            self.save_annotations()

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self._snapshot())
        self._restore_snapshot(self._undo_stack.pop())
        self._update_undo_buttons()

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self._snapshot())
        self._restore_snapshot(self._redo_stack.pop())
        self._update_undo_buttons()

    def _update_undo_buttons(self):
        self._undo_btn.setEnabled(bool(self._undo_stack))
        self._redo_btn.setEnabled(bool(self._redo_stack))

    # ── Load image ──

    def _get_pixmap(self, path: str) -> QPixmap:
        if path in self._pixmap_cache:
            # move to end (most recently used)
            self._pixmap_cache_order.remove(path)
            self._pixmap_cache_order.append(path)
            return self._pixmap_cache[path]
        pix = QPixmap(path)
        self._pixmap_cache[path] = pix
        self._pixmap_cache_order.append(path)
        if len(self._pixmap_cache_order) > self._PIXMAP_CACHE_SIZE:
            oldest = self._pixmap_cache_order.pop(0)
            self._pixmap_cache.pop(oldest, None)
        return pix

    def load_image(self, image: ImageItem):
        self._image = image
        self._pending_annotations = []
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_undo_buttons()
        self._draw_btn.setChecked(False)
        self._scene.set_draw_mode(False)
        self._view.setDragMode(QGraphicsView.RubberBandDrag)

        pix = self._get_pixmap(image.path)
        if pix.isNull():
            self._status_label.setText("Error cargando imagen")
            return

        self._scene.set_image(pix)
        self._img_label.setText(f"{image.filename}  ({image.width}×{image.height})")
        self._status_label.setText(f"Estado: {image.status}")

        if self._db:
            ann_rows = self._db.get_annotations_for_image(image.id)
            anns = [Annotation.from_db_row(r) for r in ann_rows]
            self._pending_annotations = anns
            self._scene.load_annotations(anns, self._classes)

        self._refresh_ann_list()
        self._view.fit_in_view()

    # ── Box events ──

    def _on_box_updated(self, annotation: Annotation):
        self._push_undo()
        if annotation not in self._pending_annotations:
            self._pending_annotations.append(annotation)
        self._refresh_ann_list()
        if self._autosave_cb.isChecked():
            self.save_annotations()

    def _on_box_deleted(self, annotation: Annotation):
        self._push_undo()
        if annotation in self._pending_annotations:
            self._pending_annotations.remove(annotation)
        if annotation.id and self._db:
            self._db.delete_annotation(annotation.id)
        self._refresh_ann_list()
        if self._autosave_cb.isChecked():
            self.save_annotations()

    # ── Annotation list ──

    def _refresh_ann_list(self):
        self._ann_list.clear()
        for ann in self._pending_annotations:
            tag = f" [YOLO {ann.confidence:.0%}]" if ann.source == "yolo" else ""
            item = QListWidgetItem(
                f"{ann.class_name}{tag}  [{ann.x:.2f},{ann.y:.2f}  {ann.width:.2f}×{ann.height:.2f}]"
            )
            if ann.source == "yolo":
                item.setForeground(QColor("#aaaaff"))
            item.setData(Qt.UserRole, ann)
            self._ann_list.addItem(item)

    def _on_ann_list_clicked(self, item: QListWidgetItem):
        ann = item.data(Qt.UserRole)
        if not ann:
            return
        # Select the corresponding box in the scene
        self._scene.clearSelection()
        for box in self._scene._boxes:
            if box.annotation is ann:
                box.setSelected(True)
                self._view.ensureVisible(box)
                break

    def _ann_context_menu(self, pos):
        item = self._ann_list.itemAt(pos)
        if not item:
            return
        ann = item.data(Qt.UserRole)
        menu = QMenu(self)
        change_menu = menu.addMenu("Cambiar clase")
        for class_id, (name, color) in self._classes.items():
            action = change_menu.addAction(name)
            action.setData((class_id, name, color))
            if ann.class_id == class_id:
                font = action.font()
                font.setBold(True)
                action.setFont(font)
        change_menu.triggered.connect(
            lambda act, a=ann: self._apply_class_change(a, act.data())
        )
        menu.addSeparator()
        menu.addAction("Borrar", lambda: self._delete_ann(ann))
        menu.exec(self._ann_list.mapToGlobal(pos))

    def _show_class_menu_for_box(self, box: "BoundingBoxItem"):
        """Context menu triggered by right-click on a bbox in the canvas."""
        menu = QMenu(self._view)
        for class_id, (name, color) in self._classes.items():
            action = menu.addAction(name)
            action.setData((class_id, name, color))
            if box.annotation.class_id == class_id:
                font = action.font()
                font.setBold(True)
                action.setFont(font)
        chosen = menu.exec(self._view.mapToGlobal(
            self._view.mapFromScene(box.sceneBoundingRect().topRight())
        ))
        if chosen and chosen.data():
            self._apply_class_change(box.annotation, chosen.data())

    def _apply_class_change(self, ann: Annotation, class_data: tuple):
        if not class_data:
            return
        class_id, name, color = class_data
        if ann.class_id == class_id:
            return
        self._push_undo()
        ann.class_id = class_id
        ann.class_name = name
        for box in self._scene._boxes:
            if box.annotation is ann:
                box.update_class(name, QColor(color))
                break
        self._refresh_ann_list()
        if self._autosave_cb.isChecked():
            self.save_annotations()

    # ── Delete ──

    def _delete_selected_annotation(self):
        item = self._ann_list.currentItem()
        if not item:
            # fall back to whatever is selected in the scene
            self._scene.delete_selected_boxes()
            return
        ann = item.data(Qt.UserRole)
        if ann:
            self._delete_ann(ann)

    def _delete_ann(self, ann: Annotation):
        if ann in self._pending_annotations:
            self._pending_annotations.remove(ann)
        if ann.id and self._db:
            self._db.delete_annotation(ann.id)
        for box in list(self._scene._boxes):
            if box.annotation is ann:
                self._scene._boxes.remove(box)
                self._scene.removeItem(box)
                break
        self._refresh_ann_list()

    # ── Save ──

    def save_annotations(self):
        if not self._db or not self._image:
            return
        self._db.delete_annotations_for_image(self._image.id)
        for ann in self._pending_annotations:
            new_id = self._db.insert_annotation(
                self._image.id, ann.class_id,
                ann.x, ann.y, ann.width, ann.height,
                ann.source, ann.confidence,
            )
            ann.id = new_id
        self.annotation_saved.emit(self._image.id)
        self._status_label.setText(f"Guardado — {len(self._pending_annotations)} anotaciones")

    # ── Draw mode / status ──

    def _toggle_draw_mode(self, enabled: bool):
        self._scene.set_draw_mode(enabled)
        # In draw mode disable rubber band so clicks go to scene for drawing
        self._view.setDragMode(
            QGraphicsView.NoDrag if enabled else QGraphicsView.RubberBandDrag
        )

    def _set_image_status(self, status: str):
        if self._image and self._db:
            self._db.update_image_status(self._image.id, status)
            self._image.status = status
            icons = {"reviewed": "✓", "discarded": "✗", "pending": "○"}
            self._status_label.setText(f"Estado: {status} {icons.get(status, '')}")
            self.image_status_changed.emit(self._image.id, status)

    def _mark_reviewed(self):
        self._set_image_status("reviewed")

    # ── Keyboard ──

    def eventFilter(self, obj, event):
        """Intercept navigation/status keys from the graphics view."""
        if obj is self._view and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Left:
                self.navigate_request.emit(-1)
                return True
            if event.key() == Qt.Key_Right:
                self.navigate_request.emit(1)
                return True
            if event.key() == Qt.Key_R:
                self._mark_reviewed()
                return True
            if event.key() == Qt.Key_X:
                self._set_image_status("discarded")
                return True
            if event.matches(QKeySequence.Undo):
                self.undo()
                return True
            if event.matches(QKeySequence.Redo):
                self.redo()
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.navigate_request.emit(-1)
        elif event.key() == Qt.Key_Right:
            self.navigate_request.emit(1)
        elif event.key() == Qt.Key_R:
            self._mark_reviewed()
        elif event.key() == Qt.Key_X:
            self._set_image_status("discarded")
        elif event.key() == Qt.Key_D:
            self._draw_btn.setChecked(not self._draw_btn.isChecked())
        elif event.key() == Qt.Key_F:
            self._view.fit_in_view()
        elif event.matches(QKeySequence.Save):
            self.save_annotations()
        elif event.matches(QKeySequence.Undo):
            self.undo()
        elif event.matches(QKeySequence.Redo):
            self.redo()
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._scene.delete_selected_boxes()
        else:
            super().keyPressEvent(event)
