from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsPolygonItem,
    QGraphicsItem, QListWidget, QListWidgetItem, QComboBox,
    QSplitter, QMenu, QCheckBox, QAbstractItemView,
)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QSizeF, QEvent, QThread, QObject
from PySide6.QtGui import (
    QPixmap, QPen, QBrush, QColor, QFont, QPainter, QKeySequence, QIcon,
    QPolygonF,
)

from ..models.image_item import ImageItem
from ..models.annotation import Annotation
from ..utils.config import DEFAULT_INTERACTIVE_SAM_MODEL


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
        self._polygon_item: QGraphicsPolygonItem | None = None
        self._locked = False
        self._moving = False
        self._move_start: QPointF | None = None
        self._move_start_rect: QRectF | None = None

        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setZValue(5)
        self.setCursor(Qt.SizeAllCursor)

        self._label = QGraphicsTextItem("", self)
        self._label.setFont(QFont("Arial", 9, QFont.Bold))
        # Keep label a constant on-screen size regardless of zoom
        self._label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self._label.setZValue(7)

        self._apply_style()

        for key in ["tl", "t", "tr", "r", "br", "b", "bl", "l"]:
            self._handles[key] = ResizeHandle(key, self)
        self._update_handle_positions()

    def _is_provisional(self) -> bool:
        """Anotación sugerida por IA (SAM 3 / YOLOE), pendiente de validar."""
        return self.annotation.source != "human"

    def _apply_style(self, selected: bool = False):
        style = Qt.DashLine if self._is_provisional() else Qt.SolidLine
        pen = QPen(self.box_color, 3 if selected else 2, style)
        pen.setCosmetic(True)   # ancho constante en pantalla a cualquier zoom
        self.setPen(pen)
        fill = QColor(self.box_color)
        fill.setAlpha(70 if selected else (18 if self._is_provisional() else 28))
        self.setBrush(QBrush(fill))
        # Fondo de la etiqueta con el color de la clase para que se lea siempre
        bg = QColor(self.box_color)
        bg.setAlpha(230)
        self._label.setHtml(
            f'<div style="background:{bg.name()};color:#fff;'
            f'padding:0 3px;border-radius:2px;">{self._label_inner()}</div>'
        )
        if self._polygon_item:
            poly_pen = QPen(self.box_color, 2, Qt.SolidLine)
            poly_pen.setCosmetic(True)
            self._polygon_item.setPen(poly_pen)
            poly_fill = QColor(self.box_color)
            poly_fill.setAlpha(95 if selected else 55)
            self._polygon_item.setBrush(QBrush(poly_fill))

    def add_polygon_overlay(self):
        if not self.annotation.polygon:
            return
        scene = self.scene()
        if not scene:
            return
        w, h = scene.sceneRect().width(), scene.sceneRect().height()
        polygon = QPolygonF([QPointF(x * w, y * h) for x, y in self.annotation.polygon])
        self._polygon_item = QGraphicsPolygonItem(polygon, self)
        self._polygon_item.setAcceptedMouseButtons(Qt.NoButton)
        self._polygon_item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self._polygon_item.setZValue(1)
        self._apply_style(self.isSelected())

    def _label_inner(self) -> str:
        if self._is_provisional():
            return f"{self.annotation.class_name} · {self.annotation.confidence:.0%} ⟳"
        return self.annotation.class_name

    def _update_handle_positions(self):
        r = self.rect()
        cx, cy = r.center().x(), r.center().y()
        for key, (x, y) in {
            "tl": (r.left(), r.top()), "t": (cx, r.top()), "tr": (r.right(), r.top()),
            "r":  (r.right(), cy),     "br": (r.right(), r.bottom()), "b": (cx, r.bottom()),
            "bl": (r.left(), r.bottom()), "l": (r.left(), cy),
        }.items():
            self._handles[key].setPos(x, y)
        self._label.setPos(r.topLeft())

    def set_human(self):
        """Aceptar la sugerencia: pasa a 'human'."""
        self.annotation.source = "human"
        self.annotation.confidence = 1.0
        self._apply_style(self.isSelected())

    def set_locked(self, locked: bool):
        """Bloquear: no interactivo y atenuado (para dibujar sin interferencia)."""
        self._locked = locked
        self.setFlag(QGraphicsItem.ItemIsSelectable, not locked)
        self.setAcceptedMouseButtons(Qt.NoButton if locked else Qt.AllButtons)
        if locked:
            self.setSelected(False)
            for h in self._handles.values():
                h.setVisible(False)
        self.setOpacity(0.35 if locked else 1.0)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            for h in self._handles.values():
                h.setVisible(bool(value))
            self._apply_style(selected=bool(value))
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
        if self._is_provisional():
            # Editar una sugerencia equivale a aceptarla
            self.annotation.source = "human"
            self.annotation.confidence = 1.0
        self._apply_style(self.isSelected())
        if self.scene() and hasattr(self.scene(), "box_changed"):
            self.scene().box_changed(self)

    def update_class(self, name: str, color: QColor):
        self.annotation.class_name = name
        self.box_color = color
        self._apply_style(self.isSelected())


# ─────────────────────────── AnnotationScene ───────────────────────────

class AnnotationScene(QGraphicsScene):
    box_updated = Signal(object)          # Annotation
    box_deleted = Signal(object)          # Annotation
    class_change_requested = Signal(object)  # BoundingBoxItem
    sam_point_clicked = Signal(float, float, bool)  # (px, py, positivo) en coords de imagen

    def __init__(self, parent=None):
        super().__init__(parent)
        self._draw_mode = False
        self._sam_mode = False
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
        # En modo dibujo, bloquear las cajas existentes para que no interfieran.
        for box in self._boxes:
            box.set_locked(enabled)

    def set_sam_mode(self, enabled: bool):
        self._sam_mode = enabled
        if enabled:
            self.clearSelection()
        for box in self._boxes:
            box.set_locked(enabled)

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
        box.add_polygon_overlay()
        self._boxes.append(box)
        return box

    # ── Mouse events ──

    def mousePressEvent(self, event):
        if self._sam_mode and event.button() in (Qt.LeftButton, Qt.RightButton):
            pos = event.scenePos()
            if self.sceneRect().contains(pos):
                # izquierdo = punto positivo (incluir), derecho = negativo (excluir)
                self.sam_point_clicked.emit(
                    pos.x(), pos.y(), event.button() == Qt.LeftButton)
                event.accept()
                return
        if self._draw_mode and event.button() == Qt.LeftButton:
            pos = event.scenePos()
            if self.sceneRect().contains(pos):
                self._drawing = True
                self._draw_start = pos
                preview_pen = QPen(self._current_class_color, 2, Qt.DashLine)
                preview_pen.setCosmetic(True)   # visible a cualquier zoom
                preview_fill = QColor(self._current_class_color)
                preview_fill.setAlpha(50)
                self._temp_rect = self.addRect(
                    QRectF(pos, QSizeF(0, 0)),
                    preview_pen,
                    QBrush(preview_fill),
                )
                self._temp_rect.setZValue(30)
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


# ─────────────────────────── SAM interactivo (worker) ───────────────────────────

class _SamWorker(QObject):
    """Vive en un QThread; carga SAM2, codifica la imagen y predice por punto."""
    loaded      = Signal()
    image_ready = Signal()
    box_ready   = Signal(object)   # dict {box, polygon} o None
    failed      = Signal(str)

    def __init__(self, model_name: str):
        super().__init__()
        self._model_name = model_name
        self._sam = None

    def load(self):
        try:
            from ..core.interactive_sam import InteractiveSAM
            self._sam = InteractiveSAM(self._model_name)
            self._sam.load()
            self.loaded.emit()
        except Exception as e:
            self.failed.emit(str(e))

    def set_image(self, path: str):
        try:
            self._sam.set_image(path)
            self.image_ready.emit()
        except Exception as e:
            self.failed.emit(str(e))

    def predict(self, payload):
        try:
            points, labels = payload
            self.box_ready.emit(self._sam.predict_points(points, labels))
        except Exception as e:
            self.failed.emit(str(e))


# ─────────────────────────── AnnotationEditor ───────────────────────────

class AnnotationEditor(QWidget):
    annotation_saved     = Signal(int)
    navigate_request     = Signal(int)
    image_status_changed = Signal(int, str)  # image_id, new_status

    # Solicitudes al worker SAM (cruzan al hilo del worker)
    _sam_load_requested    = Signal()
    _sam_image_requested   = Signal(str)
    _sam_predict_requested = Signal(object)   # (points, labels)

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
        self._syncing = False   # guarda contra recursión list↔scene

        # SAM interactivo (click-to-box)
        self._sam_model_name = DEFAULT_INTERACTIVE_SAM_MODEL
        self._sam_thread: QThread | None = None
        self._sam_worker: _SamWorker | None = None
        self._sam_loaded = False
        self._sam_image_path: str | None = None   # imagen ya codificada en SAM
        self._sam_pending_image: str | None = None
        self._sam_busy = False
        # Objeto activo en construcción (multi-punto)
        self._sam_points: list[tuple[float, float]] = []
        self._sam_labels: list[int] = []          # 1=positivo, 0=negativo
        self._sam_preview: dict | None = None      # {box, polygon} sin confirmar
        self._sam_overlay_items: list = []         # items temporales en la escena
        self._sam_pending_predict = False

        # Create scene and view first — _build_toolbar references self._view
        self._scene = AnnotationScene()
        self._scene.box_updated.connect(self._on_box_updated)
        self._scene.box_deleted.connect(self._on_box_deleted)
        self._scene.class_change_requested.connect(self._show_class_menu_for_box)
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)
        self._scene.sam_point_clicked.connect(self._on_sam_point)
        self._view = AnnotationView(self._scene)
        self._view.installEventFilter(self)   # intercept arrow keys
        self._view.setFocusPolicy(Qt.StrongFocus)   # poder recibir foco de teclado

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
            "D → modo dibujo   S → SAM click   A → aceptar sugerencias\n"
            "SAM: clic izq=punto +   clic der=punto −   Enter=confirmar   Esc=cancelar\n"
            "F → ajustar vista\n"
            "Del → borrar bbox seleccionado\n"
            "Ctrl+S → guardar manual\n"
            "Ctrl+Z → deshacer\n"
            "Ctrl+Y → rehacer\n"
            "Click → seleccionar bbox  (Ctrl/Shift → varias)\n"
            "Clic derecho en la lista → cambiar clase / borrar en lote\n"
            "Arrastrar vértice → redimensionar\n"
            "Rueda → zoom   Medio → pan"
        )
        hint.setStyleSheet("color: #666; font-size: 10px;")
        hint.setWordWrap(True)
        right_layout.addWidget(hint)

        # ── Visibilidad / filtro de clases ──
        self._show_labels_cb = QCheckBox("Mostrar etiquetas en imagen")
        self._show_labels_cb.setChecked(True)
        self._show_labels_cb.setToolTip(
            "Apagar todas las etiquetas sobre la imagen para ver la foto limpia")
        self._show_labels_cb.toggled.connect(self._apply_visibility)
        right_layout.addWidget(self._show_labels_cb)

        right_layout.addWidget(QLabel("Filtrar clases:"))
        self._filter_list = QListWidget()
        self._filter_list.setMaximumHeight(96)
        self._filter_list.setToolTip(
            "Tildá las clases a mostrar (afecta la imagen y la lista de abajo)")
        self._filter_list.itemChanged.connect(self._on_filter_item_changed)
        right_layout.addWidget(self._filter_list)

        right_layout.addWidget(QLabel("Anotaciones:"))
        self._ann_list = QListWidget()
        self._ann_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._ann_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._ann_list.customContextMenuRequested.connect(self._ann_context_menu)
        self._ann_list.itemSelectionChanged.connect(self._sync_list_to_scene)
        right_layout.addWidget(self._ann_list)

        # Validación rápida de sugerencias de IA (SAM 3 / YOLOE)
        sugg_row = QHBoxLayout()
        self._accept_all_btn = QPushButton("✓ Aceptar sugerencias")
        self._accept_all_btn.setStyleSheet("background: #1a6e2e; font-weight: bold;")
        self._accept_all_btn.setToolTip(
            "Aceptar las sugerencias seleccionadas, o todas si no hay selección [A]")
        self._accept_all_btn.clicked.connect(self._accept_suggestions)
        self._accept_all_btn.setVisible(False)
        sugg_row.addWidget(self._accept_all_btn)

        self._reject_all_btn = QPushButton("✗ Rechazar")
        self._reject_all_btn.setStyleSheet("background: #5c1a1a; font-weight: bold;")
        self._reject_all_btn.setToolTip(
            "Rechazar las sugerencias seleccionadas, o todas si no hay selección")
        self._reject_all_btn.clicked.connect(self._reject_suggestions)
        self._reject_all_btn.setVisible(False)
        sugg_row.addWidget(self._reject_all_btn)
        right_layout.addLayout(sugg_row)

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

        self._sam_btn = QPushButton("🪄 SAM click [S]")
        self._sam_btn.setCheckable(True)
        self._sam_btn.setToolTip(
            "SAM2 interactivo: clic izq=punto +, clic der=punto −, Enter=confirmar, Esc=cancelar")
        self._sam_btn.toggled.connect(self._toggle_sam_mode)
        row.addWidget(self._sam_btn)

        self._mode_label = QLabel()
        self._mode_label.setMinimumWidth(230)
        row.addWidget(self._mode_label)
        self._update_mode_label()

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
        self._populate_filter_list()
        self._update_scene_class()

    # ── Visibilidad / filtro por clase ──

    def _populate_filter_list(self):
        self._filter_list.blockSignals(True)
        self._filter_list.clear()
        for class_id, (name, color) in self._classes.items():
            item = QListWidgetItem(name)
            pix = QPixmap(12, 12)
            pix.fill(QColor(color))
            item.setIcon(QIcon(pix))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, class_id)
            self._filter_list.addItem(item)
        self._filter_list.blockSignals(False)

    def _class_visible(self, class_id: int) -> bool:
        if self._filter_list.count() == 0:
            return True
        for row in range(self._filter_list.count()):
            it = self._filter_list.item(row)
            if it.data(Qt.UserRole) == class_id:
                return it.checkState() == Qt.Checked
        return True

    def _apply_visibility(self):
        """Aplica el checkbox maestro + filtro de clases a las cajas del lienzo."""
        show = self._show_labels_cb.isChecked()
        for box in self._scene._boxes:
            box.setVisible(show and self._class_visible(box.annotation.class_id))

    def _on_filter_item_changed(self, _item):
        self._refresh_ann_list()   # refresca lista (respeta filtro) + visibilidad

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
            (a.class_id, a.class_name, a.x, a.y, a.width, a.height,
             a.source, a.confidence, a.polygon)
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
                       width=s[4], height=s[5], source=s[6], confidence=s[7],
                       polygon=s[8] if len(s) > 8 else None)
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
        # Garantizar que el canvas tenga el foco para que ← → R X funcionen
        self._view.setFocus(Qt.OtherFocusReason)
        # Si el modo SAM está activo, descartar objeto activo y re-codificar
        if self._sam_btn.isChecked():
            self._sam_reset()
            self._sam_request_current_image()

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
        self._ann_list.blockSignals(True)
        self._ann_list.clear()
        for ann in self._pending_annotations:
            if not self._class_visible(ann.class_id):
                continue
            provisional = ann.source != "human"
            color = self._classes.get(ann.class_id, (ann.class_name, "#FF0000"))[1]
            if provisional:
                label = f"⟳ {ann.class_name}  ·  {ann.confidence:.0%}"
            else:
                label = f"✓ {ann.class_name}"
            item = QListWidgetItem(label)
            # Punto de color de la clase para distinguir etiquetas de un vistazo
            pix = QPixmap(12, 12)
            pix.fill(QColor(color))
            item.setIcon(QIcon(pix))
            if provisional:
                item.setForeground(QColor("#ffcf66"))   # ámbar = pendiente de validar
            item.setData(Qt.UserRole, ann)
            self._ann_list.addItem(item)
        self._ann_list.blockSignals(False)
        self._apply_visibility()
        self._update_suggestion_buttons()

    def _box_for_ann(self, ann):
        for box in self._scene._boxes:
            if box.annotation is ann:
                return box
        return None

    def _selected_provisional(self) -> list:
        """Sugerencias actualmente seleccionadas en el lienzo."""
        return [b.annotation for b in self._scene.selectedItems()
                if isinstance(b, BoundingBoxItem) and b.annotation.source != "human"]

    def _all_provisional(self) -> list:
        return [a for a in self._pending_annotations if a.source != "human"]

    def _sync_list_to_scene(self):
        """Selección en la lista → seleccionar las cajas correspondientes."""
        if self._syncing:
            return
        self._syncing = True
        sel_ids = {id(it.data(Qt.UserRole)) for it in self._ann_list.selectedItems()}
        first_box = None
        for box in self._scene._boxes:
            on = id(box.annotation) in sel_ids
            box.setSelected(on)
            if on and first_box is None:
                first_box = box
        if first_box:
            self._view.ensureVisible(first_box)
        self._syncing = False
        self._update_suggestion_buttons()

    def _on_scene_selection_changed(self):
        """Selección en el lienzo → resaltar las filas correspondientes."""
        if self._syncing:
            self._update_suggestion_buttons()
            return
        self._syncing = True
        sel_ids = {id(i.annotation) for i in self._scene.selectedItems()
                   if isinstance(i, BoundingBoxItem)}
        for row in range(self._ann_list.count()):
            it = self._ann_list.item(row)
            it.setSelected(id(it.data(Qt.UserRole)) in sel_ids)
        self._syncing = False
        self._update_suggestion_buttons()

    def _update_suggestion_buttons(self):
        total = self._all_provisional()
        has = bool(total)
        self._accept_all_btn.setVisible(has)
        self._reject_all_btn.setVisible(has)
        if not has:
            return
        sel = self._selected_provisional()
        if sel:
            self._accept_all_btn.setText(f"✓ Aceptar selección ({len(sel)})")
            self._reject_all_btn.setText(f"✗ Rechazar ({len(sel)})")
        else:
            self._accept_all_btn.setText(f"✓ Aceptar todas ({len(total)})")
            self._reject_all_btn.setText("✗ Rechazar todas")

    def _selected_annotations(self) -> list:
        """Anotaciones seleccionadas en la lista (sincronizada con el lienzo)."""
        return [it.data(Qt.UserRole) for it in self._ann_list.selectedItems()]

    def _ann_context_menu(self, pos):
        item = self._ann_list.itemAt(pos)
        if not item:
            return
        clicked = item.data(Qt.UserRole)
        selected = self._selected_annotations()
        # Si el clickeado forma parte de una selección múltiple, operar sobre toda
        # la selección (batch edit); si no, solo sobre el clickeado.
        targets = selected if (clicked in selected and len(selected) > 1) else [clicked]
        multi = len(targets) > 1

        menu = QMenu(self)
        prov = [a for a in targets if a.source != "human"]
        if prov:
            menu.addAction(
                f"✓ Aceptar sugerencias ({len(prov)})" if multi else "✓ Aceptar sugerencia",
                lambda: self._accept_anns(prov))
            menu.addAction(
                f"✗ Rechazar sugerencias ({len(prov)})" if multi else "✗ Rechazar sugerencia",
                lambda: self._delete_anns(prov))
            menu.addSeparator()

        change_title = (f"Cambiar clase de {len(targets)} seleccionadas"
                        if multi else "Cambiar clase")
        change_menu = menu.addMenu(change_title)
        for class_id, (name, color) in self._classes.items():
            action = change_menu.addAction(name)
            action.setData((class_id, name, color))
            if not multi and clicked.class_id == class_id:
                font = action.font()
                font.setBold(True)
                action.setFont(font)
        change_menu.triggered.connect(
            lambda act, ts=targets: self._apply_class_change_many(ts, act.data())
        )
        menu.addSeparator()
        menu.addAction(
            f"Borrar {len(targets)} seleccionadas" if multi else "Borrar",
            lambda: self._delete_anns(targets))
        menu.exec(self._ann_list.mapToGlobal(pos))

    def _accept_anns(self, anns: list):
        """Aceptar (provisional → human) un conjunto de anotaciones, con un solo undo."""
        prov = [a for a in anns if a.source != "human"]
        if not prov:
            return
        self._push_undo()
        for ann in prov:
            ann.source = "human"
            ann.confidence = 1.0
            box = self._box_for_ann(ann)
            if box:
                box.set_human()
        self._refresh_ann_list()
        if self._autosave_cb.isChecked():
            self.save_annotations()
        self._status_label.setText(f"{len(prov)} sugerencia(s) aceptada(s)")

    def _delete_anns(self, anns: list):
        """Borrar un conjunto de anotaciones con un solo undo."""
        if not anns:
            return
        self._push_undo()
        for ann in list(anns):
            self._delete_ann(ann, push_undo=False)
        if self._autosave_cb.isChecked():
            self.save_annotations()
        self._status_label.setText(f"{len(anns)} etiqueta(s) borrada(s)")

    def _apply_class_change_many(self, anns: list, class_data: tuple):
        """Cambiar la clase de varias anotaciones a la vez, con un solo undo."""
        if not class_data or not anns:
            return
        class_id, name, color = class_data
        changed = [a for a in anns if a.class_id != class_id]
        if not changed:
            return
        self._push_undo()
        for ann in changed:
            ann.class_id = class_id
            ann.class_name = name
            for box in self._scene._boxes:
                if box.annotation is ann:
                    box.update_class(name, QColor(color))
                    break
        self._refresh_ann_list()
        if self._autosave_cb.isChecked():
            self.save_annotations()
        self._status_label.setText(f"{len(changed)} etiqueta(s) → {name}")

    def _accept_ann(self, ann: Annotation):
        if ann.source == "human":
            return
        self._push_undo()
        ann.source = "human"
        ann.confidence = 1.0
        box = self._box_for_ann(ann)
        if box:
            box.set_human()
        self._refresh_ann_list()
        if self._autosave_cb.isChecked():
            self.save_annotations()

    def _accept_suggestions(self):
        """Aceptar las sugerencias seleccionadas; si no hay selección, todas."""
        targets = self._selected_provisional() or self._all_provisional()
        if not targets:
            return
        self._push_undo()
        for ann in targets:
            ann.source = "human"
            ann.confidence = 1.0
            box = self._box_for_ann(ann)
            if box:
                box.set_human()
        self._refresh_ann_list()
        if self._autosave_cb.isChecked():
            self.save_annotations()
        self._status_label.setText(f"{len(targets)} sugerencia(s) aceptada(s)")

    def _reject_suggestions(self):
        """Rechazar las sugerencias seleccionadas; si no hay selección, todas."""
        targets = self._selected_provisional() or self._all_provisional()
        if not targets:
            return
        self._push_undo()
        for ann in list(targets):
            self._delete_ann(ann, push_undo=False)
        if self._autosave_cb.isChecked():
            self.save_annotations()
        self._status_label.setText(f"{len(targets)} sugerencia(s) rechazada(s)")

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
        sel = self._selected_annotations()
        if sel:
            self._delete_anns(sel)   # batch (o una sola), con un solo undo
        else:
            # fall back to whatever is selected in the scene
            self._scene.delete_selected_boxes()

    def _delete_ann(self, ann: Annotation, push_undo: bool = False):
        if push_undo:
            self._push_undo()
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
                ann.source, ann.confidence, polygon=ann.polygon,
            )
            ann.id = new_id
        self.annotation_saved.emit(self._image.id)
        self._status_label.setText(f"Guardado — {len(self._pending_annotations)} anotaciones")

    # ── Draw mode / status ──

    def _toggle_draw_mode(self, enabled: bool):
        if enabled and self._sam_btn.isChecked():
            self._sam_btn.setChecked(False)   # exclusión mutua (dispara su toggle)
        self._scene.set_draw_mode(enabled)
        # In draw mode disable rubber band so clicks go to scene for drawing
        self._view.setDragMode(
            QGraphicsView.NoDrag if enabled else QGraphicsView.RubberBandDrag
        )
        self._update_mode_label()

    def _update_mode_label(self):
        if self._sam_btn.isChecked():
            self._mode_label.setText("🪄  MODO SAM — izq=+  der=−  Enter=confirmar  Esc=cancelar")
            self._mode_label.setStyleSheet(
                "background:#5a2a8a; color:#fff; font-weight:bold;"
                " border-radius:3px; padding:3px 8px;")
        elif self._draw_btn.isChecked():
            self._mode_label.setText("✏️  MODO DIBUJO — click y arrastrá para crear bbox")
            self._mode_label.setStyleSheet(
                "background:#8a5a00; color:#fff; font-weight:bold;"
                " border-radius:3px; padding:3px 8px;")
        else:
            self._mode_label.setText("↔  MODO EDICIÓN — click para seleccionar / mover")
            self._mode_label.setStyleSheet(
                "background:#234e7a; color:#fff; font-weight:bold;"
                " border-radius:3px; padding:3px 8px;")

    # ── SAM interactivo (click-to-box) ──

    def _toggle_sam_mode(self, enabled: bool):
        if enabled and self._draw_btn.isChecked():
            self._draw_btn.setChecked(False)
        self._scene.set_sam_mode(enabled)
        self._view.setDragMode(
            QGraphicsView.NoDrag if enabled else QGraphicsView.RubberBandDrag
        )
        if enabled:
            self._ensure_sam_thread()
            if self._sam_loaded:
                self._sam_request_current_image()
            else:
                self._status_label.setText("SAM: cargando modelo (primera vez descarga)...")
        else:
            self._sam_reset()
        self._update_mode_label()

    def _ensure_sam_thread(self):
        if self._sam_thread is not None:
            return
        self._sam_thread = QThread()
        self._sam_worker = _SamWorker(self._sam_model_name)
        self._sam_worker.moveToThread(self._sam_thread)
        self._sam_load_requested.connect(self._sam_worker.load)
        self._sam_image_requested.connect(self._sam_worker.set_image)
        self._sam_predict_requested.connect(self._sam_worker.predict)
        self._sam_worker.loaded.connect(self._on_sam_loaded)
        self._sam_worker.image_ready.connect(self._on_sam_image_ready)
        self._sam_worker.box_ready.connect(self._on_sam_box_ready)
        self._sam_worker.failed.connect(self._on_sam_failed)
        self._sam_thread.start()
        self._sam_load_requested.emit()

    def _sam_request_current_image(self):
        if not self._sam_loaded or self._image is None:
            return
        path = self._image.path
        if self._sam_image_path == path and not self._sam_busy:
            self._status_label.setText("SAM: clickeá un objeto")
            return
        self._sam_pending_image = path
        self._sam_busy = True
        self._status_label.setText("SAM: preparando imagen...")
        self._sam_image_requested.emit(path)

    def _on_sam_loaded(self):
        self._sam_loaded = True
        if self._sam_btn.isChecked() and self._image is not None:
            self._sam_request_current_image()

    def _on_sam_image_ready(self):
        self._sam_image_path = self._sam_pending_image
        self._sam_busy = False
        self._status_label.setText("SAM: clickeá un objeto")

    def _on_sam_point(self, px: float, py: float, positive: bool):
        if not self._sam_btn.isChecked():
            return
        if not self._sam_loaded:
            self._status_label.setText("SAM: cargando modelo, esperá...")
            return
        if self._image is not None and self._sam_image_path != self._image.path:
            self._sam_request_current_image()
            return
        # Acumular el punto en el objeto activo y re-predecir
        self._sam_points.append((px, py))
        self._sam_labels.append(1 if positive else 0)
        self._redraw_sam_overlay()
        self._request_sam_predict()

    def _request_sam_predict(self):
        if not self._sam_points:
            return
        if 1 not in self._sam_labels:
            self._status_label.setText(
                "SAM: agregá un punto positivo (clic izquierdo) primero")
            return
        if self._sam_busy:
            self._sam_pending_predict = True
            return
        self._sam_busy = True
        self._sam_pending_predict = False
        n_pos = sum(self._sam_labels)
        n_neg = len(self._sam_labels) - n_pos
        self._status_label.setText(
            f"SAM: segmentando ({n_pos}+/{n_neg}-)…  Enter=confirmar  Esc=cancelar")
        self._sam_predict_requested.emit(
            (list(self._sam_points), list(self._sam_labels)))

    def _on_sam_box_ready(self, res):
        self._sam_busy = False
        if res:
            self._sam_preview = res
            self._status_label.setText(
                "SAM: Enter=confirmar · clic izq=+  der=−  · Esc=cancelar")
        else:
            self._sam_preview = None
            self._status_label.setText("SAM: sin máscara para esos puntos")
        self._redraw_sam_overlay()
        if self._sam_pending_predict:
            self._request_sam_predict()

    def _on_sam_failed(self, msg: str):
        self._sam_busy = False
        self._status_label.setText(f"SAM error: {msg[:80]}")

    # ── Overlay temporal del objeto activo ──

    def _clear_sam_overlay(self):
        for it in self._sam_overlay_items:
            try:
                self._scene.removeItem(it)
            except Exception:
                pass
        self._sam_overlay_items = []

    def _redraw_sam_overlay(self):
        self._clear_sam_overlay()
        if self._image is None:
            return
        W = self._scene.sceneRect().width()
        H = self._scene.sceneRect().height()
        # Caja preview (punteada)
        if self._sam_preview and self._sam_preview.get("box"):
            x, y, w, h = self._sam_preview["box"]
            pen = QPen(QColor("#00e5ff"), 2, Qt.DashLine)
            pen.setCosmetic(True)
            fill = QColor(0, 229, 255, 40)
            rect_item = self._scene.addRect(x * W, y * H, w * W, h * H,
                                            pen, QBrush(fill))
            rect_item.setZValue(40)
            self._sam_overlay_items.append(rect_item)
        # Marcadores de puntos (verde=+, rojo=−)
        r = max(4.0, min(W, H) * 0.006)
        for (px, py), lab in zip(self._sam_points, self._sam_labels):
            col = QColor("#2ecc71") if lab == 1 else QColor("#e74c3c")
            dot = self._scene.addEllipse(px - r, py - r, 2 * r, 2 * r,
                                         QPen(QColor("#000"), 1), QBrush(col))
            dot.setZValue(41)
            self._sam_overlay_items.append(dot)

    def _sam_reset(self):
        self._sam_points = []
        self._sam_labels = []
        self._sam_preview = None
        self._sam_pending_predict = False
        self._clear_sam_overlay()

    def _sam_commit(self):
        if not self._sam_preview or not self._sam_preview.get("box"):
            self._sam_reset()
            return
        box = self._sam_preview["box"]
        polygon = self._sam_preview.get("polygon")
        self._sam_reset()
        self._add_sam_box(box, polygon)
        self._status_label.setText("SAM: caja confirmada ✓ — clickeá otro objeto")

    def _add_sam_box(self, box, polygon):
        if self._image is None:
            return
        x, y, bw, bh = box
        cid = self._scene._current_class_id
        name = self._scene._current_class_name
        color = self._scene._current_class_color
        ann = Annotation(class_id=cid, class_name=name, x=x, y=y,
                         width=bw, height=bh, source="human", confidence=1.0,
                         polygon=polygon)
        w = self._scene.sceneRect().width()
        h = self._scene.sceneRect().height()
        rect = QRectF(x * w, y * h, bw * w, bh * h)
        box_item = self._scene._add_box(rect, ann, QColor(color))
        box_item.add_polygon_overlay()
        self._on_box_updated(ann)     # push undo + append + refresh + autosave
        box_item.setSelected(True)

    def shutdown_sam(self):
        if self._sam_thread is not None:
            self._sam_thread.quit()
            self._sam_thread.wait(3000)
            self._sam_thread = None

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
            if self._sam_btn.isChecked() and event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._sam_commit()
                return True
            if self._sam_btn.isChecked() and event.key() == Qt.Key_Escape:
                self._sam_reset()
                self._status_label.setText("SAM: objeto cancelado")
                return True
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
            if event.key() == Qt.Key_A:
                self._accept_suggestions()
                return True
            if event.key() == Qt.Key_D:
                self._draw_btn.setChecked(not self._draw_btn.isChecked())
                return True
            if event.key() == Qt.Key_S and not (event.modifiers() & Qt.ControlModifier):
                self._sam_btn.setChecked(not self._sam_btn.isChecked())
                return True
            if event.matches(QKeySequence.Undo):
                self.undo()
                return True
            if event.matches(QKeySequence.Redo):
                self.redo()
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if self._sam_btn.isChecked() and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._sam_commit()
        elif self._sam_btn.isChecked() and event.key() == Qt.Key_Escape:
            self._sam_reset()
            self._status_label.setText("SAM: objeto cancelado")
        elif event.key() == Qt.Key_Left:
            self.navigate_request.emit(-1)
        elif event.key() == Qt.Key_Right:
            self.navigate_request.emit(1)
        elif event.key() == Qt.Key_R:
            self._mark_reviewed()
        elif event.key() == Qt.Key_X:
            self._set_image_status("discarded")
        elif event.key() == Qt.Key_D:
            self._draw_btn.setChecked(not self._draw_btn.isChecked())
        elif event.key() == Qt.Key_S and not (event.modifiers() & Qt.ControlModifier):
            self._sam_btn.setChecked(not self._sam_btn.isChecked())
        elif event.key() == Qt.Key_A:
            self._accept_suggestions()
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
