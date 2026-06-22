from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QDialogButtonBox,
    QColorDialog, QInputDialog, QComboBox, QCheckBox, QGroupBox,
    QStackedWidget, QRadioButton, QButtonGroup, QMenu, QListWidget,
    QListWidgetItem, QDoubleSpinBox, QSpinBox, QFormLayout, QWidget,
    QPlainTextEdit,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap

from ..utils.config import DEFAULT_SAM3_MODEL, CLUSTER_COLORS


class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nuevo Proyecto")
        self.setMinimumWidth(500)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        layout.addWidget(QLabel("Nombre del proyecto:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("mi_dataset")
        layout.addWidget(self._name_edit)

        layout.addWidget(QLabel("Carpeta de destino:"))
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Seleccionar carpeta...")
        dir_row.addWidget(self._dir_edit)
        browse_btn = QPushButton("Examinar...")
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta")
        if d:
            self._dir_edit.setText(d)

    def _validate(self):
        name = self._name_edit.text().strip()
        directory = self._dir_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "El nombre no puede estar vacío.")
            return
        if not directory or not Path(directory).exists():
            QMessageBox.warning(self, "Error", "Selecciona una carpeta válida.")
            return
        project_path = Path(directory) / name
        if project_path.exists():
            QMessageBox.warning(self, "Error", f"Ya existe un proyecto en:\n{project_path}")
            return
        self.accept()

    @property
    def project_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def project_dir(self) -> Path:
        return Path(self._dir_edit.text().strip())


class ClassManagerDialog(QDialog):
    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Administrar Clases")
        self.setMinimumSize(440, 400)
        self._pending_color = self._next_color()
        self._setup_ui()
        self._refresh()

    def _next_color(self) -> str:
        """Primer color de la paleta no usado por una clase existente."""
        used = {c.upper() for _, _, c in self.db.get_all_classes()}
        for color in CLUSTER_COLORS:
            if color.upper() not in used:
                return color
        # Si se agotó la paleta, cicla por cantidad de clases
        return CLUSTER_COLORS[len(used) % len(CLUSTER_COLORS)]

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Clases existentes (doble clic para cambiar color):"))
        self._list = QListWidget()
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._context_menu)
        self._list.itemDoubleClicked.connect(self._edit_color)
        layout.addWidget(self._list)

        layout.addWidget(QLabel("Nueva clase:"))
        add_row = QHBoxLayout()
        self._new_name = QLineEdit()
        self._new_name.setPlaceholderText("Nombre de clase...")
        self._new_name.returnPressed.connect(self._add_class)
        add_row.addWidget(self._new_name)

        self._color_btn = QPushButton()
        self._color_btn.setFixedWidth(36)
        self._color_btn.setToolTip("Elegir color")
        self._color_btn.clicked.connect(self._pick_new_color)
        self._apply_color_btn_style()
        add_row.addWidget(self._color_btn)

        add_btn = QPushButton("Agregar")
        add_btn.clicked.connect(self._add_class)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.accept)
        layout.addWidget(btn_box)

    def _apply_color_btn_style(self):
        self._color_btn.setStyleSheet(
            f"background-color: {self._pending_color}; border: 1px solid #555; border-radius: 3px;"
        )

    def _pick_new_color(self):
        color = QColorDialog.getColor(QColor(self._pending_color), self, "Elegir color")
        if color.isValid():
            self._pending_color = color.name()
            self._apply_color_btn_style()

    def _refresh(self):
        self._list.clear()
        for row in self.db.get_all_classes():
            class_id, name, color = row
            item = QListWidgetItem(f"  {name}")
            item.setData(Qt.UserRole, (class_id, name, color))
            pix = QPixmap(16, 16)
            pix.fill(QColor(color))
            item.setIcon(QIcon(pix))
            self._list.addItem(item)

    def _add_class(self):
        name = self._new_name.text().strip()
        if not name:
            return
        self.db.get_or_create_class(name, self._pending_color)
        self._new_name.clear()
        # Avanzar al próximo color distinto para la siguiente clase
        self._pending_color = self._next_color()
        self._apply_color_btn_style()
        self._refresh()

    def _edit_color(self, item):
        data = item.data(Qt.UserRole)
        if not data:
            return
        class_id, name, current_color = data
        color = QColorDialog.getColor(QColor(current_color), self, f"Color para '{name}'")
        if color.isValid():
            self.db.update_class_color(class_id, color.name())
            self._refresh()

    def _context_menu(self, pos):
        item = self._list.itemAt(pos)
        if not item:
            return
        data = item.data(Qt.UserRole)
        if not data:
            return
        class_id, name, color = data
        menu = QMenu(self)
        menu.addAction("Cambiar color...", lambda: self._edit_color(item))
        menu.addSeparator()
        delete_action = menu.addAction(f"Eliminar '{name}'")
        delete_action.triggered.connect(lambda: self._delete_class(class_id, name))
        menu.exec(self._list.mapToGlobal(pos))

    def _delete_class(self, class_id: int, name: str):
        n = self.db.count_annotations_for_class(class_id)
        if n > 0:
            msg = QMessageBox.question(
                self, "Confirmar eliminación",
                f"La clase '{name}' tiene {n} anotación(es).\n"
                "¿Eliminarla junto con todas sus anotaciones?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if msg != QMessageBox.Yes:
                return
        self.db.delete_class(class_id)
        self._refresh()


class ExportDialog(QDialog):
    def __init__(self, parent=None, default_name: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Exportar Dataset")
        self.setMinimumWidth(450)
        self._default_name = default_name
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Formato:"))
        self._format_combo = QComboBox()
        self._format_combo.addItems(["YOLO Detection", "COCO Detection"])
        layout.addWidget(self._format_combo)

        layout.addWidget(QLabel("Nombre del dataset:"))
        self._name_edit = QLineEdit(self._default_name)
        self._name_edit.setPlaceholderText("mi_dataset")
        layout.addWidget(self._name_edit)

        layout.addWidget(QLabel("Carpeta de exportación:"))
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        dir_row.addWidget(self._dir_edit)
        browse_btn = QPushButton("Examinar...")
        browse_btn.clicked.connect(self._browse)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        self._only_reviewed = QCheckBox("Solo imágenes revisadas")
        layout.addWidget(self._only_reviewed)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Carpeta de exportación")
        if d:
            self._dir_edit.setText(d)

    def _validate(self):
        if not self._name_edit.text().strip():
            QMessageBox.warning(self, "Error", "Ingresa un nombre para el dataset.")
            return
        d = self._dir_edit.text().strip()
        if not d:
            QMessageBox.warning(self, "Error", "Selecciona una carpeta de exportación.")
            return
        self.accept()

    @property
    def export_format(self) -> str:
        return "yolo" if self._format_combo.currentIndex() == 0 else "coco"

    @property
    def dataset_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def export_dir(self) -> Path:
        return Path(self._dir_edit.text().strip()) / self.dataset_name

    @property
    def only_reviewed(self) -> bool:
        return self._only_reviewed.isChecked()


class LoadAnnotationsDialog(QDialog):
    """Dialog to select annotation format and source file/directory."""

    FORMATS = ["COCO JSON", "YOLO", "LabelMe JSON"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cargar Anotaciones Existentes")
        self.setMinimumWidth(520)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # --- Format selector ---
        layout.addWidget(QLabel("Formato de anotaciones:"))
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(self.FORMATS)
        self._fmt_combo.currentIndexChanged.connect(self._on_format_changed)
        layout.addWidget(self._fmt_combo)

        # --- Stacked options per format ---
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        # Page 0: COCO JSON
        coco_page = QGroupBox("Archivo JSON COCO")
        coco_layout = QVBoxLayout(coco_page)
        coco_row = QHBoxLayout()
        self._coco_path = QLineEdit()
        self._coco_path.setPlaceholderText("annotations.json / instances_train.json ...")
        coco_row.addWidget(self._coco_path)
        btn = QPushButton("Examinar...")
        btn.clicked.connect(self._browse_coco)
        coco_row.addWidget(btn)
        coco_layout.addLayout(coco_row)
        coco_layout.addWidget(QLabel(
            "Formato: COCO Detection JSON con campos images, annotations, categories."
        ))
        self._stack.addWidget(coco_page)

        # Page 1: YOLO
        yolo_page = QGroupBox("Directorio de etiquetas YOLO")
        yolo_layout = QVBoxLayout(yolo_page)

        yolo_layout.addWidget(QLabel("Carpeta con archivos .txt:"))
        labels_row = QHBoxLayout()
        self._yolo_labels_dir = QLineEdit()
        self._yolo_labels_dir.setPlaceholderText("labels/ ...")
        labels_row.addWidget(self._yolo_labels_dir)
        btn2 = QPushButton("Examinar...")
        btn2.clicked.connect(self._browse_yolo_labels)
        labels_row.addWidget(btn2)
        yolo_layout.addLayout(labels_row)

        yolo_layout.addWidget(QLabel("Archivo de nombres de clases (opcional):"))
        names_row = QHBoxLayout()
        self._yolo_names = QLineEdit()
        self._yolo_names.setPlaceholderText("classes.txt / obj.names  (auto-detectado si se omite)")
        names_row.addWidget(self._yolo_names)
        btn3 = QPushButton("Examinar...")
        btn3.clicked.connect(self._browse_yolo_names)
        names_row.addWidget(btn3)
        yolo_layout.addLayout(names_row)
        yolo_layout.addWidget(QLabel(
            "Formato: un .txt por imagen, cada línea: class_idx cx cy w h (normalizado)."
        ))
        self._stack.addWidget(yolo_page)

        # Page 2: LabelMe
        lm_page = QGroupBox("Directorio de archivos LabelMe JSON")
        lm_layout = QVBoxLayout(lm_page)
        lm_row = QHBoxLayout()
        self._lm_dir = QLineEdit()
        self._lm_dir.setPlaceholderText("carpeta con .json por imagen...")
        lm_row.addWidget(self._lm_dir)
        btn4 = QPushButton("Examinar...")
        btn4.clicked.connect(self._browse_lm)
        lm_row.addWidget(btn4)
        lm_layout.addLayout(lm_row)
        lm_layout.addWidget(QLabel(
            "Soporta shapes: rectangle y polygon (usa bounding rect del polígono)."
        ))
        self._stack.addWidget(lm_page)

        # --- Options ---
        self._overwrite = QCheckBox(
            "Reemplazar anotaciones existentes en imágenes coincidentes"
        )
        layout.addWidget(self._overwrite)

        layout.addWidget(QLabel(
            "Nota: solo se importan anotaciones de imágenes ya indexadas en el proyecto.\n"
            "Carga primero la carpeta de imágenes si aún no lo hiciste."
        ))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_format_changed(self, idx: int):
        self._stack.setCurrentIndex(idx)

    # --- Browse buttons ---
    def _browse_coco(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar COCO JSON", "", "JSON (*.json)"
        )
        if f:
            self._coco_path.setText(f)

    def _browse_yolo_labels(self):
        d = QFileDialog.getExistingDirectory(self, "Carpeta de etiquetas YOLO")
        if d:
            self._yolo_labels_dir.setText(d)

    def _browse_yolo_names(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Archivo de nombres de clases", "",
            "Texto (*.txt *.names)"
        )
        if f:
            self._yolo_names.setText(f)

    def _browse_lm(self):
        d = QFileDialog.getExistingDirectory(self, "Carpeta de archivos LabelMe")
        if d:
            self._lm_dir.setText(d)

    # --- Validation ---
    def _validate(self):
        fmt = self._fmt_combo.currentIndex()
        if fmt == 0:
            if not self._coco_path.text().strip():
                QMessageBox.warning(self, "Error", "Selecciona el archivo JSON de COCO.")
                return
            if not Path(self._coco_path.text().strip()).exists():
                QMessageBox.warning(self, "Error", "El archivo JSON no existe.")
                return
        elif fmt == 1:
            if not self._yolo_labels_dir.text().strip():
                QMessageBox.warning(self, "Error", "Selecciona la carpeta de etiquetas YOLO.")
                return
            if not Path(self._yolo_labels_dir.text().strip()).is_dir():
                QMessageBox.warning(self, "Error", "La carpeta de etiquetas no existe.")
                return
        elif fmt == 2:
            if not self._lm_dir.text().strip():
                QMessageBox.warning(self, "Error", "Selecciona la carpeta de JSONs LabelMe.")
                return
            if not Path(self._lm_dir.text().strip()).is_dir():
                QMessageBox.warning(self, "Error", "La carpeta no existe.")
                return
        self.accept()

    # --- Result properties ---
    @property
    def format(self) -> str:
        return ["coco", "yolo", "labelme"][self._fmt_combo.currentIndex()]

    @property
    def coco_json_path(self) -> Path:
        return Path(self._coco_path.text().strip())

    @property
    def yolo_labels_dir(self) -> Path:
        return Path(self._yolo_labels_dir.text().strip())

    @property
    def yolo_names_file(self) -> Path | None:
        t = self._yolo_names.text().strip()
        return Path(t) if t else None

    @property
    def labelme_dir(self) -> Path:
        return Path(self._lm_dir.text().strip())

    @property
    def overwrite(self) -> bool:
        return self._overwrite.isChecked()


class AutoLabelDialog(QDialog):
    """Configura el auto-etiquetado: por texto (SAM 3) o por ejemplo visual (YOLOE)."""

    def __init__(self, parent=None, n_pending: int = 0, n_total: int = 0,
                 exemplar_classes: list | None = None,
                 text_only: bool = False,
                 fixed_scope_label: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Auto-etiquetar")
        self.setMinimumWidth(480)
        self._n_pending = n_pending
        self._n_total = n_total
        # exemplar_classes: [(class_id, name, n_human_boxes), ...]
        self._exemplar_classes = exemplar_classes or []
        self._text_only = text_only
        self._fixed_scope_label = fixed_scope_label
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Modo:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Por texto — SAM 3 (open-vocab)", "text")
        if not self._text_only:
            self._mode_combo.addItem("Por ejemplo visual — YOLOE (cross-image)", "visual")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self._mode_combo)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        # Página 0: texto (SAM 3)
        text_page = QGroupBox("Conceptos por texto")
        tl = QVBoxLayout(text_page)
        tl.addWidget(QLabel("Conceptos a detectar (separados por coma):"))
        self._prompts_edit = QLineEdit()
        self._prompts_edit.setPlaceholderText("insulator, transmission tower, bird nest")
        tl.addWidget(self._prompts_edit)
        tl.addWidget(QLabel(
            "Cada concepto se crea como clase y SAM 3 detecta sus instancias.\n"
            "Frases simples en inglés ('red apple', 'person wearing a hat')."
        ))
        tl.addWidget(QLabel("Modelo SAM 3 (.pt):"))
        model_row = QHBoxLayout()
        self._model_edit = QLineEdit(DEFAULT_SAM3_MODEL)
        self._model_edit.setPlaceholderText("ruta a sam3.pt")
        model_row.addWidget(self._model_edit)
        model_btn = QPushButton("Examinar...")
        model_btn.clicked.connect(self._browse_model)
        model_row.addWidget(model_btn)
        tl.addLayout(model_row)
        self._save_polygons_cb = QCheckBox("Guardar y mostrar polígonos de segmentación")
        self._save_polygons_cb.setChecked(True)
        tl.addWidget(self._save_polygons_cb)
        self._stack.addWidget(text_page)

        # Página 1: ejemplo visual (YOLOE)
        visual_page = QGroupBox("Ejemplo visual")
        vl = QVBoxLayout(visual_page)
        vl.addWidget(QLabel("Clase de referencia (usa sus cajas anotadas como ejemplo):"))
        self._class_combo = QComboBox()
        for class_id, name, n in self._exemplar_classes:
            self._class_combo.addItem(f"{name}  ({n} cajas)", class_id)
        vl.addWidget(self._class_combo)
        if not self._exemplar_classes:
            vl.addWidget(QLabel(
                "No hay clases con anotaciones humanas. Anotá al menos una caja\n"
                "a mano para usar el modo por ejemplo visual."
            ))
        else:
            vl.addWidget(QLabel(
                "YOLOE toma la imagen con más cajas de esa clase como referencia\n"
                "y busca objetos similares en las imágenes objetivo."
            ))
        self._stack.addWidget(visual_page)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confianza mínima:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.40)
        conf_row.addWidget(self._conf_spin)
        conf_row.addStretch()
        layout.addLayout(conf_row)

        self._scope_combo = QComboBox()
        if self._fixed_scope_label:
            layout.addWidget(QLabel("Aplicar a:"))
            layout.addWidget(QLabel(self._fixed_scope_label))
            self._scope_combo.addItem(self._fixed_scope_label, "fixed")
        else:
            layout.addWidget(QLabel("Aplicar a:"))
            self._scope_combo.addItem(
                f"Solo imágenes pendientes ({self._n_pending})", "pending")
            self._scope_combo.addItem(
                f"Todas las imágenes ({self._n_total})", "all")
            layout.addWidget(self._scope_combo)

        layout.addWidget(QLabel("Guardar como:"))
        self._source_combo = QComboBox()
        self._source_combo.addItem("Sugerencias (para revisar)", "suggested")
        self._source_combo.addItem("Etiquetas (source=yolo)", "yolo")
        layout.addWidget(self._source_combo)

        imgsz_row = QHBoxLayout()
        imgsz_row.addWidget(QLabel("Tamaño de imagen:"))
        self._imgsz_combo = QComboBox()
        for sz in (512, 640, 768, 1024):
            self._imgsz_combo.addItem(str(sz), sz)
        self._imgsz_combo.setCurrentIndex(1)   # 640 (seguro para GPUs de 8 GB)
        self._imgsz_combo.setToolTip(
            "Bajalo si la GPU se queda sin memoria (8 GB → 640 o 512). "
            "Más grande = más preciso pero más VRAM.")
        imgsz_row.addWidget(self._imgsz_combo)
        imgsz_row.addStretch()
        layout.addLayout(imgsz_row)

        layout.addWidget(QLabel(
            "Modelos pesados: se recomienda GPU (CUDA). La primera vez descarga el modelo.\n"
            "Nota: SAM 3 (sam3.pt) requiere acceso aprobado en HuggingFace; YOLOE se baja solo."
        ))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Auto-etiquetar")
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_mode_changed(self, idx: int):
        self._stack.setCurrentIndex(idx)
        # YOLOE cross-image suele necesitar confianza más baja que SAM 3 por texto
        self._conf_spin.setValue(0.20 if self.mode == "visual" else 0.40)

    def _browse_model(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar modelo SAM 3", self._model_edit.text(),
            "PyTorch (*.pt)")
        if f:
            self._model_edit.setText(f)

    def _validate(self):
        if self.mode == "text":
            if not self.prompts:
                QMessageBox.warning(self, "Error", "Ingresá al menos un concepto a detectar.")
                return
            if not Path(self.model_path).is_file():
                QMessageBox.warning(
                    self, "Modelo no encontrado",
                    f"No se encontró el modelo SAM 3 en:\n{self.model_path}\n\n"
                    "Indicá la ruta a sam3.pt con 'Examinar...'.")
                return
        else:
            if not self._exemplar_classes:
                QMessageBox.warning(
                    self, "Error",
                    "No hay clases con anotaciones humanas para usar como ejemplo.")
                return
        self.accept()

    @property
    def mode(self) -> str:
        return self._mode_combo.currentData()

    @property
    def prompts(self) -> list[str]:
        raw = self._prompts_edit.text().replace("\n", ",")
        return [p.strip() for p in raw.split(",") if p.strip()]

    @property
    def model_path(self) -> str:
        return self._model_edit.text().strip()

    @property
    def class_id(self) -> int:
        return self._class_combo.currentData()

    @property
    def class_name(self) -> str:
        return self._class_combo.currentText().split("  (")[0]

    @property
    def conf(self) -> float:
        return self._conf_spin.value()

    @property
    def scope(self) -> str:
        return self._scope_combo.currentData()

    @property
    def source(self) -> str:
        return self._source_combo.currentData()

    @property
    def save_polygons(self) -> bool:
        return self.mode == "text" and self._save_polygons_cb.isChecked()

    @property
    def imgsz(self) -> int:
        return self._imgsz_combo.currentData()


class TrainModelDialog(QDialog):
    """Configura el entrenamiento rápido de un YOLO nano con las cajas revisadas
    del proyecto, para asistir el etiquetado de las pendientes."""

    MODELS = [
        ("YOLO11n (nano, más rápido)", "yolo11n.pt"),
        ("YOLO11s (small, más preciso)", "yolo11s.pt"),
        ("YOLOv8n (nano, clásico)", "yolov8n.pt"),
    ]

    def __init__(self, parent=None, n_reviewed: int = 0, n_pending: int = 0,
                 classes: list[tuple] | None = None,
                 class_counts: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Entrenar modelo asistente (YOLO)")
        self.setMinimumWidth(520)
        self._classes = classes or []
        self._class_counts = class_counts or {}
        self._setup_ui(n_reviewed, n_pending)

    def _setup_ui(self, n_reviewed: int, n_pending: int):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        info = QLabel(
            f"Se entrenará un modelo con tus <b>{n_reviewed}</b> imágenes revisadas "
            f"(cajas hechas/aceptadas a mano) y luego puede pre-etiquetar las "
            f"<b>{n_pending}</b> pendientes como sugerencias para revisar."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#bbb;")
        layout.addWidget(info)

        if n_reviewed < 5:
            warn = QLabel(
                "⚠ Tenés pocas imágenes revisadas. El modelo será débil; "
                "revisá/anotá más antes de entrenar para mejores sugerencias."
            )
            warn.setWordWrap(True)
            warn.setStyleSheet("color:#e0a030;")
            layout.addWidget(warn)

        # ── Clases a entrenar (con cantidad de cajas revisadas) ──
        cls_box = QGroupBox("Clases a entrenar")
        cls_layout = QVBoxLayout(cls_box)
        cls_hint = QLabel(
            "Tildá las clases a incluir. Destildá las que tengan pocos ejemplos."
        )
        cls_hint.setStyleSheet("color:#888; font-size:11px;")
        cls_hint.setWordWrap(True)
        cls_layout.addWidget(cls_hint)

        self._train_classes_list = QListWidget()
        self._train_classes_list.setMaximumHeight(130)
        for class_id, name, color in self._classes:
            n = int(self._class_counts.get(class_id, 0))
            item = QListWidgetItem(f"{name}  ({n} cajas)")
            pix = QPixmap(12, 12)
            pix.fill(QColor(color))
            item.setIcon(QIcon(pix))
            item.setData(Qt.UserRole, name)
            if n > 0:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                if n < 5:
                    item.setForeground(QColor("#e0a030"))   # pocas: aviso visual
            else:
                # Sin ejemplos revisados → no se puede entrenar esta clase
                item.setFlags(Qt.ItemIsUserCheckable)        # deshabilitada
                item.setCheckState(Qt.Unchecked)
                item.setForeground(QColor("#666"))
            self._train_classes_list.addItem(item)
        cls_layout.addWidget(self._train_classes_list)

        cls_btn_row = QHBoxLayout()
        all_btn = QPushButton("Todas")
        all_btn.clicked.connect(lambda: self._set_all_train_classes(True))
        none_btn = QPushButton("Ninguna")
        none_btn.clicked.connect(lambda: self._set_all_train_classes(False))
        cls_btn_row.addWidget(all_btn)
        cls_btn_row.addWidget(none_btn)
        cls_btn_row.addStretch()
        cls_layout.addLayout(cls_btn_row)
        layout.addWidget(cls_box)

        # ── Modelo base + hiperparámetros ──
        form_box = QGroupBox("Modelo y entrenamiento")
        form = QFormLayout(form_box)

        self._model_combo = QComboBox()
        for label, value in self.MODELS:
            self._model_combo.addItem(label, value)
        form.addRow("Modelo base:", self._model_combo)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(5, 1000)
        self._epochs_spin.setValue(50)
        form.addRow("Épocas:", self._epochs_spin)

        self._imgsz_combo = QComboBox()
        for sz in (512, 640, 768, 1024):
            self._imgsz_combo.addItem(str(sz), sz)
        self._imgsz_combo.setCurrentIndex(1)   # 640
        form.addRow("Tamaño de imagen:", self._imgsz_combo)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 64)
        self._batch_spin.setValue(8)
        self._batch_spin.setToolTip("Bajalo si te quedás sin memoria de GPU")
        form.addRow("Batch:", self._batch_spin)

        self._val_spin = QDoubleSpinBox()
        self._val_spin.setRange(0.0, 0.5)
        self._val_spin.setSingleStep(0.05)
        self._val_spin.setDecimals(2)
        self._val_spin.setValue(0.2)
        form.addRow("Fracción validación:", self._val_spin)

        layout.addWidget(form_box)

        # ── Pre-etiquetado tras entrenar ──
        self._predict_cb = QCheckBox("Pre-etiquetar imágenes al terminar")
        self._predict_cb.setChecked(True)
        self._predict_cb.toggled.connect(self._toggle_predict)
        layout.addWidget(self._predict_cb)

        pred_box = QGroupBox("Pre-etiquetado")
        pred_form = QFormLayout(pred_box)

        scope_lbl = QLabel("Solo imágenes pendientes")
        scope_lbl.setStyleSheet("color:#888; font-size:11px;")
        pred_form.addRow("Alcance:", scope_lbl)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(0.25)
        pred_form.addRow("Confianza mínima:", self._conf_spin)

        self._pred_box = pred_box
        layout.addWidget(pred_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Entrenar")
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _toggle_predict(self, on: bool):
        self._pred_box.setEnabled(on)

    def _set_all_train_classes(self, checked: bool):
        for row in range(self._train_classes_list.count()):
            it = self._train_classes_list.item(row)
            if it.flags() & Qt.ItemIsUserCheckable and it.flags() & Qt.ItemIsEnabled:
                it.setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _validate(self):
        if not self.train_classes:
            QMessageBox.warning(
                self, "Entrenar modelo",
                "Seleccioná al menos una clase con ejemplos para entrenar.")
            return
        self.accept()

    # ── Resultados ──

    @property
    def model_name(self) -> str:
        return self._model_combo.currentData()

    @property
    def epochs(self) -> int:
        return self._epochs_spin.value()

    @property
    def imgsz(self) -> int:
        return self._imgsz_combo.currentData()

    @property
    def batch(self) -> int:
        return self._batch_spin.value()

    @property
    def val_fraction(self) -> float:
        return self._val_spin.value()

    @property
    def predict_after(self) -> bool:
        return self._predict_cb.isChecked()

    @property
    def predict_scope(self) -> str:
        return "pending"

    @property
    def predict_conf(self) -> float:
        return self._conf_spin.value()

    @property
    def train_classes(self) -> list[str]:
        """Nombres de clases tildadas para entrenar."""
        names = []
        for row in range(self._train_classes_list.count()):
            it = self._train_classes_list.item(row)
            if it.checkState() == Qt.Checked:
                names.append(it.data(Qt.UserRole))
        return names


class PredictWithModelDialog(QDialog):
    """Usa un modelo .pt ya entrenado (de este u otro proyecto) para pre-etiquetar
    las imágenes del proyecto actual. Las clases se mapean por nombre."""

    def __init__(self, parent=None, n_pending: int = 0,
                 default_dir: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Pre-etiquetar con modelo entrenado (.pt)")
        self.setMinimumWidth(540)
        self._default_dir = default_dir
        self._setup_ui(n_pending)

    def _setup_ui(self, n_pending: int):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        info = QLabel(
            "Elegí un modelo <b>.pt</b> entrenado antes (en este u otro proyecto). "
            "Se usará para sugerir cajas; las clases del modelo se crean en este "
            "proyecto si no existen (mapeo por nombre)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#bbb;")
        layout.addWidget(info)

        layout.addWidget(QLabel("Modelo (.pt):"))
        file_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Seleccionar best.pt...")
        file_row.addWidget(self._path_edit)
        browse = QPushButton("Examinar...")
        browse.clicked.connect(self._browse)
        file_row.addWidget(browse)
        layout.addLayout(file_row)

        form_box = QGroupBox("Pre-etiquetado")
        form = QFormLayout(form_box)

        scope_lbl = QLabel("Solo imágenes pendientes")
        scope_lbl.setStyleSheet("color:#888; font-size:11px;")
        form.addRow("Alcance:", scope_lbl)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(0.25)
        form.addRow("Confianza mínima:", self._conf_spin)

        self._imgsz_combo = QComboBox()
        for sz in (512, 640, 768, 1024):
            self._imgsz_combo.addItem(str(sz), sz)
        self._imgsz_combo.setCurrentIndex(1)
        form.addRow("Tamaño de imagen:", self._imgsz_combo)

        layout.addWidget(form_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Pre-etiquetar")
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        start = self._default_dir or ""
        f, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar modelo YOLO", start, "Modelos PyTorch (*.pt)")
        if f:
            self._path_edit.setText(f)

    def _validate(self):
        p = self._path_edit.text().strip()
        if not p or not Path(p).exists():
            QMessageBox.warning(self, "Modelo", "Seleccioná un archivo .pt válido.")
            return
        self.accept()

    @property
    def weights_path(self) -> Path:
        return Path(self._path_edit.text().strip())

    @property
    def scope(self) -> str:
        return "pending"

    @property
    def conf(self) -> float:
        return self._conf_spin.value()

    @property
    def imgsz(self) -> int:
        return self._imgsz_combo.currentData()


class NearDuplicatesDialog(QDialog):
    """Configura la detección de imágenes casi-duplicadas (frames repetidos de
    video de drone) usando los embeddings ya calculados."""

    def __init__(self, parent=None, n_with_embeddings: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Imágenes duplicadas (near-duplicates)")
        self.setMinimumWidth(540)
        self._setup_ui(n_with_embeddings)

    def _setup_ui(self, n_with_embeddings: int):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        info = QLabel(
            f"Agrupa imágenes muy parecidas usando los embeddings "
            f"(<b>{n_with_embeddings}</b> disponibles). Ideal para frames casi "
            f"idénticos de video: etiquetás uno por grupo y propagás al resto."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#bbb;")
        layout.addWidget(info)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Umbral de similitud:"))
        self._thr_spin = QDoubleSpinBox()
        self._thr_spin.setRange(0.80, 0.999)
        self._thr_spin.setSingleStep(0.005)
        self._thr_spin.setDecimals(3)
        self._thr_spin.setValue(0.95)
        self._thr_spin.setToolTip(
            "Más alto = solo casi idénticas. Más bajo = agrupa más (pero arriesga "
            "juntar imágenes distintas).")
        thr_row.addWidget(self._thr_spin)
        thr_row.addStretch()
        layout.addLayout(thr_row)

        action_box = QGroupBox("Acción")
        action_layout = QVBoxLayout(action_box)
        self._action_group = QButtonGroup(self)

        self._rb_report = QRadioButton("Solo reportar (no modifica nada)")
        self._rb_report.setChecked(True)
        self._rb_collapse = QRadioButton(
            "Descartar duplicados — dejar 1 por grupo para etiquetar")
        self._rb_propagate = QRadioButton(
            "Propagar etiquetas dentro de cada grupo (usar tras etiquetar)")
        for rb in (self._rb_report, self._rb_collapse, self._rb_propagate):
            self._action_group.addButton(rb)
            action_layout.addWidget(rb)

        hint = QLabel(
            "Flujo sugerido: 1) Descartar duplicados → 2) etiquetar los "
            "representantes que quedan → 3) Propagar etiquetas para completar el "
            "resto del grupo como revisadas."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:11px;")
        action_layout.addWidget(hint)
        layout.addWidget(action_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Ejecutar")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def threshold(self) -> float:
        return self._thr_spin.value()

    @property
    def action(self) -> str:
        if self._rb_collapse.isChecked():
            return "collapse"
        if self._rb_propagate.isChecked():
            return "propagate"
        return "report"


class AutoLabelHubDialog(QDialog):
    """Centro unificado de auto-etiquetado: elegí motor (SAM 3 texto, YOLOE visual,
    entrenar YOLO nano, o modelo .pt) y alcance (imagen actual / grupo / pendientes
    / todas), todo desde un mismo lugar."""

    TRAIN_MODELS = [
        ("YOLO11n (nano, más rápido)", "yolo11n.pt"),
        ("YOLO11s (small, más preciso)", "yolo11s.pt"),
        ("YOLOv8n (nano, clásico)", "yolov8n.pt"),
    ]

    def __init__(self, parent=None, classes=None, exemplar_classes=None,
                 train_counts=None, default_sam3_model=DEFAULT_SAM3_MODEL,
                 models_dir=None, has_current=False, has_group=False,
                 n_pending=0, n_total=0, n_reviewed=0):
        super().__init__(parent)
        self.setWindowTitle("Auto-etiquetado / Modelos")
        self.setMinimumWidth(560)
        self._classes = classes or []
        self._exemplar_classes = exemplar_classes or []
        self._train_counts = train_counts or {}
        self._default_sam3_model = default_sam3_model
        self._models_dir = models_dir
        self._has_current = has_current
        self._has_group = has_group
        self._n_pending = n_pending
        self._n_total = n_total
        self._n_reviewed = n_reviewed
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        eng_row = QHBoxLayout()
        eng_row.addWidget(QLabel("Motor:"))
        self._engine_combo = QComboBox()
        self._engine_combo.addItem("SAM 3 — detección por texto", "sam3")
        self._engine_combo.addItem("YOLOE — prompt visual (cross-image)", "yoloe")
        self._engine_combo.addItem("Entrenar YOLO nano (con revisadas)", "train")
        self._engine_combo.addItem("Modelo .pt existente", "ptmodel")
        self._engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        eng_row.addWidget(self._engine_combo, 1)
        layout.addLayout(eng_row)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Alcance:"))
        self._scope_combo = QComboBox()
        if self._has_current:
            self._scope_combo.addItem("Imagen actual", "current")
        if self._has_group:
            self._scope_combo.addItem("Grupo seleccionado", "group")
        self._scope_combo.addItem(f"Solo pendientes ({self._n_pending})", "pending")
        self._scope_combo.addItem(f"Todas ({self._n_total})", "all")
        idx = self._scope_combo.findData("pending")
        if idx >= 0:
            self._scope_combo.setCurrentIndex(idx)
        scope_row.addWidget(self._scope_combo, 1)
        layout.addLayout(scope_row)
        self._scope_note = QLabel("")
        self._scope_note.setStyleSheet("color:#888; font-size:11px;")
        self._scope_note.setWordWrap(True)
        layout.addWidget(self._scope_note)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_sam3_page())
        self._stack.addWidget(self._build_yoloe_page())
        self._stack.addWidget(self._build_train_page())
        self._stack.addWidget(self._build_ptmodel_page())
        layout.addWidget(self._stack)

        common = QGroupBox("Opciones")
        cform = QFormLayout(common)
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(0.40)
        cform.addRow("Confianza mínima:", self._conf_spin)

        self._imgsz_combo = QComboBox()
        for sz in (512, 640, 768, 1024):
            self._imgsz_combo.addItem(str(sz), sz)
        self._imgsz_combo.setCurrentIndex(1)
        self._imgsz_combo.setToolTip("Bajalo si la GPU se queda sin memoria (8 GB → 640/512)")
        cform.addRow("Tamaño de imagen:", self._imgsz_combo)

        self._source_combo = QComboBox()
        self._source_combo.addItem("Sugerencias (para revisar)", "suggested")
        self._source_combo.addItem("Etiquetas (source=yolo)", "yolo")
        cform.addRow("Guardar como:", self._source_combo)
        layout.addWidget(common)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Ejecutar")
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_engine_changed(0)

    def _build_sam3_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Conceptos a detectar (separados por coma):"))
        self._prompts_edit = QLineEdit()
        self._prompts_edit.setPlaceholderText("insulator, transmission tower")
        lay.addWidget(self._prompts_edit)
        lay.addWidget(QLabel("Modelo SAM 3 (.pt):"))
        model_row = QHBoxLayout()
        self._model_edit = QLineEdit(self._default_sam3_model)
        model_row.addWidget(self._model_edit)
        browse = QPushButton("Examinar...")
        browse.clicked.connect(self._browse_sam3)
        model_row.addWidget(browse)
        lay.addLayout(model_row)
        self._save_poly_cb = QCheckBox("Guardar polígonos (máscaras) además de cajas")
        lay.addWidget(self._save_poly_cb)
        return w

    def _build_yoloe_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        if not self._exemplar_classes:
            warn = QLabel("No hay clases con cajas humanas para usar como ejemplo.\n"
                          "Anotá al menos un objeto a mano y guardá.")
            warn.setStyleSheet("color:#e0a030;")
            warn.setWordWrap(True)
            lay.addWidget(warn)
        lay.addWidget(QLabel("Clase de ejemplo (usa tus cajas humanas):"))
        self._yoloe_class_combo = QComboBox()
        for cid, name, n in self._exemplar_classes:
            self._yoloe_class_combo.addItem(f"{name}  ({n} cajas)", cid)
        lay.addWidget(self._yoloe_class_combo)
        return w

    def _build_train_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel(
            f"Entrena con tus {self._n_reviewed} imágenes revisadas y pre-etiqueta "
            f"el alcance elegido."))
        lay.addWidget(QLabel("Clases a entrenar:"))
        self._train_classes_list = QListWidget()
        self._train_classes_list.setMaximumHeight(110)
        for cid, name, color in self._classes:
            n = int(self._train_counts.get(cid, 0))
            it = QListWidgetItem(f"{name}  ({n} cajas)")
            pix = QPixmap(12, 12)
            pix.fill(QColor(color))
            it.setIcon(QIcon(pix))
            it.setData(Qt.UserRole, name)
            if n > 0:
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked)
                if n < 5:
                    it.setForeground(QColor("#e0a030"))
            else:
                it.setFlags(Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Unchecked)
                it.setForeground(QColor("#666"))
            self._train_classes_list.addItem(it)
        lay.addWidget(self._train_classes_list)

        form = QFormLayout()
        self._train_model_combo = QComboBox()
        for label, value in self.TRAIN_MODELS:
            self._train_model_combo.addItem(label, value)
        form.addRow("Modelo base:", self._train_model_combo)
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(5, 1000)
        self._epochs_spin.setValue(50)
        form.addRow("Épocas:", self._epochs_spin)
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 64)
        self._batch_spin.setValue(8)
        form.addRow("Batch:", self._batch_spin)
        self._val_spin = QDoubleSpinBox()
        self._val_spin.setRange(0.0, 0.5)
        self._val_spin.setSingleStep(0.05)
        self._val_spin.setDecimals(2)
        self._val_spin.setValue(0.2)
        form.addRow("Fracción validación:", self._val_spin)
        lay.addLayout(form)

        self._predict_after_cb = QCheckBox("Pre-etiquetar el alcance al terminar")
        self._predict_after_cb.setChecked(True)
        lay.addWidget(self._predict_after_cb)
        return w

    def _build_ptmodel_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Modelo .pt (de este u otro proyecto):"))
        row = QHBoxLayout()
        self._pt_edit = QLineEdit()
        self._pt_edit.setPlaceholderText("best.pt")
        row.addWidget(self._pt_edit)
        browse = QPushButton("Examinar...")
        browse.clicked.connect(self._browse_pt)
        row.addWidget(browse)
        lay.addLayout(row)
        note = QLabel("Las clases del modelo se crean en el proyecto si no existen "
                      "(mapeo por nombre).")
        note.setStyleSheet("color:#888; font-size:11px;")
        note.setWordWrap(True)
        lay.addWidget(note)
        return w

    def _on_engine_changed(self, _idx):
        eng = self.engine
        order = {"sam3": 0, "yoloe": 1, "train": 2, "ptmodel": 3}
        self._stack.setCurrentIndex(order[eng])
        self._conf_spin.setValue(0.20 if eng == "yoloe" else 0.40)
        if eng == "train":
            self._source_combo.setCurrentIndex(0)
            self._scope_note.setText(
                "El entrenamiento usa SOLO las imágenes revisadas; el alcance define "
                "dónde pre-etiquetar después.")
        else:
            self._scope_note.setText("")

    def _browse_sam3(self):
        f, _ = QFileDialog.getOpenFileName(self, "Modelo SAM 3", self._model_edit.text(),
                                           "PyTorch (*.pt)")
        if f:
            self._model_edit.setText(f)

    def _browse_pt(self):
        start = self._models_dir or ""
        f, _ = QFileDialog.getOpenFileName(self, "Modelo YOLO (.pt)", start,
                                           "Modelos PyTorch (*.pt)")
        if f:
            self._pt_edit.setText(f)

    def _validate(self):
        eng = self.engine
        if eng == "sam3":
            if not self.prompts:
                QMessageBox.warning(self, "Error", "Ingresá al menos un concepto.")
                return
            if not Path(self.model_path).is_file():
                QMessageBox.warning(self, "Modelo no encontrado",
                                    f"No se encontró SAM 3 en:\n{self.model_path}")
                return
        elif eng == "yoloe":
            if not self._exemplar_classes:
                QMessageBox.warning(self, "Error",
                                    "No hay clases con cajas humanas de ejemplo.")
                return
        elif eng == "train":
            if not self.train_classes:
                QMessageBox.warning(self, "Error",
                                    "Seleccioná al menos una clase con ejemplos.")
                return
        elif eng == "ptmodel":
            if not self.weights_path or not Path(self.weights_path).is_file():
                QMessageBox.warning(self, "Error", "Seleccioná un archivo .pt válido.")
                return
        self.accept()

    @property
    def engine(self) -> str:
        return self._engine_combo.currentData()

    @property
    def scope(self) -> str:
        return self._scope_combo.currentData()

    @property
    def conf(self) -> float:
        return self._conf_spin.value()

    @property
    def imgsz(self) -> int:
        return self._imgsz_combo.currentData()

    @property
    def source(self) -> str:
        return self._source_combo.currentData()

    @property
    def prompts(self) -> list:
        raw = self._prompts_edit.text().replace("\n", ",")
        return [p.strip() for p in raw.split(",") if p.strip()]

    @property
    def model_path(self) -> str:
        return self._model_edit.text().strip()

    @property
    def save_polygons(self) -> bool:
        return self._save_poly_cb.isChecked()

    @property
    def class_id(self) -> int:
        return self._yoloe_class_combo.currentData()

    @property
    def class_name(self) -> str:
        return self._yoloe_class_combo.currentText().split("  (")[0]

    @property
    def train_model(self) -> str:
        return self._train_model_combo.currentData()

    @property
    def epochs(self) -> int:
        return self._epochs_spin.value()

    @property
    def batch(self) -> int:
        return self._batch_spin.value()

    @property
    def val_fraction(self) -> float:
        return self._val_spin.value()

    @property
    def predict_after(self) -> bool:
        return self._predict_after_cb.isChecked()

    @property
    def train_classes(self) -> list:
        names = []
        for row in range(self._train_classes_list.count()):
            it = self._train_classes_list.item(row)
            if it.checkState() == Qt.Checked:
                names.append(it.data(Qt.UserRole))
        return names

    @property
    def weights_path(self) -> str:
        return self._pt_edit.text().strip()


class ActiveLearningDialog(QDialog):
    """Active learning: auto-acepta sugerencias muy confiables y prioriza la
    revisión de las dudosas. Opera sobre las imágenes pendientes."""

    def __init__(self, parent=None, n_pending: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Active learning — priorizar revisión")
        self.setMinimumWidth(520)
        self._setup_ui(n_pending)

    def _setup_ui(self, n_pending: int):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        info = QLabel(
            f"Sobre las <b>{n_pending}</b> imágenes pendientes con sugerencias de IA: "
            f"acepta automáticamente las cajas muy confiables y ordena el resto por "
            f"incertidumbre para que revises primero lo más dudoso."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#bbb;")
        layout.addWidget(info)

        # Auto-aceptar
        self._accept_cb = QCheckBox("Auto-aceptar sugerencias con confianza ≥")
        self._accept_cb.setChecked(True)
        acc_row = QHBoxLayout()
        acc_row.addWidget(self._accept_cb)
        self._accept_spin = QDoubleSpinBox()
        self._accept_spin.setRange(0.50, 0.99)
        self._accept_spin.setSingleStep(0.05)
        self._accept_spin.setDecimals(2)
        self._accept_spin.setValue(0.90)
        acc_row.addWidget(self._accept_spin)
        acc_row.addStretch()
        layout.addLayout(acc_row)

        self._reviewed_cb = QCheckBox(
            "Marcar como revisada si quedan solo cajas aceptadas (sin dudosas)")
        self._reviewed_cb.setChecked(True)
        layout.addWidget(self._reviewed_cb)

        # Priorizar
        self._prioritize_cb = QCheckBox(
            "Recalcular prioridad por incertidumbre y ordenar la grilla")
        self._prioritize_cb.setChecked(True)
        layout.addWidget(self._prioritize_cb)

        center_row = QHBoxLayout()
        center_lbl = QLabel("Centro de incertidumbre:")
        center_lbl.setToolTip(
            "Las cajas con confianza cerca de este valor son las más ambiguas y se "
            "muestran primero.")
        center_row.addWidget(center_lbl)
        self._center_spin = QDoubleSpinBox()
        self._center_spin.setRange(0.10, 0.90)
        self._center_spin.setSingleStep(0.05)
        self._center_spin.setDecimals(2)
        self._center_spin.setValue(0.50)
        center_row.addWidget(self._center_spin)
        center_row.addStretch()
        layout.addLayout(center_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Ejecutar")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def accept_enabled(self) -> bool:
        return self._accept_cb.isChecked()

    @property
    def accept_threshold(self) -> float:
        return self._accept_spin.value()

    @property
    def mark_reviewed(self) -> bool:
        return self._reviewed_cb.isChecked()

    @property
    def prioritize(self) -> bool:
        return self._prioritize_cb.isChecked()

    @property
    def center(self) -> float:
        return self._center_spin.value()


class ConceptDiscoveryDialog(QDialog):
    """Descubrimiento por diccionario: SigLIP2 rankea candidatas por concepto y
    SAM3 las pre-etiqueta. Solo validás después."""

    def __init__(self, parent=None, default_text: str = "",
                 default_sam3_model: str = DEFAULT_SAM3_MODEL,
                 n_indexed: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Descubrimiento por diccionario (SigLIP2 → SAM 3)")
        self.setMinimumWidth(580)
        self._default_sam3_model = default_sam3_model
        self._setup_ui(default_text, n_indexed)

    def _setup_ui(self, default_text: str, n_indexed: int):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        info = QLabel(
            f"Por cada concepto: SigLIP2 rankea las top-K imágenes candidatas "
            f"(de {n_indexed} indexadas) y SAM 3 las pre-etiqueta. "
            f"Después solo validás las sugerencias."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#bbb;")
        layout.addWidget(info)

        layout.addWidget(QLabel(
            "Diccionario — un concepto por línea. Formato: "
            "<i>prompt</i>  o  <i>prompt = ClaseDestino</i>"))
        self._text = QPlainTextEdit()
        self._text.setPlainText(default_text)
        self._text.setPlaceholderText(
            "chemical storage tank = tank\n"
            "transmission tower\n"
            "drilling rig\n"
            "# las líneas con # se ignoran")
        self._text.setMinimumHeight(140)
        layout.addWidget(self._text)

        form_box = QGroupBox("Parámetros")
        form = QFormLayout(form_box)

        self._topk_spin = QSpinBox()
        self._topk_spin.setRange(5, 2000)
        self._topk_spin.setValue(50)
        self._topk_spin.setToolTip(
            "Candidatas por concepto que SigLIP pasa a SAM 3. "
            "Subilo para objetos chicos (menos confiables en SigLIP).")
        form.addRow("Top-K candidatas/concepto:", self._topk_spin)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setValue(0.40)
        form.addRow("Confianza SAM 3:", self._conf_spin)

        self._imgsz_combo = QComboBox()
        for sz in (512, 640, 768, 1024):
            self._imgsz_combo.addItem(str(sz), sz)
        self._imgsz_combo.setCurrentIndex(1)
        form.addRow("Tamaño de imagen:", self._imgsz_combo)

        self._pending_cb = QCheckBox("Solo sobre imágenes pendientes")
        self._pending_cb.setChecked(True)
        form.addRow("", self._pending_cb)

        self._source_combo = QComboBox()
        self._source_combo.addItem("Sugerencias (para revisar)", "suggested")
        self._source_combo.addItem("Etiquetas (source=yolo)", "yolo")
        form.addRow("Guardar como:", self._source_combo)

        layout.addWidget(form_box)

        layout.addWidget(QLabel("Modelo SAM 3 (.pt):"))
        model_row = QHBoxLayout()
        self._model_edit = QLineEdit(self._default_sam3_model)
        model_row.addWidget(self._model_edit)
        browse = QPushButton("Examinar...")
        browse.clicked.connect(self._browse_model)
        model_row.addWidget(browse)
        layout.addLayout(model_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Descubrir")
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_model(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Modelo SAM 3", self._model_edit.text(), "PyTorch (*.pt)")
        if f:
            self._model_edit.setText(f)

    def _validate(self):
        from ..core.concept_discovery import parse_concepts
        if not parse_concepts(self.raw_text):
            QMessageBox.warning(self, "Error", "Agregá al menos un concepto.")
            return
        if not Path(self.model_path).is_file():
            QMessageBox.warning(
                self, "Modelo no encontrado",
                f"No se encontró SAM 3 en:\n{self.model_path}")
            return
        self.accept()

    @property
    def raw_text(self) -> str:
        return self._text.toPlainText()

    @property
    def top_k(self) -> int:
        return self._topk_spin.value()

    @property
    def conf(self) -> float:
        return self._conf_spin.value()

    @property
    def imgsz(self) -> int:
        return self._imgsz_combo.currentData()

    @property
    def restrict_status(self) -> str | None:
        return "pending" if self._pending_cb.isChecked() else None

    @property
    def source(self) -> str:
        return self._source_combo.currentData()

    @property
    def model_path(self) -> str:
        return self._model_edit.text().strip()
