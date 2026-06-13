from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QDialogButtonBox,
    QColorDialog, QInputDialog, QComboBox, QCheckBox, QGroupBox,
    QStackedWidget, QRadioButton, QButtonGroup, QMenu, QListWidget,
    QListWidgetItem,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap


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
        self._pending_color = "#3498DB"
        self._setup_ui()
        self._refresh()

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
