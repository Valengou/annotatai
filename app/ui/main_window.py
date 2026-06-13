from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QFileDialog, QMessageBox, QProgressBar, QLabel,
    QStatusBar, QDockWidget, QToolBar, QComboBox, QPushButton,
    QSizePolicy, QButtonGroup,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence

from .cluster_panel import ClusterPanel
from .image_grid import ImageGrid
from .graph_view import GraphView
from .annotation_editor import AnnotationEditor
from .bbox_view import BBoxOutlierView
from .analysis_lab import AnalysisLabDialog
from .dialogs import NewProjectDialog, ClassManagerDialog, ExportDialog, LoadAnnotationsDialog
from ..core.project import Project
from ..core.image_indexer import index_images
from ..core.embeddings import EmbeddingGenerator
from ..core.clustering import cluster_images
from ..core.dataset_analysis import AnalysisOptions, run_dataset_analysis
from ..core.annotations import export_yolo, export_coco
from ..core.loaders import load_coco, load_yolo, load_labelme, LoadResult
from ..core.bbox_embeddings import generate_bbox_embeddings, run_bbox_umap
from ..models.image_item import ImageItem
from ..utils.config import CLUSTER_COLORS


# ────────────────────────────── Workers ──────────────────────────────

class IndexWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, folder: Path, db, project_path: Path):
        super().__init__()
        self.folder = folder
        self.db = db
        self.project_path = project_path

    def run(self):
        try:
            ids = index_images(self.folder, self.db, self.project_path,
                               progress_callback=self.progress.emit)
            self.finished.emit(ids)
        except Exception as e:
            self.error.emit(str(e))


class EmbeddingWorker(QObject):
    progress = Signal(int, int)
    finished = Signal()
    error = Signal(str)

    def __init__(self, db):
        super().__init__()
        self.db = db
        self._gen = EmbeddingGenerator()

    def run(self):
        try:
            self._gen.load()
            self._gen.generate_all(self.db, progress_callback=self.progress.emit)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class ClusterWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, db):
        super().__init__()
        self.db = db

    def run(self):
        try:
            n = cluster_images(self.db, progress_callback=self.progress.emit)
            self.finished.emit(n)
        except Exception as e:
            self.error.emit(str(e))


class AnalysisLabWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, db, project_path: Path, options: AnalysisOptions):
        super().__init__()
        self.db = db
        self.project_path = project_path
        self.options = options

    def run(self):
        try:
            result = run_dataset_analysis(
                self.db,
                self.project_path,
                self.options,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class BBoxEmbedWorker(QObject):
    progress = Signal(int, int)
    finished = Signal()
    error    = Signal(str)

    def __init__(self, db):
        super().__init__()
        self.db = db
        self._gen = EmbeddingGenerator()

    def run(self):
        try:
            self._gen.load()
            generate_bbox_embeddings(self.db, self._gen,
                                     progress_callback=self.progress.emit)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class BBoxUmapWorker(QObject):
    progress = Signal(str, int)
    finished = Signal(object)   # dict result
    error    = Signal(str)

    def __init__(self, db):
        super().__init__()
        self.db = db

    def run(self):
        try:
            result = run_bbox_umap(self.db,
                                   progress_callback=self.progress.emit)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class LoadAnnotationsWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object)   # LoadResult
    error = Signal(str)

    def __init__(self, db, fmt: str, **kwargs):
        super().__init__()
        self.db = db
        self.fmt = fmt
        self.kwargs = kwargs

    def run(self):
        try:
            if self.fmt == "coco":
                result = load_coco(
                    self.kwargs["json_path"], self.db,
                    overwrite=self.kwargs.get("overwrite", False),
                    progress_callback=self.progress.emit,
                )
            elif self.fmt == "yolo":
                result = load_yolo(
                    self.kwargs["labels_dir"], self.db,
                    names_file=self.kwargs.get("names_file"),
                    overwrite=self.kwargs.get("overwrite", False),
                    progress_callback=self.progress.emit,
                )
            else:
                result = load_labelme(
                    self.kwargs["json_dir"], self.db,
                    overwrite=self.kwargs.get("overwrite", False),
                    progress_callback=self.progress.emit,
                )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ────────────────────────────── Main Window ──────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._project: Project | None = None
        self._current_images: list[ImageItem] = []
        self._image_by_id: dict[int, ImageItem] = {}
        self._current_editor_idx: int = 0
        self._lasso_nav_ids: list[int] = []   # when non-empty, navigation uses this
        self._lasso_nav_idx: int = 0
        self._thread: QThread | None = None
        self._worker = None

        self.setWindowTitle("AnnotatAI — Anotación Visual Inteligente")
        self.setMinimumSize(1280, 800)
        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self._set_project_active(False)

    # ─── UI setup ───

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Global status filter bar ──
        filter_bar = QWidget()
        filter_bar.setStyleSheet(
            "background:#181818; border-bottom:1px solid #333;"
        )
        fb = QHBoxLayout(filter_bar)
        fb.setContentsMargins(8, 3, 8, 3)
        fb.setSpacing(6)

        fb.addWidget(QLabel("Filtro:"))

        _FILTERS  = ["todas",    "pending",     "reviewed",  "discarded"]
        _LABELS   = ["Todas",    "Pendientes",  "Revisadas", "Descartadas"]
        _COLORS   = ["#555555",  "#808080",     "#2ECC71",   "#E74C3C"]

        self._status_filter = "todas"
        self._filter_btn_group = QButtonGroup(self)
        self._filter_btn_group.setExclusive(True)
        self._filter_btns: list[QPushButton] = []

        for i, (key, label, color) in enumerate(zip(_FILTERS, _LABELS, _COLORS)):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.setFixedHeight(22)
            btn.setStyleSheet(f"""
                QPushButton {{
                    border: 1px solid {color}; border-radius: 3px;
                    color: {color}; padding: 0 10px;
                    background: transparent; font-size: 11px;
                }}
                QPushButton:checked {{
                    background: {color}; color: #fff; font-weight: bold;
                }}
                QPushButton:hover {{ background: {color}33; }}
            """)
            btn.clicked.connect(lambda _, k=key: self._apply_global_filter(k))
            self._filter_btn_group.addButton(btn, i)
            self._filter_btns.append(btn)
            fb.addWidget(btn)

        fb.addStretch()
        self._filter_count_lbl = QLabel("")
        self._filter_count_lbl.setStyleSheet("color:#555; font-size:11px;")
        fb.addWidget(self._filter_count_lbl)

        outer.addWidget(filter_bar)

        # ── Main splitter ──
        splitter_row = QHBoxLayout()
        splitter_row.setContentsMargins(0, 0, 0, 0)
        splitter_row.setSpacing(0)
        outer.addLayout(splitter_row, stretch=1)

        self._splitter = QSplitter(Qt.Horizontal)
        splitter_row.addWidget(self._splitter)

        # Left dock - cluster panel
        self._cluster_panel = ClusterPanel()
        self._cluster_panel.cluster_selected.connect(self._on_cluster_selected)
        self._cluster_panel.cluster_delete_requested.connect(self._on_delete_cluster)
        self._cluster_panel.setMinimumWidth(180)
        self._cluster_panel.setMaximumWidth(280)
        self._splitter.addWidget(self._cluster_panel)

        # Central - tabs
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.North)

        self._image_grid = ImageGrid()
        self._image_grid.image_opened.connect(self._open_annotation_editor)
        self._image_grid.status_changed.connect(self._on_status_changed)
        self._image_grid.delete_requested.connect(self._on_delete_images)
        self._image_grid.batch_status_requested.connect(self._on_batch_status_from_grid)
        self._image_grid.batch_label_requested.connect(self._on_batch_label_from_grid)
        self._tabs.addTab(self._image_grid, "Grilla de Imágenes")

        self._graph_view = GraphView()
        self._graph_view.image_clicked.connect(self._on_graph_image_clicked)
        self._graph_view.lasso_selection_changed.connect(self._on_lasso_selection_changed)
        self._graph_view.images_status_changed.connect(self._on_umap_status_changed)
        self._graph_view.images_label_changed.connect(self._on_batch_label_from_grid)
        self._graph_view.images_delete_requested.connect(self._on_delete_images)
        self._tabs.addTab(self._graph_view, "Vista UMAP")

        self._annotation_editor = AnnotationEditor()
        self._annotation_editor.annotation_saved.connect(self._on_annotation_saved)
        self._annotation_editor.navigate_request.connect(self._navigate_images)
        self._annotation_editor.image_status_changed.connect(self._on_image_status_changed)
        self._tabs.addTab(self._annotation_editor, "Editor de Anotaciones")

        self._bbox_view = BBoxOutlierView()
        self._bbox_view.open_image_requested.connect(self._on_bbox_open_image)
        self._bbox_view.images_status_changed.connect(self._on_bbox_status_changed)
        self._bbox_view.lasso_selection_changed.connect(self._on_lasso_selection_changed)
        self._tabs.addTab(self._bbox_view, "Bboxes Raras")

        self._splitter.addWidget(self._tabs)
        self._splitter.setSizes([220, 1060])

        # Status bar
        self._status_bar = self.statusBar()
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(250)
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)
        self._status_label = QLabel("Sin proyecto")
        self._status_bar.addWidget(self._status_label)

    def _build_menus(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("Archivo")
        self._act_new = QAction("Nuevo Proyecto...", self)
        self._act_new.setShortcut("Ctrl+N")
        self._act_new.triggered.connect(self.new_project)
        file_menu.addAction(self._act_new)

        self._act_open = QAction("Abrir Proyecto...", self)
        self._act_open.setShortcut("Ctrl+O")
        self._act_open.triggered.connect(self.open_project)
        file_menu.addAction(self._act_open)

        file_menu.addSeparator()

        self._act_load_folder = QAction("Cargar Carpeta de Imágenes...", self)
        self._act_load_folder.setShortcut("Ctrl+Shift+O")
        self._act_load_folder.triggered.connect(self.load_folder)
        file_menu.addAction(self._act_load_folder)

        self._act_load_annotations = QAction("Cargar Anotaciones Existentes...", self)
        self._act_load_annotations.setShortcut("Ctrl+Shift+A")
        self._act_load_annotations.triggered.connect(self.load_annotations)
        file_menu.addAction(self._act_load_annotations)

        file_menu.addSeparator()

        self._act_export = QAction("Exportar Dataset...", self)
        self._act_export.setShortcut("Ctrl+E")
        self._act_export.triggered.connect(self.export_dataset)
        file_menu.addAction(self._act_export)

        tools_menu = menu_bar.addMenu("Herramientas")
        self._act_embed = QAction("Generar Embeddings CLIP...", self)
        self._act_embed.triggered.connect(self.generate_embeddings)
        tools_menu.addAction(self._act_embed)

        self._act_cluster = QAction("Agrupar por Similitud (UMAP + HDBSCAN)...", self)
        self._act_cluster.triggered.connect(self.run_clustering)
        tools_menu.addAction(self._act_cluster)

        self._act_analysis_lab = QAction("Dataset Analysis Lab...", self)
        self._act_analysis_lab.triggered.connect(self.run_analysis_lab)
        tools_menu.addAction(self._act_analysis_lab)

        tools_menu.addSeparator()

        self._act_bbox_embed = QAction("Embeddings de Crops de Bboxes...", self)
        self._act_bbox_embed.triggered.connect(self.generate_bbox_embeddings)
        tools_menu.addAction(self._act_bbox_embed)

        self._act_bbox_umap = QAction("Analizar Bboxes Raras (UMAP)...", self)
        self._act_bbox_umap.triggered.connect(self.run_bbox_umap)
        tools_menu.addAction(self._act_bbox_umap)

        tools_menu.addSeparator()
        self._act_classes = QAction("Administrar Clases...", self)
        self._act_classes.triggered.connect(self.manage_classes)
        tools_menu.addAction(self._act_classes)

    def _build_toolbar(self):
        tb = QToolBar("Principal")
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addAction(self._act_new)
        tb.addAction(self._act_open)
        tb.addSeparator()
        tb.addAction(self._act_load_folder)
        tb.addAction(self._act_load_annotations)
        tb.addSeparator()
        tb.addAction(self._act_embed)
        tb.addAction(self._act_cluster)
        tb.addSeparator()
        tb.addAction(self._act_export)

    def _set_project_active(self, active: bool):
        for act in [self._act_load_folder, self._act_load_annotations,
                    self._act_embed, self._act_cluster, self._act_analysis_lab,
                    self._act_bbox_embed, self._act_bbox_umap,
                    self._act_export, self._act_classes]:
            act.setEnabled(active)

    # ─── Project actions ───

    def new_project(self):
        dlg = NewProjectDialog(self)
        if dlg.exec():
            try:
                self._project = Project.create(dlg.project_dir, dlg.project_name)
                self._on_project_loaded()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo crear el proyecto:\n{e}")

    def open_project(self, path=None):
        if not isinstance(path, Path):
            path = None
        if path is None:
            d = QFileDialog.getExistingDirectory(self, "Abrir Proyecto")
            if not d:
                return
            path = Path(d)
        if not Project.is_valid_project(path):
            QMessageBox.warning(self, "Error",
                                "La carpeta seleccionada no contiene un proyecto válido.\n"
                                "Crea uno nuevo con Archivo → Nuevo Proyecto.")
            return
        try:
            if self._project:
                self._project.close()
            self._project = Project.open(path)
            self._on_project_loaded()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo abrir el proyecto:\n{e}")

    def _on_project_loaded(self):
        self._set_project_active(True)
        self.setWindowTitle(f"AnnotatAI — {self._project.name}")
        self._cluster_panel.set_db(self._project.db)
        self._annotation_editor.set_db(self._project.db)
        self._bbox_view.set_db(self._project.db)
        classes = self._project.db.get_all_classes()
        self._image_grid.set_classes(classes)
        self._graph_view.set_classes(classes)
        self._load_all_images()
        self._status_label.setText(f"Proyecto: {self._project.name}")

    def _load_all_images(self):
        rows = self._project.db.get_all_images()
        self._current_images = [ImageItem.from_db_row(r) for r in rows]
        self._image_by_id = {img.id: img for img in self._current_images}
        ann_cache = self._project.db.get_annotations_for_grid()
        self._image_grid.load_images(self._current_images, ann_cache)
        self._graph_view.set_ann_cache(ann_cache)
        self._image_grid._set_status_filter(self._status_filter)
        self._status_label.setText(
            f"{self._project.name} — {len(self._current_images)} imágenes"
        )
        self._update_filter_count()
        self._try_load_graph()
        self._try_load_bbox_projections()

    # ─── Load folder ───

    def load_folder(self):
        if not self._project:
            return
        d = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes")
        if not d:
            return
        self._run_worker(
            IndexWorker(Path(d), self._project.db, self._project.project_path),
            on_progress=self._on_index_progress,
            on_finished=self._on_index_finished,
            on_error=self._on_worker_error,
        )
        self._status_label.setText("Indexando imágenes...")
        self._show_progress(True)

    def _on_index_progress(self, done: int, total: int):
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(done)
        self._status_label.setText(f"Indexando: {done}/{total}")

    def _on_index_finished(self, ids: list):
        self._show_progress(False)
        self._load_all_images()
        self._cluster_panel.refresh()
        self._status_label.setText(f"Listo — {len(ids)} imágenes indexadas")

    # ─── Load annotations ───

    def load_annotations(self):
        if not self._project:
            return
        if self._project.db.count_images() == 0:
            QMessageBox.warning(
                self, "Sin imágenes",
                "Primero carga una carpeta de imágenes para indexarlas en el proyecto."
            )
            return

        dlg = LoadAnnotationsDialog(self)
        if not dlg.exec():
            return

        fmt = dlg.format
        overwrite = dlg.overwrite
        kwargs = {"overwrite": overwrite}

        if fmt == "coco":
            kwargs["json_path"] = dlg.coco_json_path
        elif fmt == "yolo":
            kwargs["labels_dir"] = dlg.yolo_labels_dir
            kwargs["names_file"] = dlg.yolo_names_file
        else:
            kwargs["json_dir"] = dlg.labelme_dir

        self._run_worker(
            LoadAnnotationsWorker(self._project.db, fmt, **kwargs),
            on_progress=self._on_load_ann_progress,
            on_finished=self._on_load_ann_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText("Cargando anotaciones...")

    def _on_load_ann_progress(self, done: int, total: int):
        self._progress_bar.setMaximum(max(total, 1))
        self._progress_bar.setValue(done)
        self._status_label.setText(f"Importando: {done}/{total}")

    def _on_load_ann_finished(self, result: LoadResult):
        self._show_progress(False)
        self._annotation_editor._refresh_classes()
        self._status_label.setText(
            f"Anotaciones importadas: {result.annotations_loaded} en {result.images_matched} imágenes"
        )
        QMessageBox.information(self, "Carga completa", result.summary())

    # ─── Embeddings ───

    def generate_embeddings(self):
        if not self._project:
            return
        n_images = self._project.db.count_images()
        n_emb = self._project.db.count_embeddings()
        pending = n_images - n_emb
        if pending == 0:
            QMessageBox.information(self, "Embeddings", "Todos los embeddings ya están generados.")
            return

        reply = QMessageBox.question(
            self, "Generar Embeddings",
            f"Se generarán embeddings CLIP para {pending} imágenes.\n"
            f"Esto puede tardar varios minutos en la primera vez.\n¿Continuar?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._run_worker(
            EmbeddingWorker(self._project.db),
            on_progress=self._on_embed_progress,
            on_finished=self._on_embed_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText("Generando embeddings CLIP...")

    def _on_embed_progress(self, done: int, total: int):
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(done)
        self._status_label.setText(f"Embeddings: {done}/{total}")

    def _on_embed_finished(self):
        self._show_progress(False)
        n = self._project.db.count_embeddings()
        self._status_label.setText(f"Embeddings listos: {n} imágenes")
        QMessageBox.information(self, "Listo", f"Embeddings generados: {n} imágenes.")

    # ─── Clustering ───

    def run_clustering(self):
        if not self._project:
            return
        n_emb = self._project.db.count_embeddings()
        if n_emb < 3:
            QMessageBox.warning(self, "Clustering",
                                "Se necesitan al menos 3 embeddings. Genera embeddings primero.")
            return

        self._run_worker(
            ClusterWorker(self._project.db),
            on_progress=self._on_cluster_progress,
            on_finished=self._on_cluster_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText("Ejecutando clustering...")

    def _on_cluster_progress(self, msg: str, pct: int):
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(pct)
        self._status_label.setText(msg)

    def _on_cluster_finished(self, n_clusters: int):
        self._show_progress(False)
        self._cluster_panel.refresh()
        self._load_all_images()
        self._try_load_graph()
        self._status_label.setText(f"Clustering completo: {n_clusters} grupos encontrados")
        QMessageBox.information(self, "Clustering", f"Se encontraron {n_clusters} grupos.")

    # ─── Dataset Analysis Lab ───

    def run_analysis_lab(self):
        if not self._project:
            return
        if self._project.db.count_images() < 3:
            QMessageBox.warning(
                self,
                "Dataset Analysis Lab",
                "Se necesitan al menos 3 imagenes en el proyecto.",
            )
            return

        dlg = AnalysisLabDialog(self._project.project_path, self)
        if not dlg.exec():
            return

        self._run_worker(
            AnalysisLabWorker(self._project.db, self._project.project_path, dlg.options),
            on_progress=self._on_analysis_lab_progress,
            on_finished=self._on_analysis_lab_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText("Ejecutando Dataset Analysis Lab...")

    def _on_analysis_lab_progress(self, msg: str, pct: int):
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(pct)
        self._status_label.setText(msg)

    def _on_analysis_lab_finished(self, result):
        self._show_progress(False)
        self._cluster_panel.refresh()
        self._load_all_images()
        self._try_load_graph()
        self._tabs.setCurrentWidget(self._graph_view)
        self._status_label.setText(
            f"Analysis Lab listo: {result.cluster_count} clusters en {result.run_dir}"
        )
        QMessageBox.information(
            self,
            "Dataset Analysis Lab",
            "Analisis terminado.\n\n"
            f"Imagenes: {result.image_count}\n"
            f"Clusters: {result.cluster_count}\n"
            f"Run: {result.run_dir}\n"
            f"Reporte: {result.report_html}",
        )

    # ─── Bbox embeddings & outlier analysis ───

    def generate_bbox_embeddings(self):
        if not self._project:
            return
        n_ann = 0
        with self._project.db.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM annotations")
            n_ann = cur.fetchone()[0]
        if n_ann == 0:
            QMessageBox.warning(self, "Sin anotaciones",
                                "No hay anotaciones en el proyecto todavía.")
            return
        n_done = self._project.db.count_bbox_embeddings()
        pending = n_ann - n_done
        if pending == 0:
            QMessageBox.information(self, "Bbox Embeddings",
                                    "Todos los crops ya tienen embedding.")
            return
        reply = QMessageBox.question(
            self, "Embeddings de Crops",
            f"Se generarán embeddings CLIP para {pending} crops de bboxes.\n"
            f"(ya procesados: {n_done})\n¿Continuar?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._run_worker(
            BBoxEmbedWorker(self._project.db),
            on_progress=self._on_bbox_embed_progress,
            on_finished=self._on_bbox_embed_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText("Generando embeddings de crops...")

    def _on_bbox_embed_progress(self, done: int, total: int):
        self._progress_bar.setMaximum(max(total, 1))
        self._progress_bar.setValue(done)
        self._status_label.setText(f"Bbox embeddings: {done}/{total}")

    def _on_bbox_embed_finished(self):
        self._show_progress(False)
        n = self._project.db.count_bbox_embeddings()
        self._status_label.setText(f"Bbox embeddings listos: {n}")
        QMessageBox.information(self, "Listo",
                                f"Embeddings de crops generados: {n}")

    def run_bbox_umap(self):
        if not self._project:
            return
        n = self._project.db.count_bbox_embeddings()
        if n < 4:
            QMessageBox.warning(self, "Sin datos",
                                "Primero generá los embeddings de crops de bboxes.")
            return
        self._run_worker(
            BBoxUmapWorker(self._project.db),
            on_progress=self._on_bbox_umap_progress,
            on_finished=self._on_bbox_umap_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText("Analizando bboxes raras...")

    def _on_bbox_umap_progress(self, msg: str, pct: int):
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(pct)
        self._status_label.setText(msg)

    def _on_bbox_umap_finished(self, result: dict):
        self._show_progress(False)
        proj_rows = self._project.db.get_all_bbox_projections()
        self._bbox_view.load_projections(proj_rows)
        self._tabs.setCurrentWidget(self._bbox_view)
        n_out = result.get("n_outliers", 0)
        total = result.get("total", 0)
        self._status_label.setText(
            f"Análisis listo — {n_out} bboxes raras de {total}"
        )

    def _on_bbox_open_image(self, image_id: int, annotation_id: int):
        # Set position within lasso list so ← → continue from here
        if self._lasso_nav_ids and image_id in self._lasso_nav_ids:
            self._lasso_nav_idx = self._lasso_nav_ids.index(image_id)
        self._open_annotation_editor(image_id)
        self._tabs.setCurrentIndex(2)

    def _on_lasso_selection_changed(self, image_ids: list):
        self._lasso_nav_ids = image_ids
        self._lasso_nav_idx = 0
        if image_ids:
            self._status_label.setText(
                f"Lazo activo — {len(image_ids)} imágenes  |  ← → para navegar entre ellas"
            )
        else:
            self._status_label.setText("Selección de lazo limpiada")

    def _on_umap_status_changed(self, image_ids: list, status: str):
        if not self._project:
            return
        for img_id in image_ids:
            self._project.db.update_image_status(img_id, status)
            self._image_grid.update_card_status(img_id, status)
            if img_id in self._image_by_id:
                self._image_by_id[img_id].status = status
            self._bbox_view.apply_status_filter(self._status_filter)
        self._update_filter_count()
        self._status_label.setText(
            f"{len(image_ids)} imagen(es) marcadas como {status}"
        )

    def _on_bbox_status_changed(self, image_ids: list, status: str):
        if not self._project:
            return
        for img_id in image_ids:
            self._project.db.update_image_status(img_id, status)
            self._image_grid.update_card_status(img_id, status)
            if img_id in self._image_by_id:
                self._image_by_id[img_id].status = status
        self._status_label.setText(
            f"{len(image_ids)} imagen(es) marcadas como {status}"
        )

    def _try_load_bbox_projections(self):
        if not self._project:
            return
        proj_rows = self._project.db.get_all_bbox_projections()
        if proj_rows:
            self._bbox_view.load_projections(proj_rows)

    def _try_load_graph(self):
        if not self._project:
            return
        proj_rows = self._project.db.get_all_projections()
        if not proj_rows:
            return
        cluster_rows = self._project.db.get_all_clusters()
        cluster_colors = {}
        for row in cluster_rows:
            cluster_colors[row[0]] = row[2]
        self._graph_view.load_projections(proj_rows, cluster_colors)

    # ─── Annotation editor ───

    def _open_annotation_editor(self, image_id: int):
        if not self._project:
            return
        # Use cached list — no DB scan
        ids = [img.id for img in self._current_images]
        try:
            self._current_editor_idx = ids.index(image_id)
        except ValueError:
            self._current_editor_idx = 0
        self._load_image_in_editor(image_id)
        self._tabs.setCurrentIndex(2)

    def _load_image_in_editor(self, image_id: int):
        if not self._project:
            return
        # O(1) lookup from cached dict
        img = self._image_by_id.get(image_id)
        if img is None:
            return
        self._annotation_editor.load_image(img)
        self._image_grid.highlight_image(image_id)
        self._graph_view.highlight_point(image_id)

    def _navigate_images(self, delta: int):
        # Priority 1: lasso selection from bbox view
        if self._lasso_nav_ids:
            self._lasso_nav_idx = (
                (self._lasso_nav_idx + delta) % len(self._lasso_nav_ids)
            )
            next_id = self._lasso_nav_ids[self._lasso_nav_idx]
            full_ids = [img.id for img in self._current_images]
            try:
                self._current_editor_idx = full_ids.index(next_id)
            except ValueError:
                pass
            self._load_image_in_editor(next_id)
            self._status_label.setText(
                f"Lazo: imagen {self._lasso_nav_idx + 1} / {len(self._lasso_nav_ids)}"
            )
            return

        # Priority 2: grid status + cluster filter
        visible_ids = self._image_grid.get_visible_image_ids()
        if not visible_ids:
            return

        current_id = (
            self._current_images[self._current_editor_idx].id
            if 0 <= self._current_editor_idx < len(self._current_images)
            else None
        )
        try:
            pos = visible_ids.index(current_id)
        except ValueError:
            pos = 0

        next_id = visible_ids[(pos + delta) % len(visible_ids)]
        full_ids = [img.id for img in self._current_images]
        try:
            self._current_editor_idx = full_ids.index(next_id)
        except ValueError:
            pass

        self._load_image_in_editor(next_id)
        new_pos = visible_ids.index(next_id)
        self._status_label.setText(
            f"Imagen {new_pos + 1} / {len(visible_ids)}"
            + ("" if len(visible_ids) == len(self._current_images)
               else f"  (filtradas de {len(self._current_images)})")
        )

    def _on_annotation_saved(self, image_id: int):
        self._status_label.setText(f"Anotaciones guardadas — imagen ID {image_id}")
        # Refresh bbox overlay on the grid card
        anns = self._project.db.get_annotations_for_grid_image(image_id)
        self._image_grid.refresh_card_annotations(image_id, anns)

    def _apply_global_filter(self, status: str):
        self._status_filter = status
        self._cluster_panel.set_status_filter(status)
        self._image_grid._set_status_filter(status)
        self._graph_view.apply_status_filter(status)
        self._bbox_view.apply_status_filter(status)
        self._update_filter_count()

    def _update_filter_count(self):
        if not self._current_images:
            return
        if self._status_filter == "todas":
            self._filter_count_lbl.setText(
                f"{len(self._current_images)} imágenes"
            )
        else:
            n = sum(1 for img in self._current_images
                    if img.status == self._status_filter)
            self._filter_count_lbl.setText(
                f"{n} / {len(self._current_images)} imágenes"
            )

    def _on_image_status_changed(self, image_id: int, status: str):
        self._image_grid.update_card_status(image_id, status)
        self._graph_view.update_image_status(image_id, status)
        if image_id in self._image_by_id:
            self._image_by_id[image_id].status = status
        self._status_label.setText(f"Imagen {image_id} → {status}")
        self._update_filter_count()

    def _on_graph_image_clicked(self, image_id: int):
        self._image_grid.highlight_image(image_id)
        self._open_annotation_editor(image_id)

    def _on_cluster_selected(self, cluster_id: int):
        self._image_grid.filter_by_cluster(cluster_id)
        self._tabs.setCurrentIndex(0)

    def _on_status_changed(self, image_id: int, status: str):
        if self._project:
            self._project.db.update_image_status(image_id, status)

    def _on_batch_status_from_grid(self, image_ids: list, status: str):
        if not self._project or not image_ids:
            return
        for img_id in image_ids:
            self._project.db.update_image_status(img_id, status)
            self._image_grid.update_card_status(img_id, status)
            self._graph_view.update_image_status(img_id, status)
            if img_id in self._image_by_id:
                self._image_by_id[img_id].status = status
        self._image_grid._deselect_all()
        self._update_filter_count()
        self._status_label.setText(
            f"{len(image_ids)} imagen(es) marcadas como {status}"
        )

    def _on_batch_label_from_grid(self, image_ids: list, class_id: int):
        if not self._project or not image_ids:
            return
        db = self._project.db
        classes = {row[0]: (row[1], row[2]) for row in db.get_all_classes()}
        class_name, class_color = classes.get(class_id, ("unknown", "#FF0000"))
        ann_entry = [(0.0, 0.0, 1.0, 1.0, class_color, class_name)]
        for img_id in image_ids:
            db.delete_annotations_for_image(img_id)
            db.insert_annotation(img_id, class_id, 0.0, 0.0, 1.0, 1.0, source="human")
            self._image_grid.refresh_card_annotations(img_id, ann_entry)
            self._graph_view.set_ann_cache({**self._graph_view._ann_cache, img_id: ann_entry})
        self._image_grid._deselect_all()
        self._status_label.setText(
            f"{len(image_ids)} imagen(es) etiquetadas como '{class_name}'"
        )

    def _on_delete_images(self, image_ids: list):
        if not self._project or not image_ids:
            return
        reply = QMessageBox.question(
            self, "Eliminar imágenes",
            f"¿Eliminar {len(image_ids)} imagen(es) del proyecto?\n"
            "Se borran del proyecto (no del disco).\n"
            "También se borran sus anotaciones y embeddings.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        n = self._project.db.delete_images(image_ids)
        self._load_all_images()
        self._cluster_panel.refresh()
        self._status_label.setText(f"{n} imagen(es) eliminada(s)")

    def _on_delete_cluster(self, cluster_id: int):
        if not self._project:
            return
        # Get cluster name for the dialog
        clusters = self._project.db.get_all_clusters()
        cluster_name = next(
            (r[1] for r in clusters if r[0] == cluster_id), f"ID {cluster_id}"
        )
        rows_in_cluster = next(
            (r[4] for r in clusters if r[0] == cluster_id), 0
        )
        reply = QMessageBox.question(
            self, "Eliminar grupo",
            f"¿Eliminar el grupo «{cluster_name}» y sus {rows_in_cluster} imágenes del proyecto?\n"
            "Las imágenes NO se borran del disco.\n"
            "Se borran anotaciones y embeddings de esas imágenes.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        n_imgs, _ = self._project.db.delete_cluster_with_images(cluster_id)
        self._load_all_images()
        self._cluster_panel.refresh()
        self._status_label.setText(f"Grupo eliminado — {n_imgs} imagen(es) removida(s)")

    # ─── Classes ───

    def manage_classes(self):
        if not self._project:
            return
        dlg = ClassManagerDialog(self._project.db, self)
        dlg.exec()
        classes = self._project.db.get_all_classes()
        self._annotation_editor._refresh_classes()
        self._image_grid.set_classes(classes)
        self._graph_view.set_classes(classes)

    # ─── Export ───

    def export_dataset(self):
        if not self._project:
            return
        dlg = ExportDialog(self, default_name=self._project.name)
        if not dlg.exec():
            return

        out_dir = dlg.export_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        only_reviewed = dlg.only_reviewed
        image_filter = (lambda r: r[6] == "reviewed") if only_reviewed else None

        try:
            if dlg.export_format == "yolo":
                n_img, n_ann = export_yolo(self._project.db, out_dir, image_filter)
                fmt_name = "YOLO"
            else:
                n_img, n_ann = export_coco(self._project.db, out_dir, image_filter)
                fmt_name = "COCO"

            QMessageBox.information(
                self, "Exportación completa",
                f"Formato: {fmt_name}\n"
                f"Imágenes exportadas: {n_img}\n"
                f"Anotaciones: {n_ann}\n"
                f"Destino: {out_dir}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))

    # ─── Worker helpers ───

    def _run_worker(self, worker_obj, on_progress, on_finished, on_error):
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    QMessageBox.warning(self, "Ocupado", "Ya hay una tarea en ejecución.")
                    return
            except RuntimeError:
                pass  # C++ object already deleted — safe to replace
            self._thread = None

        self._thread = QThread()
        self._worker = worker_obj
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)

        if hasattr(self._worker, "progress"):
            self._worker.progress.connect(on_progress)
        self._worker.finished.connect(on_finished)
        self._worker.error.connect(on_error)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._clear_thread)
        self._thread.start()

    def _clear_thread(self):
        if self._thread is not None:
            try:
                self._thread.deleteLater()
            except RuntimeError:
                pass
        self._thread = None

    def _on_worker_error(self, msg: str):
        self._show_progress(False)
        QMessageBox.critical(self, "Error", msg)
        self._status_label.setText("Error en tarea.")

    def _show_progress(self, visible: bool):
        self._progress_bar.setVisible(visible)
        if not visible:
            self._progress_bar.setValue(0)

    def closeEvent(self, event):
        if self._project:
            self._project.close()
        event.accept()
