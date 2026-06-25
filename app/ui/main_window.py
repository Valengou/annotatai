from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QFileDialog, QMessageBox, QProgressBar, QLabel,
    QStatusBar, QDockWidget, QToolBar, QComboBox, QPushButton,
    QSizePolicy, QButtonGroup, QInputDialog,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence

from .cluster_panel import ClusterPanel
from .image_grid import ImageGrid
from .graph_view import GraphView
from .annotation_editor import AnnotationEditor
from .bbox_view import BBoxOutlierView
from .geometry_outliers_dialog import GeometryOutliersDialog
from .analysis_lab import AnalysisLabDialog
from .dialogs import (
    NewProjectDialog, ClassManagerDialog, ExportDialog, LoadAnnotationsDialog,
    AutoLabelDialog, TrainModelDialog, PredictWithModelDialog,
    NearDuplicatesDialog, AutoLabelHubDialog, ActiveLearningDialog,
    ConceptDiscoveryDialog,
)
from ..core.project import Project
from ..core.project_manifest import open_project_from_manifest
from ..core.image_indexer import index_images
from ..core.embeddings import EmbeddingGenerator, generate_project_embeddings
from ..core.embedding_backends import create_embedding_backend
from ..core.clustering import cluster_images
from ..core.dataset_analysis import AnalysisOptions, run_dataset_analysis
from ..core.annotations import export_yolo, export_coco
from ..core.loaders import load_coco, load_yolo, load_labelme, LoadResult
from ..core.bbox_embeddings import generate_bbox_embeddings, run_bbox_umap
from ..core.auto_label import run_auto_label, run_auto_label_visual
from ..core.yolo_trainer import (
    run_quick_train, run_predict_only, TrainOptions, TrainResult,
)
from ..core.near_duplicates import run_near_duplicates
from ..core.active_learning import run_active_learning
from ..core.semantic_search import (
    generate_search_embeddings, SemanticSearchEngine, SEARCH_MODEL,
)
from ..core.concept_discovery import run_concept_discovery, parse_concepts
from ..core.box_cleanup import cleanup_overlapping
from ..models.image_item import ImageItem
from ..utils.config import CLUSTER_COLORS


# ────────────────────────────── Workers ──────────────────────────────

class WorkerCancelled(Exception):
    """Se lanza desde el callback de progreso cuando se pidió cancelar."""


class BaseWorker(QObject):
    """Worker cancelable: el callback de progreso revisa el flag y, si se pidió
    cancelar, lanza WorkerCancelled para abortar el bucle del core."""
    cancelled = Signal()

    def __init__(self):
        super().__init__()
        self._cancelled = False

    def request_cancel(self):
        self._cancelled = True

    def _cb(self, *args):
        """Pasar como progress_callback en lugar de self.progress.emit."""
        if self._cancelled:
            raise WorkerCancelled()
        self.progress.emit(*args)


class IndexWorker(BaseWorker):
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
                               progress_callback=self._cb)
            self.finished.emit(ids)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class EmbeddingWorker(BaseWorker):
    progress = Signal(int, int)
    finished = Signal()
    error = Signal(str)

    def __init__(self, db, backend_name: str):
        super().__init__()
        self.db = db
        self.backend_name = backend_name

    def run(self):
        try:
            backend = create_embedding_backend(self.backend_name)
            generate_project_embeddings(
                self.db,
                backend,
                model_name=backend.name,
                progress_callback=self._cb,
            )
            self.finished.emit()
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class ClusterWorker(BaseWorker):
    progress = Signal(str, int)
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, db):
        super().__init__()
        self.db = db

    def run(self):
        try:
            n = cluster_images(self.db, progress_callback=self._cb)
            self.finished.emit(n)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class AnalysisLabWorker(BaseWorker):
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
                progress_callback=self._cb,
            )
            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class BBoxEmbedWorker(BaseWorker):
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
                                     progress_callback=self._cb)
            self.finished.emit()
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class BBoxUmapWorker(BaseWorker):
    progress = Signal(str, int)
    finished = Signal(object)   # dict result
    error    = Signal(str)

    def __init__(self, db):
        super().__init__()
        self.db = db

    def run(self):
        try:
            result = run_bbox_umap(self.db,
                                   progress_callback=self._cb)
            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class AutoLabelWorker(BaseWorker):
    progress = Signal(int, int)
    finished = Signal(object)   # dict summary
    error    = Signal(str)

    def __init__(self, db, image_rows: list, prompts: list,
                 conf: float, source: str, model_name: str,
                 save_polygons: bool = False, imgsz: int = 1024):
        super().__init__()
        self.db = db
        self.image_rows = image_rows
        self.prompts = prompts
        self.conf = conf
        self.source = source
        self.model_name = model_name
        self.save_polygons = save_polygons
        self.imgsz = imgsz

    def run(self):
        try:
            result = run_auto_label(
                self.db, self.image_rows, self.prompts, self.conf, self.source,
                model_name=self.model_name, save_polygons=self.save_polygons,
                imgsz=self.imgsz, progress_callback=self._cb,
            )
            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class AutoLabelVisualWorker(BaseWorker):
    progress = Signal(int, int)
    finished = Signal(object)   # dict summary
    error    = Signal(str)

    def __init__(self, db, class_id: int, class_name: str,
                 target_rows: list, conf: float, source: str,
                 exemplar_image_ids: set | None = None, imgsz: int = 1024):
        super().__init__()
        self.db = db
        self.class_id = class_id
        self.class_name = class_name
        self.target_rows = target_rows
        self.conf = conf
        self.source = source
        self.exemplar_image_ids = exemplar_image_ids
        self.imgsz = imgsz

    def run(self):
        try:
            result = run_auto_label_visual(
                self.db, self.class_id, self.class_name, self.target_rows,
                self.conf, self.source,
                exemplar_image_ids=self.exemplar_image_ids,
                imgsz=self.imgsz,
                progress_callback=self._cb,
            )
            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class TrainModelWorker(BaseWorker):
    progress = Signal(str, int)
    finished = Signal(object)   # TrainResult
    error    = Signal(str)

    def __init__(self, db, project_path: Path, options: TrainOptions,
                 predict_targets: list | None = None):
        super().__init__()
        self.db = db
        self.project_path = project_path
        self.options = options
        self.predict_targets = predict_targets

    def run(self):
        try:
            result = run_quick_train(
                self.db, self.project_path, self.options,
                predict_targets=self.predict_targets,
                progress_callback=self._cb,
            )
            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class PredictModelWorker(BaseWorker):
    progress = Signal(str, int)
    finished = Signal(object)   # TrainResult
    error    = Signal(str)

    def __init__(self, db, weights: Path, conf: float, imgsz: int,
                 scope: str = "pending", targets: list | None = None,
                 source: str = "yolo"):
        super().__init__()
        self.db = db
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.scope = scope
        self.targets = targets
        self.source = source

    def run(self):
        try:
            result = run_predict_only(
                self.db, self.weights, conf=self.conf, imgsz=self.imgsz,
                scope=self.scope, targets=self.targets, source=self.source,
                progress_callback=self._cb,
            )
            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class ConceptDiscoveryWorker(BaseWorker):
    progress = Signal(str, int)
    finished = Signal(object)   # dict summary
    error    = Signal(str)

    def __init__(self, db, concepts, top_k, conf, imgsz, sam_model,
                 source, restrict_status):
        super().__init__()
        self.db = db
        self.concepts = concepts
        self.top_k = top_k
        self.conf = conf
        self.imgsz = imgsz
        self.sam_model = sam_model
        self.source = source
        self.restrict_status = restrict_status

    def run(self):
        try:
            result = run_concept_discovery(
                self.db, self.concepts, self.top_k, self.conf, self.imgsz,
                self.sam_model, source=self.source,
                restrict_status=self.restrict_status,
                progress_callback=self._cb,
            )
            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class SearchIndexWorker(BaseWorker):
    progress = Signal(int, int)
    finished = Signal(int)
    error    = Signal(str)

    def __init__(self, db):
        super().__init__()
        self.db = db

    def run(self):
        try:
            n = generate_search_embeddings(self.db, progress_callback=self._cb)
            self.finished.emit(n)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class SearchQueryWorker(BaseWorker):
    progress = Signal(str, int)
    finished = Signal(object)   # list[(image_id, score)]
    error    = Signal(str)

    def __init__(self, engine, mode: str, query):
        super().__init__()
        self.engine = engine
        self.mode = mode      # 'text' | 'image'
        self.query = query

    def run(self):
        try:
            if self.mode == "image":
                results = self.engine.search_image(self.query, progress_callback=self._cb)
            else:
                results = self.engine.search_text(self.query, progress_callback=self._cb)
            self.finished.emit(results)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class ActiveLearningWorker(BaseWorker):
    progress = Signal(str, int)
    finished = Signal(object)   # dict summary
    error    = Signal(str)

    def __init__(self, db, accept_enabled: bool, accept_threshold: float,
                 mark_reviewed: bool, prioritize: bool, center: float):
        super().__init__()
        self.db = db
        self.accept_enabled = accept_enabled
        self.accept_threshold = accept_threshold
        self.mark_reviewed = mark_reviewed
        self.prioritize = prioritize
        self.center = center

    def run(self):
        try:
            result = run_active_learning(
                self.db,
                accept_enabled=self.accept_enabled,
                accept_threshold=self.accept_threshold,
                mark_reviewed=self.mark_reviewed,
                prioritize=self.prioritize,
                center=self.center,
                progress_callback=self._cb,
            )
            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class NearDuplicatesWorker(BaseWorker):
    progress = Signal(str, int)
    finished = Signal(object)   # dict summary
    error    = Signal(str)

    def __init__(self, db, threshold: float, action: str):
        super().__init__()
        self.db = db
        self.threshold = threshold
        self.action = action

    def run(self):
        try:
            result = run_near_duplicates(
                self.db, self.threshold, self.action,
                progress_callback=self._cb,
            )
            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))


class LoadAnnotationsWorker(BaseWorker):
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
                    progress_callback=self._cb,
                )
            elif self.fmt == "yolo":
                result = load_yolo(
                    self.kwargs["labels_dir"], self.db,
                    names_file=self.kwargs.get("names_file"),
                    overwrite=self.kwargs.get("overwrite", False),
                    progress_callback=self._cb,
                )
            else:
                result = load_labelme(
                    self.kwargs["json_dir"], self.db,
                    overwrite=self.kwargs.get("overwrite", False),
                    progress_callback=self._cb,
                )
            self.finished.emit(result)
        except WorkerCancelled:
            self.cancelled.emit()
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
        self._selected_cluster_id: int | None = None
        self._auto_label_reload_image_id: int | None = None
        self._lasso_nav_ids: list[int] = []   # when non-empty, navigation uses this
        self._lasso_nav_idx: int = 0
        self._thread: QThread | None = None
        self._worker = None
        self._search_engine: SemanticSearchEngine | None = None

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
        self._cluster_panel.cluster_autolabel_requested.connect(self._on_cluster_autolabel)
        self._cluster_panel.cluster_sam3_requested.connect(self._on_cluster_sam3)
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
        self._image_grid.search_requested.connect(self._on_search_text)
        self._image_grid.search_similar_requested.connect(self._on_search_similar)
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
        self._stop_btn = QPushButton("⏹ Detener")
        self._stop_btn.setStyleSheet(
            "QPushButton { background:#8B1A1A; color:white; font-weight:bold;"
            " border-radius:3px; padding:1px 10px; }"
            "QPushButton:disabled { background:#5a3030; color:#bbb; }"
        )
        self._stop_btn.setVisible(False)
        self._stop_btn.clicked.connect(self._cancel_current_task)
        self._status_bar.addPermanentWidget(self._stop_btn)
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

        self._act_open_manifest = QAction("Abrir dataset.json...", self)
        self._act_open_manifest.triggered.connect(self.open_manifest)
        file_menu.addAction(self._act_open_manifest)

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
        self._act_embed = QAction("Generar Embeddings...", self)
        self._act_embed.triggered.connect(self.generate_embeddings)
        tools_menu.addAction(self._act_embed)

        self._act_cluster = QAction("Agrupar por Similitud (UMAP + HDBSCAN)...", self)
        self._act_cluster.triggered.connect(self.run_clustering)
        tools_menu.addAction(self._act_cluster)

        self._act_analysis_lab = QAction("Dataset Analysis Lab...", self)
        self._act_analysis_lab.triggered.connect(self.run_analysis_lab)
        tools_menu.addAction(self._act_analysis_lab)

        self._act_near_dups = QAction("Imágenes duplicadas (near-duplicates)...", self)
        self._act_near_dups.triggered.connect(self.run_near_duplicates)
        tools_menu.addAction(self._act_near_dups)

        self._act_search_index = QAction("Generar índice de búsqueda (SigLIP2)...", self)
        self._act_search_index.triggered.connect(self.run_search_index)
        tools_menu.addAction(self._act_search_index)

        self._act_concept_discovery = QAction("Descubrimiento por diccionario (SigLIP2 → SAM 3)...", self)
        self._act_concept_discovery.triggered.connect(self.open_concept_discovery)
        tools_menu.addAction(self._act_concept_discovery)

        self._act_active_learning = QAction("Active learning — priorizar revisión...", self)
        self._act_active_learning.triggered.connect(self.open_active_learning)
        tools_menu.addAction(self._act_active_learning)

        tools_menu.addSeparator()

        self._act_autolabel_hub = QAction("Auto-etiquetado / Modelos...", self)
        self._act_autolabel_hub.setShortcut("Ctrl+L")
        self._act_autolabel_hub.triggered.connect(self.run_autolabel_hub)
        tools_menu.addAction(self._act_autolabel_hub)

        tools_menu.addSeparator()

        self._act_bbox_embed = QAction("Embeddings de Crops de Bboxes...", self)
        self._act_bbox_embed.triggered.connect(self.generate_bbox_embeddings)
        tools_menu.addAction(self._act_bbox_embed)

        self._act_bbox_umap = QAction("Analizar Bboxes Raras (UMAP visual)...", self)
        self._act_bbox_umap.triggered.connect(self.run_bbox_umap)
        tools_menu.addAction(self._act_bbox_umap)

        self._act_bbox_geom = QAction("Bboxes raras por geometría (por clase)...", self)
        self._act_bbox_geom.triggered.connect(self.open_geometry_outliers)
        tools_menu.addAction(self._act_bbox_geom)

        self._act_cleanup_iou = QAction("Limpiar detecciones solapadas (NMS por clase)...", self)
        self._act_cleanup_iou.triggered.connect(self.run_cleanup_overlaps)
        tools_menu.addAction(self._act_cleanup_iou)

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
                    self._act_autolabel_hub, self._act_near_dups,
                    self._act_active_learning, self._act_search_index,
                    self._act_concept_discovery,
                    self._act_bbox_embed, self._act_bbox_umap, self._act_bbox_geom,
                    self._act_cleanup_iou,
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
        is_manifest = Project.is_manifest_file(path)
        if not is_manifest and not Project.is_valid_project(path):
            QMessageBox.warning(self, "Error",
                                "La carpeta seleccionada no contiene un proyecto válido.\n"
                                "Crea uno nuevo con Archivo → Nuevo Proyecto o abre un dataset.json.")
            return
        try:
            if self._project:
                self._project.close()
            if is_manifest:
                self._project = open_project_from_manifest(path)
            else:
                self._project = Project.open(path)
            self._on_project_loaded()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo abrir el proyecto:\n{e}")

    def open_manifest(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir dataset.json",
            "",
            "Dataset manifest (*.json);;Todos los archivos (*)",
        )
        if f:
            self.open_project(Path(f))

    def _on_project_loaded(self):
        self._set_project_active(True)
        self.setWindowTitle(f"AnnotatAI — {self._project.name}")
        self._cluster_panel.set_db(self._project.db)
        self._annotation_editor.set_db(self._project.db)
        self._bbox_view.set_db(self._project.db)
        classes = self._project.db.get_all_classes()
        self._image_grid.set_classes(classes)
        self._graph_view.set_classes(classes)
        self._bbox_view.set_classes(classes)
        self._load_all_images()
        self._status_label.setText(f"Proyecto: {self._project.name}")

    def _load_all_images(self):
        # Mantener detection_min/avg_confidence al día con las detecciones reales
        # para que el filtro de confianza de la galería funcione.
        self._project.db.recompute_detection_confidence()
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
        if n_images == 0:
            QMessageBox.warning(self, "Embeddings", "Primero carga imágenes en el proyecto.")
            return

        backend_options = [
            ("DINOv3 ViT-S/16", "dinov3"),
            ("OpenCLIP ViT-B/32", "openclip"),
            ("DINOv2 ViT-S/14", "dinov2"),
            ("SigLIP base", "siglip"),
        ]
        labels = [label for label, _ in backend_options]
        choice, ok = QInputDialog.getItem(
            self,
            "Generar Embeddings",
            "Modelo de embeddings:",
            labels,
            0,
            False,
        )
        if not ok:
            return

        backend_name = dict(backend_options)[choice]
        selected_model = create_embedding_backend(backend_name).name
        n_emb = self._project.db.count_embeddings(selected_model)
        pending = n_images - n_emb
        if pending == 0:
            QMessageBox.information(
                self,
                "Embeddings",
                f"Todos los embeddings ya están generados con {choice}.",
            )
            return

        reply = QMessageBox.question(
            self, "Generar Embeddings",
            f"Se generarán embeddings {choice} para {pending} imágenes.\n"
            "Si existen embeddings de otro modelo, serán reemplazados para esas imágenes.\n\n"
            "Esto puede tardar varios minutos la primera vez.\n¿Continuar?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._run_worker(
            EmbeddingWorker(self._project.db, backend_name),
            on_progress=self._on_embed_progress,
            on_finished=self._on_embed_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText(f"Generando embeddings {choice}...")

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

    # ─── Imágenes duplicadas (near-duplicates) ───

    def run_near_duplicates(self):
        if not self._project:
            return
        n_emb = self._project.db.count_embeddings()
        if n_emb < 2:
            QMessageBox.warning(
                self, "Imágenes duplicadas",
                "Se necesitan embeddings. Generalos primero (Herramientas → "
                "Generar Embeddings).")
            return
        dlg = NearDuplicatesDialog(self, n_with_embeddings=n_emb)
        if not dlg.exec():
            return
        if dlg.action != "report":
            warn = (
                "Se descartarán los duplicados (1 por grupo queda activo)."
                if dlg.action == "collapse" else
                "Se copiarán las etiquetas dentro de cada grupo y los duplicados "
                "quedarán como revisados."
            )
            reply = QMessageBox.question(
                self, "Imágenes duplicadas", f"{warn}\n¿Continuar?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        self._run_worker(
            NearDuplicatesWorker(self._project.db, dlg.threshold, dlg.action),
            on_progress=self._on_cluster_progress,
            on_finished=self._on_near_dups_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText("Buscando imágenes duplicadas...")

    def _on_near_dups_finished(self, result: dict):
        self._show_progress(False)
        action = result.get("action", "report")
        n_groups = result.get("groups", 0)
        n_dups = result.get("duplicates", 0)
        affected = result.get("affected", 0)
        if action != "report":
            self._load_all_images()
            self._cluster_panel.refresh()
        self._status_label.setText(
            f"Near-duplicates: {n_groups} grupos, {n_dups} duplicados"
        )
        msg = (f"Grupos de casi-duplicados: {n_groups}\n"
               f"Imágenes duplicadas (sobre el representante): {n_dups}\n")
        if action == "collapse":
            msg += f"\nDescartadas: {affected}. Quedan los representantes para etiquetar."
        elif action == "propagate":
            msg += f"\nEtiquetas propagadas a {affected} imágenes (marcadas revisadas)."
        elif n_groups:
            msg += ("\nUsá 'Descartar duplicados' para reducir el trabajo, o "
                    "'Propagar etiquetas' tras etiquetar los representantes.")
        QMessageBox.information(self, "Imágenes duplicadas", msg)

    # ─── Búsqueda semántica (SigLIP2) ───

    def run_search_index(self):
        if not self._project:
            return
        if self._project.db.count_images() == 0:
            QMessageBox.warning(self, "Búsqueda", "Primero cargá imágenes.")
            return
        n_done = self._project.db.count_search_embeddings(SEARCH_MODEL)
        n_total = self._project.db.count_images()
        pending = n_total - n_done
        if pending == 0:
            QMessageBox.information(
                self, "Índice de búsqueda",
                f"El índice SigLIP2 ya está completo ({n_done} imágenes).")
            return
        reply = QMessageBox.question(
            self, "Índice de búsqueda (SigLIP2)",
            f"Se generarán embeddings SigLIP2 para {pending} imágenes "
            f"(separados de los de clustering).\n"
            "La primera vez descarga el modelo. ¿Continuar?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self._run_worker(
            SearchIndexWorker(self._project.db),
            on_progress=self._on_embed_progress,
            on_finished=self._on_search_index_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText("Generando índice de búsqueda (SigLIP2)...")

    def _on_search_index_finished(self, n: int):
        self._show_progress(False)
        self._search_engine = None   # forzar recarga de la matriz en la próxima búsqueda
        total = self._project.db.count_search_embeddings(SEARCH_MODEL)
        self._status_label.setText(f"Índice de búsqueda listo: {total} imágenes")
        QMessageBox.information(
            self, "Índice de búsqueda",
            f"Índice SigLIP2 listo ({total} imágenes).\n\n"
            "Usá la barra 🔎 en la grilla para buscar por texto, o clic derecho → "
            "«Buscar similares» sobre una imagen.")

    def _search_ready_or_prompt(self) -> bool:
        """True si hay índice; si no, ofrece generarlo."""
        if self._project.db.count_search_embeddings(SEARCH_MODEL) > 0:
            return True
        reply = QMessageBox.question(
            self, "Búsqueda semántica",
            "Todavía no hay índice de búsqueda (SigLIP2).\n¿Generarlo ahora?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.run_search_index()
        return False

    def _ensure_search_engine(self) -> SemanticSearchEngine:
        if self._search_engine is None:
            self._search_engine = SemanticSearchEngine(self._project.db)
        return self._search_engine

    def _on_search_text(self, text: str):
        if not self._project or not self._search_ready_or_prompt():
            return
        self._run_worker(
            SearchQueryWorker(self._ensure_search_engine(), "text", text),
            on_progress=self._on_cluster_progress,
            on_finished=self._on_search_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText(f"Buscando: «{text}»...")

    def _on_search_similar(self, image_id: int):
        if not self._project or not self._search_ready_or_prompt():
            return
        self._run_worker(
            SearchQueryWorker(self._ensure_search_engine(), "image", image_id),
            on_progress=self._on_cluster_progress,
            on_finished=self._on_search_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText("Buscando imágenes similares...")

    def _on_search_finished(self, results: list):
        self._show_progress(False)
        ids = [iid for iid, _score in results]
        if not ids:
            QMessageBox.information(self, "Búsqueda", "Sin resultados.")
            return
        self._image_grid.set_search_results(ids)
        self._tabs.setCurrentWidget(self._image_grid)
        self._status_label.setText(f"Búsqueda: {len(ids)} resultados (ordenados por similitud)")

    # ─── Descubrimiento por diccionario (SigLIP2 → SAM 3) ───

    def open_concept_discovery(self):
        if not self._project:
            return
        if self._project.db.count_search_embeddings(SEARCH_MODEL) == 0:
            reply = QMessageBox.question(
                self, "Descubrimiento por diccionario",
                "Necesitás el índice de búsqueda (SigLIP2) primero.\n¿Generarlo ahora?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.run_search_index()
            return

        default_text = self._project._meta.get("discovery_concepts", "")
        n_indexed = self._project.db.count_search_embeddings(SEARCH_MODEL)
        dlg = ConceptDiscoveryDialog(self, default_text=default_text,
                                     n_indexed=n_indexed)
        if not dlg.exec():
            return

        concepts = parse_concepts(dlg.raw_text)
        if not concepts:
            return
        # Persistir el diccionario en el proyecto
        self._project._meta["discovery_concepts"] = dlg.raw_text
        try:
            self._project.save_meta()
        except Exception:
            pass

        self._run_worker(
            ConceptDiscoveryWorker(
                self._project.db, concepts, dlg.top_k, dlg.conf, dlg.imgsz,
                dlg.model_path, dlg.source, dlg.restrict_status),
            on_progress=self._on_cluster_progress,
            on_finished=self._on_concept_discovery_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText("Descubrimiento por diccionario...")

    def _on_concept_discovery_finished(self, result: dict):
        self._show_progress(False)
        classes = self._project.db.get_all_classes()
        self._annotation_editor._refresh_classes()
        self._image_grid.set_classes(classes)
        self._graph_view.set_classes(classes)
        self._load_all_images()

        n_boxes = result.get("boxes", 0)
        n_images = result.get("images", 0)
        per_class = result.get("per_class", {})
        oom = result.get("oom_skipped", 0)
        self._status_label.setText(
            f"Descubrimiento: {n_boxes} cajas en {n_images} imágenes"
        )
        lines = "\n".join(f"  • {k}: {v}" for k, v in per_class.items()) or "  (ninguna)"
        msg = (f"Conceptos procesados: {result.get('concepts', 0)}\n"
               f"Cajas sugeridas: {n_boxes}\n"
               f"Imágenes con detecciones: {n_images}\n\n"
               f"Por clase:\n{lines}\n\n"
               "Revisá y aceptá/rechazá las sugerencias.")
        if oom:
            msg += f"\n\n⚠ {oom} candidata(s) saltadas por falta de memoria (bajá el tamaño de imagen)."
        QMessageBox.information(self, "Descubrimiento por diccionario", msg)

    # ─── Active learning (priorizar revisión) ───

    def open_active_learning(self):
        if not self._project:
            return
        n_pending = sum(1 for img in self._current_images if img.status == "pending")
        if n_pending == 0:
            QMessageBox.information(
                self, "Active learning",
                "No hay imágenes pendientes para procesar.")
            return
        dlg = ActiveLearningDialog(self, n_pending=n_pending)
        if not dlg.exec():
            return
        self._al_prioritized = dlg.prioritize
        self._run_worker(
            ActiveLearningWorker(
                self._project.db, dlg.accept_enabled, dlg.accept_threshold,
                dlg.mark_reviewed, dlg.prioritize, dlg.center),
            on_progress=self._on_cluster_progress,
            on_finished=self._on_active_learning_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText("Active learning...")

    def _on_active_learning_finished(self, result: dict):
        self._show_progress(False)
        accepted = result.get("accepted", 0)
        auto_reviewed = result.get("auto_reviewed", 0)
        prioritized = result.get("prioritized", 0)

        # Refrescar grilla/editor y, si se priorizó, ordenar por prioridad
        classes = self._project.db.get_all_classes()
        self._image_grid.set_classes(classes)
        self._load_all_images()
        if getattr(self, "_al_prioritized", False):
            self._image_grid.set_sort_mode("priority")
            self._tabs.setCurrentWidget(self._image_grid)

        self._status_label.setText(
            f"Active learning: {accepted} cajas aceptadas, {auto_reviewed} auto-revisadas"
        )
        QMessageBox.information(
            self, "Active learning",
            f"Sugerencias auto-aceptadas: {accepted}\n"
            f"Imágenes auto-revisadas: {auto_reviewed}\n"
            f"Imágenes priorizadas: {prioritized}\n\n"
            + ("La grilla quedó ordenada por incertidumbre (revisá primero lo de "
               "arriba)." if getattr(self, "_al_prioritized", False) else "")
        )

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

    # ─── Auto-etiquetado / Modelos (centro unificado) ───

    def _resolve_scope_rows(self, scope: str) -> list | None:
        """Resuelve el 'alcance' a filas de imágenes (formato get_all_images)."""
        db = self._project.db
        if scope == "current":
            img = self._annotation_editor._image
            if img is None:
                return None
            return [r for r in db.get_all_images() if r[0] == img.id]
        if scope == "group":
            if self._selected_cluster_id is None:
                return None
            return db.get_images_for_cluster(self._selected_cluster_id)
        if scope == "pending":
            return [r for r in db.get_all_images() if r[6] == "pending"]
        return db.get_all_images()

    def run_autolabel_hub(self):
        if not self._project:
            return
        db = self._project.db
        classes = db.get_all_classes()
        counts = db.count_human_boxes_per_class()
        exemplar_classes = [(cid, name, counts.get(cid, 0))
                            for cid, name, _ in classes if counts.get(cid, 0) > 0]
        train_counts = db.count_human_boxes_in_reviewed_per_class()
        n_pending = sum(1 for img in self._current_images if img.status == "pending")
        n_total = len(self._current_images)
        n_reviewed = db.count_reviewed_with_boxes()
        has_current = self._annotation_editor._image is not None
        has_group = self._selected_cluster_id is not None

        dlg = AutoLabelHubDialog(
            self, classes=classes, exemplar_classes=exemplar_classes,
            train_counts=train_counts,
            models_dir=str(self._project.project_path / "models"),
            has_current=has_current, has_group=has_group,
            n_pending=n_pending, n_total=n_total, n_reviewed=n_reviewed,
        )
        if not dlg.exec():
            return

        rows = self._resolve_scope_rows(dlg.scope)
        if not rows:
            QMessageBox.information(self, "Auto-etiquetado",
                                   "El alcance seleccionado no tiene imágenes.")
            return

        engine = dlg.engine
        # Si la imagen abierta en el editor entra en el alcance, recargarla al final
        cur = self._annotation_editor._image
        reload_id = cur.id if cur and cur.id in {r[0] for r in rows} else None

        if engine == "sam3":
            self._auto_label_reload_image_id = reload_id
            worker = AutoLabelWorker(
                db, rows, dlg.prompts, dlg.conf, dlg.source, dlg.model_path,
                save_polygons=dlg.save_polygons, imgsz=dlg.imgsz)
            self._run_worker(worker, on_progress=self._on_auto_label_progress,
                             on_finished=self._on_auto_label_finished,
                             on_error=self._on_worker_error)
            status = "Auto-etiquetando con SAM 3..."

        elif engine == "yoloe":
            self._auto_label_reload_image_id = reload_id
            worker = AutoLabelVisualWorker(
                db, dlg.class_id, dlg.class_name, rows, dlg.conf, dlg.source,
                imgsz=dlg.imgsz)
            self._run_worker(worker, on_progress=self._on_auto_label_progress,
                             on_finished=self._on_auto_label_finished,
                             on_error=self._on_worker_error)
            status = f"Auto-etiquetando '{dlg.class_name}' con YOLOE..."

        elif engine == "ptmodel":
            targets = [(r[0], r[1]) for r in rows]
            worker = PredictModelWorker(
                db, Path(dlg.weights_path), dlg.conf, dlg.imgsz,
                targets=targets, source=dlg.source)
            self._run_worker(worker, on_progress=self._on_train_progress,
                             on_finished=self._on_predict_finished,
                             on_error=self._on_worker_error)
            status = "Pre-etiquetando con modelo .pt..."

        else:  # train
            options = TrainOptions(
                model=dlg.train_model, epochs=dlg.epochs, imgsz=dlg.imgsz,
                batch=dlg.batch, val_fraction=dlg.val_fraction,
                predict_after=dlg.predict_after, predict_conf=dlg.conf,
                train_classes=dlg.train_classes)
            targets = [(r[0], r[1]) for r in rows] if dlg.predict_after else None
            worker = TrainModelWorker(db, self._project.project_path, options,
                                      predict_targets=targets)
            self._run_worker(worker, on_progress=self._on_train_progress,
                             on_finished=self._on_train_finished,
                             on_error=self._on_worker_error)
            status = "Entrenando modelo asistente..."

        self._show_progress(True)
        self._status_label.setText(status)

    # ─── Auto-etiquetado (SAM 3) — helpers legacy (menú de clusters) ───

    def run_auto_label(self):
        if not self._project:
            return
        rows = self._project.db.get_all_images()
        if not rows:
            QMessageBox.warning(self, "Sin imágenes",
                                "Primero cargá una carpeta de imágenes.")
            return
        n_pending = sum(1 for r in rows if r[6] == "pending")
        counts = self._project.db.count_human_boxes_per_class()
        exemplar_classes = [
            (cid, name, counts[cid])
            for cid, name, _ in self._project.db.get_all_classes()
            if counts.get(cid, 0) > 0
        ]
        dlg = AutoLabelDialog(self, n_pending=n_pending, n_total=len(rows),
                              exemplar_classes=exemplar_classes)
        if not dlg.exec():
            return

        image_rows = rows if dlg.scope == "all" else [r for r in rows if r[6] == "pending"]
        if not image_rows:
            QMessageBox.information(self, "Auto-etiquetar",
                                    "No hay imágenes en el alcance seleccionado.")
            return

        if dlg.mode == "visual":
            worker = AutoLabelVisualWorker(
                self._project.db, dlg.class_id, dlg.class_name,
                image_rows, dlg.conf, dlg.source, imgsz=dlg.imgsz)
            status = f"Auto-etiquetando '{dlg.class_name}' con YOLOE..."
        else:
            worker = AutoLabelWorker(
                self._project.db, image_rows, dlg.prompts, dlg.conf, dlg.source,
                dlg.model_path, save_polygons=dlg.save_polygons, imgsz=dlg.imgsz)
            status = "Auto-etiquetando con SAM 3..."

        self._run_worker(
            worker,
            on_progress=self._on_auto_label_progress,
            on_finished=self._on_auto_label_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText(status)

    def run_auto_label_current_image(self):
        if not self._project:
            return
        image = self._annotation_editor._image
        if image is None:
            QMessageBox.warning(
                self,
                "Auto-etiquetar imagen actual",
                "Abrí una imagen en el editor antes de ejecutar SAM 3 sobre la imagen actual.",
            )
            return
        rows = [row for row in self._project.db.get_all_images() if row[0] == image.id]
        if not rows:
            return
        self._run_sam3_text_autolabel(
            rows,
            fixed_scope_label=f"Imagen actual: {image.filename}",
            status_text=f"Auto-etiquetando imagen actual con SAM 3...",
            reload_image_id=image.id,
        )

    def run_auto_label_selected_cluster(self):
        if not self._project:
            return
        if self._selected_cluster_id is None:
            QMessageBox.warning(
                self,
                "Auto-etiquetar grupo",
                "Seleccioná un grupo/cluster en el panel izquierdo antes de ejecutar SAM 3.",
            )
            return
        rows = self._project.db.get_images_for_cluster(self._selected_cluster_id)
        if not rows:
            QMessageBox.information(self, "Auto-etiquetar grupo", "El grupo seleccionado no tiene imágenes.")
            return
        self._run_sam3_text_autolabel(
            rows,
            fixed_scope_label=f"Grupo seleccionado ({len(rows)} imágenes)",
            status_text="Auto-etiquetando grupo seleccionado con SAM 3...",
            reload_image_id=self._annotation_editor._image.id
            if self._annotation_editor._image and
            self._annotation_editor._image.id in {row[0] for row in rows}
            else None,
        )

    def _on_cluster_sam3(self, cluster_id: int):
        """SAM 3 por texto sobre un grupo, desde el menú del panel de clusters."""
        if not self._project:
            return
        rows = self._project.db.get_images_for_cluster(cluster_id)
        if not rows:
            QMessageBox.information(self, "Auto-etiquetar grupo",
                                    "El grupo seleccionado no tiene imágenes.")
            return
        current = self._annotation_editor._image
        reload_id = current.id if current and current.id in {r[0] for r in rows} else None
        self._run_sam3_text_autolabel(
            rows,
            fixed_scope_label=f"Grupo ({len(rows)} imágenes)",
            status_text="Detectando por texto en el grupo con SAM 3...",
            reload_image_id=reload_id,
        )

    def _run_sam3_text_autolabel(self, image_rows: list, fixed_scope_label: str,
                                 status_text: str, reload_image_id: int | None = None):
        dlg = AutoLabelDialog(
            self,
            n_pending=sum(1 for row in image_rows if row[6] == "pending"),
            n_total=len(image_rows),
            text_only=True,
            fixed_scope_label=fixed_scope_label,
        )
        if not dlg.exec():
            return

        self._auto_label_reload_image_id = reload_image_id
        self._run_worker(
            AutoLabelWorker(
                self._project.db,
                image_rows,
                dlg.prompts,
                dlg.conf,
                dlg.source,
                dlg.model_path,
                save_polygons=dlg.save_polygons,
                imgsz=dlg.imgsz,
            ),
            on_progress=self._on_auto_label_progress,
            on_finished=self._on_auto_label_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText(status_text)

    def _on_auto_label_progress(self, done: int, total: int):
        self._progress_bar.setMaximum(max(total, 1))
        self._progress_bar.setValue(done)
        self._status_label.setText(f"Auto-etiquetando: {done}/{total}")

    def _on_auto_label_finished(self, result: dict):
        self._show_progress(False)
        classes = self._project.db.get_all_classes()
        self._annotation_editor._refresh_classes()
        self._image_grid.set_classes(classes)
        self._graph_view.set_classes(classes)
        self._load_all_images()
        if self._auto_label_reload_image_id:
            self._load_image_in_editor(self._auto_label_reload_image_id)
            self._tabs.setCurrentWidget(self._annotation_editor)
            self._auto_label_reload_image_id = None
        n_boxes = result.get("boxes", 0)
        n_images = result.get("images", 0)
        self._status_label.setText(
            f"Auto-etiquetado listo — {n_boxes} cajas en {n_images} imágenes"
        )
        new_cls = result.get("classes", [])
        oom = result.get("oom_skipped", 0)
        refs = result.get("references")
        msg = (
            f"Cajas generadas: {n_boxes}\n"
            f"Imágenes con detecciones: {n_images}\n"
            + (f"Imágenes de ejemplo usadas (YOLOE): {refs}\n" if refs else "")
            + (f"Clases nuevas: {', '.join(new_cls)}\n" if new_cls else "")
        )
        if oom:
            msg += (f"\n⚠ {oom} imagen(es) se saltaron por falta de memoria de GPU.\n"
                    "Probá un tamaño de imagen menor (p. ej. 768 o 640) al re-ejecutar.")
        QMessageBox.information(self, "Auto-etiquetado completo", msg)

    def _on_cluster_autolabel(self, cluster_id: int):
        if not self._project:
            return
        db = self._project.db
        human_classes = db.get_human_classes_in_cluster(cluster_id)
        if not human_classes:
            QMessageBox.information(
                self, "Propagar etiquetas",
                "Este grupo no tiene anotaciones hechas a mano.\n\n"
                "Anotá al menos un objeto (p. ej. una bomba) en una imagen del grupo,\n"
                "guardá, y volvé a propagar.")
            return

        if len(human_classes) == 1:
            class_id, class_name, _ = human_classes[0]
        else:
            labels = [f"{name}  ({n} cajas)" for _, name, n in human_classes]
            choice, ok = QInputDialog.getItem(
                self, "Propagar etiquetas",
                "¿Qué clase querés propagar al grupo?", labels, 0, False)
            if not ok:
                return
            idx = labels.index(choice)
            class_id, class_name, _ = human_classes[idx]

        target_rows = db.get_images_for_cluster(cluster_id)
        exemplar_ids = {r[0] for r in target_rows}

        reply = QMessageBox.question(
            self, "Propagar etiquetas al grupo",
            f"Usar tus cajas de «{class_name}» como ejemplo y detectar objetos\n"
            f"similares en las {len(target_rows)} imágenes del grupo con YOLOE.\n\n"
            "Se guardan como sugerencias para revisar. ¿Continuar?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._run_worker(
            AutoLabelVisualWorker(
                db, class_id, class_name, target_rows, 0.20, "suggested",
                exemplar_image_ids=exemplar_ids),
            on_progress=self._on_auto_label_progress,
            on_finished=self._on_auto_label_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText(
            f"Propagando '{class_name}' al grupo con YOLOE...")

    # ─── Entrenar modelo asistente (YOLO) ───

    def run_train_model(self):
        if not self._project:
            return
        n_reviewed = self._project.db.count_reviewed_with_boxes()
        if n_reviewed == 0:
            QMessageBox.warning(
                self, "Entrenar modelo",
                "No hay imágenes revisadas con cajas hechas a mano.\n\n"
                "Anotá y marcá como «Revisada» (R) algunas imágenes primero;\n"
                "esas son los ejemplos con los que aprende el modelo.")
            return
        n_pending = sum(1 for img in self._current_images if img.status == "pending")
        classes = self._project.db.get_all_classes()
        class_counts = self._project.db.count_human_boxes_in_reviewed_per_class()
        dlg = TrainModelDialog(self, n_reviewed=n_reviewed, n_pending=n_pending,
                               classes=classes, class_counts=class_counts)
        if not dlg.exec():
            return

        options = TrainOptions(
            model=dlg.model_name,
            epochs=dlg.epochs,
            imgsz=dlg.imgsz,
            batch=dlg.batch,
            val_fraction=dlg.val_fraction,
            predict_after=dlg.predict_after,
            predict_conf=dlg.predict_conf,
            predict_scope=dlg.predict_scope,
            train_classes=dlg.train_classes,
        )
        self._run_worker(
            TrainModelWorker(self._project.db, self._project.project_path, options),
            on_progress=self._on_train_progress,
            on_finished=self._on_train_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText("Entrenando modelo asistente...")

    def _on_train_progress(self, msg: str, pct: int):
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(pct)
        self._status_label.setText(msg)

    def _on_train_finished(self, result: TrainResult):
        self._show_progress(False)
        if result.error:
            QMessageBox.warning(self, "Entrenar modelo", result.error)
            self._status_label.setText("Entrenamiento cancelado.")
            return

        # Refrescar grilla/UMAP para mostrar las sugerencias importadas
        classes = self._project.db.get_all_classes()
        self._annotation_editor._refresh_classes()
        self._image_grid.set_classes(classes)
        self._graph_view.set_classes(classes)
        self._load_all_images()

        self._status_label.setText(
            f"Modelo listo — {result.pred_boxes} sugerencias en {result.pred_images} imágenes"
        )
        msg = (
            f"Entrenamiento completo.\n\n"
            f"Imágenes de entrenamiento: {result.n_train}\n"
            f"Validación: {result.n_val}\n"
            f"Clases: {', '.join(result.names) if result.names else '-'}\n"
            f"Modelo: {result.weights}\n"
        )
        if result.pred_boxes:
            msg += (f"\nPre-etiquetado: {result.pred_boxes} cajas sugeridas en "
                    f"{result.pred_images} imágenes (revisalas y aceptá/rechazá).")
        QMessageBox.information(self, "Entrenar modelo", msg)

    def run_predict_with_model(self):
        if not self._project:
            return
        n_pending = sum(1 for img in self._current_images if img.status == "pending")
        # Empezar a explorar desde la carpeta de modelos del proyecto actual
        default_dir = str(self._project.project_path / "models")
        dlg = PredictWithModelDialog(self, n_pending=n_pending, default_dir=default_dir)
        if not dlg.exec():
            return
        self._run_worker(
            PredictModelWorker(self._project.db, dlg.weights_path,
                               dlg.conf, dlg.imgsz, dlg.scope),
            on_progress=self._on_train_progress,
            on_finished=self._on_predict_finished,
            on_error=self._on_worker_error,
        )
        self._show_progress(True)
        self._status_label.setText("Pre-etiquetando con modelo entrenado...")

    def _on_predict_finished(self, result: TrainResult):
        self._show_progress(False)
        if result.error:
            QMessageBox.warning(self, "Pre-etiquetar con modelo", result.error)
            self._status_label.setText("Pre-etiquetado cancelado.")
            return
        classes = self._project.db.get_all_classes()
        self._annotation_editor._refresh_classes()
        self._image_grid.set_classes(classes)
        self._graph_view.set_classes(classes)
        self._load_all_images()
        self._status_label.setText(
            f"Pre-etiquetado listo — {result.pred_boxes} sugerencias en {result.pred_images} imágenes"
        )
        QMessageBox.information(
            self, "Pre-etiquetar con modelo",
            f"Modelo: {result.weights}\n"
            f"Clases detectadas: {', '.join(result.names) if result.names else '-'}\n\n"
            f"{result.pred_boxes} cajas sugeridas en {result.pred_images} imágenes.\n"
            "Revisalas en el editor y aceptá/rechazá."
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

    def open_geometry_outliers(self):
        if not self._project:
            return
        with self._project.db.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM annotations")
            if cur.fetchone()[0] == 0:
                QMessageBox.information(
                    self, "Bboxes raras por geometría",
                    "No hay anotaciones todavía.")
                return
        classes = self._project.db.get_all_classes()
        dlg = GeometryOutliersDialog(self._project.db, classes, self)
        dlg.open_image_requested.connect(self._on_bbox_open_image)
        self._geometry_dialog = dlg   # mantener referencia (modeless)
        dlg.show()

    def run_cleanup_overlaps(self):
        if not self._project:
            return
        iou, ok = QInputDialog.getDouble(
            self, "Limpiar detecciones solapadas",
            "Umbral IoU — cajas de la MISMA clase que se solapan más que esto\n"
            "colapsan a la de mayor confianza. Las cajas humanas nunca se borran.",
            0.50, 0.10, 0.95, 2)
        if not ok:
            return
        result = cleanup_overlapping(self._project.db, iou_thresh=float(iou))
        self._load_all_images()
        n = result.get("removed", 0)
        imgs = result.get("images", 0)
        self._status_label.setText(
            f"Limpieza IoU: {n} cajas solapadas eliminadas en {imgs} imágenes")
        QMessageBox.information(
            self, "Limpiar detecciones solapadas",
            f"Cajas solapadas eliminadas: {n}\nImágenes afectadas: {imgs}")

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
        # Activar la pestaña del editor antes de cargar, para que el canvas sea
        # visible y pueda tomar el foco del teclado (← → R X).
        self._tabs.setCurrentIndex(2)
        self._load_image_in_editor(image_id)

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
        ordered_ids = self._image_grid.get_ordered_image_ids()
        visible_set = set(self._image_grid.get_visible_image_ids())
        if not ordered_ids or not visible_set:
            return

        current_id = (
            self._current_images[self._current_editor_idx].id
            if 0 <= self._current_editor_idx < len(self._current_images)
            else None
        )
        # Anchor on the current image's position in the full display order, even
        # if it just left the filter (e.g. marked reviewed). Then walk in the
        # requested direction to the next still-visible image.
        try:
            pos = ordered_ids.index(current_id)
        except ValueError:
            pos = -delta % len(ordered_ids)   # so first step lands on index 0

        next_id = None
        i = pos
        for _ in range(len(ordered_ids)):
            i = (i + delta) % len(ordered_ids)
            if ordered_ids[i] in visible_set:
                next_id = ordered_ids[i]
                break
        if next_id is None:
            return

        full_ids = [img.id for img in self._current_images]
        try:
            self._current_editor_idx = full_ids.index(next_id)
        except ValueError:
            pass

        self._load_image_in_editor(next_id)
        visible_ids = self._image_grid.get_visible_image_ids()
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

        # "Revisar y continuar": avanzar automáticamente a la siguiente imagen
        # tras marcar Revisada/Descartar en el editor (aplica a ambas).
        editing = self._annotation_editor._image
        if self._tabs.currentIndex() != 2 or editing is None or editing.id != image_id:
            return

        if self._lasso_nav_ids:
            # Navegación por lazo (vista UMAP): avanzar dentro de la selección.
            self._navigate_images(1)
        elif image_id not in set(self._image_grid.get_visible_image_ids()):
            # Filtro de grilla: avanzar solo si la imagen dejó de pasar el filtro.
            self._navigate_images(1)

    def _on_graph_image_clicked(self, image_id: int):
        self._image_grid.highlight_image(image_id)
        self._open_annotation_editor(image_id)

    def _on_cluster_selected(self, cluster_id: int):
        self._selected_cluster_id = cluster_id if cluster_id >= 0 else None
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
        if hasattr(self._worker, "cancelled"):
            self._worker.cancelled.connect(self._on_worker_cancelled)
            self._worker.cancelled.connect(self._thread.quit)
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

    def _cancel_current_task(self):
        if self._worker is not None and hasattr(self._worker, "request_cancel"):
            self._worker.request_cancel()
            self._stop_btn.setEnabled(False)
            self._stop_btn.setText("Cancelando...")
            self._status_label.setText("Cancelando — esperá a que termine el paso actual...")

    def _on_worker_cancelled(self):
        self._show_progress(False)
        self._status_label.setText("Operación cancelada.")
        # Reflejar cualquier escritura parcial en la DB
        if self._project:
            self._load_all_images()
            self._cluster_panel.refresh()

    def _show_progress(self, visible: bool):
        self._progress_bar.setVisible(visible)
        self._stop_btn.setVisible(visible)
        self._stop_btn.setEnabled(visible)
        self._stop_btn.setText("⏹ Detener")
        if not visible:
            self._progress_bar.setValue(0)

    def closeEvent(self, event):
        try:
            self._annotation_editor.shutdown_sam()
        except Exception:
            pass
        if self._project:
            self._project.close()
        event.accept()
