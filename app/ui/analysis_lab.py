from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..core.dataset_analysis import AnalysisOptions, safe_run_name
from ..utils.config import (
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
)


class AnalysisLabDialog(QDialog):
    def __init__(self, project_path: Path, parent=None):
        super().__init__(parent)
        self.project_path = Path(project_path)
        self.setWindowTitle("Dataset Analysis Lab")
        self.setMinimumWidth(620)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QLabel(
            "Genera embeddings, UMAP/HDBSCAN, reportes CSV/HTML y carpetas de review "
            "para el proyecto actual."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form_group = QGroupBox("Run")
        form = QFormLayout(form_group)

        default_name = datetime.now().strftime("analysis_%Y%m%d_%H%M%S")
        self._run_name = QLineEdit(default_name)
        form.addRow("Nombre:", self._run_name)

        self._backend = QComboBox()
        self._backend.addItem("OpenCLIP ViT-B/32", "openclip")
        self._backend.addItem("DINOv2 ViT-S/14", "dinov2")
        self._backend.addItem("DINOv3 ViT-S/16", "dinov3")
        self._backend.addItem("SigLIP base", "siglip")
        self._backend.addItem("ConvNeXt checkpoint", "convnext_checkpoint")
        self._backend.currentIndexChanged.connect(self._update_backend_fields)
        form.addRow("Embedding:", self._backend)

        self._checkpoint = QLineEdit()
        self._checkpoint.setPlaceholderText("best.pt para ConvNeXt")
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(self._checkpoint, stretch=1)
        checkpoint_btn = QPushButton("Examinar...")
        checkpoint_btn.clicked.connect(self._browse_checkpoint)
        checkpoint_row.addWidget(checkpoint_btn)
        form.addRow("Checkpoint:", checkpoint_row)
        self._checkpoint_btn = checkpoint_btn

        self._predictions = QLineEdit()
        self._predictions.setPlaceholderText("Opcional: CSV con true_class/pred_class/path")
        predictions_row = QHBoxLayout()
        predictions_row.addWidget(self._predictions, stretch=1)
        predictions_btn = QPushButton("Examinar...")
        predictions_btn.clicked.connect(lambda: self._browse_csv(self._predictions))
        predictions_row.addWidget(predictions_btn)
        form.addRow("Predicciones:", predictions_row)

        self._metadata = QLineEdit()
        self._metadata.setPlaceholderText("Opcional: metadata.csv del proyecto/dataset")
        metadata_row = QHBoxLayout()
        metadata_row.addWidget(self._metadata, stretch=1)
        metadata_btn = QPushButton("Examinar...")
        metadata_btn.clicked.connect(lambda: self._browse_csv(self._metadata))
        metadata_row.addWidget(metadata_btn)
        form.addRow("Metadata:", metadata_row)

        layout.addWidget(form_group)

        params_group = QGroupBox("Parametros")
        params = QFormLayout(params_group)

        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 256)
        self._batch_size.setValue(32)
        params.addRow("Batch size:", self._batch_size)

        self._umap_neighbors = QSpinBox()
        self._umap_neighbors.setRange(2, 200)
        self._umap_neighbors.setValue(UMAP_N_NEIGHBORS)
        params.addRow("UMAP vecinos:", self._umap_neighbors)

        self._umap_min_dist = QDoubleSpinBox()
        self._umap_min_dist.setRange(0.0, 0.99)
        self._umap_min_dist.setDecimals(2)
        self._umap_min_dist.setSingleStep(0.05)
        self._umap_min_dist.setValue(UMAP_MIN_DIST)
        params.addRow("UMAP min_dist:", self._umap_min_dist)

        self._hdbscan_min_cluster = QSpinBox()
        self._hdbscan_min_cluster.setRange(2, 500)
        self._hdbscan_min_cluster.setValue(HDBSCAN_MIN_CLUSTER_SIZE)
        params.addRow("HDBSCAN min cluster:", self._hdbscan_min_cluster)

        self._hdbscan_min_samples = QSpinBox()
        self._hdbscan_min_samples.setRange(1, 200)
        self._hdbscan_min_samples.setValue(HDBSCAN_MIN_SAMPLES)
        params.addRow("HDBSCAN min samples:", self._hdbscan_min_samples)

        layout.addWidget(params_group)

        self._apply_to_project = QCheckBox("Aplicar clusters y UMAP al proyecto")
        self._apply_to_project.setChecked(True)
        layout.addWidget(self._apply_to_project)

        self._export_review = QCheckBox("Exportar carpetas de review de clusters prioritarios")
        self._export_review.setChecked(True)
        layout.addWidget(self._export_review)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_backend_fields()

    @property
    def options(self) -> AnalysisOptions:
        checkpoint = self._path_or_none(self._checkpoint.text())
        predictions = self._path_or_none(self._predictions.text())
        metadata = self._path_or_none(self._metadata.text())
        return AnalysisOptions(
            run_name=safe_run_name(self._run_name.text()),
            backend_name=self._backend.currentData(),
            checkpoint_path=checkpoint,
            predictions_csv=predictions,
            metadata_csv=metadata,
            batch_size=self._batch_size.value(),
            umap_neighbors=self._umap_neighbors.value(),
            umap_min_dist=self._umap_min_dist.value(),
            hdbscan_min_cluster_size=self._hdbscan_min_cluster.value(),
            hdbscan_min_samples=self._hdbscan_min_samples.value(),
            apply_to_project=self._apply_to_project.isChecked(),
            export_review_clusters=self._export_review.isChecked(),
        )

    def _update_backend_fields(self):
        needs_checkpoint = self._backend.currentData() == "convnext_checkpoint"
        self._checkpoint.setEnabled(needs_checkpoint)
        self._checkpoint_btn.setEnabled(needs_checkpoint)

    def _browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar checkpoint ConvNeXt",
            str(self.project_path),
            "PyTorch checkpoints (*.pt *.pth);;Todos (*.*)",
        )
        if path:
            self._checkpoint.setText(path)

    def _browse_csv(self, target: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar CSV",
            str(self.project_path),
            "CSV (*.csv);;Todos (*.*)",
        )
        if path:
            target.setText(path)

    def _validate(self):
        if not safe_run_name(self._run_name.text()):
            QMessageBox.warning(self, "Analysis Lab", "El nombre del run no puede estar vacio.")
            return
        if self._backend.currentData() == "convnext_checkpoint":
            checkpoint = self._path_or_none(self._checkpoint.text())
            if not checkpoint or not checkpoint.is_file():
                QMessageBox.warning(self, "Analysis Lab", "Selecciona un checkpoint ConvNeXt valido.")
                return
        for label, edit in (("predicciones", self._predictions), ("metadata", self._metadata)):
            value = self._path_or_none(edit.text())
            if value and not value.is_file():
                QMessageBox.warning(self, "Analysis Lab", f"El CSV de {label} no existe:\n{value}")
                return
        self.accept()

    @staticmethod
    def _path_or_none(value: str) -> Path | None:
        value = value.strip()
        return Path(value) if value else None
