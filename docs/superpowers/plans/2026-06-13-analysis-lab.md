# Analysis Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable Dataset Analysis Lab to AnnotatAI for CLIP/DINOv2/SigLIP/ConvNeXt embedding analysis, UMAP/HDBSCAN clustering, error summaries, and review exports.

**Architecture:** Keep model feature extraction in isolated backend classes, keep clustering/report generation in a core module callable from scripts or UI, and expose a small PySide dialog plus worker from `main_window.py`. The analysis writes a timestamped run folder under the current project and can optionally apply the generated clusters/projections back to the project database.

**Tech Stack:** PySide6, PyTorch, OpenCLIP, optional Torch Hub DINOv2, optional Transformers SigLIP, timm/torchvision ConvNeXt checkpoints, UMAP, HDBSCAN, matplotlib, pytest.

---

### Task 1: Core Analysis Helpers

**Files:**
- Create: `app/core/dataset_analysis.py`
- Test: `tests/test_dataset_analysis.py`

- [x] **Step 1: Write failing tests**

Tests should cover metadata matching, prediction error labels, cluster summaries, and review priority scoring without loading ML models.

- [x] **Step 2: Run tests to verify failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_dataset_analysis.py -q`
Expected: FAIL because `app.core.dataset_analysis` does not exist yet.

- [x] **Step 3: Implement minimal helper functions**

Create dataclasses/options and pure helper functions for reading CSV metadata, merging assignments, summarizing clusters, and writing CSV/HTML outputs.

- [x] **Step 4: Run tests to verify pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_dataset_analysis.py -q`
Expected: PASS.

### Task 2: Embedding Backends

**Files:**
- Create: `app/core/embedding_backends.py`
- Modify: `app/core/dataset_analysis.py`

- [x] **Step 1: Add backend factory**

Implement `openclip`, `dinov2`, `siglip`, and `convnext_checkpoint` backends behind one `embed_paths()` interface. Optional backends raise actionable `RuntimeError` messages when dependencies are missing.

- [x] **Step 2: Wire analysis runner**

Run selected backend, normalize embeddings, run UMAP/HDBSCAN, save `embeddings.npz`, `umap.csv`, `cluster_assignments.csv`, `cluster_summary.csv`, `report.html`, and optional review folders.

### Task 3: UI Integration

**Files:**
- Create: `app/ui/analysis_lab.py`
- Modify: `app/ui/main_window.py`

- [x] **Step 1: Add configuration dialog**

Expose run name, backend, optional checkpoint, predictions CSV, metadata CSV, and apply-to-project option.

- [x] **Step 2: Add worker and menu action**

Add `Herramientas -> Dataset Analysis Lab...`, run analysis in a `QThread`, refresh cluster panel/grid/UMAP graph when clusters are applied.

### Task 4: Verification

**Files:**
- All touched files

- [x] **Step 1: Run unit tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_dataset_analysis.py -q`

- [x] **Step 2: Run import smoke checks**

Run: `.venv\Scripts\python.exe -c "from app.ui.main_window import MainWindow; from app.core.dataset_analysis import AnalysisOptions"`

- [x] **Step 3: Report exact commands and results**

Include output paths and any optional dependency caveats.
