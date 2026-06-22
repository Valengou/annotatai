# Dataset Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `dataset.json` support so AnnotatAI can open projects backed by source image paths without copying images.

**Architecture:** Keep manifest parsing in `app/core/project_manifest.py`, reuse existing `Project` and thumbnail/indexing code, and only adapt the UI/CLI to pass manifest files through the core API. Image files remain outside the project; project state stays in the sidecar `.annotatai` directory.

**Tech Stack:** Python, PySide6, SQLite, unittest, Pillow test images.

---

### Task 1: Core Manifest Behavior

**Files:**
- Create: `tests/test_project_manifest.py`
- Create: `app/core/project_manifest.py`
- Modify: `app/core/image_indexer.py`

- [x] **Step 1: Write failing tests**

Create tests that build temporary images, write a `dataset.json`, assert relative paths resolve, roots expand, source images are not copied into the sidecar project, and re-opening is idempotent.

- [x] **Step 2: Run tests in red**

Run: `python -m unittest discover -s tests -p test_project_manifest.py -v`

Expected: import failure for `app.core.project_manifest`.

- [x] **Step 3: Implement minimal core code**

Add `resolve_manifest_image_paths`, `project_path_for_manifest`, and `open_project_from_manifest`. Add `index_image_paths` so roots and explicit paths can share the existing thumbnail/database logic.

- [x] **Step 4: Run focused tests in green**

Run: `python -m unittest discover -s tests -p test_project_manifest.py -v`

Expected: all manifest tests pass.

### Task 2: UI And CLI Opening

**Files:**
- Modify: `app/core/project.py`
- Modify: `app/ui/main_window.py`
- Modify: `app/main.py`

- [x] **Step 1: Write failing tests for path classification**

Add unit coverage for project path classification so `.json` manifests are recognized separately from project directories.

- [x] **Step 2: Run tests in red**

Run: `python -m unittest discover -s tests -p test_project_manifest.py -v`

Expected: missing classification API.

- [x] **Step 3: Implement opening support**

Add `Project.is_manifest_file`. In `MainWindow.open_project`, if a selected path is a manifest, call `open_project_from_manifest`; otherwise keep existing valid-project behavior. In `app.main`, pass the CLI path through the same method.

- [x] **Step 4: Run focused tests in green**

Run: `python -m unittest discover -s tests -p test_project_manifest.py -v`

Expected: all manifest tests pass.

### Task 3: Full Verification

**Files:**
- Verify only.

- [x] **Step 1: Run existing tests**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 2: Smoke-test import path**

Run: `python -m app.main <path-to-dataset.json>` manually when a GUI session is desired.

Expected: app opens the sidecar project and shows indexed images.
