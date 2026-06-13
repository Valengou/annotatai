"""
Bootstrap script: indexes a dataset folder + imports COCO annotations into an
AnnotatAI project, then launches the GUI.

Usage:
  python bootstrap_dataset.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

DATASET_DIR   = Path(r"C:\Users\valen\Documents\Valen\Drone_AI\Code\DAIS-inspection-pipeline\inspection_pipeline\output\datasets\insulator_detector_combined_4recordings_dedup_phash2")
IMAGES_DIR    = DATASET_DIR / "images"
COCO_JSON     = DATASET_DIR / "annotations" / "instances_train.json"
PROJECT_PARENT = ROOT / "projects"
PROJECT_NAME  = "insulator_detector_combined"


def progress_bar(done, total, prefix=""):
    pct = done / max(total, 1)
    filled = int(pct * 30)
    bar = "#" * filled + "-" * (30 - filled)
    print(f"\r{prefix} [{bar}] {done}/{total}", end="", flush=True)


def main():
    from app.core.project import Project
    from app.core.image_indexer import index_images
    from app.core.loaders import load_coco

    project_path = PROJECT_PARENT / PROJECT_NAME

    # ── 1. Create or open project ──
    if project_path.exists():
        print(f"Abriendo proyecto: {project_path}")
        project = Project.open(project_path)
    else:
        PROJECT_PARENT.mkdir(parents=True, exist_ok=True)
        print(f"Creando proyecto: {project_path}")
        project = Project.create(PROJECT_PARENT, PROJECT_NAME)

    db = project.db

    # ── 2. Index images (idempotent — INSERT OR IGNORE) ──
    total_imgs = sum(1 for _ in IMAGES_DIR.glob("*.jpg"))
    n_existing = db.count_images()
    if n_existing >= total_imgs:
        print(f"Ya indexadas: {n_existing} imagenes.")
    else:
        pending = total_imgs - n_existing
        print(f"\nIndexando {pending} imagenes nuevas (total {total_imgs}) + thumbnails...")
        t0 = time.time()
        ids = index_images(
            IMAGES_DIR, db, project_path,
            progress_callback=lambda d, t: progress_bar(d, t, "Thumbnails"),
        )
        print(f"\nIndexadas: {len(ids)} imagenes en {time.time()-t0:.1f}s")

    # ── 3. Load COCO annotations ──
    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM annotations")
        n_existing_ann = cur.fetchone()[0]

    # Re-import if annotations look incomplete (COCO has 2149 annotations)
    import json as _json
    coco_ann_count = len(_json.loads(COCO_JSON.read_text())["annotations"])
    if n_existing_ann >= coco_ann_count:
        print(f"Ya cargadas: {n_existing_ann} anotaciones.")
    else:
        print(f"\nImportando anotaciones COCO: {COCO_JSON.name} ({coco_ann_count} anotaciones)...")
        t0 = time.time()
        result = load_coco(
            COCO_JSON, db, overwrite=True,
            progress_callback=lambda d, t: progress_bar(d, t, "Anotaciones"),
        )
        print(f"\n{result.summary()}")
        print(f"Tiempo: {time.time()-t0:.1f}s")

    project.close()

    # ── 4. Launch GUI ──
    print(f"\nLanzando AnnotatAI con proyecto: {project_path}\n")

    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QPalette, QColor
    from app.main import apply_dark_theme
    from app.ui.main_window import MainWindow
    from app.core.project import Project as Proj

    app = QApplication(sys.argv)
    app.setApplicationName("AnnotatAI")
    apply_dark_theme(app)

    window = MainWindow()

    # Auto-open the project
    proj = Proj.open(project_path)
    window._project = proj
    window._on_project_loaded()

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
