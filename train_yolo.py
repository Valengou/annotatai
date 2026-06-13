"""
train_yolo.py — Train YOLOv8n on reviewed images and pre-annotate pending ones.

Usage:
    python train_yolo.py [--project PROJECT_DIR] [--epochs 50] [--conf 0.25] [--imgsz 640] [--predict-only]

Steps:
    1. Export reviewed images to YOLO format (train/val split)
    2. Train YOLOv8n
    3. Run inference on pending images
    4. Import predictions into the DB (source='yolo', skips images that already have yolo preds)
"""

import argparse
import random
import shutil
import sqlite3
import sys
from pathlib import Path

VENV_PYTHON = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
PROJECT_DIR = Path(__file__).parent / "projects" / "insulator_detector_combined"
DB_PATH = PROJECT_DIR / "project.db"


# ── helpers ──────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def export_yolo_dataset(output_dir: Path, val_fraction: float = 0.15):
    """Export reviewed images as YOLO dataset. Returns (train_count, val_count)."""
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name FROM classes ORDER BY name")
    classes = cur.fetchall()
    class_idx = {row[0]: i for i, row in enumerate(classes)}
    class_names = [row[1] for row in classes]
    print(f"  Classes: {class_names}")

    cur.execute(
        "SELECT id, path, filename, width, height FROM images WHERE status='reviewed'"
    )
    reviewed = cur.fetchall()
    print(f"  Reviewed images: {len(reviewed)}")

    # Shuffle and split
    random.shuffle(reviewed)
    n_val = max(1, int(len(reviewed) * val_fraction))
    val_set = set(r[0] for r in reviewed[:n_val])

    splits = {"train": [], "val": []}
    for row in reviewed:
        split = "val" if row[0] in val_set else "train"
        splits[split].append(row)

    for split_name, rows in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_id, path, filename, width, height in rows:
            src = Path(path)
            if not src.exists():
                continue

            shutil.copy2(src, img_dir / filename)

            cur.execute(
                "SELECT a.x, a.y, a.width, a.height, a.class_id "
                "FROM annotations a WHERE a.image_id=?",
                (img_id,),
            )
            anns = cur.fetchall()
            lines = []
            for x, y, w, h, class_id in anns:
                idx = class_idx.get(class_id)
                if idx is None:
                    continue
                cx = x + w / 2
                cy = y + h / 2
                lines.append(f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if lines:
                (lbl_dir / (Path(filename).stem + ".txt")).write_text(
                    "\n".join(lines), encoding="utf-8"
                )

    # data.yaml
    yaml_content = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    (output_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

    conn.close()
    return len(splits["train"]), len(splits["val"])


def train(data_yaml: Path, epochs: int, imgsz: int, output_dir: Path) -> Path:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(output_dir / "runs"),
        name="train",
        exist_ok=True,
        patience=15,
        batch=-1,        # auto batch size
        cache=False,
        workers=4,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n  Best model: {best}")
    return best


def predict_and_import(model_path: Path, conf_threshold: float, imgsz: int):
    from ultralytics import YOLO
    model = YOLO(str(model_path))

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, name FROM classes ORDER BY name")
    classes = cur.fetchall()
    yolo_idx_to_class_id = {i: row[0] for i, row in enumerate(classes)}

    # Pending images (no annotation or only status='pending')
    cur.execute(
        "SELECT id, path, filename, width, height FROM images WHERE status='pending'"
    )
    pending = cur.fetchall()
    print(f"\n  Pending images: {len(pending)}")

    # Skip images that already have yolo predictions
    cur.execute(
        "SELECT DISTINCT image_id FROM annotations WHERE source='yolo'"
    )
    already_predicted = {r[0] for r in cur.fetchall()}
    to_predict = [r for r in pending if r[0] not in already_predicted]
    print(f"  To predict (skipping {len(already_predicted)} already done): {len(to_predict)}")

    if not to_predict:
        print("  Nothing to predict.")
        conn.close()
        return 0, 0

    n_images = 0
    n_boxes = 0

    for img_id, path, filename, width, height in to_predict:
        src = Path(path)
        if not src.exists():
            print(f"  [SKIP] Not found: {path}")
            continue

        results = model.predict(
            source=str(src),
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
        )

        boxes_inserted = 0
        for result in results:
            if result.boxes is None:
                continue
            img_w = result.orig_shape[1]
            img_h = result.orig_shape[0]

            for box in result.boxes:
                cls_idx = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_id = yolo_idx_to_class_id.get(cls_idx)
                if class_id is None:
                    continue

                # xywhn: normalized cx, cy, w, h
                xywhn = box.xywhn[0].tolist()
                cx, cy, bw, bh = xywhn
                x = cx - bw / 2
                y = cy - bh / 2

                cur.execute(
                    "INSERT INTO annotations (image_id, class_id, x, y, width, height, source, confidence) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'yolo', ?)",
                    (img_id, class_id, x, y, bw, bh, conf),
                )
                boxes_inserted += 1

        if boxes_inserted > 0:
            n_images += 1
            n_boxes += boxes_inserted

        conn.commit()

    conn.close()
    print(f"\n  Imported {n_boxes} boxes into {n_images} images.")
    return n_images, n_boxes


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train YOLO + pre-annotate pending images")
    parser.add_argument("--project", default=str(PROJECT_DIR))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for importing predictions")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip training, use existing best.pt")
    parser.add_argument("--model", default=None,
                        help="Path to existing .pt file (used with --predict-only)")
    args = parser.parse_args()

    project_dir = Path(args.project)
    export_dir = project_dir / "yolo_dataset"
    runs_dir = project_dir / "yolo_dataset"

    if not args.predict_only:
        print("=" * 60)
        print("STEP 1 — Exporting reviewed images to YOLO format")
        print("=" * 60)
        export_dir.mkdir(parents=True, exist_ok=True)
        n_train, n_val = export_yolo_dataset(export_dir, val_fraction=args.val_fraction)
        print(f"  Exported: {n_train} train / {n_val} val")

        print("\n" + "=" * 60)
        print("STEP 2 — Training YOLOv8n")
        print("=" * 60)
        best_pt = train(export_dir / "data.yaml", args.epochs, args.imgsz, runs_dir)
    else:
        if args.model:
            best_pt = Path(args.model)
        else:
            best_pt = runs_dir / "runs" / "train" / "weights" / "best.pt"
        if not best_pt.exists():
            print(f"ERROR: model not found at {best_pt}")
            sys.exit(1)
        print(f"  Using existing model: {best_pt}")

    print("\n" + "=" * 60)
    print("STEP 3 — Predicting pending images")
    print("=" * 60)
    predict_and_import(best_pt, args.conf, args.imgsz)

    print("\nDone. Open AnnotatAI to review the pre-annotations.")


if __name__ == "__main__":
    main()
