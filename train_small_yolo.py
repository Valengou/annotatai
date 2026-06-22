"""Entrena un YOLOv8n multi-clase con las cajas HUMANAS del proyecto y pre-etiqueta
clases elegidas (por defecto wellhead) sobre las candidatas, importándolas con
source='yolo'.

A diferencia de train_yolo.py:
  - exporta imágenes con cajas humanas (status='reviewed'), multi-clase;
  - al predecir NO saltea imágenes que ya tienen cajas yolo (pump jack via YOLOE),
    e importa SOLO las clases indicadas con --classes;
  - batch fijo y modesto para no derramar VRAM en GPUs de 8 GB.

Uso:
    python train_small_yolo.py --train                 # exporta + entrena
    python train_small_yolo.py --predict --classes wellhead --conf 0.30
"""

from __future__ import annotations

import argparse
import random
import shutil
import sqlite3
import sys
from pathlib import Path

from PIL import Image, ImageDraw

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_DIR = Path("projects/LLL-1794h-rgb")
DB_PATH = PROJECT_DIR / "project.db"
DATASET_DIR = PROJECT_DIR / "yolo_wellhead"
BEST_PT = DATASET_DIR / "runs" / "train" / "weights" / "best.pt"
PREVIEW_DIR = Path("analysis_out/review_trained")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def classes_ordered(conn):
    rows = conn.execute("SELECT id, name FROM classes ORDER BY name").fetchall()
    return rows, {row[0]: i for i, row in enumerate(rows)}, [r[1] for r in rows]


def export_dataset(val_fraction: float = 0.2, seed_shuffle: bool = True):
    conn = get_db()
    rows, class_idx, names = classes_ordered(conn)
    print(f"  Clases ({len(names)}): {names}")

    labeled = conn.execute(
        "SELECT id, path, filename FROM images WHERE status='reviewed'"
    ).fetchall()
    labeled = [r for r in labeled if Path(r[1]).exists()]
    print(f"  Imágenes etiquetadas: {len(labeled)}")

    if seed_shuffle:
        random.Random(0).shuffle(labeled)
    n_val = max(1, int(len(labeled) * val_fraction))
    val_ids = {r[0] for r in labeled[:n_val]}

    for split, rows_s in (("val", [r for r in labeled if r[0] in val_ids]),
                          ("train", [r for r in labeled if r[0] not in val_ids])):
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_id, path, filename in rows_s:
            shutil.copy2(path, img_dir / filename)
            anns = conn.execute(
                "SELECT x, y, width, height, class_id FROM annotations "
                "WHERE image_id=? AND source='human'", (img_id,)
            ).fetchall()
            lines = []
            for x, y, w, h, cid in anns:
                idx = class_idx.get(cid)
                if idx is None:
                    continue
                lines.append(f"{idx} {x + w/2:.6f} {y + h/2:.6f} {w:.6f} {h:.6f}")
            if lines:
                (lbl_dir / (Path(filename).stem + ".txt")).write_text(
                    "\n".join(lines), encoding="utf-8")

    (DATASET_DIR / "data.yaml").write_text(
        f"path: {DATASET_DIR.resolve()}\n"
        f"train: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names}\n", encoding="utf-8")
    conn.close()


def train(epochs: int, imgsz: int, batch: int):
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=epochs, imgsz=imgsz, batch=batch,
        project=str(DATASET_DIR / "runs"), name="train", exist_ok=True,
        patience=30, cache=False, workers=4, verbose=True,
    )
    print(f"\n  Mejor modelo: {Path(results.save_dir) / 'weights' / 'best.pt'}")


def predict_import(class_names: list[str], conf: float, imgsz: int,
                   do_import: bool):
    import torch
    from ultralytics import YOLO

    best = BEST_PT
    if not best.exists():
        matches = sorted(Path(".").glob("**/yolo_wellhead/**/weights/best.pt"))
        if not matches:
            sys.exit("No se encontró best.pt. Corré --train primero.")
        best = matches[-1]
    print(f"  Modelo: {best}")

    conn = get_db()
    rows, _, names = classes_ordered(conn)
    idx_to_cid = {i: rows[i][0] for i in range(len(rows))}
    cid_to_name = {row[0]: row[1] for row in rows}
    wanted_cids = {row[0] for row in rows if row[1] in class_names}
    if not wanted_cids:
        sys.exit(f"Clases {class_names} no encontradas. Disponibles: {names}")

    # candidatas = pending con archivo existente
    pend = conn.execute(
        "SELECT id, path FROM images WHERE status='pending'").fetchall()
    targets = [(i, p) for i, p in pend if Path(p).exists()]
    print(f"  Candidatas a predecir: {len(targets)}")

    model = YOLO(str(best))
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    per_class = Counter()
    n_imgs = n_boxes = 0
    preview_budget = 16
    chunk = 8
    for start in range(0, len(targets), chunk):
        batch_rows = targets[start:start + chunk]
        results = model.predict([p for _, p in batch_rows], conf=conf, imgsz=imgsz,
                                verbose=False, stream=True, save=False)
        for (img_id, path), r in zip(batch_rows, results):
            dets = []
            if r.boxes is not None:
                for b in r.boxes:
                    cid = idx_to_cid.get(int(b.cls[0].item()))
                    if cid not in wanted_cids:
                        continue
                    cx, cy, bw, bh = b.xywhn[0].tolist()
                    dets.append((cid, cx - bw/2, cy - bh/2, bw, bh,
                                 float(b.conf[0].item())))
            if not dets:
                continue
            if do_import:
                for cid, x, y, bw, bh, c in dets:
                    conn.execute(
                        "INSERT INTO annotations (image_id, class_id, x, y, width, "
                        "height, source, confidence) VALUES (?,?,?,?,?,?, 'yolo', ?)",
                        (img_id, cid, x, y, bw, bh, c))
                conn.commit()
            n_imgs += 1
            n_boxes += len(dets)
            for cid, *_ in dets:
                per_class[cid_to_name[cid]] += 1
            if preview_budget > 0:
                _draw_preview(path, dets, cid_to_name)
                preview_budget -= 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    conn.close()
    verb = "importadas" if do_import else "detectadas (sin importar)"
    print(f"\n  {n_boxes} cajas {verb} en {n_imgs} imágenes.")
    for name, n in per_class.most_common():
        print(f"    {name}: {n}")
    print(f"  Previews: {PREVIEW_DIR}")


def _draw_preview(path, dets, cid_to_name):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    scale = 1600 / max(W, H)
    im = im.resize((int(W * scale), int(H * scale)))
    d = ImageDraw.Draw(im)
    w, h = im.size
    for cid, x, y, bw, bh, c in dets:
        d.rectangle([x*w, y*h, (x+bw)*w, (y+bh)*h], outline="#00E5FF", width=4)
        d.text((x*w + 3, max(0, y*h - 14)), f"{cid_to_name[cid]} {c:.2f}", fill="#00E5FF")
    im.save(PREVIEW_DIR / f"{Path(path).stem}.jpg", quality=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--import", dest="do_import", action="store_true",
                    help="con --predict, además inserta en la DB")
    ap.add_argument("--classes", default="wellhead")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.30)
    args = ap.parse_args()

    if args.train:
        print("=== Exportando dataset YOLO ===")
        export_dataset()
        print("=== Entrenando YOLOv8n ===")
        train(args.epochs, args.imgsz, args.batch)
    if args.predict:
        print("=== Prediciendo ===")
        predict_import([c.strip() for c in args.classes.split(",") if c.strip()],
                       args.conf, args.imgsz, args.do_import)
    if not args.train and not args.predict:
        ap.error("Indicá --train y/o --predict")


if __name__ == "__main__":
    main()
