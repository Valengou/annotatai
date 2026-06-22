"""Auto-etiquetado de oil spill con SAM 3 por texto (con máscaras de polígono).

Corre SAM 3 (SAM3SemanticPredictor) con el prompt de texto "oil spill" sobre las
candidatas rankeadas, guarda bbox + polígono de segmentación con source="yolo", y
renderiza previews con el polígono dibujado para revisar.

Uso:
    python batch_sam3_oilspill.py --start 0 --count 6          # probe
    python batch_sam3_oilspill.py --start 0 --count 700 --import
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from app.core.database import Database
from app.core.image_indexer import index_image_paths
from app.core.auto_label import AutoLabeler
from app.utils.config import DEFAULT_SAM3_MODEL

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_DIR = Path(r"projects/LLL-1794h-rgb")
PROJECT_DB = PROJECT_DIR / "project.db"
RANKING_CSV = Path("analysis_out/candidatas_ranking.csv")
PREVIEW_DIR = Path("analysis_out/review_sam3_oilspill")
PHRASE = "oil spill"
CLASS_NAME = "oil spill"
SOURCE = "yolo"


def ranked_candidates() -> list[str]:
    rows = list(csv.DictReader(open(RANKING_CSV, encoding="utf-8")))
    rows.sort(key=lambda r: int(r["rank"]))
    return [r["candidata"] for r in rows]


def draw_preview(path, dets):
    im = Image.open(path).convert("RGB")
    W, H = im.size
    scale = 1600 / max(W, H)
    im = im.resize((int(W * scale), int(H * scale)))
    d = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    for x, y, bw, bh, conf, polygon in dets:
        if polygon:
            pts = [(px * w, py * h) for px, py in polygon]
            d.polygon(pts, outline=(0, 229, 255, 255), fill=(0, 229, 255, 70))
        d.rectangle([x*w, y*h, (x+bw)*w, (y+bh)*h], outline=(255, 80, 80, 255), width=3)
        d.text((x*w + 3, max(0, y*h - 14)), f"oil spill {conf:.2f}", fill=(255, 80, 80, 255))
    im.save(PREVIEW_DIR / f"{Path(path).stem}.jpg", quality=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=DEFAULT_SAM3_MODEL)
    ap.add_argument("--import", dest="do_import", action="store_true",
                    help="además de previsualizar, inserta en la DB")
    args = ap.parse_args()

    db = Database(PROJECT_DB)
    db.connect()

    cands = ranked_candidates()
    batch = [Path(p) for p in cands[args.start:args.start + args.count] if Path(p).is_file()]
    print(f"Tanda SAM 3: candidatas {args.start}–{args.start + len(batch)} ({len(batch)} imgs)")

    ids = sorted(set(index_image_paths(batch, db, PROJECT_DIR)))
    all_rows = {r[0]: r for r in db.get_all_images()}
    target_rows = [all_rows[i] for i in ids if i in all_rows]

    cid = db.get_or_create_class(CLASS_NAME)
    print(f"clase '{CLASS_NAME}' id={cid} | modelo={args.model}")

    labeler = AutoLabeler(model_name=args.model, device=args.device, conf=args.conf,
                          imgsz=args.imgsz)
    print("Cargando SAM 3...")
    labeler.load()

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    n_imgs = n_boxes = 0
    import torch
    for done, row in enumerate(target_rows, 1):
        img_id, path = row[0], row[1]
        dets = labeler.detect_text_with_polygons(Path(path), PHRASE)
        if dets:
            for x, y, w, h, conf, polygon in dets:
                if args.do_import:
                    db.insert_annotation(img_id, cid, x, y, w, h, source=SOURCE,
                                         confidence=conf, polygon=polygon)
            draw_preview(path, dets)
            n_imgs += 1
            n_boxes += len(dets)
        print(f"\r  {done}/{len(target_rows)}  (cajas: {n_boxes})", end="", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print()

    verb = "importadas" if args.do_import else "detectadas (sin importar)"
    print(f"{n_boxes} cajas {verb} en {n_imgs} imágenes.")
    print(f"Previews: {PREVIEW_DIR}")
    db.close()


if __name__ == "__main__":
    main()
