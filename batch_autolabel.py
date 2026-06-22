"""Auto-etiquetado por tandas con YOLOE (prompt visual) sobre candidatas rankeadas.

Toma las candidatas ordenadas por similitud (analysis_out/candidatas_ranking.csv),
indexa una tanda al proyecto, corre YOLOE usando las cajas HUMANAS de cada clase como
prompt visual (cross-image), guarda las detecciones con source="yolo" y renderiza
previews con las cajas dibujadas para revisar.

Uso:
    python batch_autolabel.py --start 0   --count 100              # tanda 1 (top 100)
    python batch_autolabel.py --start 100 --count 100              # tanda 2
    python batch_autolabel.py --classes "pump jack,wellhead" --conf 0.25
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.core.database import Database
from app.core.image_indexer import index_image_paths
from app.core.auto_label import run_auto_label_visual, DEFAULT_YOLOE_MODEL

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_DIR = Path(r"projects/LLL-1794h-rgb")
PROJECT_DB = PROJECT_DIR / "project.db"
RANKING_CSV = Path("analysis_out/candidatas_ranking.csv")
MODEL_PATH = Path("yoloe-v8l-seg.pt").resolve()
SOURCE = "yolo"


def ranked_candidates() -> list[str]:
    rows = list(csv.DictReader(open(RANKING_CSV, encoding="utf-8")))
    rows.sort(key=lambda r: int(r["rank"]))
    return [r["candidata"] for r in rows]


def class_id_by_name(db: Database) -> dict[str, int]:
    return {row[1]: row[0] for row in db.get_all_classes()}


def render_previews(db: Database, image_ids: list[int], out_dir: Path,
                    max_imgs: int = 16):
    out_dir.mkdir(parents=True, exist_ok=True)
    placeholders = ",".join("?" * len(image_ids))
    rows = db._conn.execute(
        f"""SELECT a.image_id, i.path, i.width, i.height, c.name, c.color,
                   a.x, a.y, a.width, a.height, a.confidence
            FROM annotations a
            JOIN images i ON i.id = a.image_id
            JOIN classes c ON c.id = a.class_id
            WHERE a.source = ? AND a.image_id IN ({placeholders})
            ORDER BY a.image_id""",
        (SOURCE, *image_ids),
    ).fetchall()

    by_img: dict[int, list] = {}
    for r in rows:
        by_img.setdefault(r[0], []).append(r)

    # imágenes con más cajas primero (más para revisar)
    ranked = sorted(by_img.items(), key=lambda kv: -len(kv[1]))[:max_imgs]
    for img_id, dets in ranked:
        src = dets[0][1]
        im = Image.open(src).convert("RGB")
        W, H = im.size
        scale = 1600 / max(W, H)
        im = im.resize((int(W * scale), int(H * scale)))
        draw = ImageDraw.Draw(im)
        w, h = im.size
        for d in dets:
            _, _, _, _, cname, color, x, y, bw, bh, conf = d
            x0, y0, x1, y1 = x * w, y * h, (x + bw) * w, (y + bh) * h
            draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
            draw.text((x0 + 3, max(0, y0 - 14)), f"{cname} {conf:.2f}", fill=color)
        im.save(out_dir / f"{Path(src).stem}.jpg", quality=80)
    print(f"  previews: {len(ranked)} imágenes en {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--classes", default="pump jack,wellhead")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="tamaño de inferencia (más chico = menos VRAM)")
    ap.add_argument("--chunk", type=int, default=8,
                    help="imágenes por chunk antes de liberar VRAM")
    args = ap.parse_args()

    target_classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    db = Database(PROJECT_DB)
    db.connect()

    cands = ranked_candidates()
    batch = cands[args.start:args.start + args.count]
    batch_paths = [Path(p) for p in batch if Path(p).is_file()]
    print(f"Tanda: candidatas {args.start}–{args.start + len(batch_paths)} "
          f"({len(batch_paths)} imágenes)")

    print("Indexando al proyecto (thumbnails)...")
    ids = index_image_paths(batch_paths, db, PROJECT_DIR)
    batch_ids = sorted(set(ids))
    print(f"  {len(batch_ids)} imágenes en el proyecto (nuevas o ya existentes)")

    all_rows = db.get_all_images()
    target_rows = [r for r in all_rows if r[0] in set(batch_ids)]

    cid_by_name = class_id_by_name(db)
    for cname in target_classes:
        if cname not in cid_by_name:
            print(f"  [skip] clase '{cname}' no existe en el proyecto")
            continue
        cid = cid_by_name[cname]
        print(f"\nYOLOE → '{cname}' (conf={args.conf}) ...")
        res = run_auto_label_visual(
            db, cid, cname, target_rows, conf=args.conf, source=SOURCE,
            model_name=str(MODEL_PATH), device=args.device,
            imgsz=args.imgsz, chunk_size=args.chunk,
            progress_callback=lambda d, t: print(f"\r  {d}/{t}", end="", flush=True),
        )
        print()
        print(f"  resultado: {res}")

    out_dir = Path(f"analysis_out/review_b{args.start}_{args.start + len(batch_paths)}")
    print("\nRenderizando previews para revisar...")
    render_previews(db, batch_ids, out_dir)
    db.close()


if __name__ == "__main__":
    main()
