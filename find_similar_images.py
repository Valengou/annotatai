"""Busca imágenes similares a las etiquetadas de un proyecto para expandir el dataset.

Embebe las imágenes de referencia (las que tienen anotaciones en el proyecto) y un
conjunto de imágenes candidatas con el mismo backend, calcula la similitud coseno de
cada candidata contra la referencia más parecida, rankea y opcionalmente escribe un
manifiesto dataset.json (abrible en AnnotatAI) con las seleccionadas.

Los embeddings se cachean en .npy, así re-rankear con otro corte es instantáneo.

Uso:
    # 1ª pasada: embebe, rankea, escribe CSV e imprime la distribución
    python find_similar_images.py

    # 2ª pasada: genera el manifiesto con un corte (instantáneo, usa el cache)
    python find_similar_images.py --threshold 0.55
    python find_similar_images.py --top 200
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.core.embedding_backends import create_embedding_backend

PROJECT_DB = Path(r"projects/LLL-1794h-rgb/project.db")
VUELOS_ROOT = Path(
    r"C:\Users\valen\OneDrive - USS Servicios Unidos de Seguridad S.A"
    r"\YPF-Confluencia\datasets\Vuelos_prueba"
)
DAYS = ["20260526", "20260527", "20260528"]
RGB_SUBDIR = "rgb"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

BACKEND = "dinov3"
OUT_DIR = Path("analysis_out")
CSV_PATH = OUT_DIR / "candidatas_ranking.csv"
MANIFEST_PATH = OUT_DIR / "dataset_candidatas.json"
REF_CACHE = OUT_DIR / "ref_embeddings.npy"
CAND_CACHE = OUT_DIR / "cand_embeddings.npy"
CAND_PATHS_CACHE = OUT_DIR / "cand_paths.json"


def labeled_reference_paths(db_path: Path) -> list[Path]:
    db = sqlite3.connect(str(db_path))
    rows = db.execute(
        "SELECT path FROM images WHERE id IN (SELECT DISTINCT image_id FROM annotations)"
    ).fetchall()
    db.close()
    return [Path(r[0]) for r in rows]


def project_image_paths(db_path: Path) -> set[str]:
    db = sqlite3.connect(str(db_path))
    rows = db.execute("SELECT path FROM images").fetchall()
    db.close()
    return {str(Path(r[0]).resolve()).lower() for r in rows}


def candidate_paths(already_in_project: set[str]) -> list[Path]:
    out: list[Path] = []
    for day in DAYS:
        day_dir = VUELOS_ROOT / day
        if not day_dir.is_dir():
            continue
        for site in sorted(p for p in day_dir.iterdir() if p.is_dir()):
            rgb_dir = site / RGB_SUBDIR
            if not rgb_dir.is_dir():
                continue
            for img in sorted(rgb_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                if str(img.resolve()).lower() in already_in_project:
                    continue
                out.append(img)
    return out


def embed(paths: list[Path], cache: Path) -> np.ndarray:
    if cache.is_file():
        cached = np.load(cache)
        if cached.shape[0] == len(paths):
            print(f"  (cache) {cache.name}: {cached.shape}")
            return cached
    backend = create_embedding_backend(BACKEND)
    backend.load()
    total = len(paths)

    def progress(done: int, tot: int):
        print(f"\r  embeddings {min(done, total)}/{total}", end="", flush=True)

    vectors = backend.embed_paths(paths, progress_callback=progress)
    print()
    vectors = np.asarray(vectors, dtype=np.float32)
    np.save(cache, vectors)
    return vectors


def site_of(path: Path) -> str:
    # .../Vuelos_prueba/<day>/<site>/rgb/<file>
    parts = path.parts
    try:
        i = parts.index(RGB_SUBDIR)
        return f"{parts[i - 2]}/{parts[i - 1]}"
    except (ValueError, IndexError):
        return str(path.parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None,
                    help="similitud mínima para incluir en el manifiesto")
    ap.add_argument("--top", type=int, default=None,
                    help="cantidad de candidatas top para el manifiesto")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    refs = labeled_reference_paths(PROJECT_DB)
    refs = [p for p in refs if p.is_file()]
    print(f"Referencias etiquetadas: {len(refs)}")

    in_project = project_image_paths(PROJECT_DB)
    cands = candidate_paths(in_project)
    print(f"Candidatas RGB (excluyendo las del proyecto): {len(cands)}")

    print("Embebiendo referencias...")
    ref_vecs = embed(refs, REF_CACHE)
    print("Embebiendo candidatas...")
    cand_vecs = embed(cands, CAND_CACHE)
    CAND_PATHS_CACHE.write_text(
        json.dumps([str(p) for p in cands], ensure_ascii=False), encoding="utf-8"
    )

    # vectores ya L2-normalizados -> coseno = producto punto
    sims = cand_vecs @ ref_vecs.T            # (n_cand, n_ref)
    best_idx = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(cands)), best_idx]
    order = np.argsort(-best_sim)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "similitud_max", "candidata", "sitio", "referencia_mas_parecida"])
        for rank, i in enumerate(order, 1):
            w.writerow([rank, f"{best_sim[i]:.4f}", str(cands[i]),
                        site_of(cands[i]), refs[best_idx[i]].name])
    print(f"\nCSV: {CSV_PATH}")

    pct = np.percentile(best_sim, [50, 75, 90, 95, 99])
    print("\nDistribución de similitud máxima:")
    print(f"  min={best_sim.min():.3f}  p50={pct[0]:.3f}  p75={pct[1]:.3f} "
          f" p90={pct[2]:.3f}  p95={pct[3]:.3f}  max={best_sim.max():.3f}")
    print("\nCandidatas por encima de cada umbral:")
    for t in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        print(f"  >= {t:.2f}: {(best_sim >= t).sum()}")

    print("\nTop sitios entre las 150 más parecidas:")
    from collections import Counter
    top_sites = Counter(site_of(cands[i]) for i in order[:150])
    for s, n in top_sites.most_common():
        print(f"  {n:3d}  {s}")

    if args.threshold is None and args.top is None:
        print("\nSin --threshold/--top: no se generó manifiesto. "
              "Revisá la distribución y volvé a correr con un corte.")
        return

    if args.top is not None:
        selected = [cands[i] for i in order[:args.top]]
    else:
        selected = [cands[i] for i in order if best_sim[i] >= args.threshold]

    manifest = {
        "image_paths": [str(p) for p in selected],
        "_note": f"Candidatas similares a referencias de {PROJECT_DB.parent.name} "
                 f"(backend={BACKEND}, threshold={args.threshold}, top={args.top})",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"\nManifiesto: {MANIFEST_PATH}  ({len(selected)} imágenes)")
    print("Abrilo en AnnotatAI: Archivo → Abrir dataset.json...")


if __name__ == "__main__":
    main()
