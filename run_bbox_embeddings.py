"""Run CLIP bbox crop embeddings with GPU batching, then UMAP outlier analysis."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core.project import Project
from app.core.embeddings import EmbeddingGenerator
from app.core.bbox_embeddings import generate_bbox_embeddings, run_bbox_umap, BATCH_SIZE

PROJECT_PATH = Path(__file__).parent / "projects" / "insulator_detector_combined"

proj = Project.open(PROJECT_PATH)
db   = proj.db

n_ann  = db.execute_scalar("SELECT COUNT(*) FROM annotations")
n_done = db.count_bbox_embeddings()
pending = n_ann - n_done
print(f"Anotaciones totales : {n_ann}")
print(f"Ya embedidas        : {n_done}")
print(f"Pendientes          : {pending}")
print(f"Batch size          : {BATCH_SIZE}")

if pending > 0:
    print("\nCargando modelo CLIP...")
    gen = EmbeddingGenerator()
    gen.load()
    print(f"Dispositivo: {gen._device}")

    t0 = time.time()
    def prog(done, total):
        pct = done / max(total, 1)
        bar = "#" * int(pct * 30) + "-" * (30 - int(pct * 30))
        eta = (time.time() - t0) / max(pct, 0.001) * (1 - pct)
        print(f"\r  [{bar}] {done}/{total}  ETA {eta:.0f}s", end="", flush=True)

    generate_bbox_embeddings(db, gen, progress_callback=prog)
    elapsed = time.time() - t0
    print(f"\nEmbeddings listos en {elapsed:.1f}s  "
          f"({elapsed/max(pending,1)*1000:.0f}ms/crop)")

print("\nEjecutando UMAP + outlier scoring...")
t0 = time.time()
def prog2(msg, pct):
    print(f"  [{pct:3d}%] {msg}", flush=True)

result = run_bbox_umap(db, progress_callback=prog2)
print(f"Listo en {time.time()-t0:.1f}s")
print(f"  Total bboxes analizadas : {result.get('total', 0)}")
print(f"  Outliers (HDBSCAN -1)   : {result.get('n_outliers', 0)}")

proj.close()
print("\nAbri la app y ve al tab 'Bboxes Raras'.")
