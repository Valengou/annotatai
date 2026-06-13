"""Run UMAP + HDBSCAN clustering on existing embeddings and exit."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core.project import Project
from app.core.clustering import cluster_images

PROJECT_PATH = Path(__file__).parent / "projects" / "insulator_detector_combined"

proj = Project.open(PROJECT_PATH)
db = proj.db

print(f"Embeddings disponibles: {db.count_embeddings()}")
print("Ejecutando UMAP + HDBSCAN...")

t0 = time.time()

def progress(msg, pct):
    print(f"  [{pct:3d}%] {msg}", flush=True)

n = cluster_images(db, progress_callback=progress)
print(f"\nClusters encontrados: {n}")
print(f"Tiempo total: {time.time() - t0:.1f}s\n")

for r in db.get_all_clusters():
    print(f"  [{r[0]:2d}] {r[1]:<20s}  {r[4]:4d} imgs  {r[2]}")

proj.close()
print("\nListo. Reabri la app para ver los grupos.")
