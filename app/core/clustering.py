import numpy as np
from typing import Callable
from ..utils.config import (
    UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
    CLUSTER_COLORS,
)
from .embeddings import EmbeddingGenerator


def run_umap(embeddings: np.ndarray) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(UMAP_N_NEIGHBORS, len(embeddings) - 1),
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def run_hdbscan(embeddings_2d: np.ndarray) -> np.ndarray:
    import hdbscan
    min_size = min(HDBSCAN_MIN_CLUSTER_SIZE, max(2, len(embeddings_2d) // 5))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size,
        min_samples=min(HDBSCAN_MIN_SAMPLES, min_size),
    )
    return clusterer.fit_predict(embeddings_2d)


def cluster_images(db, progress_callback: Callable | None = None) -> int:
    """Run UMAP + HDBSCAN on stored embeddings, persist clusters and projections.
    Returns number of clusters created."""
    rows = db.get_all_embeddings()
    if len(rows) < 3:
        return 0

    if progress_callback:
        progress_callback("Cargando embeddings...", 10)

    image_ids = [r[0] for r in rows]
    vectors = np.array([EmbeddingGenerator.bytes_to_vector(r[1]) for r in rows])

    if progress_callback:
        progress_callback("Ejecutando UMAP...", 30)

    projections_2d = run_umap(vectors)

    if progress_callback:
        progress_callback("Ejecutando HDBSCAN...", 70)

    labels = run_hdbscan(projections_2d)

    if progress_callback:
        progress_callback("Guardando clusters...", 90)

    db.clear_clusters()

    unique_labels = sorted(set(labels))
    label_to_cluster_id: dict[int, int] = {}

    noise_cluster_id = None
    for lbl in unique_labels:
        if lbl == -1:
            color = "#555555"
            name = "Sin cluster"
            cid = db.insert_cluster(name, color)
            noise_cluster_id = cid
        else:
            color = CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)]
            name = f"Grupo {lbl + 1}"
            cid = db.insert_cluster(name, color)
        label_to_cluster_id[lbl] = cid

    for img_id, lbl in zip(image_ids, labels):
        db.update_image_cluster(img_id, label_to_cluster_id[lbl])

    proj_data = [
        (img_id, float(x), float(y))
        for img_id, (x, y) in zip(image_ids, projections_2d)
    ]
    db.save_projections(proj_data)

    if progress_callback:
        progress_callback("Listo", 100)

    n_real_clusters = sum(1 for lbl in unique_labels if lbl != -1)
    return n_real_clusters
