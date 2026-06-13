"""
Generate CLIP embeddings for bounding-box crops, run UMAP, and compute
outlier scores so the user can spot weird/mislabeled boxes quickly.
"""

import numpy as np
from pathlib import Path
from typing import Callable

MIN_CROP_PX = 24   # skip boxes smaller than this in px


def crop_bbox(image_path: Path, x: float, y: float,
              w: float, h: float, img_w: int = 0, img_h: int = 0):
    """Return a PIL crop for a normalized bbox, or None if too small / error."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        iw, ih = img.size
        x1 = int(max(0, x * iw))
        y1 = int(max(0, y * ih))
        x2 = int(min(iw, (x + w) * iw))
        y2 = int(min(ih, (y + h) * ih))
        if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
            return None
        return img.crop((x1, y1, x2, y2))
    except Exception:
        return None


BATCH_SIZE = 32


def generate_bbox_embeddings(db, generator,
                              progress_callback: Callable | None = None):
    """Embed all bbox crops not yet in bbox_embeddings table, using GPU batching."""
    import torch

    ann_rows = db.get_annotations_with_image_paths()
    total = len(ann_rows)
    done = 0

    # Separate already-done from pending
    pending = []
    for row in ann_rows:
        ann_id = row[0]
        if db.has_bbox_embedding(ann_id):
            done += 1
            if progress_callback:
                progress_callback(done, total)
        else:
            pending.append(row)

    # Process pending in batches
    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch_rows = pending[batch_start:batch_start + BATCH_SIZE]

        ids_in_batch: list[int] = []
        tensors: list = []

        for row in batch_rows:
            ann_id, img_id, x, y, w, h, img_path, img_w, img_h = row
            crop = crop_bbox(Path(img_path), x, y, w, h)
            if crop is None:
                continue
            try:
                t = generator._preprocess(crop)
                tensors.append(t)
                ids_in_batch.append(ann_id)
            except Exception:
                pass

        if tensors:
            try:
                batch_tensor = torch.stack(tensors).to(generator._device)
                with torch.no_grad():
                    features = generator._model.encode_image(batch_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)
                vecs = features.cpu().numpy()

                for ann_id, vec in zip(ids_in_batch, vecs):
                    db.save_bbox_embedding(
                        ann_id,
                        generator.vector_to_bytes(vec),
                        generator.model_name,
                    )
            except Exception:
                pass

        done += len(batch_rows)
        if progress_callback:
            progress_callback(done, total)


def run_bbox_umap(db, progress_callback: Callable | None = None) -> dict:
    """UMAP + HDBSCAN outlier scoring on bbox embeddings.
    Returns {'total': N, 'n_outliers': K}.
    """
    import umap as umap_mod
    import hdbscan as hdbscan_mod
    from .embeddings import EmbeddingGenerator

    if progress_callback:
        progress_callback("Cargando embeddings de crops...", 10)

    rows = db.get_all_bbox_embeddings()
    if len(rows) < 4:
        return {}

    ann_ids = [r[0] for r in rows]
    vectors = np.array([EmbeddingGenerator.bytes_to_vector(r[1]) for r in rows])

    if progress_callback:
        progress_callback("UMAP sobre crops...", 30)

    reducer = umap_mod.UMAP(
        n_components=2,
        n_neighbors=min(15, len(vectors) - 1),
        min_dist=0.05,
        metric="cosine",
    )
    proj = reducer.fit_transform(vectors)

    if progress_callback:
        progress_callback("Calculando outlier scores...", 70)

    min_size = max(2, min(10, len(vectors) // 20))
    clusterer = hdbscan_mod.HDBSCAN(
        min_cluster_size=min_size,
        min_samples=2,
        prediction_data=True,
    )
    clusterer.fit(proj)
    outlier_scores = np.nan_to_num(clusterer.outlier_scores_, nan=0.0)

    if progress_callback:
        progress_callback("Guardando proyecciones...", 90)

    db.save_bbox_projections([
        (int(ann_id), float(px), float(py), float(score))
        for ann_id, (px, py), score in zip(ann_ids, proj, outlier_scores)
    ])

    if progress_callback:
        progress_callback("Listo", 100)

    n_outliers = int(np.sum(clusterer.labels_ == -1))
    return {"total": len(ann_ids), "n_outliers": n_outliers}
