"""Detección de imágenes casi-duplicadas (near-duplicates) usando los embeddings
ya calculados del proyecto.

Útil para video de drone: muchos frames son casi idénticos. Agrupando los que
superan un umbral de similitud coseno, se etiqueta solo un representante por grupo
y se descarta o se propaga al resto.

Acciones:
  - report:    solo cuenta grupos/duplicados.
  - collapse:  deja 1 representante por grupo y marca el resto como 'discarded'.
  - propagate: copia las cajas del miembro etiquetado a los demás del grupo y los
               marca 'reviewed' (para usar DESPUÉS de etiquetar los representantes).
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np

from .embeddings import EmbeddingGenerator


ProgressCb = Callable[[str, int], None]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def find_near_duplicate_groups(db, threshold: float = 0.95,
                               progress_callback: ProgressCb | None = None
                               ) -> list[list[int]]:
    """Agrupa imágenes cuyo coseno de embeddings >= threshold (componentes
    conexas). Devuelve solo grupos con más de 1 imagen, cada uno como lista de
    image_ids ordenada."""
    rows = db.get_all_embeddings()
    if len(rows) < 2:
        return []

    image_ids = [r[0] for r in rows]
    vectors = np.array(
        [EmbeddingGenerator.bytes_to_vector(r[1]) for r in rows],
        dtype=np.float32,
    )
    # Normalizar para que el producto punto sea el coseno
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    n = len(image_ids)
    uf = _UnionFind(n)
    # Procesar por filas: para cada i, similitud contra todos j>i. O(n^2) en
    # cómputo pero O(n) en memoria (evita matriz n×n completa).
    for i in range(n):
        sims = vectors[i + 1:] @ vectors[i]
        for offset in np.nonzero(sims >= threshold)[0]:
            uf.union(i, i + 1 + int(offset))
        if progress_callback and (i % 50 == 0 or i == n - 1):
            progress_callback(f"Comparando embeddings {i + 1}/{n}",
                              int((i + 1) / n * 80))

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(uf.find(idx), []).append(image_ids[idx])

    result = [sorted(g) for g in groups.values() if len(g) > 1]
    result.sort(key=len, reverse=True)
    return result


def _pick_representative(group: list[int], status_by_id: dict[int, str],
                        boxes_by_id: dict[int, int]) -> int:
    """Mejor representante del grupo: revisado > más cajas humanas > menor id."""
    def key(img_id: int):
        reviewed = 1 if status_by_id.get(img_id) == "reviewed" else 0
        return (reviewed, boxes_by_id.get(img_id, 0), -img_id)
    return max(group, key=key)


def run_near_duplicates(db, threshold: float, action: str,
                        progress_callback: ProgressCb | None = None) -> dict:
    """Orquesta la detección + acción. Devuelve un resumen."""
    groups = find_near_duplicate_groups(db, threshold, progress_callback)
    n_groups = len(groups)
    n_dups = sum(len(g) - 1 for g in groups)

    summary = {"groups": n_groups, "duplicates": n_dups,
               "threshold": threshold, "action": action,
               "affected": 0}
    if not groups or action == "report":
        return summary

    status_by_id = {r[0]: r[6] for r in db.get_all_images()}
    boxes_by_id = db.count_human_boxes_per_image()

    affected = 0
    total = len(groups)
    for gi, group in enumerate(groups, start=1):
        rep = _pick_representative(group, status_by_id, boxes_by_id)

        if action == "collapse":
            # Dejar solo el representante para etiquetar; descartar el resto.
            for img_id in group:
                if img_id != rep:
                    db.update_image_status(img_id, "discarded")
                    affected += 1

        elif action == "propagate":
            # Copiar las cajas del miembro mejor etiquetado al resto del grupo.
            src = max(group, key=lambda i: boxes_by_id.get(i, 0))
            if boxes_by_id.get(src, 0) == 0:
                continue  # nada que propagar en este grupo
            src_anns = db.get_annotations_for_image(src)
            for img_id in group:
                if img_id == src:
                    continue
                db.delete_annotations_for_image(img_id)
                for a in src_anns:
                    # (id, image_id, class_id, name, x, y, w, h, source, conf, poly)
                    polygon = json.loads(a[10]) if a[10] else None
                    db.insert_annotation(img_id, a[2], a[4], a[5], a[6], a[7],
                                         source="human", confidence=1.0,
                                         polygon=polygon)
                db.update_image_status(img_id, "reviewed")
                affected += 1

        if progress_callback:
            progress_callback(f"Procesando grupos {gi}/{total}",
                              80 + int(gi / total * 20))

    summary["affected"] = affected
    return summary
