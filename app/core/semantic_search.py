"""Búsqueda semántica con SigLIP2 (texto ↔ imagen) sobre un índice de embeddings
propio, separado de los de clustering.

- generate_search_embeddings: codifica todas las imágenes con SigLIP2 y las guarda
  en la tabla `search_embeddings` (model='siglip2').
- SemanticSearchEngine: mantiene el modelo y la matriz en memoria para responder
  consultas de texto o por imagen de ejemplo rápidamente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .embedding_backends import create_embedding_backend


SEARCH_MODEL = "siglip2"


def generate_search_embeddings(db, progress_callback: Callable | None = None) -> int:
    """Genera embeddings de búsqueda (SigLIP2) para las imágenes que falten."""
    backend = create_embedding_backend(SEARCH_MODEL)
    rows = db.get_all_images()
    total = len(rows)
    pending = [r for r in rows if not db.has_search_embedding(r[0], SEARCH_MODEL)]
    if not pending:
        if progress_callback:
            progress_callback(total, total)
        return 0

    paths = [Path(r[1]) for r in pending]

    def cb(done: int, backend_total: int):
        if progress_callback:
            progress_callback(done, backend_total)

    vectors = backend.embed_paths(paths, progress_callback=cb)
    for r, v in zip(pending, vectors):
        db.save_search_embedding(
            r[0], np.asarray(v, dtype=np.float32).tobytes(), SEARCH_MODEL)

    if progress_callback:
        progress_callback(total, total)
    return len(pending)


class SemanticSearchEngine:
    """Modelo + matriz de embeddings en memoria para consultas rápidas."""

    def __init__(self, db, model_name: str = SEARCH_MODEL):
        self.db = db
        self.model_name = model_name
        self._backend = None
        self._ids: list[int] | None = None
        self._matrix: np.ndarray | None = None

    def is_indexed(self) -> bool:
        return self.db.count_search_embeddings(self.model_name) > 0

    def load(self, progress_callback: Callable | None = None):
        if progress_callback:
            progress_callback("Cargando modelo SigLIP2...", 10)
        if self._backend is None:
            self._backend = create_embedding_backend(self.model_name)
            self._backend.load()
        if progress_callback:
            progress_callback("Cargando índice de búsqueda...", 60)
        rows = self.db.get_all_search_embeddings(self.model_name)
        self._ids = [r[0] for r in rows]
        if rows:
            self._matrix = np.vstack(
                [np.frombuffer(r[1], dtype=np.float32) for r in rows])
        else:
            self._matrix = None

    def _ensure(self, progress_callback: Callable | None = None):
        if self._backend is None or self._matrix is None or self._ids is None:
            self.load(progress_callback)

    def _rank(self, scores: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        order = np.argsort(-scores)[:top_k]
        return [(self._ids[i], float(scores[i])) for i in order]

    def search_text(self, query: str, top_k: int = 300,
                    progress_callback: Callable | None = None) -> list[tuple[int, float]]:
        self._ensure(progress_callback)
        if self._matrix is None:
            return []
        if progress_callback:
            progress_callback("Buscando...", 90)
        q = self._backend.embed_text([query])[0].astype(np.float32)
        scores = self._matrix @ q
        return self._rank(scores, top_k)

    def search_image(self, image_id: int, top_k: int = 300,
                     progress_callback: Callable | None = None) -> list[tuple[int, float]]:
        self._ensure(progress_callback)
        if self._matrix is None or self._ids is None or image_id not in self._ids:
            return []
        idx = self._ids.index(image_id)
        scores = self._matrix @ self._matrix[idx]
        ranked = self._rank(scores, top_k + 1)
        return [(i, s) for i, s in ranked if i != image_id][:top_k]
