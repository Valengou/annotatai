"""Descubrimiento por diccionario: cierra el loop SigLIP2 → SAM3 → validar.

Para cada concepto del diccionario:
  1. SigLIP2 rankea las top-K imágenes candidatas (búsqueda semántica).
  2. SAM3 (texto) localiza el concepto en esas candidatas y genera cajas.
  3. Se importan como sugerencias para que el usuario solo valide.

Las dos fases corren secuencialmente y se libera la VRAM de SigLIP antes de
cargar SAM3 (clave en GPUs de 8 GB).
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

from ..utils.config import CLUSTER_COLORS
from .semantic_search import SemanticSearchEngine


ProgressCb = Callable[[str, int], None]


def _free_vram():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_concept_discovery(db, concepts: list[dict], top_k: int,
                          conf: float, imgsz: int, sam_model: str,
                          source: str = "suggested",
                          restrict_status: str | None = "pending",
                          progress_callback: ProgressCb | None = None) -> dict:
    """`concepts`: [{'prompt': str, 'class': str|None}, ...].
    Devuelve resumen {boxes, images, per_class, concepts}."""
    if not concepts:
        return {"boxes": 0, "images": 0, "per_class": {}, "concepts": 0}

    rows = db.get_all_images()
    path_by_id = {r[0]: r[1] for r in rows}
    status_by_id = {r[0]: r[6] for r in rows}

    # ── Fase 1: candidatas por SigLIP2 ──
    if progress_callback:
        progress_callback("Cargando SigLIP2...", 2)
    engine = SemanticSearchEngine(db)
    engine.load()

    fetch = max(top_k * 4, 200)
    candidates: list[list[int]] = []
    for ci, c in enumerate(concepts):
        results = engine.search_text(c["prompt"], top_k=fetch)
        ids = [iid for iid, _ in results]
        if restrict_status:
            ids = [i for i in ids if status_by_id.get(i) == restrict_status]
        candidates.append(ids[:top_k])
        if progress_callback:
            progress_callback(f"Buscando candidatas: {ci + 1}/{len(concepts)}",
                              2 + int((ci + 1) / len(concepts) * 28))

    engine = None          # liberar VRAM de SigLIP antes de SAM3
    _free_vram()

    # Limpiar sugerencias previas del mismo source en las candidatas (evita pilas)
    union_ids = set().union(*candidates) if candidates else set()
    for iid in union_ids:
        db.delete_annotations_by_source(iid, source)

    # ── Fase 2: SAM3 sobre las candidatas ──
    from .auto_label import AutoLabeler, _is_oom_error

    if progress_callback:
        progress_callback("Cargando SAM 3...", 32)
    labeler = AutoLabeler(model_name=sam_model, conf=conf, imgsz=imgsz)
    labeler.load()

    existing = {row[1]: row[0] for row in db.get_all_classes()}
    palette_i = len(existing)
    per_class: dict[str, int] = {}
    images_with_boxes: set[int] = set()
    n_boxes = 0
    oom = 0

    total = sum(len(c) for c in candidates) or 1
    done = 0
    for ci, c in enumerate(concepts):
        cls_name = (c.get("class") or c["prompt"]).strip()
        cid = existing.get(cls_name)
        if cid is None:
            color = CLUSTER_COLORS[palette_i % len(CLUSTER_COLORS)]
            cid = db.get_or_create_class(cls_name, color)
            existing[cls_name] = cid
            palette_i += 1

        for iid in candidates[ci]:
            path = path_by_id.get(iid)
            if not path or not Path(path).exists():
                done += 1
                continue
            try:
                dets = labeler.detect_text(Path(path), c["prompt"])
            except Exception as exc:
                if _is_oom_error(exc):
                    oom += 1
                    _free_vram()
                    done += 1
                    continue
                raise
            for x, y, w, h, dconf in dets:
                db.insert_annotation(iid, cid, x, y, w, h,
                                     source=source, confidence=dconf)
                n_boxes += 1
                per_class[cls_name] = per_class.get(cls_name, 0) + 1
                images_with_boxes.add(iid)
            done += 1
            _free_vram()
            if progress_callback:
                progress_callback(f"SAM 3: {done}/{total} candidatas",
                                  32 + int(done / total * 68))

    if progress_callback:
        progress_callback("Listo", 100)
    return {
        "boxes": n_boxes,
        "images": len(images_with_boxes),
        "per_class": per_class,
        "concepts": len(concepts),
        "oom_skipped": oom,
    }


def parse_concepts(text: str) -> list[dict]:
    """Una línea por concepto. Formato: 'prompt' o 'prompt = ClaseDestino'.
    Líneas vacías o que empiezan con # se ignoran."""
    out: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            prompt, cls = line.split("=", 1)
            prompt, cls = prompt.strip(), cls.strip()
        else:
            prompt, cls = line, ""
        if prompt:
            out.append({"prompt": prompt, "class": cls or prompt})
    return out
