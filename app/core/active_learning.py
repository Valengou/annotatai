"""Active learning para acelerar la revisión de sugerencias.

Sobre las imágenes pendientes (que tienen sugerencias de IA, source != 'human'):

  - Auto-aceptar: las cajas con confianza >= umbral pasan a 'human' (se dan por
    buenas), reduciendo clicks.
  - Auto-revisar: si tras auto-aceptar una imagen queda solo con cajas humanas y
    al menos una, se marca 'reviewed'.
  - Priorizar: calcula una métrica de incertidumbre por imagen y la guarda en
    `review_sort_confidence`, de modo que el orden "Prioridad baja conf" de la
    grilla muestre primero lo más dudoso (cajas con confianza cerca del centro
    de decisión).
"""

from __future__ import annotations

from typing import Callable

ProgressCb = Callable[[str, int], None]

# Índices de get_annotations_for_image:
# (id, image_id, class_id, name, x, y, w, h, source, confidence, points_json)
_ID, _SOURCE, _CONF = 0, 8, 9


def run_active_learning(db, accept_enabled: bool = True,
                        accept_threshold: float = 0.90,
                        mark_reviewed: bool = True,
                        prioritize: bool = True,
                        center: float = 0.5,
                        progress_callback: ProgressCb | None = None) -> dict:
    rows = db.get_images_by_status("pending")
    total = max(1, len(rows))
    accepted = 0
    auto_reviewed = 0
    prioritized = 0

    for done, (img_id, _path, _fname) in enumerate(rows, start=1):
        anns = db.get_annotations_for_image(img_id)

        if accept_enabled:
            for a in anns:
                conf = a[_CONF]
                if a[_SOURCE] != "human" and conf is not None and conf >= accept_threshold:
                    db.update_annotation_source(a[_ID], "human", 1.0)
                    accepted += 1
            anns = db.get_annotations_for_image(img_id)

        provisional = [a for a in anns if a[_SOURCE] != "human"]
        human = [a for a in anns if a[_SOURCE] == "human"]

        if mark_reviewed and human and not provisional:
            db.update_image_status(img_id, "reviewed")
            auto_reviewed += 1
            # Ya resuelta: no necesita prioridad de revisión
            db.update_review_sort_confidence(img_id, None)
        elif prioritize:
            if provisional:
                # Caja más ambigua = la más cercana al centro de decisión.
                score = min(
                    abs((a[_CONF] if a[_CONF] is not None else center) - center)
                    for a in provisional
                )
            else:
                score = 1.0   # sin sugerencias dudosas → al fondo de la cola
            db.update_review_sort_confidence(img_id, score)
            prioritized += 1

        if progress_callback:
            progress_callback(f"Analizando {done}/{len(rows)}",
                              int(done / total * 100))

    return {
        "images": len(rows),
        "accepted": accepted,
        "auto_reviewed": auto_reviewed,
        "prioritized": prioritized,
    }
