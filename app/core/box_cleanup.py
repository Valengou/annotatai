"""Limpieza de detecciones solapadas (NMS por clase, a nivel proyecto).

Por cada imagen y clase, si dos cajas se solapan (IoU > umbral) se conserva la de
mayor confianza y se borran las demás. Las cajas humanas son anclas: nunca se
borran y siempre ganan (confianza 1.0).
"""

from __future__ import annotations

from typing import Callable

from .auto_label import _iou_xywh


def cleanup_overlapping(db, iou_thresh: float = 0.5,
                        progress_callback: Callable | None = None) -> dict:
    rows = db.get_annotations_for_cleanup()
    # agrupar por (image_id, class_id)
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r[1], r[2]), []).append(r)

    to_delete: list[int] = []
    images_affected: set[int] = set()
    total = max(1, len(groups))
    for gi, ((img_id, _cid), boxes) in enumerate(groups.items(), start=1):
        # humanas primero (anclas), luego sugeridas por confianza descendente
        boxes.sort(key=lambda b: (b[7] == "human", b[8] if b[8] is not None else 0.0),
                   reverse=True)
        kept: list = []
        for b in boxes:
            xywh = (b[3], b[4], b[5], b[6])
            overlaps = any(_iou_xywh(xywh, (k[3], k[4], k[5], k[6])) > iou_thresh
                           for k in kept)
            if overlaps and b[7] != "human":
                to_delete.append(b[0])
                images_affected.add(img_id)
            else:
                kept.append(b)
        if progress_callback and gi % 50 == 0:
            progress_callback(f"Analizando grupos {gi}/{len(groups)}",
                              int(gi / total * 90))

    removed = db.delete_annotations(to_delete) if to_delete else 0
    if progress_callback:
        progress_callback("Listo", 100)
    return {"removed": removed, "images": len(images_affected)}
