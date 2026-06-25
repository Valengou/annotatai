"""Outliers geométricos de bounding boxes, por clase (sin modelo).

Para cada clase compara cada caja contra la distribución de su clase en varias
features (área, aspect ratio, ancho, alto, posición) usando z-score robusto
(mediana + MAD). Marca las cajas con |z| por encima de un umbral — típicamente
errores de anotación: caja demasiado chica/grande/elongada o mal ubicada.
"""

from __future__ import annotations

import math

import numpy as np


# Índices de get_annotations_full:
# (ann_id, image_id, class_id, name, color, x, y, w, h, path, filename, status)
_ID, _IMG, _CID, _NAME, _COLOR, _X, _Y, _W, _H, _PATH, _FNAME, _STATUS = range(12)


def _robust_z(vals: np.ndarray) -> np.ndarray:
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = float(vals.std())
    if scale < 1e-9:
        return np.zeros_like(vals)
    return (vals - med) / scale


def _reason(feature: str, z: float) -> str:
    hi = z > 0
    return {
        "área":    "área anormalmente grande" if hi else "área anormalmente chica",
        "aspect":  "muy alargada (horizontal)" if hi else "muy alargada (vertical)",
        "ancho":   "ancho atípico (grande)" if hi else "ancho atípico (chico)",
        "alto":    "alto atípico (grande)" if hi else "alto atípico (chico)",
        "pos-x":   "posición horizontal atípica",
        "pos-y":   "posición vertical atípica",
    }.get(feature, "atípica")


def find_geometric_outliers(db, z_thresh: float = 3.5, min_per_class: int = 5,
                            class_name: str | None = None) -> list[dict]:
    """Devuelve outliers ordenados por score (|z| máximo) descendente."""
    rows = db.get_annotations_full()
    by_class: dict[str, list] = {}
    for r in rows:
        by_class.setdefault(r[_NAME], []).append(r)

    results: list[dict] = []
    for cname, items in by_class.items():
        if class_name and cname != class_name:
            continue
        if len(items) < min_per_class:
            continue

        w = np.array([it[_W] for it in items], dtype=float)
        h = np.array([it[_H] for it in items], dtype=float)
        x = np.array([it[_X] for it in items], dtype=float)
        y = np.array([it[_Y] for it in items], dtype=float)
        w_safe = np.clip(w, 1e-6, None)
        h_safe = np.clip(h, 1e-6, None)

        feats = {
            "área":   np.log(np.clip(w * h, 1e-9, None)),
            "aspect": np.log(w_safe / h_safe),
            "ancho":  w,
            "alto":   h,
            "pos-x":  x + w / 2.0,
            "pos-y":  y + h / 2.0,
        }
        zmaps = {f: _robust_z(v) for f, v in feats.items()}

        for i, it in enumerate(items):
            best_f, best_z = None, 0.0
            for f, zv in zmaps.items():
                if abs(zv[i]) > abs(best_z):
                    best_z, best_f = float(zv[i]), f
            if best_f is None or abs(best_z) < z_thresh:
                continue
            results.append({
                "ann_id": it[_ID],
                "image_id": it[_IMG],
                "class_id": it[_CID],
                "class_name": cname,
                "class_color": it[_COLOR],
                "x": it[_X], "y": it[_Y], "w": it[_W], "h": it[_H],
                "path": it[_PATH], "filename": it[_FNAME], "status": it[_STATUS],
                "score": abs(best_z),
                "feature": best_f,
                "reason": _reason(best_f, best_z),
            })

    results.sort(key=lambda d: d["score"], reverse=True)
    return results
