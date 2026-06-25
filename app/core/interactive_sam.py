"""SAM interactivo por punto (click-to-box) para el editor.

El usuario hace click sobre un objeto y SAM2 propone la máscara/caja ajustada.
Se cachea el embedding de la imagen (set_image) para que cada click sea rápido.

Nota: el prompt por PUNTO lo soporta SAM/SAM2, no SAM3 (que es por concepto/texto).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class InteractiveSAM:
    def __init__(self, model_name: str, imgsz: int = 1024,
                 device: Optional[str] = None):
        self.model_name = model_name
        self.imgsz = imgsz
        self._device = device
        self._predictor = None
        self._image_path: str | None = None

    def load(self):
        """Crea un predictor SAM con set_image (caché de embedding por imagen)."""
        name = self.model_name.lower()
        if "sam2" in name:
            from ultralytics.models.sam import SAM2Predictor as SAMPredictor
        else:
            from ultralytics.models.sam import Predictor as SAMPredictor

        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"

        overrides = dict(
            conf=0.25, task="segment", mode="predict",
            model=self.model_name, imgsz=self.imgsz, save=False,
            device=self._device, verbose=False,
        )
        self._predictor = SAMPredictor(overrides=overrides)

    def set_image(self, path: str):
        """Codifica la imagen una vez; los clicks posteriores reusan el embedding."""
        if self._predictor is None:
            self.load()
        self._image_path = path
        self._predictor.set_image(path)

    def predict_point(self, px: float, py: float) -> dict | None:
        """Atajo: un solo punto positivo."""
        return self.predict_points([(px, py)], [1])

    def predict_points(self, points: list, labels: list) -> dict | None:
        """Predice a partir de varios puntos (en píxeles) con labels 1=positivo,
        0=negativo. Devuelve {'box': (x,y,w,h) normalizado, 'polygon': [...]|None}
        o None si no hay máscara."""
        if self._predictor is None or self._image_path is None or not points:
            return None
        pts = [[float(p[0]), float(p[1])] for p in points]
        lbls = [int(l) for l in labels]
        results = self._predictor(points=pts, labels=lbls)
        if not results:
            return None
        r = results[0]

        box = None
        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            x1, y1, x2, y2 = r.boxes.xyxyn[0].tolist()
            box = (max(0.0, x1), max(0.0, y1),
                   min(1.0, x2 - x1), min(1.0, y2 - y1))

        polygon = None
        masks = getattr(r, "masks", None)
        if masks is not None and getattr(masks, "xyn", None) is not None and len(masks.xyn):
            pts = masks.xyn[0]
            pts = pts.tolist() if hasattr(pts, "tolist") else pts
            polygon = [(float(x), float(y)) for x, y in pts]
            if box is None and polygon:
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

        if box is None or box[2] <= 0 or box[3] <= 0:
            return None
        return {"box": box, "polygon": polygon}

    def reset(self):
        if self._predictor is not None:
            try:
                self._predictor.reset_image()
            except Exception:
                pass
        self._image_path = None
