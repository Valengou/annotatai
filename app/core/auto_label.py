"""
Auto-etiquetado con SAM 3 (Promptable Concept Segmentation).

Genera bounding boxes para imágenes a partir de prompts de concepto:
  - texto: frases tipo "insulator", "transmission tower"
  - exemplar visual: una o más bboxes de ejemplo dentro de la misma imagen

SAM 3 devuelve máscara + caja para cada instancia; por ahora solo se persiste
la bbox (las máscaras se ignoran). Las coordenadas se devuelven normalizadas
(x, y, width, height en 0-1, esquina superior izquierda) igual que el modelo
Annotation del proyecto.
"""

from pathlib import Path
from typing import Callable, Optional

from ..utils.config import CLUSTER_COLORS

DEFAULT_MODEL = "sam3.pt"
DEFAULT_YOLOE_MODEL = "yoloe-v8l-seg.pt"


class AutoLabeler:
    def __init__(self, model_name: str = DEFAULT_MODEL,
                 device: Optional[str] = None, conf: float = 0.25,
                 imgsz: int = 1024):
        self.model_name = model_name
        self._device = device
        self._conf = conf
        self._imgsz = imgsz
        self._predictor = None

    def load(self):
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = dict(
            conf=self._conf, task="segment", mode="predict", model=self.model_name,
            imgsz=self._imgsz, save=False,
        )
        if self._device:
            overrides["device"] = self._device
        self._predictor = SAM3SemanticPredictor(overrides=overrides)

    def detect_text(self, image_path: Path, phrase: str) -> list[tuple]:
        """Detecta todas las instancias del concepto `phrase` en la imagen."""
        self._predictor.set_image(str(image_path))
        results = self._predictor(text=[phrase])
        return self._extract_boxes(results)

    def detect_text_with_polygons(self, image_path: Path, phrase: str) -> list[tuple]:
        self._predictor.set_image(str(image_path))
        results = self._predictor(text=[phrase])
        return self._extract_detections(results, include_polygons=True)

    def detect_exemplar(self, image_path: Path,
                        bboxes_xyxy: list[list[float]]) -> list[tuple]:
        """Detecta instancias similares a las bboxes de ejemplo (en px, xyxy)
        dentro de la misma imagen."""
        self._predictor.set_image(str(image_path))
        results = self._predictor(bboxes=bboxes_xyxy)
        return self._extract_boxes(results)

    @staticmethod
    def _extract_boxes(results) -> list[tuple]:
        """De los Results de ultralytics a [(x, y, w, h, conf), ...] normalizado."""
        return [d[:5] for d in AutoLabeler._extract_detections(results)]

    @staticmethod
    def _extract_detections(results, include_polygons: bool = False) -> list[tuple]:
        """De Results de ultralytics a detecciones normalizadas.

        Sin polígonos: [(x, y, w, h, conf), ...].
        Con polígonos: [(x, y, w, h, conf, [(x, y), ...] | None), ...].
        """
        out: list[tuple] = []
        for r in results:
            if r.boxes is None:
                continue
            xywhn = r.boxes.xywhn.tolist()
            confs = r.boxes.conf.tolist()
            polygons = AutoLabeler._extract_polygons(r) if include_polygons else []
            for idx, ((cx, cy, bw, bh), c) in enumerate(zip(xywhn, confs)):
                detection = (cx - bw / 2, cy - bh / 2, bw, bh, float(c))
                if include_polygons:
                    polygon = polygons[idx] if idx < len(polygons) else None
                    detection = (*detection, polygon)
                out.append(detection)
        return out

    @staticmethod
    def _extract_polygons(result) -> list[list[tuple[float, float]]]:
        masks = getattr(result, "masks", None)
        if masks is None or getattr(masks, "xyn", None) is None:
            return []
        polygons = []
        for poly in masks.xyn:
            points = poly.tolist() if hasattr(poly, "tolist") else poly
            polygons.append([(float(x), float(y)) for x, y in points])
        return polygons


class YoloeLabeler:
    """Auto-etiquetado por prompt visual con YOLOE (transfiere cross-image).

    Un conjunto de bboxes de ejemplo (en una imagen de referencia) define el
    concepto, y YOLOE detecta instancias similares en las imágenes objetivo.
    """

    def __init__(self, model_name: str = DEFAULT_YOLOE_MODEL,
                 device: Optional[str] = None, conf: float = 0.25):
        self.model_name = model_name
        self._device = device
        self._conf = conf
        self._model = None

    def load(self):
        from ultralytics import YOLOE

        self._model = YOLOE(self.model_name)
        if self._device:
            self._model.to(self._device)

    def detect(self, target_paths: list[str], ref_path: str,
               exemplar_xyxy: list[list[float]], class_name: str,
               imgsz: int = 1024):
        """Generador: por cada imagen objetivo emite (path, [(x,y,w,h,conf), ...]).

        `imgsz` acota el tamaño de inferencia: las máscaras del modelo -seg escalan
        con esto, así que bajarlo reduce mucho el pico de VRAM en imágenes grandes.
        """
        import numpy as np
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        visual_prompts = dict(
            bboxes=np.array(exemplar_xyxy, dtype=float),
            cls=np.zeros(len(exemplar_xyxy), dtype=int),
        )
        results = self._model.predict(
            target_paths,
            refer_image=ref_path,
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            conf=self._conf,
            imgsz=imgsz,
            retina_masks=False,
            stream=True,
            verbose=False,
            save=False,
        )
        # Con refer_image + lista, ultralytics renombra r.path (image0.jpg, ...);
        # los resultados llegan en el mismo orden que target_paths.
        for path, r in zip(target_paths, results):
            yield path, AutoLabeler._extract_boxes([r])


MAX_REFS = 8   # imágenes de referencia máximas para el ensemble de prompt visual


def _iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: list, iou_thresh: float = 0.5) -> list:
    """NMS simple sobre [(x, y, w, h, conf), ...] (normalizado, esquina sup-izq)."""
    keep: list = []
    for b in sorted(boxes, key=lambda d: d[4], reverse=True):
        if all(_iou_xywh(b, k) < iou_thresh for k in keep):
            keep.append(b)
    return keep


def run_auto_label_visual(db, class_id: int, class_name: str, target_rows: list,
                          conf: float, source: str,
                          model_name: str = DEFAULT_YOLOE_MODEL,
                          device: Optional[str] = None,
                          exemplar_image_ids: Optional[set] = None,
                          imgsz: int = 1024,
                          chunk_size: int = 16,
                          progress_callback: Callable | None = None) -> dict:
    """Auto-etiqueta `target_rows` usando las cajas humanas de `class_id` como
    prompt visual (YOLOE cross-image). `target_rows` son filas de
    `get_all_images` (0=id, 1=path). Si se pasa `exemplar_image_ids`, la imagen
    de referencia se elige solo entre esas (p. ej. las de un cluster).
    Devuelve {'images', 'boxes'}."""
    box_rows = db.get_boxes_for_class(class_id, source="human")
    if exemplar_image_ids is not None:
        box_rows = [r for r in box_rows if r[0] in exemplar_image_ids]
    if not box_rows:
        return {"images": 0, "boxes": 0, "error": "sin_exemplar"}

    # Imágenes de referencia: hasta MAX_REFS, las que más cajas humanas tienen.
    # Se usan TODAS como ejemplos (ensemble), no solo una → más robusto.
    by_image: dict[int, list] = {}
    ref_dims: dict[int, tuple] = {}
    ref_path_by_image: dict[int, str] = {}
    for image_id, path, iw, ih, x, y, w, h in box_rows:
        by_image.setdefault(image_id, []).append((x, y, w, h))
        ref_dims[image_id] = (iw, ih)
        ref_path_by_image[image_id] = path

    ref_ids = sorted(by_image, key=lambda k: len(by_image[k]), reverse=True)[:MAX_REFS]
    references = []   # [(ref_id, ref_path, exemplar_xyxy), ...]
    for rid in ref_ids:
        iw, ih = ref_dims[rid]
        xyxy = [[x * iw, y * ih, (x + w) * iw, (y + h) * ih]
                for x, y, w, h in by_image[rid]]
        references.append((rid, ref_path_by_image[rid], xyxy))

    ref_id_set = {r[0] for r in references}
    id_by_path = {row[1]: row[0] for row in target_rows}
    target_paths = [row[1] for row in target_rows
                    if Path(row[1]).exists() and row[0] not in ref_id_set]
    if not target_paths or not references:
        return {"images": 0, "boxes": 0}

    labeler = YoloeLabeler(model_name=model_name, device=device, conf=conf)
    labeler.load()

    total = len(target_paths)
    n_images = 0
    n_boxes = 0
    done = 0
    # Procesa en chunks y libera la cache de VRAM entre cada uno (GPUs de 8 GB).
    for start in range(0, total, max(1, chunk_size)):
        chunk = target_paths[start:start + chunk_size]
        merged: dict[str, list] = {p: [] for p in chunk}
        # Ensemble: cada imagen de referencia aporta sus detecciones
        for _rid, ref_path, xyxy in references:
            for path, boxes in labeler.detect(chunk, ref_path, xyxy,
                                              class_name, imgsz=imgsz):
                if boxes:
                    merged.setdefault(path, []).extend(boxes)
            _empty_cuda_cache()
        # Fusiona las detecciones de todas las referencias con NMS
        for path in chunk:
            boxes = _nms(merged.get(path, []), iou_thresh=0.5)
            img_id = id_by_path.get(path)
            if img_id is not None and boxes:
                for x, y, w, h, c in boxes:
                    db.insert_annotation(img_id, class_id, x, y, w, h,
                                         source=source, confidence=c)
                n_images += 1
                n_boxes += len(boxes)
            done += 1
            if progress_callback:
                progress_callback(done, total)

    return {"images": n_images, "boxes": n_boxes,
            "references": len(references)}


def _empty_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_oom_error(exc: Exception) -> bool:
    """True si la excepción es por falta de memoria de GPU (CUDA OOM)."""
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error" in msg


def run_auto_label(db, image_rows: list, phrases: list[str], conf: float,
                   source: str, model_name: str = DEFAULT_MODEL,
                   device: Optional[str] = None,
                   save_polygons: bool = False,
                   imgsz: int = 1024,
                   progress_callback: Callable | None = None) -> dict:
    """Auto-etiqueta `image_rows` con prompts de texto y escribe en la DB.

    `image_rows` son filas de `get_all_images` (se usan índices 0=id, 1=path).
    Cada frase se mapea a una clase (creándola si no existe) y todas las
    detecciones de esa frase se guardan con esa clase y el `source` indicado.
    Devuelve {'images': N, 'boxes': K, 'classes': [...]}.
    """
    labeler = AutoLabeler(model_name=model_name, device=device, conf=conf, imgsz=imgsz)
    labeler.load()

    existing = {row[1] for row in db.get_all_classes()}
    class_ids: dict[str, int] = {}
    for i, phrase in enumerate(phrases):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        class_ids[phrase] = db.get_or_create_class(phrase, color)

    total = len(image_rows)
    n_images = 0
    n_boxes = 0
    oom_skipped = 0

    for done, row in enumerate(image_rows, start=1):
        img_id, path = row[0], row[1]
        src = Path(path)
        if not src.exists():
            if progress_callback:
                progress_callback(done, total)
            continue

        boxes_for_image = 0
        try:
            for phrase in phrases:
                detections = (
                    labeler.detect_text_with_polygons(src, phrase)
                    if save_polygons else
                    labeler.detect_text(src, phrase)
                )
                for detection in detections:
                    x, y, w, h, c = detection[:5]
                    polygon = detection[5] if save_polygons and len(detection) > 5 else None
                    db.insert_annotation(img_id, class_ids[phrase], x, y, w, h,
                                         source=source, confidence=c, polygon=polygon)
                    boxes_for_image += 1
        except Exception as exc:
            # Si la GPU se queda sin memoria, liberamos y saltamos esta imagen
            # en lugar de tumbar todo el proceso.
            if _is_oom_error(exc):
                oom_skipped += 1
                _empty_cuda_cache()
            else:
                raise

        if boxes_for_image:
            n_images += 1
            n_boxes += boxes_for_image

        # Liberar la cache de VRAM entre imágenes: en GPUs chicas (8 GB) evita que
        # el pico crezca hasta derramar a memoria compartida (y crashear).
        _empty_cuda_cache()

        if progress_callback:
            progress_callback(done, total)

    new_classes = [p for p in phrases if p not in existing]
    return {"images": n_images, "boxes": n_boxes, "classes": new_classes,
            "oom_skipped": oom_skipped}
