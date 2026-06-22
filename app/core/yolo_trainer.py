"""Entrenamiento rápido de un YOLO nano con las cajas humanas del proyecto, para
asistir el etiquetado.

Flujo:
  1. export_yolo_dataset  → vuelca las imágenes 'reviewed' con cajas 'human' a un
     dataset YOLO (train/val) multi-clase.
  2. train_yolo           → fine-tunea un modelo nano (yolo11n/yolov8n) sobre eso.
  3. predict_and_import   → corre el modelo sobre las pendientes e inserta las
     detecciones como sugerencias (source='yolo'), que aparecen punteadas en el
     editor para aceptar/rechazar.

`run_quick_train` orquesta los tres pasos y reporta progreso por
`progress_callback(mensaje: str, porcentaje: int)`.
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..utils.paths import project_models_dir


ProgressCb = Callable[[str, int], None]


@dataclass
class TrainOptions:
    model: str = "yolo11n.pt"      # modelo base nano a fine-tunear
    epochs: int = 50
    imgsz: int = 640
    batch: int = 8
    val_fraction: float = 0.2
    predict_after: bool = True     # predecir sobre pendientes al terminar
    predict_conf: float = 0.25
    predict_scope: str = "pending"  # 'pending' | 'all'
    train_classes: list[str] | None = None  # clases a entrenar (None = todas)
    run_name: str = "assist"


@dataclass
class TrainResult:
    weights: Path | None = None
    n_train: int = 0
    n_val: int = 0
    names: list[str] = field(default_factory=list)
    pred_images: int = 0
    pred_boxes: int = 0
    dataset_dir: Path | None = None
    error: str | None = None


def _classes_ordered(db, only_names: list[str] | None = None) -> tuple[dict[int, int], list[str]]:
    """{class_id: indice_contiguo}, [nombres] — en orden de get_all_classes.

    Si `only_names` se pasa, restringe a esas clases y re-indexa contiguamente."""
    rows = db.get_all_classes()   # (id, name, color) ORDER BY name
    if only_names is not None:
        wanted = set(only_names)
        rows = [r for r in rows if r[1] in wanted]
    class_idx = {row[0]: i for i, row in enumerate(rows)}
    names = [row[1] for row in rows]
    return class_idx, names


def export_yolo_dataset(db, out_dir: Path, val_fraction: float = 0.2,
                        train_classes: list[str] | None = None,
                        progress_callback: ProgressCb | None = None) -> tuple[int, int, list[str]]:
    """Exporta imágenes 'reviewed' con sus cajas 'human' a un dataset YOLO.

    Solo se exportan las cajas de las clases en `train_classes` (None = todas),
    re-indexadas contiguamente. Devuelve (n_train, n_val, names). Las imágenes
    revisadas sin cajas de esas clases se incluyen como negativos (background).
    """
    class_idx, names = _classes_ordered(db, train_classes)
    if not names:
        return 0, 0, []

    labeled = [r for r in db.get_images_by_status("reviewed") if Path(r[1]).exists()]
    if val_fraction > 0:
        random.Random(0).shuffle(labeled)
    n_val = int(len(labeled) * val_fraction)
    # Garantizar al menos 1 de validación si hay material suficiente
    if labeled and n_val == 0 and len(labeled) > 1:
        n_val = 1
    val_ids = {r[0] for r in labeled[:n_val]}

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    splits = {
        "val":   [r for r in labeled if r[0] in val_ids],
        "train": [r for r in labeled if r[0] not in val_ids],
    }
    total = max(1, len(labeled))
    done = 0
    for split, rows_s in splits.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_id, path, filename in rows_s:
            shutil.copy2(path, img_dir / filename)
            lines = []
            for row in db.get_annotations_for_image(img_id):
                # (id, image_id, class_id, name, x, y, w, h, source, conf, poly)
                cid, x, y, w, h, source = row[2], row[4], row[5], row[6], row[7], row[8]
                if source != "human":
                    continue
                idx = class_idx.get(cid)
                if idx is None:
                    continue
                lines.append(f"{idx} {x + w/2:.6f} {y + h/2:.6f} {w:.6f} {h:.6f}")
            if lines:
                (lbl_dir / (Path(filename).stem + ".txt")).write_text(
                    "\n".join(lines), encoding="utf-8")
            done += 1
            if progress_callback:
                progress_callback(f"Exportando dataset {done}/{total}",
                                  int(done / total * 15))

    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names}\n", encoding="utf-8")

    return len(splits["train"]), len(splits["val"]), names


def train_yolo(dataset_dir: Path, model: str, epochs: int, imgsz: int, batch: int,
               device: str | None = None,
               progress_callback: ProgressCb | None = None) -> Path:
    """Fine-tunea un YOLO nano. Devuelve la ruta a best.pt."""
    from ultralytics import YOLO

    yolo = YOLO(model)

    if progress_callback:
        def _on_epoch_end(trainer):
            ep = int(getattr(trainer, "epoch", 0)) + 1
            total = int(getattr(trainer, "epochs", epochs)) or epochs
            pct = 15 + int(ep / max(1, total) * 70)   # 15% → 85%
            progress_callback(f"Entrenando época {ep}/{total}", min(85, pct))
        yolo.add_callback("on_train_epoch_end", _on_epoch_end)

    results = yolo.train(
        data=str(Path(dataset_dir) / "data.yaml"),
        epochs=epochs, imgsz=imgsz, batch=batch,
        project=str(Path(dataset_dir) / "runs"), name="train", exist_ok=True,
        patience=max(10, epochs // 4), cache=False, workers=4, verbose=False,
        device=device,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    return best


def predict_and_import(db, weights: Path, conf: float, imgsz: int,
                       scope: str = "pending",
                       targets: list | None = None,
                       import_classes: list[str] | None = None,
                       source: str = "yolo",
                       device: str | None = None,
                       chunk: int = 8,
                       progress_base: int = 85, progress_span: int = 15,
                       progress_callback: ProgressCb | None = None) -> tuple[int, int, list[str]]:
    """Predice con un modelo YOLO e importa las detecciones como sugerencias.

    Las clases se mapean por NOMBRE (leídas de `model.names`) a las clases del
    proyecto, creándolas si no existen — así el mismo .pt sirve en otro proyecto.
    `import_classes` (None = todas) filtra qué nombres importar.

    `targets` (lista de (image_id, path)) define el alcance explícito; si es None
    se resuelve desde `scope` ('pending' | 'all').

    Devuelve (n_imagenes, n_cajas, nombres_importados). Reemplaza sugerencias
    previas del mismo `source` en cada imagen para no acumular duplicados.
    """
    from ultralytics import YOLO
    from ..utils.config import CLUSTER_COLORS

    model = YOLO(str(weights))
    model_names = model.names   # {idx: name} o lista
    if isinstance(model_names, (list, tuple)):
        model_names = {i: n for i, n in enumerate(model_names)}

    # Mapa índice-de-modelo → class_id del proyecto (por nombre, crea si falta)
    existing = {row[1]: row[0] for row in db.get_all_classes()}
    palette_i = len(existing)
    idx_to_cid: dict[int, int] = {}
    imported_names: list[str] = []
    for idx, name in model_names.items():
        if import_classes and name not in import_classes:
            continue
        cid = existing.get(name)
        if cid is None:
            color = CLUSTER_COLORS[palette_i % len(CLUSTER_COLORS)]
            cid = db.get_or_create_class(name, color)
            existing[name] = cid
            palette_i += 1
        idx_to_cid[idx] = cid
        imported_names.append(name)

    if targets is None:
        status = "pending" if scope == "pending" else None
        if status:
            targets = [(r[0], r[1]) for r in db.get_images_by_status(status)
                       if Path(r[1]).exists()]
        else:
            targets = [(r[0], r[1]) for r in db.get_all_images() if Path(r[1]).exists()]
    else:
        targets = [(i, p) for (i, p) in targets if Path(p).exists()]
    if not targets:
        return 0, 0, imported_names

    total = len(targets)
    n_imgs = n_boxes = done = 0

    for start in range(0, total, chunk):
        batch_rows = targets[start:start + chunk]
        results = model.predict([p for _, p in batch_rows], conf=conf, imgsz=imgsz,
                                device=device, verbose=False, stream=True, save=False)
        for (img_id, _path), r in zip(batch_rows, results):
            dets = []
            if r.boxes is not None:
                for b in r.boxes:
                    cid = idx_to_cid.get(int(b.cls[0].item()))
                    if cid is None:
                        continue
                    cx, cy, bw, bh = b.xywhn[0].tolist()
                    dets.append((cid, cx - bw/2, cy - bh/2, bw, bh,
                                 float(b.conf[0].item())))
            db.delete_annotations_by_source(img_id, source)
            if dets:
                for cid, x, y, bw, bh, c in dets:
                    db.insert_annotation(img_id, cid, x, y, bw, bh,
                                         source=source, confidence=c)
                n_imgs += 1
                n_boxes += len(dets)
            done += 1
            if progress_callback:
                progress_callback(f"Pre-etiquetando {done}/{total}",
                                  progress_base + int(done / total * progress_span))
        _empty_cuda_cache()

    return n_imgs, n_boxes, imported_names


def _detect_device() -> str | None:
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return None


def _empty_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_quick_train(db, project_path: Path, options: TrainOptions,
                    predict_targets: list | None = None,
                    progress_callback: ProgressCb | None = None) -> TrainResult:
    """Orquesta export → train → (opcional) predict. Pensado para correr en un
    QThread; reporta progreso 0-100. `predict_targets` (lista de (id, path))
    define el alcance del pre-etiquetado; si None, usa options.predict_scope."""
    device = _detect_device()
    models_root = project_models_dir(project_path)
    run_dir = models_root / options.run_name
    dataset_dir = run_dir / "dataset"

    if progress_callback:
        progress_callback("Exportando dataset...", 1)
    n_train, n_val, names = export_yolo_dataset(
        db, dataset_dir, options.val_fraction, options.train_classes,
        progress_callback)

    if not names:
        return TrainResult(error="No seleccionaste ninguna clase para entrenar.",
                           dataset_dir=dataset_dir)
    if n_train == 0:
        return TrainResult(error="No hay imágenes revisadas con cajas de las clases "
                                 "seleccionadas para entrenar.",
                           names=names, dataset_dir=dataset_dir)

    if progress_callback:
        progress_callback("Iniciando entrenamiento...", 15)
    weights = train_yolo(
        dataset_dir, options.model, options.epochs, options.imgsz,
        options.batch, device=device, progress_callback=progress_callback)

    result = TrainResult(weights=weights, n_train=n_train, n_val=n_val,
                         names=names, dataset_dir=dataset_dir)

    # Copiar best.pt a un lugar estable del proyecto
    if weights and weights.exists():
        stable = run_dir / "best.pt"
        try:
            shutil.copy2(weights, stable)
            result.weights = stable
        except Exception:
            pass

    if options.predict_after and result.weights and result.weights.exists():
        if progress_callback:
            progress_callback("Pre-etiquetando imágenes...", 85)
        n_imgs, n_boxes, _ = predict_and_import(
            db, result.weights, options.predict_conf, options.imgsz,
            scope=options.predict_scope, targets=predict_targets, device=device,
            progress_callback=progress_callback)
        result.pred_images = n_imgs
        result.pred_boxes = n_boxes

    if progress_callback:
        progress_callback("Listo", 100)
    return result


def run_predict_only(db, weights: Path, conf: float = 0.25, imgsz: int = 640,
                     scope: str = "pending", targets: list | None = None,
                     import_classes: list[str] | None = None,
                     source: str = "yolo",
                     progress_callback: ProgressCb | None = None) -> TrainResult:
    """Usa un modelo .pt ya entrenado (de este u otro proyecto) para pre-etiquetar.
    Pensado para correr en un QThread. `targets` (id, path) define el alcance."""
    weights = Path(weights)
    if not weights.exists():
        return TrainResult(error=f"No se encontró el modelo:\n{weights}")
    device = _detect_device()
    if progress_callback:
        progress_callback("Cargando modelo...", 3)
    n_imgs, n_boxes, names = predict_and_import(
        db, weights, conf, imgsz, scope=scope, targets=targets,
        import_classes=import_classes, source=source,
        device=device, progress_base=3, progress_span=96,
        progress_callback=progress_callback)
    if progress_callback:
        progress_callback("Listo", 100)
    return TrainResult(weights=weights, names=names,
                       pred_images=n_imgs, pred_boxes=n_boxes)
