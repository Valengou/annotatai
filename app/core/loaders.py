"""
Loaders for existing annotation formats: COCO JSON, YOLO, LabelMe.

All loaders return a LoadResult with counters and warnings so the
caller can surface the outcome to the user without knowing format details.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class LoadResult:
    images_matched: int = 0
    images_skipped: int = 0
    annotations_loaded: int = 0
    classes_created: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Imágenes con anotaciones: {self.images_matched}",
            f"Imágenes no encontradas en proyecto: {self.images_skipped}",
            f"Anotaciones cargadas: {self.annotations_loaded}",
            f"Clases nuevas creadas: {self.classes_created}",
        ]
        if self.warnings:
            lines.append(f"\nAdvertencias ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                lines.append(f"  • {w}")
            if len(self.warnings) > 10:
                lines.append(f"  ... y {len(self.warnings) - 10} más")
        return "\n".join(lines)


# ─────────────────────────────── helpers ───────────────────────────────

def _build_filename_index(db) -> dict[str, tuple[int, int, int]]:
    """Return {filename_lower: (image_id, width, height)} for all images in DB."""
    rows = db.get_all_images()
    index: dict[str, tuple[int, int, int]] = {}
    for row in rows:
        img_id, path, filename, width, height = row[0], row[1], row[2], row[3], row[4]
        index[filename.lower()] = (img_id, width or 0, height or 0)
        # also index by stem in case annotations omit extension
        stem = Path(filename).stem.lower()
        if stem not in index:
            index[stem] = (img_id, width or 0, height or 0)
    return index


def _get_or_create_class(db, name: str, result: LoadResult) -> int:
    existing = {row[1]: row[0] for row in db.get_all_classes()}
    if name in existing:
        return existing[name]
    class_id = db.get_or_create_class(name)
    result.classes_created += 1
    return class_id


def _insert_annotation(db, image_id: int, class_id: int,
                        x: float, y: float, w: float, h: float,
                        source: str = "loaded") -> bool:
    """Clamp coords to [0,1] and skip degenerate boxes."""
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    if w < 0.001 or h < 0.001:
        return False
    db.insert_annotation(image_id, class_id, x, y, w, h, source=source)
    return True


# ─────────────────────────────── COCO ───────────────────────────────

def load_coco(json_path: Path, db,
              overwrite: bool = False,
              progress_callback: Callable | None = None) -> LoadResult:
    """Load COCO detection JSON into the project DB.

    Matches images by file_name (basename). Skips images not indexed yet.
    bbox format: [x_min, y_min, width, height] in pixels.
    """
    result = LoadResult()

    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)

    categories: dict[int, str] = {
        cat["id"]: cat["name"] for cat in coco.get("categories", [])
    }
    if not categories:
        result.warnings.append("No se encontraron categorías en el JSON.")
        return result

    # coco image_id → (db_image_id, width, height)
    filename_index = _build_filename_index(db)
    coco_to_db: dict[int, tuple[int, int, int]] = {}

    for coco_img in coco.get("images", []):
        fname = Path(coco_img["file_name"]).name.lower()
        stem = Path(fname).stem
        entry = filename_index.get(fname) or filename_index.get(stem)
        if entry:
            w = coco_img.get("width") or entry[1] or 1
            h = coco_img.get("height") or entry[2] or 1
            coco_to_db[coco_img["id"]] = (entry[0], w, h)
        else:
            result.images_skipped += 1

    if overwrite:
        for db_id, _, _ in coco_to_db.values():
            db.delete_annotations_for_image(db_id)

    class_cache: dict[str, int] = {}
    total_anns = len(coco.get("annotations", []))

    for i, ann in enumerate(coco.get("annotations", [])):
        if progress_callback:
            progress_callback(i + 1, total_anns)

        coco_img_id = ann.get("image_id")
        if coco_img_id not in coco_to_db:
            continue

        cat_name = categories.get(ann.get("category_id", -1))
        if not cat_name:
            continue

        if cat_name not in class_cache:
            class_cache[cat_name] = _get_or_create_class(db, cat_name, result)
        class_id = class_cache[cat_name]

        db_img_id, img_w, img_h = coco_to_db[coco_img_id]
        if img_w == 0 or img_h == 0:
            result.warnings.append(f"Imagen ID {db_img_id}: dimensiones desconocidas, saltando.")
            continue

        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue
        bx, by, bw, bh = bbox
        x_norm = bx / img_w
        y_norm = by / img_h
        w_norm = bw / img_w
        h_norm = bh / img_h

        if _insert_annotation(db, db_img_id, class_id, x_norm, y_norm, w_norm, h_norm):
            result.annotations_loaded += 1

    result.images_matched = len(coco_to_db)
    return result


# ─────────────────────────────── YOLO ───────────────────────────────

def load_yolo(labels_dir: Path, db,
              names_file: Path | None = None,
              overwrite: bool = False,
              progress_callback: Callable | None = None) -> LoadResult:
    """Load YOLO .txt annotation files.

    Expects labels_dir/*.txt where each line is: class_idx cx cy w h (normalized).
    names_file should be classes.txt / obj.names listing class names one per line.
    If omitted, class names default to "class_0", "class_1", etc.
    """
    result = LoadResult()

    # Build class index
    class_names: list[str] = []
    if names_file and names_file.exists():
        class_names = [
            line.strip() for line in names_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        # try common locations relative to labels_dir
        for candidate in [
            labels_dir.parent / "classes.txt",
            labels_dir.parent / "obj.names",
            labels_dir.parent / "data" / "obj.names",
            labels_dir / "classes.txt",
        ]:
            if candidate.exists():
                class_names = [
                    l.strip() for l in candidate.read_text(encoding="utf-8").splitlines()
                    if l.strip()
                ]
                break

    filename_index = _build_filename_index(db)
    txt_files = sorted(labels_dir.glob("*.txt"))
    total = len(txt_files)

    class_cache: dict[int, int] = {}  # yolo_idx → db class_id

    for i, txt_path in enumerate(txt_files):
        if progress_callback:
            progress_callback(i + 1, total)

        stem = txt_path.stem.lower()
        entry = filename_index.get(stem)
        if not entry:
            # try with common extensions
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                entry = filename_index.get(stem + ext)
                if entry:
                    break
        if not entry:
            result.images_skipped += 1
            continue

        db_img_id, img_w, img_h = entry

        if overwrite:
            db.delete_annotations_for_image(db_img_id)

        lines = txt_path.read_text(encoding="utf-8").splitlines()
        matched_this = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                yolo_idx = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                result.warnings.append(f"{txt_path.name}: línea mal formada: '{line}'")
                continue

            if yolo_idx not in class_cache:
                if yolo_idx < len(class_names):
                    name = class_names[yolo_idx]
                else:
                    name = f"class_{yolo_idx}"
                    if result.classes_created < 3:
                        result.warnings.append(
                            f"Clase {yolo_idx} no encontrada en nombres, usando '{name}'"
                        )
                class_cache[yolo_idx] = _get_or_create_class(db, name, result)

            class_id = class_cache[yolo_idx]
            # YOLO uses center coords → convert to top-left
            x = cx - bw / 2
            y = cy - bh / 2

            if _insert_annotation(db, db_img_id, class_id, x, y, bw, bh):
                result.annotations_loaded += 1
                matched_this = True

        if matched_this:
            result.images_matched += 1

    return result


# ─────────────────────────────── LabelMe ───────────────────────────────

def load_labelme(json_dir: Path, db,
                 overwrite: bool = False,
                 progress_callback: Callable | None = None) -> LoadResult:
    """Load LabelMe JSON annotation files (one .json per image).

    Supports shape types: rectangle, polygon (uses bounding rect of polygon).
    """
    result = LoadResult()
    filename_index = _build_filename_index(db)
    json_files = sorted(json_dir.glob("*.json"))
    total = len(json_files)
    class_cache: dict[str, int] = {}

    for i, json_path in enumerate(json_files):
        if progress_callback:
            progress_callback(i + 1, total)

        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            result.warnings.append(f"{json_path.name}: error al leer: {e}")
            continue

        img_path = data.get("imagePath", "")
        fname = Path(img_path).name.lower()
        stem = Path(fname).stem

        entry = filename_index.get(fname) or filename_index.get(stem)
        if not entry:
            result.images_skipped += 1
            continue

        db_img_id, img_w, img_h = entry
        img_w = data.get("imageWidth") or img_w or 1
        img_h = data.get("imageHeight") or img_h or 1

        if overwrite:
            db.delete_annotations_for_image(db_img_id)

        matched_this = False
        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip()
            if not label:
                continue
            shape_type = shape.get("shape_type", "rectangle")
            points = shape.get("points", [])

            if shape_type == "rectangle" and len(points) == 2:
                (x1, y1), (x2, y2) = points[0], points[1]
            elif shape_type in ("polygon", "rectangle") and len(points) >= 2:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            else:
                result.warnings.append(
                    f"{json_path.name}: shape type '{shape_type}' no soportado"
                )
                continue

            x_norm = min(x1, x2) / img_w
            y_norm = min(y1, y2) / img_h
            w_norm = abs(x2 - x1) / img_w
            h_norm = abs(y2 - y1) / img_h

            if label not in class_cache:
                class_cache[label] = _get_or_create_class(db, label, result)
            class_id = class_cache[label]

            if _insert_annotation(db, db_img_id, class_id, x_norm, y_norm, w_norm, h_norm):
                result.annotations_loaded += 1
                matched_this = True

        if matched_this:
            result.images_matched += 1

    return result
