import json
import shutil
from pathlib import Path
from ..models.annotation import Annotation
from ..models.image_item import ImageItem


def export_yolo(db, output_dir: Path, image_filter=None):
    """Export annotations in YOLO format with images. Returns (n_images, n_annotations) exported."""
    classes_rows = db.get_all_classes()
    if not classes_rows:
        return 0, 0

    class_name_to_index = {row[1]: i for i, row in enumerate(classes_rows)}
    names_file = output_dir / "classes.txt"
    names_file.write_text("\n".join(r[1] for r in classes_rows), encoding="utf-8")

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    all_images = db.get_all_images()
    if image_filter:
        all_images = [r for r in all_images if image_filter(r)]

    n_images = 0
    n_annotations = 0

    for row in all_images:
        img_id = row[0]
        img_path = Path(row[1])
        ann_rows = db.get_annotations_for_image(img_id)

        if not ann_rows:
            continue

        anns = [Annotation.from_db_row(r) for r in ann_rows]
        label_lines = []
        for ann in anns:
            idx = class_name_to_index.get(ann.class_name)
            if idx is None:
                continue
            label_lines.append(ann.to_yolo_line(idx))
            n_annotations += 1

        if label_lines:
            label_file = labels_dir / (img_path.stem + ".txt")
            label_file.write_text("\n".join(label_lines), encoding="utf-8")
            if img_path.exists():
                shutil.copy2(img_path, images_dir / img_path.name)
            n_images += 1

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {output_dir}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: {len(classes_rows)}\n"
        f"names: {[r[1] for r in classes_rows]}\n",
        encoding="utf-8",
    )

    return n_images, n_annotations


def export_coco(db, output_dir: Path, image_filter=None):
    """Export annotations in COCO JSON format with images copied to images/ subfolder."""
    classes_rows = db.get_all_classes()
    categories = [
        {"id": i + 1, "name": row[1], "supercategory": "object"}
        for i, row in enumerate(classes_rows)
    ]
    class_name_to_coco_id = {row[1]: i + 1 for i, row in enumerate(classes_rows)}

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    all_images = db.get_all_images()
    if image_filter:
        all_images = [r for r in all_images if image_filter(r)]

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for row in all_images:
        img_id = row[0]
        img_path = Path(row[1])
        width = row[3] or 1
        height = row[4] or 1
        ann_rows = db.get_annotations_for_image(img_id)

        if not ann_rows:
            continue

        if img_path.exists():
            shutil.copy2(img_path, images_dir / img_path.name)

        coco_images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        for ann_row in ann_rows:
            ann = Annotation.from_db_row(ann_row)
            cat_id = class_name_to_coco_id.get(ann.class_name)
            if cat_id is None:
                continue
            px = ann.x * width
            py = ann.y * height
            pw = ann.width * width
            ph = ann.height * height
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [px, py, pw, ph],
                "area": pw * ph,
                "iscrowd": 0,
            })
            ann_id += 1

    coco_data = {
        "info": {"description": "Exported from AnnotatAI"},
        "categories": categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }
    out_file = output_dir / "annotations.json"
    out_file.write_text(json.dumps(coco_data, indent=2), encoding="utf-8")
    return len(coco_images), len(coco_annotations)
