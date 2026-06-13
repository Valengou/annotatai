from pathlib import Path
from PIL import Image
from .config import THUMBNAIL_SIZE


def generate_thumbnail(image_path: Path, thumbnail_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
            img = img.convert("RGB")
            img.save(thumbnail_path, "JPEG", quality=85, optimize=True)
        return True
    except Exception:
        return False


def get_image_size(image_path: Path) -> tuple[int, int]:
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return (0, 0)
