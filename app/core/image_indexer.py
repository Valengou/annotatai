from pathlib import Path
from typing import Generator
from ..utils.config import IMAGE_EXTENSIONS
from ..utils.thumbnails import generate_thumbnail, get_image_size
from ..utils.paths import thumbnail_path_for
from .database import Database


def iter_images(folder: Path) -> Generator[Path, None, None]:
    for ext in IMAGE_EXTENSIONS:
        yield from folder.rglob(f"*{ext}")
        yield from folder.rglob(f"*{ext.upper()}")


def index_images(folder: Path, db: Database, project_path: Path,
                 progress_callback=None) -> list[int]:
    """Index all images in folder, generate thumbnails, save to DB.
    Returns list of inserted image IDs."""
    image_paths = sorted(set(iter_images(folder)))
    return index_image_paths(image_paths, db, project_path, progress_callback)


def index_image_paths(image_paths: list[Path], db: Database, project_path: Path,
                      progress_callback=None) -> list[int]:
    """Index image paths, generate thumbnails, save to DB.
    Returns list of image IDs."""
    total = len(image_paths)
    inserted_ids = []

    for i, img_path in enumerate(image_paths):
        try:
            thumb_path = thumbnail_path_for(project_path, img_path)
            if not thumb_path.exists():
                generate_thumbnail(img_path, thumb_path)

            w, h = get_image_size(img_path)
            img_id = db.insert_image(
                path=str(img_path),
                filename=img_path.name,
                width=w,
                height=h,
                thumbnail_path=str(thumb_path) if thumb_path.exists() else None,
            )
            inserted_ids.append(img_id)
        except Exception:
            pass

        if progress_callback:
            progress_callback(i + 1, total)

    return inserted_ids
