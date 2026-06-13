from pathlib import Path


def project_thumbnails_dir(project_path: Path) -> Path:
    d = project_path / "thumbnails"
    d.mkdir(parents=True, exist_ok=True)
    return d


def project_embeddings_dir(project_path: Path) -> Path:
    d = project_path / "embeddings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def project_exports_dir(project_path: Path) -> Path:
    d = project_path / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def project_db_path(project_path: Path) -> Path:
    return project_path / "project.db"


def thumbnail_path_for(project_path: Path, image_path: Path) -> Path:
    stem = image_path.stem
    thumbs_dir = project_thumbnails_dir(project_path)
    return thumbs_dir / f"{stem}_{hash(str(image_path)) & 0xFFFFFF:06x}.jpg"
