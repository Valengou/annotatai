from pathlib import Path
from hashlib import sha1


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


def project_models_dir(project_path: Path) -> Path:
    d = project_path / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def project_db_path(project_path: Path) -> Path:
    return project_path / "project.db"


def thumbnail_path_for(project_path: Path, image_path: Path) -> Path:
    stem = image_path.stem
    thumbs_dir = project_thumbnails_dir(project_path)
    digest = sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return thumbs_dir / f"{stem}_{digest}.jpg"
