import json
from pathlib import Path

from .image_indexer import index_image_paths, iter_images
from .project import Project


def project_path_for_manifest(manifest_path: Path) -> Path:
    manifest_path = Path(manifest_path)
    return manifest_path.with_suffix(".annotatai")


def resolve_manifest_image_paths(manifest_path: Path) -> list[Path]:
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("El manifiesto debe ser un objeto JSON.")

    base_dir = manifest_path.parent
    image_paths = _read_path_list(data, "image_paths")
    image_roots = _read_path_list(data, "image_roots")
    if not image_paths and not image_roots:
        raise ValueError("El manifiesto debe incluir image_paths o image_roots.")

    resolved: list[Path] = []
    for value in image_paths:
        path = _resolve_path(base_dir, value)
        if not path.is_file():
            raise ValueError(f"No se encontró la imagen: {path}")
        resolved.append(path.resolve())

    for value in image_roots:
        root = _resolve_path(base_dir, value)
        if not root.is_dir():
            raise ValueError(f"No se encontró la carpeta de imágenes: {root}")
        resolved.extend(path.resolve() for path in iter_images(root))

    return sorted(set(resolved))


def open_project_from_manifest(manifest_path: Path) -> Project:
    manifest_path = Path(manifest_path)
    image_paths = resolve_manifest_image_paths(manifest_path)
    project_path = project_path_for_manifest(manifest_path)

    if Project.is_valid_project(project_path):
        project = Project.open(project_path)
    else:
        project = Project.create(project_path.parent, project_path.name)

    try:
        index_image_paths(image_paths, project.db, project.project_path)
    except Exception:
        project.close()
        raise

    return project


def _read_path_list(data: dict, key: str) -> list[str]:
    value = data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} debe ser una lista de strings.")
    return value


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path
