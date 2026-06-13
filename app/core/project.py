import json
from pathlib import Path
from .database import Database
from ..utils.paths import project_db_path, project_thumbnails_dir, project_embeddings_dir, project_exports_dir
from ..utils.config import DEFAULT_CLASSES


class Project:
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.db = Database(project_db_path(project_path))
        self._meta: dict = {}

    @property
    def name(self) -> str:
        return self._meta.get("name", self.project_path.name)

    @classmethod
    def create(cls, parent_dir: Path, name: str) -> "Project":
        project_path = parent_dir / name
        project_path.mkdir(parents=True, exist_ok=True)

        project_thumbnails_dir(project_path)
        project_embeddings_dir(project_path)
        project_exports_dir(project_path)

        meta = {"name": name, "version": "1.0"}
        (project_path / "project.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        proj = cls(project_path)
        proj._meta = meta
        proj.db.connect()

        for cls_def in DEFAULT_CLASSES:
            proj.db.get_or_create_class(cls_def["name"], cls_def["color"])

        return proj

    @classmethod
    def open(cls, project_path: Path) -> "Project":
        meta_file = project_path / "project.json"
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        proj = cls(project_path)
        proj._meta = meta
        proj.db.connect()

        project_thumbnails_dir(project_path)
        project_embeddings_dir(project_path)
        project_exports_dir(project_path)

        return proj

    def close(self):
        self.db.close()

    def save_meta(self):
        (self.project_path / "project.json").write_text(
            json.dumps(self._meta, indent=2), encoding="utf-8"
        )

    @staticmethod
    def is_valid_project(path: Path) -> bool:
        return (path / "project.json").exists() and (path / "project.db").exists()
