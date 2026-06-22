import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.core.project import Project
from app.core.project_manifest import (
    open_project_from_manifest,
    project_path_for_manifest,
    resolve_manifest_image_paths,
)


class ProjectManifestTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _image(self, path: Path, color=(120, 80, 40)) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), color=color).save(path)
        return path

    def _manifest(self, payload: dict) -> Path:
        path = self.root / "dataset.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_resolves_relative_image_paths_and_expands_roots(self):
        explicit = self._image(self.root / "single.png")
        rooted = self._image(self.root / "images" / "nested" / "flight.JPG")
        (self.root / "images" / "notes.txt").write_text("ignored", encoding="utf-8")
        manifest = self._manifest({
            "image_paths": ["single.png"],
            "image_roots": ["images"],
        })

        paths = resolve_manifest_image_paths(manifest)

        self.assertEqual(paths, [rooted.resolve(), explicit.resolve()])

    def test_opens_sidecar_project_without_copying_images_and_reopens_idempotently(self):
        first = self._image(self.root / "src" / "img_001.jpg")
        second = self._image(self.root / "src" / "img_002.jpg")
        manifest = self._manifest({"image_roots": ["src"]})
        sidecar = project_path_for_manifest(manifest)

        project = open_project_from_manifest(manifest)
        try:
            self.assertEqual(project.project_path, sidecar)
            self.assertEqual(project.db.count_images(), 2)
            rows = project.db.get_all_images()
            self.assertEqual({Path(row[1]).resolve() for row in rows}, {first.resolve(), second.resolve()})
            self.assertFalse((sidecar / first.name).exists())
            self.assertFalse((sidecar / second.name).exists())
        finally:
            project.close()

        project = open_project_from_manifest(manifest)
        try:
            self.assertEqual(project.db.count_images(), 2)
        finally:
            project.close()

    def test_project_classifies_dataset_manifest_files(self):
        manifest = self._manifest({"image_paths": []})

        self.assertTrue(Project.is_manifest_file(manifest))
        self.assertFalse(Project.is_manifest_file(self.root))


if __name__ == "__main__":
    unittest.main()
