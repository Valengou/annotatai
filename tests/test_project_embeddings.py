import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.database import Database
from app.core.embeddings import generate_project_embeddings


class FakeBackend:
    name = "fake_backend"

    def __init__(self):
        self.seen_paths = []

    def embed_paths(self, paths, progress_callback=None):
        self.seen_paths = [Path(path) for path in paths]
        rows = []
        total = len(self.seen_paths)
        for i, _ in enumerate(self.seen_paths):
            rows.append(np.array([i + 1, i + 2], dtype=np.float32))
            if progress_callback:
                progress_callback(i + 1, total)
        return np.vstack(rows)


class ProjectEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.db = Database(self.root / "project.db")
        self.db.connect()

    def tearDown(self):
        self.db.close()
        self._tmpdir.cleanup()

    def _insert_image(self, name: str) -> int:
        path = self.root / name
        Image.new("RGB", (16, 16), color=(20, 40, 60)).save(path)
        return self.db.insert_image(str(path), name, 16, 16, None)

    def test_database_counts_embeddings_by_model(self):
        first = self._insert_image("first.jpg")
        second = self._insert_image("second.jpg")
        self.db.save_embedding(first, np.array([1], dtype=np.float32).tobytes(), "openclip")
        self.db.save_embedding(second, np.array([2], dtype=np.float32).tobytes(), "dinov3")

        self.assertEqual(self.db.count_embeddings(), 2)
        self.assertEqual(self.db.count_embeddings("openclip"), 1)
        self.assertEqual(self.db.count_embeddings("dinov3"), 1)
        self.assertTrue(self.db.has_embedding(first, "openclip"))
        self.assertFalse(self.db.has_embedding(first, "dinov3"))

    def test_generate_project_embeddings_replaces_other_backend_vectors(self):
        first = self._insert_image("first.jpg")
        second = self._insert_image("second.jpg")
        self.db.save_embedding(first, np.array([99], dtype=np.float32).tobytes(), "openclip")
        progress = []

        written = generate_project_embeddings(
            self.db,
            FakeBackend(),
            model_name="dinov3",
            progress_callback=lambda done, total: progress.append((done, total)),
        )

        self.assertEqual(written, 2)
        self.assertEqual(self.db.count_embeddings("dinov3"), 2)
        self.assertEqual(self.db.count_embeddings("openclip"), 0)
        self.assertTrue(self.db.has_embedding(first, "dinov3"))
        self.assertTrue(self.db.has_embedding(second, "dinov3"))
        self.assertEqual(progress[-1], (2, 2))


if __name__ == "__main__":
    unittest.main()
