from pathlib import Path
import tempfile
import unittest

from app.ui.image_grid import resolve_thumbnail_source
from app.utils.paths import thumbnail_path_for


class ImageGridThumbnailTests(unittest.TestCase):
    def test_resolve_thumbnail_source_prefers_existing_thumbnail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.jpg"
            thumbnail = root / "thumb.jpg"
            image.write_bytes(b"image")
            thumbnail.write_bytes(b"thumb")

            self.assertEqual(
                resolve_thumbnail_source(str(thumbnail), str(image)),
                str(thumbnail),
            )

    def test_resolve_thumbnail_source_falls_back_to_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.jpg"
            image.write_bytes(b"image")

            self.assertEqual(
                resolve_thumbnail_source(str(root / "missing.jpg"), str(image)),
                str(image),
            )

    def test_thumbnail_path_for_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project.annotatai"
            image = root / "image.jpg"
            image.write_bytes(b"image")

            self.assertEqual(
                thumbnail_path_for(project, image),
                thumbnail_path_for(project, image),
            )


if __name__ == "__main__":
    unittest.main()
