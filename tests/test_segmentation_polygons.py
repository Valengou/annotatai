import tempfile
import unittest
from pathlib import Path

from app.core.auto_label import AutoLabeler
from app.core.database import Database
from app.models.annotation import Annotation


class _FakeTensor:
    def __init__(self, value):
        self._value = value

    def tolist(self):
        return self._value


class _FakeBoxes:
    xywhn = _FakeTensor([[0.5, 0.5, 0.4, 0.2]])
    conf = _FakeTensor([0.75])


class _FakeMasks:
    xyn = [
        _FakeTensor(
            [
                [0.3, 0.4],
                [0.7, 0.4],
                [0.7, 0.6],
                [0.3, 0.6],
            ]
        )
    ]


class _FakeResult:
    boxes = _FakeBoxes()
    masks = _FakeMasks()


class SegmentationPolygonTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.db = Database(self.root / "project.db")
        self.db.connect()

    def tearDown(self):
        self.db.close()
        self._tmpdir.cleanup()

    def test_annotation_round_trips_polygon_from_database(self):
        image_id = self.db.insert_image(str(self.root / "image.jpg"), "image.jpg", 100, 50, None)
        class_id = self.db.get_or_create_class("object", "#FF0000")
        polygon = [(0.1, 0.2), (0.4, 0.2), (0.3, 0.5)]
        ann_id = self.db.insert_annotation(
            image_id, class_id, 0.1, 0.2, 0.3, 0.3,
            source="suggested", confidence=0.9, polygon=polygon,
        )

        rows = self.db.get_annotations_for_image(image_id)
        ann = Annotation.from_db_row(rows[0])

        self.assertEqual(ann.id, ann_id)
        self.assertEqual(ann.polygon, polygon)

    def test_sam3_detection_can_include_polygon_points(self):
        detections = AutoLabeler._extract_detections([_FakeResult()], include_polygons=True)

        self.assertEqual(len(detections), 1)
        x, y, w, h, conf, polygon = detections[0]
        self.assertAlmostEqual(x, 0.3)
        self.assertAlmostEqual(y, 0.4)
        self.assertAlmostEqual(w, 0.4)
        self.assertAlmostEqual(h, 0.2)
        self.assertEqual(conf, 0.75)
        self.assertEqual(
            polygon,
            [(0.3, 0.4), (0.7, 0.4), (0.7, 0.6), (0.3, 0.6)],
        )


if __name__ == "__main__":
    unittest.main()
