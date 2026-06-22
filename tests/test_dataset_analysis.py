import unittest
from pathlib import Path

from app.core.dataset_analysis import (
    build_metadata_index,
    merge_assignment_metadata,
    prediction_error_type,
    summarize_assignments,
)
from app.core.embedding_backends import available_backend_names, create_embedding_backend


class DatasetAnalysisTests(unittest.TestCase):
    def test_prediction_error_type_marks_correct_and_errors(self):
        self.assertEqual(
            prediction_error_type("not_bent_insulator", "not_bent_insulator"),
            "correct",
        )
        self.assertEqual(
            prediction_error_type("not_bent_insulator", "Polymerdirty"),
            "not_bent_insulator_pred_Polymerdirty",
        )
        self.assertEqual(prediction_error_type("Bent_Insulator", ""), "unpredicted")

    def test_build_metadata_index_matches_path_filename_and_source_path(self):
        image_path = Path(self._tmpdir.name) / "img_001.jpg"
        source_path = Path(self._tmpdir.name) / "source" / "original.jpg"
        rows = [
            {
                "path": str(image_path),
                "filename": image_path.name,
                "source_path": str(source_path),
                "true_class": "not_bent_insulator",
            }
        ]

        index = build_metadata_index(rows)

        self.assertEqual(index[str(image_path.resolve())]["true_class"], "not_bent_insulator")
        self.assertEqual(index[image_path.name]["true_class"], "not_bent_insulator")
        self.assertEqual(index[str(source_path.resolve())]["true_class"], "not_bent_insulator")

    def test_merge_assignment_metadata_computes_error_fields(self):
        image_path = Path(self._tmpdir.name) / "crop.jpg"
        assignments = [
            {
                "image_id": 7,
                "path": str(image_path),
                "filename": image_path.name,
                "cluster_id": 3,
                "cluster_name": "Grupo 3",
            }
        ]
        metadata = [
            {
                "filename": image_path.name,
                "subset": "holdout_08",
                "true_class": "not_bent_insulator",
                "pred_class": "Polymerdirty",
            }
        ]

        merged = merge_assignment_metadata(assignments, metadata)

        self.assertEqual(merged[0]["subset"], "holdout_08")
        self.assertEqual(
            merged[0]["error_type"],
            "not_bent_insulator_pred_Polymerdirty",
        )
        self.assertIs(merged[0]["is_error"], True)

    def test_summarize_assignments_prioritizes_false_positive_clusters(self):
        rows = [
            {
                "cluster_id": 1,
                "cluster_name": "Grupo 1",
                "subset": "holdout_08",
                "true_class": "not_bent_insulator",
                "pred_class": "Polymerdirty",
                "error_type": "not_bent_insulator_pred_Polymerdirty",
                "status": "pending",
            },
            {
                "cluster_id": 1,
                "cluster_name": "Grupo 1",
                "subset": "holdout_08",
                "true_class": "Polymerdirty",
                "pred_class": "Polymerdirty",
                "error_type": "correct",
                "status": "reviewed",
            },
            {
                "cluster_id": 1,
                "cluster_name": "Grupo 1",
                "subset": "holdout_08",
                "true_class": "not_bent_insulator",
                "pred_class": "not_bent_insulator",
                "error_type": "correct",
                "status": "pending",
            },
            {
                "cluster_id": 2,
                "cluster_name": "Grupo 2",
                "subset": "holdout_08",
                "true_class": "not_bent_insulator",
                "pred_class": "Bent_Insulator",
                "error_type": "not_bent_insulator_pred_Bent_Insulator",
                "status": "pending",
            },
            {
                "cluster_id": None,
                "cluster_name": "Sin cluster",
                "subset": "train_context",
                "true_class": "not_bent_insulator",
                "pred_class": "",
                "error_type": "train_context",
                "status": "discarded",
            },
        ]

        summaries = summarize_assignments(rows)

        self.assertEqual(summaries[0]["cluster_id"], "1")
        self.assertEqual(summaries[0]["total"], 3)
        self.assertEqual(summaries[0]["error_count"], 1)
        self.assertEqual(summaries[0]["normal_pred_polymerdirty"], 1)
        self.assertEqual(summaries[0]["normal_pred_bent"], 0)
        self.assertEqual(summaries[0]["true_dirty_holdout"], 1)
        self.assertGreater(summaries[0]["review_priority"], summaries[1]["review_priority"])

    def test_embedding_backend_factory_lists_reusable_backends(self):
        names = available_backend_names()
        self.assertIn("openclip", names)
        self.assertIn("dinov2", names)
        self.assertIn("siglip", names)
        self.assertIn("convnext_checkpoint", names)
        with self.assertRaises(ValueError):
            create_embedding_backend("does_not_exist")

    def setUp(self):
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
