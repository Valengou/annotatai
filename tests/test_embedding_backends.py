import unittest

import torch

from app.core.embedding_backends import (
    DinoV3Backend,
    available_backend_names,
    create_embedding_backend,
    dinov3_cls_patch_embedding,
)


class EmbeddingBackendTests(unittest.TestCase):
    def test_dinov3_backend_is_available_from_factory(self):
        self.assertIn("dinov3", available_backend_names())
        backend = create_embedding_backend("dino-v3", batch_size=4)

        self.assertIsInstance(backend, DinoV3Backend)
        self.assertEqual(backend.batch_size, 4)
        self.assertEqual(backend.name, "dinov3")

    def test_dinov3_embedding_uses_cls_and_mean_patch_tokens_skipping_registers(self):
        hidden = torch.zeros((1, 8, 2), dtype=torch.float32)
        hidden[:, 0, :] = torch.tensor([10.0, 20.0])
        hidden[:, 1:5, :] = 1000.0
        hidden[:, 5:, :] = torch.tensor(
            [
                [1.0, 3.0],
                [3.0, 5.0],
                [5.0, 7.0],
            ]
        )

        features = dinov3_cls_patch_embedding(hidden)

        expected = torch.tensor([[10.0, 20.0, 3.0, 5.0]])
        self.assertTrue(torch.equal(features, expected))


if __name__ == "__main__":
    unittest.main()
