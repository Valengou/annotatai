import numpy as np
from pathlib import Path
from typing import Callable
from ..utils.config import CLIP_MODEL, CLIP_PRETRAINED


class EmbeddingGenerator:
    def __init__(self, model_name: str = CLIP_MODEL, pretrained: str = CLIP_PRETRAINED):
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = "cpu"

    def load(self):
        import torch
        import open_clip
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._model = self._model.to(self._device)
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def embed_image(self, image_path: Path) -> np.ndarray:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        return self.embed_image_pil(img)

    def embed_image_pil(self, img) -> np.ndarray:
        import torch
        tensor = self._preprocess(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    def embed_text(self, text: str) -> np.ndarray:
        import torch
        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    @staticmethod
    def vector_to_bytes(vector: np.ndarray) -> bytes:
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_vector(data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)

    def generate_all(self, db, progress_callback: Callable | None = None):
        """Generate embeddings for all images not yet processed."""
        rows = db.get_all_images()
        total = len(rows)
        done = 0

        for row in rows:
            img_id = row[0]
            img_path = Path(row[1])

            if db.has_embedding(img_id):
                done += 1
                if progress_callback:
                    progress_callback(done, total)
                continue

            try:
                vec = self.embed_image(img_path)
                db.save_embedding(img_id, self.vector_to_bytes(vec), self.model_name)
                db.mark_embedding_ready(img_id)
            except Exception:
                pass

            done += 1
            if progress_callback:
                progress_callback(done, total)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def generate_project_embeddings(
    db,
    backend,
    model_name: str | None = None,
    progress_callback: Callable | None = None,
) -> int:
    rows = db.get_all_images()
    total = len(rows)
    target_model = model_name or backend.name
    pending = [row for row in rows if not db.has_embedding(row[0], target_model)]

    if not pending:
        if progress_callback:
            progress_callback(total, total)
        return 0

    image_paths = [Path(row[1]) for row in pending]

    def on_backend_progress(done: int, backend_total: int):
        if progress_callback:
            progress_callback(done, backend_total)

    vectors = backend.embed_paths(image_paths, progress_callback=on_backend_progress)
    for row, vector in zip(pending, vectors):
        image_id = row[0]
        db.save_embedding(
            image_id,
            np.asarray(vector, dtype=np.float32).tobytes(),
            target_model,
        )
        db.mark_embedding_ready(image_id)

    if progress_callback:
        progress_callback(total, total)
    return len(pending)
