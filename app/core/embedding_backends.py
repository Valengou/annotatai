from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..utils.config import CLIP_MODEL, CLIP_PRETRAINED


BatchProgress = Callable[[int, int], None]


def available_backend_names() -> list[str]:
    return ["openclip", "dinov2", "dinov3", "siglip", "siglip2", "convnext_checkpoint"]


def create_embedding_backend(
    name: str,
    checkpoint_path: Path | None = None,
    batch_size: int = 32,
) -> "ImageEmbeddingBackend":
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"clip", "open_clip", "openclip"}:
        return OpenCLIPBackend(batch_size=batch_size)
    if normalized in {"dinov2", "dino_v2", "dino"}:
        return DinoV2Backend(batch_size=batch_size)
    if normalized in {"dinov3", "dino_v3"}:
        return DinoV3Backend(batch_size=batch_size)
    if normalized in {"siglip"}:
        return SigLIPBackend(batch_size=batch_size)
    if normalized in {"siglip2", "siglip_2"}:
        return SigLIP2Backend(batch_size=batch_size)
    if normalized in {"convnext", "convnext_checkpoint", "convnext_features"}:
        return ConvNeXtCheckpointBackend(checkpoint_path=checkpoint_path, batch_size=batch_size)
    raise ValueError(f"Unknown embedding backend: {name}")


class ImageEmbeddingBackend:
    name = "base"
    supports_text = False   # True si el backend tiene encoder de texto (búsqueda)

    def __init__(self, batch_size: int = 32):
        self.batch_size = max(1, int(batch_size))
        self._loaded = False
        self._device = "cpu"

    def load(self) -> None:
        self._loaded = True

    def embed_paths(self, paths: Iterable[Path], progress_callback: BatchProgress | None = None) -> np.ndarray:
        raise NotImplementedError

    def embed_text(self, texts: Iterable[str]) -> np.ndarray:
        """Embeddings de texto, normalizados, en el mismo espacio que las imágenes."""
        raise NotImplementedError(f"El backend '{self.name}' no soporta texto.")

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()


class OpenCLIPBackend(ImageEmbeddingBackend):
    name = "openclip"

    def __init__(self, batch_size: int = 32, model_name: str = CLIP_MODEL, pretrained: str = CLIP_PRETRAINED):
        super().__init__(batch_size=batch_size)
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None

    def load(self) -> None:
        import torch
        import open_clip

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        self._model = self._model.to(self._device)
        self._model.eval()
        self._loaded = True

    def embed_paths(self, paths: Iterable[Path], progress_callback: BatchProgress | None = None) -> np.ndarray:
        import torch

        self._ensure_loaded()
        path_list = [Path(path) for path in paths]
        features = []
        with torch.no_grad():
            for done, batch_paths in _batched(path_list, self.batch_size):
                batch = _load_tensor_batch(batch_paths, self._preprocess).to(self._device)
                batch_features = self._model.encode_image(batch)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                features.append(batch_features.cpu().numpy())
                if progress_callback:
                    progress_callback(min(done + len(batch_paths), len(path_list)), len(path_list))
        return np.concatenate(features, axis=0)


class DinoV2Backend(ImageEmbeddingBackend):
    name = "dinov2"

    def __init__(self, batch_size: int = 32, model_name: str = "dinov2_vits14"):
        super().__init__(batch_size=batch_size)
        self.model_name = model_name
        self._model = None
        self._preprocess = None

    def load(self) -> None:
        import torch
        from torchvision import transforms

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        except Exception as exc:
            raise RuntimeError(
                "Could not load DINOv2 via torch.hub. Check internet access or pre-download "
                "facebookresearch/dinov2."
            ) from exc
        self._model = self._model.to(self._device)
        self._model.eval()
        self._preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self._loaded = True

    def embed_paths(self, paths: Iterable[Path], progress_callback: BatchProgress | None = None) -> np.ndarray:
        import torch

        self._ensure_loaded()
        path_list = [Path(path) for path in paths]
        features = []
        with torch.no_grad():
            for done, batch_paths in _batched(path_list, self.batch_size):
                batch = _load_tensor_batch(batch_paths, self._preprocess).to(self._device)
                batch_features = self._model(batch)
                if isinstance(batch_features, dict):
                    if "x_norm_clstoken" in batch_features:
                        batch_features = batch_features["x_norm_clstoken"]
                    else:
                        batch_features = next(iter(batch_features.values()))
                batch_features = _flatten_features(batch_features)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                features.append(batch_features.cpu().numpy())
                if progress_callback:
                    progress_callback(min(done + len(batch_paths), len(path_list)), len(path_list))
        return np.concatenate(features, axis=0)


class DinoV3Backend(ImageEmbeddingBackend):
    name = "dinov3"

    def __init__(
        self,
        batch_size: int = 32,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    ):
        super().__init__(batch_size=batch_size)
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._num_register_tokens = 4

    def load(self) -> None:
        import torch

        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "DINOv3 backend requires transformers. Install it in the Label_studio venv "
                "with: .venv\\Scripts\\python.exe -m pip install transformers"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(
            self.model_name,
            attn_implementation="sdpa",
        ).to(self._device)
        self._model.eval()
        self._num_register_tokens = int(getattr(self._model.config, "num_register_tokens", 4))
        self._loaded = True

    def embed_paths(self, paths: Iterable[Path], progress_callback: BatchProgress | None = None) -> np.ndarray:
        import torch

        self._ensure_loaded()
        path_list = [Path(path) for path in paths]
        features = []
        with torch.no_grad():
            for done, batch_paths in _batched(path_list, self.batch_size):
                images = [_load_image(path) for path in batch_paths]
                inputs = self._processor(images=images, return_tensors="pt").to(self._device)
                outputs = self._model(**inputs)
                batch_features = dinov3_cls_patch_embedding(
                    outputs.last_hidden_state,
                    num_register_tokens=self._num_register_tokens,
                )
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                features.append(batch_features.cpu().numpy())
                if progress_callback:
                    progress_callback(min(done + len(batch_paths), len(path_list)), len(path_list))
        return np.concatenate(features, axis=0)


class SigLIPBackend(ImageEmbeddingBackend):
    name = "siglip"

    def __init__(self, batch_size: int = 32, model_name: str = "google/siglip-base-patch16-224"):
        super().__init__(batch_size=batch_size)
        self.model_name = model_name
        self._model = None
        self._processor = None

    def load(self) -> None:
        import torch

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "SigLIP backend requires transformers. Install it in the Label_studio venv "
                "with: .venv\\Scripts\\python.exe -m pip install transformers"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
        self._model.eval()
        self._loaded = True

    def embed_paths(self, paths: Iterable[Path], progress_callback: BatchProgress | None = None) -> np.ndarray:
        import torch

        self._ensure_loaded()
        path_list = [Path(path) for path in paths]
        features = []
        with torch.no_grad():
            for done, batch_paths in _batched(path_list, self.batch_size):
                images = [_load_image(path) for path in batch_paths]
                inputs = self._processor(images=images, return_tensors="pt").to(self._device)
                if hasattr(self._model, "get_image_features"):
                    batch_features = self._model.get_image_features(**inputs)
                else:
                    outputs = self._model(**inputs)
                    batch_features = outputs.pooler_output
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                features.append(batch_features.cpu().numpy())
                if progress_callback:
                    progress_callback(min(done + len(batch_paths), len(path_list)), len(path_list))
        return np.concatenate(features, axis=0)


class SigLIP2Backend(ImageEmbeddingBackend):
    """SigLIP2 con encoder de texto + imagen, para búsqueda semántica."""
    name = "siglip2"
    supports_text = True

    def __init__(self, batch_size: int = 32,
                 model_name: str = "google/siglip2-base-patch16-224"):
        super().__init__(batch_size=batch_size)
        self.model_name = model_name
        self._model = None
        self._processor = None

    def load(self) -> None:
        import torch

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "SigLIP2 requiere transformers. Instalalo en el venv con: "
                ".venv\\Scripts\\python.exe -m pip install -U transformers"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
        self._model.eval()
        self._loaded = True

    @staticmethod
    def _as_tensor(out):
        """get_image_features/get_text_features pueden devolver un tensor o un
        objeto ModelOutput según la versión de transformers; extrae el vector."""
        import torch
        if torch.is_tensor(out):
            return out
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(out, attr, None)
            if v is not None:
                return v
        lhs = getattr(out, "last_hidden_state", None)
        if lhs is not None:
            return lhs.mean(dim=1)
        raise RuntimeError("SigLIP2: no se pudo extraer el vector de features")

    def embed_paths(self, paths: Iterable[Path], progress_callback: BatchProgress | None = None) -> np.ndarray:
        import torch

        self._ensure_loaded()
        path_list = [Path(path) for path in paths]
        features = []
        with torch.no_grad():
            for done, batch_paths in _batched(path_list, self.batch_size):
                images = [_load_image(path) for path in batch_paths]
                inputs = self._processor(images=images, return_tensors="pt").to(self._device)
                batch_features = self._as_tensor(self._model.get_image_features(**inputs))
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                features.append(batch_features.cpu().numpy())
                if progress_callback:
                    progress_callback(min(done + len(batch_paths), len(path_list)), len(path_list))
        return np.concatenate(features, axis=0)

    def embed_text(self, texts: Iterable[str]) -> np.ndarray:
        import torch

        self._ensure_loaded()
        text_list = [str(t) for t in texts]
        with torch.no_grad():
            inputs = self._processor(
                text=text_list, padding="max_length", truncation=True,
                return_tensors="pt",
            ).to(self._device)
            feats = self._as_tensor(self._model.get_text_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats.cpu().numpy()


class ConvNeXtCheckpointBackend(ImageEmbeddingBackend):
    name = "convnext_checkpoint"

    def __init__(self, checkpoint_path: Path | None, batch_size: int = 32, image_size: int = 224):
        super().__init__(batch_size=batch_size)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.image_size = image_size
        self.class_names: list[str] = []
        self._model = None
        self._preprocess = None

    def load(self) -> None:
        import torch
        from torchvision import transforms

        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "ConvNeXt checkpoint backend requires timm. Install it with: pip install timm"
            ) from exc

        if not self.checkpoint_path or not self.checkpoint_path.is_file():
            raise RuntimeError("ConvNeXt backend needs a valid checkpoint_path.")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(self.checkpoint_path, map_location=self._device, weights_only=False)
        self.class_names = list(checkpoint.get("class_names") or [])
        num_classes = int(checkpoint.get("num_classes") or len(self.class_names) or 1)
        self._model = timm.create_model("convnext_tiny", pretrained=False, num_classes=num_classes)
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        self._model.load_state_dict(state_dict, strict=True)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._preprocess = transforms.Compose(
            [
                transforms.Resize(int(self.image_size * 256 / 224)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self._loaded = True

    def embed_paths(self, paths: Iterable[Path], progress_callback: BatchProgress | None = None) -> np.ndarray:
        import torch

        self._ensure_loaded()
        path_list = [Path(path) for path in paths]
        features = []
        with torch.no_grad():
            for done, batch_paths in _batched(path_list, self.batch_size):
                batch = _load_tensor_batch(batch_paths, self._preprocess).to(self._device)
                feature_map = self._model.forward_features(batch)
                if hasattr(self._model, "forward_head"):
                    batch_features = self._model.forward_head(feature_map, pre_logits=True)
                else:
                    batch_features = _flatten_features(feature_map)
                batch_features = _flatten_features(batch_features)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                features.append(batch_features.cpu().numpy())
                if progress_callback:
                    progress_callback(min(done + len(batch_paths), len(path_list)), len(path_list))
        return np.concatenate(features, axis=0)


def _batched(items: list[Path], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def _load_tensor_batch(paths: list[Path], preprocess):
    import torch

    tensors = [preprocess(_load_image(path)) for path in paths]
    return torch.stack(tensors, dim=0)


def _load_image(path: Path):
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Could not load image: {path}") from exc


def _flatten_features(features):
    if features.ndim == 4:
        features = features.mean(dim=(2, 3))
    elif features.ndim > 2:
        features = features.flatten(start_dim=1)
    return features


def dinov3_cls_patch_embedding(last_hidden_state, num_register_tokens: int = 4):
    import torch

    cls_token = last_hidden_state[:, 0, :]
    patch_start = 1 + int(num_register_tokens)
    patch_mean = last_hidden_state[:, patch_start:, :].mean(dim=1)
    return torch.cat((cls_token, patch_mean), dim=1)
