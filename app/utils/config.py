from pathlib import Path

THUMBNAIL_SIZE = (200, 200)
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

DEFAULT_CLASSES = [
    {"name": "object", "color": "#FF0000"},
]

CLUSTER_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#34495E", "#E91E63", "#00BCD4",
    "#8BC34A", "#FF5722", "#607D8B", "#795548", "#FFC107",
    "#673AB7", "#009688", "#FFEB3B", "#03A9F4", "#4CAF50",
]

STATUS_COLORS = {
    "pending": "#808080",
    "reviewed": "#2ECC71",
    "discarded": "#E74C3C",
}
