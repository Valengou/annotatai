"""Identifica un recorte con CLIP zero-shot (open-vocabulary).

Recorta una región de una imagen y la compara contra una lista de etiquetas
candidatas usando OpenCLIP (el mismo del proyecto). Útil para nombrar objetos
desconocidos que SAM 3 segmentó sin clase.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.embeddings import EmbeddingGenerator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IMG = Path(sys.argv[1])
# caja relativa (x0,y0,x1,y1) en 0-1; default = región central de la bolsa
box = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.30, 0.48, 0.60, 0.74]

CANDIDATES = [
    "a plastic bag", "a piece of cloth or fabric", "a tarp",
    "trash or debris", "a goat", "a dog", "a sheep", "a cow",
    "a person", "a rock", "a bush or shrub", "bare dirt ground",
    "an oil spill stain", "a metal pipe", "a tire", "a cardboard box",
]


def main():
    im = Image.open(IMG).convert("RGB")
    W, H = im.size
    crop = im.crop((int(box[0]*W), int(box[1]*H), int(box[2]*W), int(box[3]*H)))
    out = Path("analysis_out/sam3_everything/_crop_identify.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out, quality=90)

    gen = EmbeddingGenerator()
    gen.load()
    img_vec = gen.embed_image_pil(crop)
    text_vecs = np.stack([gen.embed_text(f"a photo of {c}") for c in CANDIDATES])
    sims = text_vecs @ img_vec
    # softmax para legibilidad
    e = np.exp((sims - sims.max()) * 100)
    probs = e / e.sum()
    order = np.argsort(-sims)
    print(f"Recorte: {out}  ({crop.size[0]}x{crop.size[1]} px)")
    print("Ranking CLIP zero-shot:")
    for i in order[:8]:
        print(f"  {probs[i]*100:5.1f}%  sim={sims[i]:.3f}  {CANDIDATES[i]}")


if __name__ == "__main__":
    main()
