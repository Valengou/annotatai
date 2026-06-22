"""SAM 3 en modo 'segmentar todo' (automatic mask generator, sin prompt).

Corre sobre una imagen, devuelve una máscara por cada objeto saliente (class-agnostic)
y dibuja el resultado. Sirve para descubrir objetos desconocidos (no en tus clases).
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.utils.config import DEFAULT_SAM3_MODEL

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IMG = Path(sys.argv[1]) if len(sys.argv) > 1 else None
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
OUT = Path("analysis_out/sam3_everything")


def main():
    from ultralytics import SAM

    model = SAM(DEFAULT_SAM3_MODEL)
    print(f"Imagen: {IMG.name}  | imgsz={IMGSZ}")
    results = model(str(IMG), imgsz=IMGSZ, verbose=False)
    r = results[0]

    OUT.mkdir(parents=True, exist_ok=True)
    im = Image.open(IMG).convert("RGB")
    W, H = im.size
    scale = 1800 / max(W, H)
    im = im.resize((int(W * scale), int(H * scale)))
    d = ImageDraw.Draw(im, "RGBA")
    w, h = im.size

    n = 0
    if r.masks is not None and r.masks.xyn is not None:
        polys = r.masks.xyn
        n = len(polys)
        palette = [(255, 80, 80), (0, 229, 255), (124, 252, 0), (255, 215, 0),
                   (186, 85, 211), (255, 140, 0), (30, 144, 255), (255, 105, 180)]
        for i, poly in enumerate(polys):
            pts = [(float(px) * w, float(py) * h) for px, py in poly]
            if len(pts) < 3:
                continue
            c = palette[i % len(palette)]
            d.polygon(pts, outline=(*c, 255), fill=(*c, 60))
    im.save(OUT / f"{IMG.stem}_everything.jpg", quality=82)

    # tamaño de cada máscara (en px de imagen original) para ver chicos vs grandes
    sizes = []
    if r.masks is not None and r.masks.data is not None:
        for m in r.masks.data.cpu().numpy():
            sizes.append(int(m.sum()))
    print(f"Máscaras totales: {n}")
    if sizes:
        sizes.sort()
        print(f"Área (px máscara a {IMGSZ}): min={sizes[0]} mediana={sizes[len(sizes)//2]} max={sizes[-1]}")
        print(f"Máscaras chicas (<500 px): {sum(1 for s in sizes if s < 500)}")
    print(f"Preview: {OUT / (IMG.stem + '_everything.jpg')}")


if __name__ == "__main__":
    main()
