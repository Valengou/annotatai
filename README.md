# AnnotatAI — Anotación Visual Inteligente

Herramienta **local** de anotación de imágenes con agrupación automática por similitud visual (embeddings de visión + UMAP + HDBSCAN) y un **loop de etiquetado asistido** completo: descubrís objetos por diccionario, los pre-etiquetás con IA, validás, entrenás un modelo propio y dejás que active-learning priorice lo dudoso. Alternativa liviana a Label Studio, pensada para datasets de drones / infraestructura.

## Características

### Proyectos y datos
- Crear/abrir proyectos con persistencia en SQLite (WAL, FK cascade)
- Abrir un dataset desde un manifiesto `dataset.json` (`image_paths` / `image_roots`) sin copiar imágenes
- Cargar carpetas de imágenes (thumbnails automáticos)
- Importar anotaciones existentes: COCO, YOLO, LabelMe
- Exportar a YOLO y COCO

### Embeddings, agrupación y exploración
- **Embeddings visuales con múltiples backends**:
  - **DINOv3** ViT-S/16 (por defecto, vía `transformers`)
  - **OpenCLIP** ViT-B/32 · **DINOv2** ViT-S/14 · **SigLIP** base
  - **SigLIP2** (texto + imagen, para búsqueda semántica)
  - **ConvNeXt** desde checkpoint propio (timm)
- Agrupación automática UMAP + HDBSCAN; panel lateral de grupos
- Vista UMAP 2D interactiva (zoom, pan, lazo, click para abrir)
- Grilla con filtros por grupo / clase / estado / confianza, que **reagrupa los resultados de forma contigua**
- Vista de outliers por similitud de bboxes (UMAP sobre crops)
- **Dataset Analysis Lab**: embeddings + UMAP + HDBSCAN + reporte HTML y análisis de errores contra un CSV de predicciones

### Editor de anotaciones
- Bounding boxes con clases personalizables (nombre + color), Undo/Redo, cambio de clase
- **SAM click-to-box**: un click sobre un objeto y **SAM2** propone la caja ajustada (tecla `S`)
- Navegación con flechas y flujo **"revisar y continuar"**: al marcar Revisada/Descartar avanza solo a la siguiente
- Validación rápida de sugerencias de IA (aceptar/rechazar)

### Etiquetado asistido (el loop)
Todo desde **Herramientas → "Auto-etiquetado / Modelos..." (Ctrl+L)** — un solo diálogo elige **motor** y **alcance** (imagen actual / grupo / pendientes / todas):
- **SAM 3 (texto)**: open-vocabulary, escribís conceptos y los detecta (bbox + máscara opcional). Requiere `sam3.pt` (acceso aprobado en HuggingFace).
- **YOLOE (visual)**: usa tus cajas humanas de una clase como ejemplo y detecta similares cross-image.
- **Entrenar YOLO nano**: fine-tunea `yolo11n`/`yolov8n` con tus imágenes **revisadas** (clases seleccionables con su conteo) y pre-etiqueta las pendientes.
- **Modelo .pt existente**: reusa cualquier `.pt` (de este u otro proyecto); mapea clases por nombre, creándolas si faltan.

Más herramientas para cerrar el loop:
- **Imágenes duplicadas (near-duplicates)**: agrupa frames casi idénticos con los embeddings y deja 1 representante por grupo (descartar duplicados / propagar etiquetas).
- **Búsqueda semántica (SigLIP2)**: barra 🔎 en la grilla — buscás por texto ("aislador roto", "torre de alta tensión") o por imagen de ejemplo (clic derecho → "Buscar similares"); los resultados se ordenan por similitud.
- **Descubrimiento por diccionario (SigLIP2 → SAM 3)**: editás un diccionario de conceptos; SigLIP rankea las top-K imágenes candidatas por concepto y SAM 3 las pre-etiqueta. Solo validás.
- **Active learning — priorizar revisión**: auto-acepta sugerencias de alta confianza, auto-revisa imágenes sin dudosas, y ordena la grilla por incertidumbre (revisás primero lo más ambiguo).
- **Botón Detener**: cancela cualquier tarea larga (cooperativo, entre pasos).

## Instalación

```bash
git clone https://github.com/Valengou/annotatai.git
cd annotatai
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
pip install -r requirements.txt
```

> **GPU (recomendado)**: instalá PyTorch con CUDA **antes** del resto (`pip install torch` solo trae CPU):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> En GPUs de 8 GB (p. ej. RTX 4070 Laptop) usá **tamaño de imagen 640** para SAM/YOLO; 1024 puede agotar la VRAM.

Ejecutar:
```bash
python -m app.main
```

## Flujo recomendado (loop de etiquetado)

1. **Nuevo Proyecto** y **Cargar Carpeta de Imágenes**.
2. **Generar Embeddings** (DINOv3) → **Agrupar por Similitud**.
3. *(opcional)* **Near-duplicates** para no etiquetar frames repetidos.
4. **Generar índice de búsqueda (SigLIP2)** → **Descubrimiento por diccionario**: SigLIP encuentra candidatas y SAM 3 las pre-etiqueta.
5. **Validás** las sugerencias en el editor (aceptar/rechazar, o SAM click para ajustar).
6. **Entrenar YOLO nano** con lo revisado → pre-etiqueta el resto.
7. **Active learning**: auto-acepta lo confiable y prioriza lo dudoso. Repetís 5–7.
8. **Exportar Dataset** (YOLO / COCO).

## Atajos de teclado — Editor de Anotaciones

| Tecla | Acción | Tecla | Acción |
|-------|--------|-------|--------|
| `D` | Modo dibujo | `S` | SAM click-to-box |
| `A` | Aceptar sugerencias | `F` | Ajustar a la vista |
| `R` | Marcar revisada | `X` | Descartar |
| `←` / `→` | Imagen anterior / siguiente | `Del` | Borrar bbox |
| `Ctrl+S` | Guardar | `Ctrl+Z` / `Ctrl+Y` | Deshacer / Rehacer |
| Rueda | Zoom | Botón medio | Pan |
| Click derecho en bbox | Cambiar clase | | |

## Estructura del código

```
app/
├── main.py
├── models/                      # Dataclasses: ImageItem, Annotation, Cluster
├── utils/                       # config, paths, thumbnails
├── core/
│   ├── database.py              # SQLite (imágenes, embeddings, clusters, clases,
│   │                            #  anotaciones, polígonos, search_embeddings, bbox_*)
│   ├── project.py / project_manifest.py
│   ├── image_indexer.py
│   ├── embeddings.py / embedding_backends.py   # OpenCLIP/DINOv2/DINOv3/SigLIP/SigLIP2/ConvNeXt
│   ├── clustering.py            # UMAP + HDBSCAN
│   ├── bbox_embeddings.py / dataset_analysis.py
│   ├── auto_label.py            # SAM 3 (texto+máscaras) y YOLOE (visual)
│   ├── yolo_trainer.py          # Entrenar YOLO nano + predecir/importar
│   ├── near_duplicates.py       # Frames casi-duplicados
│   ├── semantic_search.py       # Búsqueda SigLIP2 (texto/imagen)
│   ├── concept_discovery.py     # Diccionario → SigLIP → SAM 3 → validar
│   ├── active_learning.py       # Auto-aceptar + priorizar por incertidumbre
│   ├── interactive_sam.py       # SAM2 click-to-box
│   ├── annotations.py / loaders.py
└── ui/
    ├── main_window.py           # Ventana principal + QThread workers (cancelables)
    ├── image_grid.py            # Grilla + filtros + barra de búsqueda 🔎
    ├── graph_view.py            # Scatter UMAP interactivo (lazo)
    ├── annotation_editor.py     # Editor bbox + SAM click + undo/redo + polígonos
    ├── bbox_view.py / cluster_panel.py / analysis_lab.py
    └── dialogs.py               # Diálogos (proyecto, clases, export/import, hub de
                                 #  auto-etiquetado, near-dups, active learning,
                                 #  búsqueda, descubrimiento)
```

Tests en `tests/` (pytest): backends de embeddings, embeddings de proyecto, manifiesto, polígonos de segmentación, thumbnails y análisis de dataset.

## Estructura de un proyecto creado

```
mi_proyecto/
├── project.json     # Metadatos (nombre, versión, diccionario de descubrimiento)
├── project.db       # SQLite
├── thumbnails/
├── models/          # Modelos YOLO entrenados (assist/best.pt) y datasets de entrenamiento
└── exports/
```

## Modelos y dónde se guardan

- **DINOv3 / SigLIP2 / SAM3**: cache del HuggingFace Hub (`~/.cache/huggingface/hub`). Se puede mover con `HF_HOME`.
- **DINOv2**: `torch.hub` (`~/.cache/torch/hub`).
- **SAM2 / YOLO base**: se descargan vía ultralytics al usarlos.
- **YOLO entrenado por vos**: `projects/<proyecto>/models/assist/best.pt`.
