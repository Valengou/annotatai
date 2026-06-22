# AGENTS.md — Guía para agentes de IA

Este archivo describe la arquitectura y convenciones del proyecto para que un agente de IA pueda trabajar en él de forma efectiva.

## Cómo correr la app

```bash
# Desde Label_studio/ con el venv activo
python -m app.main
```

## Stack

| Capa | Tecnología |
|------|-----------|
| GUI | PySide6 (Qt6) |
| Embeddings | backends intercambiables: DINOv3/SigLIP (transformers), OpenCLIP (open-clip-torch), DINOv2 (torch.hub), ConvNeXt (timm) |
| Clustering | umap-learn + hdbscan |
| Auto-etiquetado | ultralytics — SAM 3 (texto + máscaras) y YOLOE (prompt visual cross-image) |
| Visualización | matplotlib |
| Persistencia | SQLite (WAL mode, FK ON) |

### Backends de embeddings (`core/embedding_backends.py`)

`create_embedding_backend(name, checkpoint_path=None, batch_size=32)` devuelve un `ImageEmbeddingBackend` con `.name` y `.embed_paths(paths, progress_callback)` (vectores L2-normalizados). Nombres válidos en `available_backend_names()`: `openclip`, `dinov2`, `dinov3` (default en la UI), `siglip`, `convnext_checkpoint`. El nombre del backend se guarda como `model` en la tabla `embeddings`, así que embeddings de distintos modelos coexisten y `generate_project_embeddings` solo regenera los faltantes para ese modelo. DINOv3 concatena CLS + media de patches (saltando los register tokens) vía `dinov3_cls_patch_embedding`.

## Arquitectura

```
app/
├── models/      Dataclasses puras (sin dependencias Qt)
├── utils/       Constantes y helpers sin estado
├── core/        Lógica de negocio, workers, DB — sin Qt (excepto señales en workers)
└── ui/          Widgets Qt, señales, layouts
```

### Regla de dependencias
`models` ← `utils` ← `core` ← `ui`. Nunca importar `ui` desde `core`.

## Base de datos (`core/database.py`)

Todas las operaciones van por `Database`. No ejecutar SQL directo desde la UI.

Tablas clave:
- `images` — path, filename, status (`pending`/`reviewed`/`discarded`), cluster_id
- `classes` — id, name, color (hex)
- `annotations` — bbox en coordenadas normalizadas (0-1), class_id, source
- `embeddings` — vector como blob float32 + columna `model` (nombre del backend)
- `annotation_polygons` — polígono de segmentación (`points_json`) 1:1 con una anotación, FK cascade
- `projections` — coordenadas UMAP 2D por imagen
- `bbox_embeddings` / `bbox_projections` — igual pero por anotación

Las coordenadas de anotaciones **siempre son normalizadas** (x, y, width, height en 0–1 relativo al tamaño de imagen). Los puntos del polígono también van normalizados (x, y en 0–1).

## Manifiestos de dataset (`core/project_manifest.py`)

`open_project_from_manifest(dataset.json)` permite trabajar un dataset sin copiar imágenes: el manifiesto JSON declara `image_paths` y/o `image_roots` (relativos al manifiesto o absolutos) y el proyecto SQLite se crea/abre junto a él (`.annotatai`). Se accede desde `Archivo → Abrir dataset.json...`.

## Dataset Analysis Lab (`core/dataset_analysis.py` + `ui/analysis_lab.py`)

`run_dataset_analysis(db, project_path, AnalysisOptions)` corre embeddings (backend a elección) + UMAP + HDBSCAN sobre el proyecto, exporta a un `run_dir`: CSV de asignaciones/resumen, reporte HTML y clusters de revisión. Si se pasa un CSV de predicciones, calcula tipos de error contra las clases verdaderas (`prediction_error_type`).

## Workers (`ui/main_window.py`)

Las tareas pesadas corren en `QThread` con el patrón worker:

```python
worker = XxxWorker(db)
thread = QThread()
worker.moveToThread(thread)
thread.started.connect(worker.run)
worker.finished.connect(thread.quit)
thread.start()
```

Workers existentes: `IndexWorker`, `EmbeddingWorker` (recibe `backend_name`), `ClusterWorker`, `BBoxEmbedWorker`, `AnalysisLabWorker`, `AutoLabelWorker` (recibe `save_polygons`).

## Editor de anotaciones (`ui/annotation_editor.py`)

- `AnnotationScene` — `QGraphicsScene` con lógica de dibujo y selección de bboxes
- `BoundingBoxItem` — bbox con 8 handles de resize
- `AnnotationEditor` — widget principal: maneja DB, clases, undo/redo

### Undo/redo
Antes de cualquier cambio destructivo llamar `self._push_undo()`. El stack guarda snapshots de `_pending_annotations` como tuplas simples. Al restaurar se recarga la escena completa.

### Cambio de clase en bbox
- Click derecho en canvas → `AnnotationScene.class_change_requested` → `AnnotationEditor._show_class_menu_for_box`
- Click derecho en lista → `_ann_context_menu` → submenú "Cambiar clase"
- Ambos llaman `_apply_class_change(ann, (class_id, name, color))`

## Gestión de clases (`ui/dialogs.py` + `core/database.py`)

- `ClassManagerDialog` — crear (nombre + color picker), editar color (doble click), eliminar (con confirmación si hay anotaciones)
- Después de cerrar el diálogo, `main_window.manage_classes()` refresca los tres consumidores:
  1. `_annotation_editor._refresh_classes()`
  2. `_image_grid.set_classes(classes)`
  3. `_graph_view.set_classes(classes)`

Si se agrega un nuevo consumidor de clases, incluirlo en ese método.

## Convenciones

- **Sin comentarios obvios** — los nombres de variables y métodos ya explican qué hace el código.
- **Sin manejo de errores defensivo** — validar en los bordes (UI, input de usuario), no en la lógica interna.
- **No crear abstracciones prematuras** — preferir código directo sobre helpers genéricos.
- Los colores de clase se guardan y usan siempre como strings hex (`"#FF0000"`).
- El campo `source` de una anotación puede ser `"human"`, `"yolo"`, `"suggested"` o `"propagated"`.
- Auto-etiquetado (`core/auto_label.py`): `run_auto_label` (SAM 3 por texto, con `save_polygons` opcional para persistir máscaras) y `run_auto_label_visual` (YOLOE, usa las cajas humanas de una clase como prompt visual y transfiere cross-image). Las máscaras se devuelven como polígonos normalizados y se guardan vía `db.insert_annotation(..., polygon=...)`.

## Archivos que NO modificar sin cuidado

| Archivo | Razón |
|---------|-------|
| `core/database.py` → `SCHEMA` | Cambios de schema no se migran automáticamente en proyectos existentes |
| `models/annotation.py` → `from_db_row` | El orden de columnas debe coincidir exactamente con el SELECT en `get_annotations_for_image` (incluye el `LEFT JOIN annotation_polygons` en el índice 10) |
| `ui/main_window.py` → workers | Cada worker tiene un ciclo de vida QThread estricto; romper el orden de señales puede dejar threads colgados |
