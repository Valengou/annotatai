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
| Embeddings | open-clip-torch (ViT-B/32) |
| Clustering | umap-learn + hdbscan |
| Detección (opcional) | ultralytics (YOLOv8) |
| Visualización | matplotlib |
| Persistencia | SQLite (WAL mode, FK ON) |

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
- `embeddings` — vector CLIP como blob float32
- `projections` — coordenadas UMAP 2D por imagen
- `bbox_embeddings` / `bbox_projections` — igual pero por anotación

Las coordenadas de anotaciones **siempre son normalizadas** (x, y, width, height en 0–1 relativo al tamaño de imagen).

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

Workers existentes: `IndexWorker`, `EmbeddingWorker`, `ClusterWorker`, `BBoxEmbedWorker`.

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

## Archivos que NO modificar sin cuidado

| Archivo | Razón |
|---------|-------|
| `core/database.py` → `SCHEMA` | Cambios de schema no se migran automáticamente en proyectos existentes |
| `models/annotation.py` → `from_db_row` | El orden de columnas debe coincidir exactamente con el SELECT en `get_annotations_for_image` |
| `ui/main_window.py` → workers | Cada worker tiene un ciclo de vida QThread estricto; romper el orden de señales puede dejar threads colgados |
