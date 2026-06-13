# AnnotatAI — Anotación Visual Inteligente

Herramienta **local** de anotación de imágenes con agrupación automática por similitud visual usando CLIP + UMAP + HDBSCAN. Alternativa liviana a Label Studio enfocada en datasets de drones / infraestructura eléctrica.

## Características

- Crear/abrir proyectos con persistencia en SQLite
- Cargar carpetas de imágenes (genera thumbnails automáticamente)
- Embeddings visuales con CLIP (ViT-B/32)
- Agrupación automática: UMAP + HDBSCAN
- Vista de grilla con filtros por cluster y clase
- Vista UMAP 2D interactiva (zoom, pan, click para abrir)
- Editor de bounding boxes con clases personalizables (nombre + color)
- **Undo/Redo** en el editor de anotaciones (Ctrl+Z / Ctrl+Y)
- **Cambio de clase** sobre bboxes existentes (click derecho en el canvas o en la lista)
- **Gestor de clases**: crear, cambiar color y eliminar etiquetas con propagación inmediata
- Carga de anotaciones existentes: COCO, YOLO, LabelMe
- Exportación a YOLO y COCO
- Vista de outliers por similitud de bboxes (UMAP sobre crops)

## Instalación

### 1. Clonar

```bash
git clone https://github.com/<usuario>/annotatai.git
cd annotatai
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **GPU (opcional)**: Instalar PyTorch con CUDA antes del resto para acelerar CLIP:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Ejecutar

```bash
python -m app.main
```

## Flujo básico

1. **Nuevo Proyecto**: `Archivo → Nuevo Proyecto` → nombre + carpeta de destino
2. **Cargar imágenes**: botón `Cargar Carpeta de Imágenes`
3. **Generar embeddings**: `Generar Embeddings CLIP` (descarga el modelo la primera vez ~340 MB)
4. **Agrupar**: `Agrupar por Similitud (UMAP + HDBSCAN)`
5. **Explorar**: panel izquierdo para filtrar por grupo; tab "Vista UMAP" para el mapa 2D
6. **Administrar clases**: `Herramientas → Administrar Clases`
7. **Anotar**: doble click en imagen → Editor de Anotaciones
8. **Exportar**: `Exportar Dataset`

## Atajos de teclado — Editor de Anotaciones

| Tecla | Acción |
|-------|--------|
| `D` | Activar/desactivar modo dibujo |
| `F` | Ajustar imagen a la vista |
| `Ctrl+S` | Guardar anotaciones |
| `Ctrl+Z` | Deshacer |
| `Ctrl+Y` | Rehacer |
| `Delete` | Borrar bbox seleccionado |
| `←` / `→` | Imagen anterior / siguiente |
| `R` | Marcar como revisada |
| `X` | Descartar imagen |
| Rueda del mouse | Zoom |
| Botón del medio | Pan |
| Click derecho en bbox | Cambiar clase |

## Formatos de exportación

- **YOLO**: `labels/*.txt` + `classes.txt` + `data.yaml`
- **COCO**: `annotations.json`

## Estructura del código

```
app/
├── main.py                     # Punto de entrada, tema oscuro
├── models/
│   ├── image_item.py           # Dataclass ImageItem
│   ├── annotation.py           # Dataclass Annotation (bbox, export YOLO)
│   └── cluster.py              # Dataclass Cluster
├── utils/
│   ├── config.py               # Constantes globales y clases por defecto
│   ├── paths.py                # Helpers de rutas de proyecto
│   └── thumbnails.py           # Generación de thumbnails
├── core/
│   ├── database.py             # SQLite (WAL, FK cascade)
│   ├── project.py              # Crear/abrir proyectos
│   ├── image_indexer.py        # Escaneo + indexado de carpetas
│   ├── embeddings.py           # CLIP embeddings (imagen completa)
│   ├── bbox_embeddings.py      # CLIP embeddings sobre crops de bbox
│   ├── clustering.py           # UMAP + HDBSCAN
│   ├── annotations.py          # Exportadores YOLO / COCO
│   └── loaders.py              # Importadores COCO / YOLO / LabelMe
└── ui/
    ├── main_window.py          # Ventana principal + QThread workers
    ├── image_grid.py           # Grilla de thumbnails con filtros
    ├── graph_view.py           # Scatter plot UMAP interactivo
    ├── annotation_editor.py    # Editor bbox + undo/redo
    ├── bbox_view.py            # Vista de outliers por bbox
    ├── cluster_panel.py        # Panel lateral de grupos
    └── dialogs.py              # Diálogos (proyecto, clases, export, import)
```

## Estructura de un proyecto creado

```
mi_proyecto/
├── project.json     # Metadatos (nombre, versión)
├── project.db       # SQLite: imágenes, embeddings, clusters, clases, anotaciones
├── thumbnails/      # Miniaturas 200×200
├── embeddings/      # (reservado)
└── exports/         # Exports generados
```
