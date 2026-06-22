# Dataset Manifest Design

## Goal

Allow AnnotatAI projects to reference source images through a manifest file without copying the image files into the project directory.

## Current Behavior

Projects already store each image's original path in the `images.path` column. Loading a folder indexes image paths, reads dimensions, and generates thumbnails under the project directory. The original images remain in their source folder.

The missing piece is a portable, explicit manifest that can create or refresh the indexed image list without selecting the folder manually in the UI.

## Manifest Format

Use a separate `dataset.json` file instead of adding image paths to `project.json`.

```json
{
  "image_roots": [
    "C:/datasets/flight/rgb"
  ],
  "image_paths": [
    "C:/datasets/flight/rgb/DJI_0001.JPG"
  ]
}
```

Both keys are optional, but at least one must contain entries. `image_roots` recursively discovers supported image extensions. `image_paths` lists individual image files.

Relative paths are resolved from the directory containing the manifest file. Absolute paths are used as-is.

## Project Opening

Opening a normal project directory keeps the current behavior.

Opening a manifest file creates or opens a project directory next to the manifest using the manifest filename stem. For example, `C:/datasets/foo/dataset.json` opens `C:/datasets/foo/dataset.annotatai/`. The project stores database, thumbnails, embeddings, exports, and annotations there while images remain in their original locations.

If a project already exists, opening the same manifest refreshes the image index idempotently: existing image rows remain, new paths are inserted, thumbnails are generated when missing, and existing annotations/statuses are preserved.

## Core API

Add a small manifest module under `app/core` responsible for:

- Reading `dataset.json`.
- Resolving absolute image paths.
- Expanding folder roots through the existing image extension rules.
- Creating/opening the sidecar project directory.
- Calling existing indexing logic.

The UI should not parse JSON directly.

## UI Behavior

`Archivo -> Abrir Proyecto` should accept either:

- A project directory containing `project.json` and `project.db`.
- A `dataset.json` manifest file.

The command-line path passed to `python -m app.main <path>` should support the same two cases.

## Validation And Errors

Invalid manifests should raise clear `ValueError` messages before opening the project:

- JSON is not an object.
- Neither `image_roots` nor `image_paths` contains entries.
- Listed image file does not exist.
- Listed root directory does not exist.

Unsupported files under valid roots are ignored through existing extension filtering.

## Tests

Add unit tests for manifest parsing and project indexing:

- Resolves relative `image_paths` from the manifest directory.
- Expands `image_roots` using existing supported extensions.
- Creates a sidecar project and indexes images without copying source images.
- Re-opening the same manifest does not duplicate image rows.

Run the focused test file, then the full existing test suite.
