from __future__ import annotations

import csv
import html
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..utils.config import (
    CLUSTER_COLORS,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
)


ProgressCallback = Callable[[str, int], None]

NORMAL_LABELS = {"normal", "not_bent", "not_bent_insulator", "not bent insulator"}
DIRTY_LABELS = {"dirty", "polymerdirty", "polymer_dirty", "sucio", "dirty_insulator"}
BENT_LABELS = {"bent", "bent_insulator", "bend", "bend_insulator"}

METADATA_KEY_FIELDS = (
    "path",
    "image_path",
    "project_image_path",
    "source_path",
    "filename",
    "file_name",
)


@dataclass(slots=True)
class AnalysisOptions:
    run_name: str = ""
    backend_name: str = "openclip"
    checkpoint_path: Path | None = None
    predictions_csv: Path | None = None
    metadata_csv: Path | None = None
    batch_size: int = 32
    umap_neighbors: int = UMAP_N_NEIGHBORS
    umap_min_dist: float = UMAP_MIN_DIST
    hdbscan_min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE
    hdbscan_min_samples: int = HDBSCAN_MIN_SAMPLES
    apply_to_project: bool = True
    export_review_clusters: bool = True
    top_review_clusters: int = 24


@dataclass(slots=True)
class AnalysisResult:
    run_dir: Path
    backend_name: str
    image_count: int
    cluster_count: int
    embeddings_path: Path
    assignments_csv: Path
    summary_csv: Path
    report_html: Path
    review_dir: Path | None = None


def prediction_error_type(true_class: str | None, pred_class: str | None) -> str:
    true_value = (true_class or "").strip()
    pred_value = (pred_class or "").strip()
    if not pred_value:
        return "unpredicted"
    if not true_value:
        return "unknown_truth"
    if true_value == pred_value:
        return "correct"
    return f"{true_value}_pred_{pred_value}"


def read_csv_rows(path: Path | None) -> list[dict]:
    if not path:
        return []
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_metadata_index(rows: Iterable[dict]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row in rows:
        for field in METADATA_KEY_FIELDS:
            value = (row.get(field) or "").strip()
            if not value:
                continue
            for key in _metadata_keys(value):
                index.setdefault(key, row)
    return index


def merge_assignment_metadata(assignments: Iterable[dict], metadata_rows: Iterable[dict]) -> list[dict]:
    metadata_index = build_metadata_index(metadata_rows)
    merged_rows = []
    for assignment in assignments:
        row = dict(assignment)
        metadata = _find_metadata(row, metadata_index)
        if metadata:
            for key, value in metadata.items():
                if value != "" and key not in {"path", "filename"}:
                    row[key] = value

        true_class = row.get("true_class") or row.get("label") or row.get("class") or ""
        pred_class = row.get("pred_class") or row.get("prediction") or row.get("predicted_class") or ""
        row["true_class"] = true_class
        row["pred_class"] = pred_class
        row["error_type"] = row.get("error_type") or prediction_error_type(true_class, pred_class)
        row["is_error"] = _is_error(row["error_type"])
        merged_rows.append(row)
    return merged_rows


def summarize_assignments(assignments: Iterable[dict]) -> list[dict]:
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in assignments:
        cluster_id = row.get("cluster_id")
        key = "none" if cluster_id in (None, "") else str(cluster_id)
        by_cluster[key].append(row)

    summaries = []
    for cluster_key, rows in by_cluster.items():
        subset_counts = Counter(_str(row.get("subset")) for row in rows if _str(row.get("subset")))
        class_counts = Counter(_str(row.get("true_class")) for row in rows if _str(row.get("true_class")))
        pred_counts = Counter(_str(row.get("pred_class")) for row in rows if _str(row.get("pred_class")))
        status_counts = Counter(_str(row.get("status")) for row in rows if _str(row.get("status")))
        error_counts = Counter(_str(row.get("error_type")) for row in rows if _str(row.get("error_type")))

        error_count = sum(1 for row in rows if _is_error(row.get("error_type")))
        normal_pred_dirty = sum(
            1 for row in rows
            if _is_normal(row.get("true_class")) and _is_dirty(row.get("pred_class"))
        )
        normal_pred_bent = sum(
            1 for row in rows
            if _is_normal(row.get("true_class")) and _is_bent(row.get("pred_class"))
        )
        true_dirty_holdout = sum(
            1 for row in rows
            if _is_dirty(row.get("true_class")) and "holdout" in _label_key(row.get("subset"))
        )
        dirty_train = sum(
            1 for row in rows
            if _is_dirty(row.get("true_class")) and "train" in _label_key(row.get("subset"))
        )
        normal_train = sum(
            1 for row in rows
            if _is_normal(row.get("true_class")) and "train" in _label_key(row.get("subset"))
        )
        bent_train = sum(
            1 for row in rows
            if _is_bent(row.get("true_class")) and "train" in _label_key(row.get("subset"))
        )
        review_priority = (
            error_count * 10
            + normal_pred_dirty * 3
            + normal_pred_bent * 3
            + true_dirty_holdout * 2
            + min(len(rows), 50) / 100.0
        )

        summaries.append(
            {
                "cluster_id": cluster_key,
                "cluster_name": rows[0].get("cluster_name") or ("Sin cluster" if cluster_key == "none" else f"Cluster {cluster_key}"),
                "total": len(rows),
                "error_count": error_count,
                "normal_pred_polymerdirty": normal_pred_dirty,
                "normal_pred_bent": normal_pred_bent,
                "true_dirty_holdout": true_dirty_holdout,
                "train_normal": normal_train,
                "train_dirty": dirty_train,
                "train_bent": bent_train,
                "review_priority": round(review_priority, 4),
                "subset_counts": json.dumps(dict(subset_counts), ensure_ascii=False, sort_keys=True),
                "class_counts": json.dumps(dict(class_counts), ensure_ascii=False, sort_keys=True),
                "pred_counts": json.dumps(dict(pred_counts), ensure_ascii=False, sort_keys=True),
                "status_counts": json.dumps(dict(status_counts), ensure_ascii=False, sort_keys=True),
                "error_counts": json.dumps(dict(error_counts), ensure_ascii=False, sort_keys=True),
            }
        )

    summaries.sort(key=lambda row: (row["review_priority"], row["total"]), reverse=True)
    return summaries


def run_dataset_analysis(db, project_path: Path, options: AnalysisOptions, progress_callback: ProgressCallback | None = None) -> AnalysisResult:
    rows = db.get_all_images()
    if len(rows) < 3:
        raise RuntimeError("At least 3 images are required for UMAP/HDBSCAN analysis.")

    run_name = safe_run_name(options.run_name or datetime.now().strftime("analysis_%Y%m%d_%H%M%S"))
    run_dir = Path(project_path) / "analysis_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    image_ids = [row[0] for row in rows]
    image_paths = [Path(row[1]) for row in rows]

    _progress(progress_callback, f"Loading {options.backend_name} backend...", 5)
    from .embedding_backends import create_embedding_backend

    backend = create_embedding_backend(
        options.backend_name,
        checkpoint_path=options.checkpoint_path,
        batch_size=options.batch_size,
    )

    _progress(progress_callback, "Generating embeddings...", 15)
    embeddings = backend.embed_paths(
        image_paths,
        progress_callback=lambda done, total: _progress(
            progress_callback,
            f"Embeddings: {done}/{total}",
            15 + int((done / max(total, 1)) * 45),
        ),
    )
    embeddings = _normalize_matrix(np.asarray(embeddings, dtype=np.float32))
    embeddings_path = run_dir / "embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        image_ids=np.asarray(image_ids, dtype=np.int64),
        paths=np.asarray([str(path) for path in image_paths]),
        embeddings=embeddings,
        backend=options.backend_name,
    )

    _progress(progress_callback, "Running UMAP...", 65)
    projections_2d = run_umap(
        embeddings,
        n_neighbors=options.umap_neighbors,
        min_dist=options.umap_min_dist,
    )

    _progress(progress_callback, "Running HDBSCAN...", 78)
    labels = run_hdbscan(
        projections_2d,
        min_cluster_size=options.hdbscan_min_cluster_size,
        min_samples=options.hdbscan_min_samples,
    )

    if options.apply_to_project:
        _progress(progress_callback, "Applying clusters to project...", 86)
        apply_clusters_to_db(db, image_ids, labels, projections_2d)
        assignments = assignments_from_db(db)
    else:
        assignments = assignments_from_rows(rows, labels, projections_2d)

    metadata_rows = []
    metadata_rows.extend(read_csv_rows(options.predictions_csv))
    metadata_rows.extend(read_csv_rows(options.metadata_csv))
    assignments = merge_assignment_metadata(assignments, metadata_rows)
    summaries = summarize_assignments(assignments)

    _progress(progress_callback, "Writing reports...", 94)
    assignments_csv = write_assignments_csv(run_dir / "cluster_assignments.csv", assignments)
    summary_csv = write_summary_csv(run_dir / "cluster_summary.csv", summaries)
    report_html = write_html_report(run_dir / "report.html", summaries, assignments, project_path)
    review_dir = None
    if options.export_review_clusters:
        review_dir = export_review_clusters(
            run_dir / "review_clusters",
            summaries,
            assignments,
            top_n=options.top_review_clusters,
        )

    summary = {
        "run_dir": str(run_dir),
        "backend_name": options.backend_name,
        "image_count": len(rows),
        "cluster_count": len({int(lbl) for lbl in labels if int(lbl) != -1}),
        "embeddings_path": str(embeddings_path),
        "assignments_csv": str(assignments_csv),
        "summary_csv": str(summary_csv),
        "report_html": str(report_html),
        "review_dir": str(review_dir) if review_dir else "",
        "options": {
            "run_name": run_name,
            "backend_name": options.backend_name,
            "checkpoint_path": str(options.checkpoint_path) if options.checkpoint_path else "",
            "predictions_csv": str(options.predictions_csv) if options.predictions_csv else "",
            "metadata_csv": str(options.metadata_csv) if options.metadata_csv else "",
            "batch_size": options.batch_size,
            "umap_neighbors": options.umap_neighbors,
            "umap_min_dist": options.umap_min_dist,
            "hdbscan_min_cluster_size": options.hdbscan_min_cluster_size,
            "hdbscan_min_samples": options.hdbscan_min_samples,
            "apply_to_project": options.apply_to_project,
        },
    }
    (run_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _progress(progress_callback, "Analysis complete", 100)

    return AnalysisResult(
        run_dir=run_dir,
        backend_name=options.backend_name,
        image_count=len(rows),
        cluster_count=summary["cluster_count"],
        embeddings_path=embeddings_path,
        assignments_csv=assignments_csv,
        summary_csv=summary_csv,
        report_html=report_html,
        review_dir=review_dir,
    )


def run_umap(embeddings: np.ndarray, n_neighbors: int = UMAP_N_NEIGHBORS, min_dist: float = UMAP_MIN_DIST) -> np.ndarray:
    import umap

    if len(embeddings) < 3:
        raise RuntimeError("UMAP needs at least 3 embeddings.")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(max(2, n_neighbors), len(embeddings) - 1),
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def run_hdbscan(
    embeddings_2d: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    import hdbscan

    min_size = min(max(2, min_cluster_size), max(2, len(embeddings_2d) // 5))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size,
        min_samples=min(max(1, min_samples), min_size),
    )
    return clusterer.fit_predict(embeddings_2d)


def apply_clusters_to_db(db, image_ids: list[int], labels: np.ndarray, projections_2d: np.ndarray) -> int:
    db.clear_clusters()
    unique_labels = sorted(set(int(label) for label in labels))
    label_to_cluster_id: dict[int, int] = {}
    for label in unique_labels:
        if label == -1:
            name = "Sin cluster"
            color = "#555555"
        else:
            name = f"Grupo {label + 1}"
            color = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
        label_to_cluster_id[label] = db.insert_cluster(name, color)

    for image_id, label in zip(image_ids, labels):
        db.update_image_cluster(image_id, label_to_cluster_id[int(label)])

    db.save_projections(
        [
            (int(image_id), float(x), float(y))
            for image_id, (x, y) in zip(image_ids, projections_2d)
        ]
    )
    return sum(1 for label in unique_labels if label != -1)


def assignments_from_db(db) -> list[dict]:
    with db.cursor() as cur:
        cur.execute(
            """
            SELECT i.id, i.path, i.filename, i.thumbnail_path, i.status, i.cluster_id,
                   c.name AS cluster_name, c.color AS cluster_color,
                   p.x, p.y
            FROM images i
            LEFT JOIN clusters c ON c.id = i.cluster_id
            LEFT JOIN projections p ON p.image_id = i.id
            ORDER BY i.id
            """
        )
        rows = cur.fetchall()
    return [
        {
            "image_id": row[0],
            "path": row[1],
            "filename": row[2],
            "thumbnail_path": row[3],
            "status": row[4],
            "cluster_id": row[5],
            "cluster_name": row[6] or "",
            "cluster_color": row[7] or "",
            "umap_x": row[8],
            "umap_y": row[9],
        }
        for row in rows
    ]


def assignments_from_rows(rows: list, labels: np.ndarray, projections_2d: np.ndarray) -> list[dict]:
    assignments = []
    for row, label, (x, y) in zip(rows, labels, projections_2d):
        label = int(label)
        assignments.append(
            {
                "image_id": row[0],
                "path": row[1],
                "filename": row[2],
                "thumbnail_path": row[5],
                "status": row[6],
                "cluster_id": label,
                "cluster_name": "Sin cluster" if label == -1 else f"Grupo {label + 1}",
                "cluster_color": "#555555" if label == -1 else CLUSTER_COLORS[label % len(CLUSTER_COLORS)],
                "umap_x": float(x),
                "umap_y": float(y),
            }
        )
    return assignments


def write_assignments_csv(path: Path, rows: list[dict]) -> Path:
    fieldnames = _ordered_fieldnames(
        rows,
        preferred=(
            "image_id",
            "filename",
            "path",
            "thumbnail_path",
            "status",
            "cluster_id",
            "cluster_name",
            "umap_x",
            "umap_y",
            "subset",
            "split",
            "true_class",
            "pred_class",
            "error_type",
            "is_error",
        ),
    )
    write_csv(path, rows, fieldnames)
    return path


def write_summary_csv(path: Path, rows: list[dict]) -> Path:
    fieldnames = _ordered_fieldnames(
        rows,
        preferred=(
            "cluster_id",
            "cluster_name",
            "total",
            "error_count",
            "normal_pred_polymerdirty",
            "normal_pred_bent",
            "true_dirty_holdout",
            "train_normal",
            "train_dirty",
            "train_bent",
            "review_priority",
            "subset_counts",
            "class_counts",
            "pred_counts",
            "status_counts",
            "error_counts",
        ),
    )
    write_csv(path, rows, fieldnames)
    return path


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_html_report(path: Path, summaries: list[dict], assignments: list[dict], project_path: Path) -> Path:
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in assignments:
        cluster_id = "none" if row.get("cluster_id") in (None, "") else str(row.get("cluster_id"))
        by_cluster[cluster_id].append(row)

    sections = []
    for summary in summaries[:50]:
        cluster_id = str(summary["cluster_id"])
        items = by_cluster.get(cluster_id, [])
        interesting = sorted(
            items,
            key=lambda row: (
                1 if _is_error(row.get("error_type")) else 0,
                1 if _is_dirty(row.get("true_class")) else 0,
                _str(row.get("filename")),
            ),
            reverse=True,
        )[:24]
        thumbs = []
        for item in interesting:
            thumb = item.get("thumbnail_path") or item.get("path") or ""
            label = f"{item.get('true_class', '')} -> {item.get('pred_class', '') or item.get('status', '')}"
            thumbs.append(
                "<figure>"
                f"<img src='{html.escape(_relative_path(project_path, thumb))}'>"
                f"<figcaption>{html.escape(label)}<br>{html.escape(_str(item.get('error_type')))}</figcaption>"
                "</figure>"
            )

        sections.append(
            "<section>"
            f"<h2>{html.escape(_str(summary['cluster_name']))} / id {html.escape(cluster_id)}</h2>"
            "<table>"
            f"<tr><th>total</th><td>{summary['total']}</td><th>errors</th><td>{summary['error_count']}</td><th>priority</th><td>{summary['review_priority']}</td></tr>"
            f"<tr><th>normal->dirty</th><td>{summary['normal_pred_polymerdirty']}</td><th>normal->bent</th><td>{summary['normal_pred_bent']}</td><th>true dirty holdout</th><td>{summary['true_dirty_holdout']}</td></tr>"
            f"<tr><th>status</th><td colspan='5'>{html.escape(_str(summary['status_counts']))}</td></tr>"
            f"<tr><th>errors</th><td colspan='5'>{html.escape(_str(summary['error_counts']))}</td></tr>"
            "</table>"
            f"<div class='grid'>{''.join(thumbs)}</div>"
            "</section>"
        )

    css = """
    body { font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #111; color: #eee; }
    h1 { margin-bottom: 4px; }
    p { color: #bbb; }
    section { border-top: 1px solid #333; padding-top: 18px; margin-top: 24px; }
    table { border-collapse: collapse; margin-bottom: 12px; }
    th, td { border: 1px solid #333; padding: 6px 10px; }
    th { color: #bbb; text-align: left; }
    .grid { display: flex; flex-wrap: wrap; gap: 10px; }
    figure { width: 150px; margin: 0; background: #1b1b1b; padding: 6px; border-radius: 6px; }
    img { width: 150px; height: 150px; object-fit: contain; background: #000; }
    figcaption { font-size: 11px; color: #ccc; overflow-wrap: anywhere; }
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{css}</style></head><body>"
        "<h1>Dataset Analysis Lab</h1>"
        "<p>Clusters are ordered by review priority. Open the CSV files for complete rows.</p>"
        f"{''.join(sections)}"
        "</body></html>",
        encoding="utf-8",
    )
    return path


def export_review_clusters(path: Path, summaries: list[dict], assignments: list[dict], top_n: int = 24) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in assignments:
        key = "none" if row.get("cluster_id") in (None, "") else str(row.get("cluster_id"))
        by_cluster[key].append(row)

    for rank, summary in enumerate(summaries[:top_n], start=1):
        cluster_key = str(summary["cluster_id"])
        cluster_dir = path / f"{rank:02d}_cluster_{safe_run_name(cluster_key)}_priority_{summary['review_priority']}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        write_csv(cluster_dir / "items.csv", by_cluster.get(cluster_key, []), _ordered_fieldnames(by_cluster.get(cluster_key, [])))
        for idx, item in enumerate(by_cluster.get(cluster_key, [])[:120], start=1):
            src = Path(item.get("path") or "")
            if not src.is_file():
                continue
            dst = cluster_dir / f"{idx:04d}__{safe_run_name(item.get('error_type') or item.get('status') or 'item')}__{src.name}"
            try:
                shutil.copy2(src, dst)
            except OSError:
                pass
    return path


def safe_run_name(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in "._-":
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._") or "analysis"


def _find_metadata(row: dict, metadata_index: dict[str, dict]) -> dict | None:
    for field in METADATA_KEY_FIELDS:
        value = (row.get(field) or "").strip()
        if not value:
            continue
        for key in _metadata_keys(value):
            if key in metadata_index:
                return metadata_index[key]
    return None


def _metadata_keys(value: str) -> list[str]:
    path = Path(value)
    keys = {value}
    try:
        keys.add(str(path.resolve()))
    except OSError:
        pass
    if path.name:
        keys.add(path.name)
    return [key for key in keys if key]


def _ordered_fieldnames(rows: list[dict], preferred: Iterable[str] = ()) -> list[str]:
    seen = []
    for field in preferred:
        if field not in seen:
            seen.append(field)
    for row in rows:
        for field in row.keys():
            if field not in seen:
                seen.append(field)
    return seen or ["empty"]


def _normalize_matrix(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _progress(callback: ProgressCallback | None, message: str, pct: int) -> None:
    if callback:
        callback(message, max(0, min(100, int(pct))))


def _relative_path(root: Path, value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _is_error(error_type: str | None) -> bool:
    value = _str(error_type)
    return bool(value and value not in {"correct", "train_context", "unpredicted", "unknown_truth"})


def _is_normal(value: str | None) -> bool:
    key = _label_key(value)
    return key in NORMAL_LABELS or key.startswith("normal")


def _is_dirty(value: str | None) -> bool:
    key = _label_key(value)
    return key in DIRTY_LABELS or "dirty" in key or "sucio" in key


def _is_bent(value: str | None) -> bool:
    key = _label_key(value)
    if key in NORMAL_LABELS or key.startswith("not_bent") or key.startswith("normal"):
        return False
    return key in BENT_LABELS or "bent" in key or "bend" in key


def _label_key(value: str | None) -> str:
    return _str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _str(value) -> str:
    return "" if value is None else str(value)
