import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    width INTEGER DEFAULT 0,
    height INTEGER DEFAULT 0,
    thumbnail_path TEXT,
    status TEXT DEFAULT 'pending',
    cluster_id INTEGER,
    embedding_ready INTEGER DEFAULT 0,
    detection_avg_confidence REAL,
    detection_min_confidence REAL,
    classifier_confidence REAL,
    review_sort_confidence REAL,
    review_reasons TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cluster_id) REFERENCES clusters(id)
);

CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    color TEXT DEFAULT '#808080',
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embeddings (
    image_id INTEGER PRIMARY KEY,
    vector BLOB NOT NULL,
    model TEXT DEFAULT 'ViT-B-32',
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS projections (
    image_id INTEGER PRIMARY KEY,
    x REAL NOT NULL,
    y REAL NOT NULL,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    color TEXT DEFAULT '#FF0000'
);

CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    width REAL NOT NULL,
    height REAL NOT NULL,
    source TEXT DEFAULT 'human',
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY (class_id) REFERENCES classes(id)
);

CREATE TABLE IF NOT EXISTS bbox_embeddings (
    annotation_id INTEGER PRIMARY KEY,
    vector BLOB NOT NULL,
    model TEXT DEFAULT 'ViT-B-32',
    FOREIGN KEY (annotation_id) REFERENCES annotations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bbox_projections (
    annotation_id INTEGER PRIMARY KEY,
    x REAL NOT NULL,
    y REAL NOT NULL,
    outlier_score REAL DEFAULT 0.0,
    FOREIGN KEY (annotation_id) REFERENCES annotations(id) ON DELETE CASCADE
);
"""


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self):
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._ensure_review_metadata_columns()
        self._conn.commit()

    def _ensure_review_metadata_columns(self):
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(images)").fetchall()}
        columns = {
            "detection_avg_confidence": "REAL",
            "detection_min_confidence": "REAL",
            "classifier_confidence": "REAL",
            "review_sort_confidence": "REAL",
            "review_reasons": "TEXT",
        }
        for name, definition in columns.items():
            if name not in existing:
                self._conn.execute(f"ALTER TABLE images ADD COLUMN {name} {definition}")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # --- Images ---

    def insert_image(self, path: str, filename: str, width: int, height: int,
                     thumbnail_path: str | None) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO images (path, filename, width, height, thumbnail_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (path, filename, width, height, thumbnail_path),
            )
            if cur.lastrowid:
                return cur.lastrowid
            cur.execute("SELECT id FROM images WHERE path=?", (path,))
            return cur.fetchone()[0]

    def get_all_images(self) -> list:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, path, filename, width, height, thumbnail_path, "
                "status, cluster_id, embedding_ready, detection_avg_confidence, "
                "detection_min_confidence, classifier_confidence, review_sort_confidence, "
                "review_reasons FROM images ORDER BY filename"
            )
            return cur.fetchall()

    def get_images_for_cluster(self, cluster_id: int) -> list:
        with self.cursor() as cur:
            cur.execute(
                "SELECT id, path, filename, width, height, thumbnail_path, "
                "status, cluster_id, embedding_ready, detection_avg_confidence, "
                "detection_min_confidence, classifier_confidence, review_sort_confidence, "
                "review_reasons FROM images WHERE cluster_id=? ORDER BY filename",
                (cluster_id,),
            )
            return cur.fetchall()

    def update_image_status(self, image_id: int, status: str):
        with self.cursor() as cur:
            cur.execute("UPDATE images SET status=? WHERE id=?", (status, image_id))

    def update_image_cluster(self, image_id: int, cluster_id: int | None):
        with self.cursor() as cur:
            cur.execute("UPDATE images SET cluster_id=? WHERE id=?", (cluster_id, image_id))

    def mark_embedding_ready(self, image_id: int):
        with self.cursor() as cur:
            cur.execute("UPDATE images SET embedding_ready=1 WHERE id=?", (image_id,))

    # --- BBox embeddings ---

    def count_bbox_embeddings(self) -> int:
        with self.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM bbox_embeddings")
            return cur.fetchone()[0]

    def has_bbox_embedding(self, annotation_id: int) -> bool:
        with self.cursor() as cur:
            cur.execute("SELECT 1 FROM bbox_embeddings WHERE annotation_id=?", (annotation_id,))
            return cur.fetchone() is not None

    def save_bbox_embedding(self, annotation_id: int, vector_bytes: bytes, model: str):
        with self.cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO bbox_embeddings (annotation_id, vector, model) VALUES (?, ?, ?)",
                (annotation_id, vector_bytes, model),
            )

    def get_all_bbox_embeddings(self) -> list:
        with self.cursor() as cur:
            cur.execute(
                "SELECT be.annotation_id, be.vector "
                "FROM bbox_embeddings be "
                "JOIN annotations a ON a.id = be.annotation_id "
                "ORDER BY be.annotation_id"
            )
            return cur.fetchall()

    def save_bbox_projections(self, rows: list[tuple]):
        with self.cursor() as cur:
            cur.execute("DELETE FROM bbox_projections")
            cur.executemany(
                "INSERT INTO bbox_projections (annotation_id, x, y, outlier_score) VALUES (?, ?, ?, ?)",
                rows,
            )

    def get_all_bbox_projections(self) -> list:
        with self.cursor() as cur:
            cur.execute("""
                SELECT bp.annotation_id, bp.x, bp.y, bp.outlier_score,
                       a.image_id, a.x, a.y, a.width, a.height,
                       i.path, i.filename, i.status, c.name, c.color
                FROM bbox_projections bp
                JOIN annotations a ON a.id = bp.annotation_id
                JOIN images i ON i.id = a.image_id
                JOIN classes c ON c.id = a.class_id
                ORDER BY bp.outlier_score DESC
            """)
            return cur.fetchall()

    def get_annotations_for_grid(self) -> dict:
        """Single bulk query → {image_id: [(x, y, w, h, color_hex, class_name), ...]}"""
        with self.cursor() as cur:
            cur.execute("""
                SELECT a.image_id, a.x, a.y, a.width, a.height, c.color, c.name, a.confidence
                FROM annotations a
                JOIN classes c ON c.id = a.class_id
            """)
            result: dict[int, list] = {}
            for row in cur.fetchall():
                img_id = row[0]
                if img_id not in result:
                    result[img_id] = []
                result[img_id].append((row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
        return result

    def get_annotations_for_grid_image(self, image_id: int) -> list:
        """Annotations for a single image as [(x, y, w, h, color_hex, class_name), ...]"""
        with self.cursor() as cur:
            cur.execute("""
                SELECT a.x, a.y, a.width, a.height, c.color, c.name
                FROM annotations a
                JOIN classes c ON c.id = a.class_id
                WHERE a.image_id = ?
            """, (image_id,))
            return cur.fetchall()

    def get_annotations_with_image_paths(self) -> list:
        with self.cursor() as cur:
            cur.execute("""
                SELECT a.id, a.image_id, a.x, a.y, a.width, a.height,
                       i.path, i.width, i.height
                FROM annotations a
                JOIN images i ON i.id = a.image_id
            """)
            return cur.fetchall()

    # --- Image deletion ---

    def delete_images(self, image_ids: list[int]) -> int:
        """Delete images (and cascaded embeddings/projections/annotations). Returns count deleted."""
        if not image_ids:
            return 0
        with self.cursor() as cur:
            placeholders = ",".join("?" * len(image_ids))
            cur.execute(f"DELETE FROM images WHERE id IN ({placeholders})", image_ids)
            return cur.rowcount

    def delete_cluster_with_images(self, cluster_id: int) -> tuple[int, int]:
        """Delete a cluster and all its images. Returns (n_images, cluster_id)."""
        with self.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM images WHERE cluster_id=?", (cluster_id,))
            n = cur.fetchone()[0]
            cur.execute("DELETE FROM images WHERE cluster_id=?", (cluster_id,))
            cur.execute("DELETE FROM clusters WHERE id=?", (cluster_id,))
        return n, cluster_id

    def execute_scalar(self, sql: str, params=()) -> int:
        with self.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()[0]

    def count_images(self) -> int:
        with self.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM images")
            return cur.fetchone()[0]

    def count_embeddings(self) -> int:
        with self.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embeddings")
            return cur.fetchone()[0]

    # --- Clusters ---

    def insert_cluster(self, name: str, color: str) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO clusters (name, color) VALUES (?, ?)",
                (name, color),
            )
            return cur.lastrowid

    def get_all_clusters(self, status_filter: str = "todas") -> list:
        with self.cursor() as cur:
            if status_filter == "todas":
                cur.execute(
                    "SELECT c.id, c.name, c.color, c.status, COUNT(i.id) "
                    "FROM clusters c LEFT JOIN images i ON i.cluster_id=c.id "
                    "GROUP BY c.id ORDER BY c.name"
                )
            else:
                cur.execute(
                    "SELECT c.id, c.name, c.color, c.status, "
                    "COUNT(CASE WHEN i.status=? THEN 1 END) "
                    "FROM clusters c LEFT JOIN images i ON i.cluster_id=c.id "
                    "GROUP BY c.id ORDER BY c.name",
                    (status_filter,),
                )
            return cur.fetchall()

    def clear_clusters(self):
        with self.cursor() as cur:
            cur.execute("UPDATE images SET cluster_id=NULL")
            cur.execute("DELETE FROM clusters")

    # --- Embeddings ---

    def save_embedding(self, image_id: int, vector_bytes: bytes, model: str):
        with self.cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO embeddings (image_id, vector, model) VALUES (?, ?, ?)",
                (image_id, vector_bytes, model),
            )

    def get_all_embeddings(self) -> list:
        with self.cursor() as cur:
            cur.execute(
                "SELECT e.image_id, e.vector FROM embeddings e "
                "JOIN images i ON i.id=e.image_id ORDER BY i.id"
            )
            return cur.fetchall()

    def has_embedding(self, image_id: int) -> bool:
        with self.cursor() as cur:
            cur.execute("SELECT 1 FROM embeddings WHERE image_id=?", (image_id,))
            return cur.fetchone() is not None

    # --- Projections ---

    def save_projections(self, projections: list[tuple[int, float, float]]):
        with self.cursor() as cur:
            cur.execute("DELETE FROM projections")
            cur.executemany(
                "INSERT INTO projections (image_id, x, y) VALUES (?, ?, ?)", projections
            )

    def get_all_projections(self) -> list:
        with self.cursor() as cur:
            cur.execute(
                "SELECT p.image_id, p.x, p.y, i.filename, i.thumbnail_path, i.cluster_id, i.status "
                "FROM projections p JOIN images i ON i.id=p.image_id"
            )
            return cur.fetchall()

    # --- Classes ---

    def get_or_create_class(self, name: str, color: str = "#FF0000") -> int:
        with self.cursor() as cur:
            cur.execute("SELECT id FROM classes WHERE name=?", (name,))
            row = cur.fetchone()
            if row:
                return row[0]
            cur.execute("INSERT INTO classes (name, color) VALUES (?, ?)", (name, color))
            return cur.lastrowid

    def get_all_classes(self) -> list:
        with self.cursor() as cur:
            cur.execute("SELECT id, name, color FROM classes ORDER BY name")
            return cur.fetchall()

    def update_class_color(self, class_id: int, color: str):
        with self.cursor() as cur:
            cur.execute("UPDATE classes SET color=? WHERE id=?", (color, class_id))

    def count_annotations_for_class(self, class_id: int) -> int:
        with self.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM annotations WHERE class_id=?", (class_id,))
            return cur.fetchone()[0]

    def delete_class(self, class_id: int):
        with self.cursor() as cur:
            cur.execute("DELETE FROM annotations WHERE class_id=?", (class_id,))
            cur.execute("DELETE FROM classes WHERE id=?", (class_id,))

    # --- Annotations ---

    def insert_annotation(self, image_id: int, class_id: int,
                          x: float, y: float, w: float, h: float,
                          source: str = "human", confidence: float = 1.0) -> int:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO annotations (image_id, class_id, x, y, width, height, source, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (image_id, class_id, x, y, w, h, source, confidence),
            )
            return cur.lastrowid

    def update_annotation(self, ann_id: int, class_id: int,
                          x: float, y: float, w: float, h: float):
        with self.cursor() as cur:
            cur.execute(
                "UPDATE annotations SET class_id=?, x=?, y=?, width=?, height=? WHERE id=?",
                (class_id, x, y, w, h, ann_id),
            )

    def delete_annotation(self, ann_id: int):
        with self.cursor() as cur:
            cur.execute("DELETE FROM annotations WHERE id=?", (ann_id,))

    def get_annotations_for_image(self, image_id: int) -> list:
        with self.cursor() as cur:
            cur.execute(
                "SELECT a.id, a.image_id, a.class_id, c.name, a.x, a.y, a.width, a.height, "
                "a.source, a.confidence "
                "FROM annotations a JOIN classes c ON c.id=a.class_id "
                "WHERE a.image_id=? ORDER BY a.id",
                (image_id,),
            )
            return cur.fetchall()

    def delete_annotations_for_image(self, image_id: int):
        with self.cursor() as cur:
            cur.execute("DELETE FROM annotations WHERE image_id=?", (image_id,))
