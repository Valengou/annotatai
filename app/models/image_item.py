from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ImageItem:
    id: Optional[int] = None
    path: str = ""
    filename: str = ""
    width: int = 0
    height: int = 0
    thumbnail_path: Optional[str] = None
    status: str = "pending"   # pending | reviewed | discarded
    cluster_id: Optional[int] = None
    embedding_ready: bool = False
    detection_avg_confidence: Optional[float] = None
    detection_min_confidence: Optional[float] = None
    classifier_confidence: Optional[float] = None
    review_sort_confidence: Optional[float] = None
    review_reasons: str = ""

    @property
    def display_name(self) -> str:
        return Path(self.path).name

    @classmethod
    def from_db_row(cls, row) -> "ImageItem":
        return cls(
            id=row[0],
            path=row[1],
            filename=row[2],
            width=row[3] or 0,
            height=row[4] or 0,
            thumbnail_path=row[5],
            status=row[6] or "pending",
            cluster_id=row[7],
            embedding_ready=bool(row[8]) if len(row) > 8 else False,
            detection_avg_confidence=row[9] if len(row) > 9 else None,
            detection_min_confidence=row[10] if len(row) > 10 else None,
            classifier_confidence=row[11] if len(row) > 11 else None,
            review_sort_confidence=row[12] if len(row) > 12 else None,
            review_reasons=row[13] if len(row) > 13 and row[13] else "",
        )
