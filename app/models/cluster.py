from dataclasses import dataclass
from typing import Optional


@dataclass
class Cluster:
    id: Optional[int] = None
    name: str = ""
    color: str = "#808080"
    status: str = "active"   # active | reviewed | discarded
    image_count: int = 0

    @classmethod
    def from_db_row(cls, row) -> "Cluster":
        return cls(
            id=row[0],
            name=row[1],
            color=row[2] or "#808080",
            status=row[3] or "active",
            image_count=row[4] if len(row) > 4 else 0,
        )
