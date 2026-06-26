from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class Annotation:
    id: Optional[int] = None
    image_id: int = 0
    class_id: int = 0
    class_name: str = ""
    x: float = 0.0       # normalized 0-1 (top-left corner)
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    source: str = "human"   # human | suggested | propagated
    confidence: float = 1.0
    polygon: Optional[list[tuple[float, float]]] = None
    severity: str = "nula"  # nula | leve | moderada | critica

    def to_yolo_line(self, class_index: int) -> str:
        cx = self.x + self.width / 2
        cy = self.y + self.height / 2
        return f"{class_index} {cx:.6f} {cy:.6f} {self.width:.6f} {self.height:.6f}"

    @classmethod
    def from_db_row(cls, row) -> "Annotation":
        polygon = None
        if len(row) > 10 and row[10]:
            polygon = [tuple(point) for point in json.loads(row[10])]
        return cls(
            id=row[0],
            image_id=row[1],
            class_id=row[2],
            class_name=row[3],
            x=row[4],
            y=row[5],
            width=row[6],
            height=row[7],
            source=row[8] if len(row) > 8 else "human",
            confidence=row[9] if len(row) > 9 else 1.0,
            polygon=polygon,
            severity=(row[11] if len(row) > 11 and row[11] else "nula"),
        )
