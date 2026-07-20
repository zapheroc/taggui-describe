from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtGui import QIcon


@dataclass
class Image:
    path: Path
    dimensions: tuple[int, int] | None
    tags: list[str] = field(default_factory=list)
    description: str = ""
    thumbnail: QIcon | None = None
