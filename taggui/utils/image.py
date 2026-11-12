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
    # The stable blake2b content hash of the image file, set by the storage
    # backend when the image is loaded. This is the key of the image's record
    # in the database (None in text-file storage mode).
    content_hash: str | None = None
