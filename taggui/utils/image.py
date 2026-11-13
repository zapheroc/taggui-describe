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
    # How many of the loaded images have the same content as this one
    # (itself included): 2 or more when the image has duplicate file(s), 0
    # when it has no content hash (text-file storage mode) or no other loaded
    # image shares its content. Set by the image list model when a directory
    # loads. In database mode the duplicates share one record, so they always
    # show the same tags.
    duplicate_count: int = 0
    # This copy's 1-based position among the loaded files with the same
    # content (0 when it is not part of a duplicate group).
    duplicate_position: int = 0
