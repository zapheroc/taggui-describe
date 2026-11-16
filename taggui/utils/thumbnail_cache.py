"""
On-disk cache for the image list's thumbnails.

Thumbnails are decoded in background threads (see
``thumbnail_generation.py``), and the result of each decode is stored here,
keyed by the image's content hash and the display width. Re-opening a
directory — or scrolling back to a row — then reads the cached file instead
of decoding the (often much larger) original image, and identical images in
different folders share one cached thumbnail.

The cache is a disposable optimization: every entry is derived data that
can be regenerated from the image files, so failures to read, write, or
evict it are always best-effort and never surface as errors.

The cache is bounded by the ``thumbnail_cache_max_size_mb`` setting; the
least recently used entries (by file modification time, refreshed by
``mark_thumbnail_used``) are evicted once at startup.
"""
import os
import sys
import threading
from pathlib import Path

from PySide6.QtGui import QImage, QImageReader

from utils.settings import DEFAULT_SETTINGS, get_settings

# JPEG quality for the cached thumbnails: high enough that thumbnails this
# small look lossless, and measured to save ~7x faster and ~4.6x smaller
# than PNG (see `dev_local/perf_hash_thumbnails_report.md`).
THUMBNAIL_CACHE_JPEG_QUALITY = 90


def get_thumbnail_cache_dir() -> Path:
    """
    The on-disk thumbnail cache directory, created if needed.

    The platform cache directory (disposable by convention), in an
    application-specific subdirectory so that other applications' caches —
    and their eviction of their own entries — do not touch this one.
    """
    if sys.platform == 'win32':
        local_app_data = os.environ.get('LOCALAPPDATA', '')
        base = (Path(local_app_data) if local_app_data
                else Path.home() / 'AppData' / 'Local')
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Caches'
    else:
        xdg_cache = os.environ.get('XDG_CACHE_HOME', '')
        base = Path(xdg_cache) if xdg_cache else Path.home() / '.cache'
    cache_dir = base / 'taggui-describe' / 'thumbnails'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_max_cache_bytes() -> int:
    """The configured maximum size of the thumbnail cache, in bytes."""
    settings = get_settings()
    max_size_mb = settings.value(
        'thumbnail_cache_max_size_mb',
        defaultValue=DEFAULT_SETTINGS['thumbnail_cache_max_size_mb'],
        type=int)
    return max_size_mb * 1024 * 1024


def get_cache_path(content_hash: str, image_width: int) -> Path:
    """
    The cache file for the thumbnail of an image with content `content_hash`
    at display width `image_width` pixels.

    Keyed by the content hash (not the file path), so identical images in
    different folders share one cached thumbnail, and a cached entry can
    never be stale: the hash changes exactly when the image's bytes change.
    """
    return get_thumbnail_cache_dir() / f'{content_hash}_{image_width}.jpg'


def load_cached_thumbnail(cache_path: Path) -> QImage:
    """
    A cached thumbnail, or an empty `QImage` when the file is missing,
    unreadable, or not a decodable image (the caller then falls back to
    decoding the original image).
    """
    return QImageReader(str(cache_path)).read()


def save_thumbnail(thumbnail_image: QImage, cache_path: Path) -> None:
    """
    Save a thumbnail (a ``QImage``) to the cache without ever leaving a
    partly written file behind: it is written to a temporary file first and
    then swapped into place. Several threads may save the same thumbnail at
    once (two rows of the same duplicate content can both miss the cache);
    whichever finishes last wins, and both results are identical.

    Never raises: a failed cache write simply means the thumbnail is not
    cached, and the thumbnail itself is still delivered.
    """
    temp_path = cache_path.with_name(
        f'{cache_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp')
    try:
        if thumbnail_image.save(str(temp_path), 'JPEG',
                                THUMBNAIL_CACHE_JPEG_QUALITY):
            os.replace(temp_path, cache_path)
    except OSError:
        pass
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def mark_thumbnail_used(cache_path: Path) -> None:
    """
    Record that a cached thumbnail was just used, so the size limit removes
    the least recently used thumbnails first. (Windows does not reliably
    record when a file was last read, so the modified time is used.)
    """
    try:
        os.utime(cache_path)
    except OSError:
        pass


def _is_cache_file(path: Path) -> bool:
    # `.tmp` files are leftovers from an interrupted save.
    return path.is_file() and path.suffix in ('.jpg', '.tmp')


def get_cache_size_bytes() -> int:
    """The total size of all cached thumbnail files, in bytes."""
    cache_dir = get_thumbnail_cache_dir()
    try:
        return sum(path.stat().st_size
                   for path in cache_dir.iterdir() if _is_cache_file(path))
    except OSError:
        return 0


def clear_cache() -> int:
    """
    Delete all cached thumbnails (and any leftover temporary files).
    Returns the number of files deleted.
    """
    cache_dir = get_thumbnail_cache_dir()
    deleted = 0
    try:
        for path in cache_dir.iterdir():
            if _is_cache_file(path):
                try:
                    path.unlink()
                    deleted += 1
                except OSError:
                    pass
    except OSError:
        pass
    return deleted


def evict_to_limit(max_bytes: int) -> None:
    """
    Delete the least recently used cached thumbnails until the total cache
    size is at or below `max_bytes`. "Used" is the file's modified time,
    which `mark_thumbnail_used` refreshes whenever a cached thumbnail is
    shown. Called once, in a background thread, at startup so that entries
    accumulated by earlier sessions are cleaned up across sessions.
    """
    cache_dir = get_thumbnail_cache_dir()
    try:
        files = []
        for path in cache_dir.iterdir():
            if not _is_cache_file(path):
                continue
            try:
                stat_result = path.stat()
            except OSError:
                continue
            files.append((path, stat_result.st_size, stat_result.st_mtime))
    except OSError:
        return
    total = sum(size for _, size, _ in files)
    if total <= max_bytes:
        return
    # Delete least recently used entries first.
    files.sort(key=lambda entry: entry[2])
    for path, size, _ in files:
        try:
            path.unlink()
        except OSError:
            pass
        total -= size
        if total <= max_bytes:
            break
