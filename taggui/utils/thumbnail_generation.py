"""
Generate image list thumbnails on background threads.

Thumbnails are generated with a fast, scaled image decode
(``QImageReader.setScaledSize``) instead of decoding the full-resolution
image and then scaling it down. This is several times faster and uses far
less memory, which keeps the GUI thread responsive even with thousands of
images.

Each result is also stored in the on-disk thumbnail cache
(``thumbnail_cache.py``) keyed by the image's content hash, so re-opening a
directory does not re-decode the originals (see
``dev_local/perf_hash_thumbnails_report.md``).
"""
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, Signal
from PySide6.QtGui import QImage, QImageReader

from utils.thumbnail_cache import (evict_to_limit, get_cache_path,
                                   load_cached_thumbnail, mark_thumbnail_used,
                                   save_thumbnail)


class ThumbnailGenerationWorker(QObject):
    """
    Receive finished thumbnails from the worker threads and forward them to
    the image list model.

    `thumbnail_ready` is emitted from a worker thread and delivered to the
    connected slot on the GUI thread (queued connection).
    """
    # (row of the image, generated image, generation it belongs to)
    thumbnail_ready = Signal(int, QImage, int)

    def __init__(self, parent=None):
        super().__init__(parent)


class ThumbnailGenerationTask(QRunnable):
    """
    Generate the thumbnail for one image on a worker thread.

    The task is added to a `registry` set and removed from it when it is
    done. The set keeps a strong reference to the task so that it is not
    garbage collected while the thread pool still references it.
    """

    def __init__(self, row: int, image_path: Path, image_width: int,
                 generation: int, worker: ThumbnailGenerationWorker,
                 registry: set, content_hash: str | None = None):
        super().__init__()
        # The thread pool must not delete the C++ object while Python may
        # still reference it.
        self.setAutoDelete(False)
        self.row = row
        self.image_path = image_path
        self.image_width = image_width
        self.generation = generation
        self.worker = worker
        self.registry = registry
        # The image's content hash (database mode only): the key of its
        # on-disk thumbnail cache entry. None in text-file mode, where the
        # cache is skipped (the cache is keyed by content, which text-file
        # mode does not compute).
        self.content_hash = content_hash
        registry.add(self)

    def run(self):
        try:
            self._emit_thumbnail(self._generate_thumbnail())
        finally:
            self.registry.discard(self)

    def _generate_thumbnail(self) -> QImage:
        """
        The image's thumbnail, from the on-disk cache when it has an entry,
        and from a fresh (scaled) decode of the original image otherwise —
        which is then stored in the cache for the next time.
        """
        if self.content_hash is not None:
            cache_path = get_cache_path(self.content_hash, self.image_width)
            image = load_cached_thumbnail(cache_path)
            if not image.isNull():
                # A cache hit: record the use so the entry is among the last
                # evicted, and skip decoding the original image entirely.
                mark_thumbnail_used(cache_path)
                return image
            image = self._decode()
            if not image.isNull():
                save_thumbnail(image, cache_path)
            return image
        return self._decode()

    def _decode(self) -> QImage:
        """
        Decode the original image at the thumbnail's target size (a fast
        scaled decode rather than a full decode followed by a down-scale).
        """
        image_reader = QImageReader(str(self.image_path))
        # Rotate the image based on the orientation tag.
        image_reader.setAutoTransform(True)
        stored_size = image_reader.size()
        if stored_size.width() > 0:
            # `QImageReader` scales the image to *exactly* the given size
            # (without preserving the aspect ratio), so give it a target
            # size that already has the image's aspect ratio. This uses a
            # fast scaled decode instead of decoding the full image and
            # then scaling it down.
            # The target is computed from the *stored* image size because
            # `QImageReader` scales the image before applying the Exif
            # orientation transform.
            image_reader.setScaledSize(QSize(
                self.image_width,
                int(self.image_width
                    * stored_size.height() / stored_size.width())))
        image = image_reader.read()
        # If the image was rotated by the Exif orientation transform, its
        # width is not the requested width. Scale it to the requested
        # width (a small re-scale that is cheap compared to a full decode).
        if not image.isNull() and image.width() != self.image_width:
            image = image.scaledToWidth(
                self.image_width,
                Qt.TransformationMode.SmoothTransformation)
        return image

    def _emit_thumbnail(self, image: QImage):
        try:
            self.worker.thumbnail_ready.emit(self.row, image, self.generation)
        except RuntimeError:
            # The worker (and therefore the image list model) has been
            # destroyed, so there is no one to deliver the thumbnail to.
            pass


class ThumbnailCacheEvictionTask(QRunnable):
    """
    Evict the on-disk thumbnail cache down to its configured size limit.
    Runs once at startup, on a background thread, so that entries
    accumulated by earlier sessions are cleaned up across sessions without
    blocking the GUI thread.
    """

    def __init__(self, max_bytes: int, registry: set):
        super().__init__()
        # The thread pool must not delete the C++ object while Python may
        # still reference it.
        self.setAutoDelete(False)
        self.max_bytes = max_bytes
        self.registry = registry
        registry.add(self)

    def run(self):
        try:
            evict_to_limit(self.max_bytes)
        finally:
            self.registry.discard(self)
