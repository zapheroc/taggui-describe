"""
Generate image list thumbnails on background threads.

Thumbnails are generated with a fast, scaled image decode
(``QImageReader.setScaledSize``) instead of decoding the full-resolution
image and then scaling it down. This is several times faster and uses far
less memory, which keeps the GUI thread responsive even with thousands of
images.
"""
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, Signal
from PySide6.QtGui import QImage, QImageReader


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
                 registry: set):
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
        registry.add(self)

    def run(self):
        try:
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
            self._emit_thumbnail(image)
        finally:
            self.registry.discard(self)

    def _emit_thumbnail(self, image: QImage):
        try:
            self.worker.thumbnail_ready.emit(self.row, image, self.generation)
        except RuntimeError:
            # The worker (and therefore the image list model) has been
            # destroyed, so there is no one to deliver the thumbnail to.
            pass
