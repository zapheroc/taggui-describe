from pathlib import Path

from PySide6.QtCore import (QCoreApplication, QObject, QRunnable, QModelIndex,
                            QSize, QThreadPool, Qt, QTimer, Signal, Slot)
from PySide6.QtGui import QImage, QImageReader, QPixmap, QResizeEvent
from PySide6.QtWidgets import (QLabel, QSizePolicy, QVBoxLayout, QWidget)

from models.proxy_image_list_model import ProxyImageListModel
from utils.image import Image

# How long (in milliseconds) to wait for the background image load to finish
# when the application is quitting. Destroying the thread pool while its
# worker thread is still running a task makes Qt report worker-thread
# exceptions (and can even abort the process), so the in-flight load is given
# a brief, bounded chance to finish. The wait is capped so that quitting is
# never blocked for a long time: the work is read-only, so abandoning it is
# safe. (Same bound as `SHUTDOWN_TIMEOUT_MS` in `models/image_list_model.py`.)
SHUTDOWN_TIMEOUT_MS = 2000


class FullImageLoadWorker(QObject):
    """
    Receive full-resolution images loaded on worker threads and forward them
    to the image label.

    `full_image_ready` is emitted from a worker thread and delivered to the
    connected slot on the GUI thread (queued connection).
    """
    # (path of the image, loaded image)
    full_image_ready = Signal(Path, QImage)

    def __init__(self, parent=None):
        super().__init__(parent)


class FullImageLoadTask(QRunnable):
    """
    Load one full-resolution image on a worker thread.

    The task is added to a `registry` set and removed from it when it is
    done. The set keeps a strong reference to the task so that it is not
    garbage collected while the thread pool still references it.
    """

    def __init__(self, image_path: Path, worker: FullImageLoadWorker,
                 registry: set):
        super().__init__()
        # The thread pool must not delete the C++ object while Python may
        # still reference it.
        self.setAutoDelete(False)
        self.image_path = image_path
        self.worker = worker
        self.registry = registry
        registry.add(self)

    def run(self):
        try:
            image_reader = QImageReader(str(self.image_path))
            # Rotate the image according to the orientation tag.
            image_reader.setAutoTransform(True)
            try:
                self.worker.full_image_ready.emit(self.image_path,
                                                  image_reader.read())
            except RuntimeError:
                # The worker (and therefore the image label) has been
                # destroyed, so there is no one to deliver the image to.
                pass
        finally:
            self.registry.discard(self)


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        # The image currently being displayed, if any.
        self.image: Image | None = None
        # The last loaded full-resolution image. It is kept so that it can be
        # re-scaled (without re-decoding it) whenever the label is resized.
        self._full_image_pixmap: QPixmap | None = None
        # The full-resolution image is loaded on a background thread so that
        # decoding never blocks the GUI thread (which made image navigation
        # slow with large images).
        self._full_image_load_worker = FullImageLoadWorker(self)
        self._full_image_load_worker.full_image_ready.connect(
            self.on_full_image_ready)
        self._full_image_thread_pool = QThreadPool(self)
        self._full_image_thread_pool.setMaxThreadCount(2)
        # Re-scaling a large image takes a little while, so only re-scale it
        # at most every 50 ms while the label is being resized.
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(50)
        self._resize_timer.timeout.connect(self._display_current_image)
        # Strong references to running tasks so that they are not garbage
        # collected while the thread pool still references them.
        self._running_tasks: set = set()
        # When the application is about to quit, give the background image
        # load a bounded chance to finish so that the thread pool is not
        # destroyed while its worker thread is still running a task (which
        # makes Qt report worker-thread exceptions, and can even abort the
        # process).
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._wait_for_pending_work)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        # This allows the label to shrink.
        self.setMinimumSize(QSize(1, 1))

    def resizeEvent(self, event: QResizeEvent):
        """Re-scale the current image whenever the label is resized."""
        super().resizeEvent(event)
        if self._full_image_pixmap is not None:
            self._resize_timer.start()

    @Slot(Path, QImage)
    def on_full_image_ready(self, image_path: Path, image: QImage):
        # The load has been superseded by a newer load, or it failed.
        if (self.image is None or image_path != self.image.path
                or image.isNull()):
            return
        self._full_image_pixmap = QPixmap.fromImage(image)
        self._display_current_image()

    def _display_current_image(self):
        if self._full_image_pixmap is None:
            return
        dpr = self.devicePixelRatio()
        target_size = self.size() * dpr
        pixmap = self._full_image_pixmap.scaled(
            target_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        # `QPixmap.scaled()` returns a pixmap with a device pixel ratio of 1,
        # so set the ratio explicitly to display it at the correct size.
        pixmap.setDevicePixelRatio(dpr)
        self.setPixmap(pixmap)

    def load_image(self, image: Image):
        self.image = image
        self._full_image_pixmap = None
        # Show the thumbnail (if one is available) immediately as a
        # placeholder. The full-resolution image is shown when it has been
        # loaded on a background thread.
        if image.thumbnail is not None:
            dpr = self.devicePixelRatio()
            target_size = self.size() * dpr
            thumbnail_pixmap = image.thumbnail.pixmap(
                image.thumbnail.availableSizes()[0]).scaled(
                    target_size, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
            thumbnail_pixmap.setDevicePixelRatio(dpr)
            self.setPixmap(thumbnail_pixmap)
        else:
            self.clear()
        task = FullImageLoadTask(image.path, self._full_image_load_worker,
                                 self._running_tasks)
        self._full_image_thread_pool.start(task)

    @Slot()
    def _wait_for_pending_work(self):
        """Wait for the background image load to finish (app is quitting)."""
        self._full_image_thread_pool.waitForDone(SHUTDOWN_TIMEOUT_MS)


class ImageViewer(QWidget):
    def __init__(self, proxy_image_list_model: ProxyImageListModel):
        super().__init__()
        self.proxy_image_list_model = proxy_image_list_model
        self.image_label = ImageLabel()
        QVBoxLayout(self).addWidget(self.image_label)

    @Slot()
    def load_image(self, proxy_image_index: QModelIndex):
        image: Image = self.proxy_image_list_model.data(
            proxy_image_index, Qt.ItemDataRole.UserRole)
        # `image` is None when the index is invalid (for example, when the
        # model is reset while a directory is loading).
        if image is None:
            return
        self.image_label.load_image(image)
