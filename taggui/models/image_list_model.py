import os
import random
import re
import sqlite3
import sys
import threading
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import exifread
import imagesize
from PySide6.QtCore import (QAbstractListModel, QModelIndex, QCoreApplication,
                            QObject, QRunnable, QSize, Qt, QThreadPool, Signal,
                            Slot)
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QMessageBox

from models.image_storage import (ImageStorage,
                                   export_images_to_text_files,
                                   get_database_path, get_image_storage,
                                   import_images)
from utils.image import Image
from utils.settings import DEFAULT_SETTINGS, get_settings
from utils.thumbnail_generation import (ThumbnailGenerationTask,
                                        ThumbnailGenerationWorker)
from utils.utils import get_confirmation_dialog_reply, pluralize

UNDO_STACK_SIZE = 32
# How long (in milliseconds) to wait for background tasks to finish when the
# application is quitting. Destroying a thread pool while its worker threads
# are still running tasks makes Qt report worker-thread exceptions (and can
# even abort the process), so the in-flight work is given a brief, bounded
# chance to finish. The wait is capped so that quitting is never blocked for
# a long time: all of the work is read-only, so abandoning it is safe.
SHUTDOWN_TIMEOUT_MS = 2000


def get_file_paths(directory_path: Path) -> set[Path]:
    """
    Recursively get all file paths in a directory, including those in
    subdirectories.
    """
    file_paths = set()
    for path in directory_path.iterdir():
        if path.is_file():
            file_paths.add(path)
        elif path.is_dir():
            file_paths.update(get_file_paths(path))
    return file_paths


@dataclass
class HistoryItem:
    action_name: str
    tags: list[list[str]]
    descriptions: list[str]
    should_ask_for_confirmation: bool
    # The grouped tag panel's tag order from just before this change was
    # applied, so undoing/redoing it puts every tag back exactly where it was.
    # Set by the Image Tags pane; None when the change was not made in the
    # grouped view.
    group_order_state: object = None


class Scope(str, Enum):
    ALL_IMAGES = 'All images'
    FILTERED_IMAGES = 'Filtered images'
    SELECTED_IMAGES = 'Selected images'


class Location(str, Enum):
    CAPTIONS = 'Captions'
    DESCRIPTIONS = 'Descriptions'


class DirectoryLoadWorker(QObject):
    """
    Collect the images of one directory load and notify the model when the
    load is complete.

    `directory_loaded` is emitted from a worker thread and delivered to the
    connected slot on the GUI thread (queued connection).
    """
    # (generation, loaded images)
    directory_loaded = Signal(int, list)

    def __init__(self, generation: int, parent: QObject = None):
        super().__init__(parent)
        self.generation = generation
        self._lock = threading.Lock()
        self._pending_task_count = 0
        self._collected_images: list[Image] = []

    def task_started(self):
        with self._lock:
            self._pending_task_count += 1

    def task_finished(self, image: Image | None):
        with self._lock:
            if image is not None:
                self._collected_images.append(image)
            self._pending_task_count -= 1
            is_last_task = self._pending_task_count == 0
        if is_last_task:
            self.directory_loaded.emit(self.generation,
                                       self._collected_images)


class DirectoryLoadTask(QRunnable):
    """
    Load the metadata (dimensions, tags, description) of one image on a
    worker thread.

    The task is added to a `registry` set and removed from it when it is
    done. The set keeps a strong reference to the task so that it is not
    garbage collected while the thread pool still references it.
    """

    def __init__(self, image_path: Path, storage: ImageStorage,
                 worker: DirectoryLoadWorker, registry: set):
        super().__init__()
        # The thread pool must not delete the C++ object while Python may
        # still reference it.
        self.setAutoDelete(False)
        self.image_path = image_path
        self.storage = storage
        self.worker = worker
        self.registry = registry
        registry.add(self)
        worker.task_started()

    def run(self):
        image = None
        try:
            dimensions = None
            try:
                dimensions = imagesize.get(self.image_path)
                # Check the Exif orientation tag and rotate the dimensions if
                # necessary.
                with open(self.image_path, 'rb') as image_file:
                    try:
                        exif_tags = exifread.process_file(
                            image_file, details=False,
                            stop_tag='Image Orientation')
                        if 'Image Orientation' in exif_tags:
                            orientations = (exif_tags['Image Orientation']
                                            .values)
                            if any(value in orientations
                                   for value in (5, 6, 7, 8)):
                                dimensions = (dimensions[1], dimensions[0])
                    except Exception as exception:
                        print(f'Failed to get Exif tags for {self.image_path}: '
                              f'{exception}', file=sys.stderr)
            except (ValueError, OSError) as exception:
                print(f'Failed to get dimensions for {self.image_path}: '
                      f'{exception}', file=sys.stderr)
                dimensions = None
            tags, description, content_hash = (
                self.storage.load_image(self.image_path))
            image = Image(self.image_path, dimensions, tags, description,
                          content_hash=content_hash)
        except Exception as exception:
            print(f'Failed to load {self.image_path}: {exception}',
                  file=sys.stderr)
        self._notify_worker(image)
        self.registry.discard(self)

    def _notify_worker(self, image: Image | None):
        try:
            self.worker.task_finished(image)
        except RuntimeError:
            # The worker (and therefore the image list model) has been
            # destroyed, so there is no one to notify.
            pass


class ImageListModel(QAbstractListModel):
    update_undo_and_redo_actions_requested = Signal()
    # Emitted on the GUI thread when a directory has finished loading and the
    # images are available.
    directory_loaded = Signal()

    def __init__(self, image_list_image_width: int, tag_separator: str):
        super().__init__()
        self.image_list_image_width = image_list_image_width
        self.tag_separator = tag_separator
        # Where the tags and descriptions of the loaded images are read from
        # and written to (a SQLite database or caption files next to the
        # images), based on the storage backend setting.
        self.storage = get_image_storage()
        self.images: list[Image] = []
        self.undo_stack = deque(maxlen=UNDO_STACK_SIZE)
        self.redo_stack = []
        # What the most recent `add_to_undo_stack` call displaced (the oldest
        # undo entry dropped by the size limit, and the cleared redo stack), so
        # a bulk operation that ends up changing nothing can be fully undone
        # from the history. None when there is nothing to restore.
        self._last_undo_push = None
        # True only while restoring tags from the undo/redo history. Consumers
        # (e.g. the Common/Differences panel) use this to distinguish a genuine
        # new edit from a history restore, so an undone/redone tag returns to
        # its original position while a freshly added tag lands at the end.
        self.is_restoring_history = False
        # While restoring: (item being restored, its reverse item). See
        # `restore_history_tags_and_description`.
        self.restoring_history_items = None
        # The entry most recently added by `add_to_undo_stack`, until the
        # grouped tag panel has saved its tag order with it (see HistoryItem).
        self.newest_history_item = None
        # Set only while a rename-like operation reports its change: maps each
        # renamed tag to the tag(s) that replaced it, so the grouped tag panel
        # can keep the new name in the old tag's position.
        self.tag_renames: dict[str, list[str]] | None = None
        self.proxy_image_list_model = None
        self.image_list_selection_model = None
        # Thumbnails are generated on background threads so that image
        # decoding never blocks the GUI thread (which made scrolling the
        # image list slow with many images).
        self.thumbnail_worker = ThumbnailGenerationWorker(self)
        self.thumbnail_worker.thumbnail_ready.connect(self.thumbnail_ready)
        self.thumbnail_thread_pool = QThreadPool(self)
        self.thumbnail_thread_pool.setMaxThreadCount(
            min(8, os.cpu_count() or 4))
        # Directory contents are loaded on background threads.
        self.directory_load_thread_pool = QThreadPool(self)
        self.directory_load_thread_pool.setMaxThreadCount(
            min(8, os.cpu_count() or 4))
        self.directory_load_worker = None
        # Incremented for each directory load. Used to discard the results of
        # loads that are superseded by a newer load.
        self.directory_load_generation = 0
        # Incremented for each thumbnail generation pass. Used to discard the
        # results of passes that are superseded by a newer directory load.
        self.thumbnail_generation = 0
        # Strong references to running tasks so that they are not garbage
        # collected while the thread pools still reference them.
        self._running_tasks: set = set()
        # When the application is about to quit, give the background tasks a
        # bounded chance to finish so that the thread pools are not destroyed
        # while worker threads are still running tasks (which makes Qt report
        # worker-thread exceptions, and can even abort the process).
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.wait_for_pending_work)

    def rowCount(self, parent=None) -> int:
        return len(self.images)

    def data(self, index, role=None) -> Image | str | QIcon | QSize:
        image = self.images[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return image
        if role == Qt.ItemDataRole.DisplayRole:
            # The text shown next to the thumbnail in the image list.
            text = image.path.name
            if image.tags:
                caption = self.tag_separator.join(image.tags)
                text += f'\n{caption}'
            return text
        if role == Qt.ItemDataRole.ToolTipRole:
            # The full file name and tag list, for when the image list
            # delegate only shows a truncated preview.
            text = image.path.name
            if image.tags:
                text += f'\n{self.tag_separator.join(image.tags)}'
            return text
        if role == Qt.ItemDataRole.DecorationRole:
            # The thumbnail, if it has been generated yet. Thumbnails are
            # generated on background threads (see
            # `start_thumbnail_generation`), so `data()` never blocks the GUI
            # thread on image decoding.
            return image.thumbnail
        if role == Qt.ItemDataRole.SizeHintRole:
            if image.thumbnail:
                return image.thumbnail.availableSizes()[0]
            dimensions = image.dimensions
            if not dimensions:
                return QSize(self.image_list_image_width,
                             self.image_list_image_width)
            width, height = dimensions
            # Scale the dimensions to the image width.
            return QSize(self.image_list_image_width,
                         int(self.image_list_image_width * height / width))

    def load_directory(self, directory_path: Path):
        """
        Load the images in a directory.

        The load happens on background threads so that the GUI thread stays
        responsive. When the load is complete, `directory_loaded` is emitted
        on the GUI thread.
        """
        # The view now shows an empty list until the load is complete.
        self.beginResetModel()
        self.images.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._last_undo_push = None
        self.endResetModel()
        self.update_undo_and_redo_actions_requested.emit()
        # Discard the results of any in-flight thumbnail generation pass.
        self.thumbnail_generation += 1
        self.directory_load_generation += 1
        generation = self.directory_load_generation
        file_paths = get_file_paths(directory_path)
        settings = get_settings()
        image_suffixes_string = settings.value(
            'image_list_file_formats',
            defaultValue=DEFAULT_SETTINGS['image_list_file_formats'], type=str)
        image_suffixes = []
        for suffix in image_suffixes_string.split(','):
            suffix = suffix.strip().lower()
            if not suffix.startswith('.'):
                suffix = '.' + suffix
            image_suffixes.append(suffix)
        image_paths = {path for path in file_paths
                       if path.suffix.lower() in image_suffixes}
        if not image_paths:
            # There are no images to load.
            self.directory_loaded.emit()
            return
        worker = DirectoryLoadWorker(generation, self)
        self.directory_load_worker = worker
        worker.directory_loaded.connect(self.on_directory_loaded)
        # Create all tasks (which increments the worker's pending-task count)
        # before starting any of them. If tasks were started as they were
        # created, a fast task could finish and trigger the "last task" check
        # before the remaining tasks were even counted, causing
        # `directory_loaded` to be emitted (partially) more than once.
        tasks = []
        for image_path in image_paths:
            tasks.append(DirectoryLoadTask(image_path, self.storage, worker,
                                           self._running_tasks))
        for task in tasks:
            self.directory_load_thread_pool.start(task)

    @Slot(int, list)
    def on_directory_loaded(self, generation: int,
                            images: list[Image]):
        """Complete a directory load. Runs on the GUI thread."""
        # The load has been superseded by a newer load.
        if generation != self.directory_load_generation:
            return
        images.sort(key=lambda image_: image_.path)
        self.beginResetModel()
        self.images = images
        self.endResetModel()
        self.directory_loaded.emit()
        self.start_thumbnail_generation()

    def start_thumbnail_generation(self):
        """Start generating thumbnails on background threads."""
        self.thumbnail_generation += 1
        generation = self.thumbnail_generation
        for row, image in enumerate(self.images):
            task = ThumbnailGenerationTask(
                row, image.path, self.image_list_image_width, generation,
                self.thumbnail_worker, self._running_tasks)
            self.thumbnail_thread_pool.start(task)

    @Slot(int, QImage, int)
    def thumbnail_ready(self, row: int, image: QImage, generation: int):
        """Receive a generated thumbnail. Runs on the GUI thread."""
        # The pass has been superseded by a newer directory load, or the
        # image is no longer in the model.
        if generation != self.thumbnail_generation or row >= len(self.images):
            return
        image_model_image = self.images[row]
        if image_model_image.thumbnail is not None or image.isNull():
            return
        image_model_image.thumbnail = QIcon(QPixmap.fromImage(image))
        index = self.index(row, 0)
        self.dataChanged.emit(index, index,
                              [Qt.ItemDataRole.DecorationRole])

    @Slot()
    def wait_for_pending_work(self):
        """
        Shut down the background task pools when the application is quitting.

        The queued tasks are discarded and the few tasks already running are
        given a bounded chance to finish, so that the pools are destroyed
        without worker threads still inside a task (which makes Qt report
        worker-thread exceptions, and can even abort the process). All of
        the work is read-only and its results are no longer needed, so
        discarding it is safe.
        """
        for pool in (self.directory_load_thread_pool,
                     self.thumbnail_thread_pool):
            pool.clear()
            pool.waitForDone(SHUTDOWN_TIMEOUT_MS)

    def add_to_undo_stack(self, action_name: str,
                          should_ask_for_confirmation: bool):
        """Add the current state of the image tags and descriptions to the
        undo stack."""
        tags = [image.tags.copy() for image in self.images]
        descriptions = [image.description for image in self.images]
        self._push_history_item(HistoryItem(
            action_name, tags, descriptions, should_ask_for_confirmation))

    def _push_history_item(self, history_item: HistoryItem):
        dropped_item = (self.undo_stack[0]
                        if len(self.undo_stack) == self.undo_stack.maxlen
                        else None)
        self._last_undo_push = (dropped_item, self.redo_stack.copy())
        self.newest_history_item = history_item
        self.undo_stack.append(self.newest_history_item)
        self.redo_stack.clear()
        self.update_undo_and_redo_actions_requested.emit()

    def discard_last_undo_entry(self):
        """Remove the entry just added by `add_to_undo_stack`, for an operation
        that turned out to change nothing, restoring the history exactly as it
        was before (including the redo stack)."""
        if self._last_undo_push is None or not self.undo_stack:
            return
        dropped_item, previous_redo_stack = self._last_undo_push
        self._last_undo_push = None
        if self.undo_stack[-1] is self.newest_history_item:
            self.newest_history_item = None
        self.undo_stack.pop()
        if dropped_item is not None:
            self.undo_stack.appendleft(dropped_item)
        self.redo_stack[:] = previous_redo_stack
        self.update_undo_and_redo_actions_requested.emit()

    @Slot(Image)
    def write_image_tags_to_disk(self, image: Image):
        try:
            self.storage.save_image(image)
        except (OSError, sqlite3.Error) as exception:
            print(f'Failed to save tags for {image.path}: {exception}',
                  file=sys.stderr)
            error_message_box = QMessageBox()
            error_message_box.setWindowTitle('Error')
            error_message_box.setIcon(QMessageBox.Icon.Critical)
            error_message_box.setText(f'Failed to save tags for {image.path}.')
            error_message_box.exec()

    def set_storage(self, storage: ImageStorage):
        """
        Switch the active storage backend (called after a successful storage
        mode switch in the settings).
        """
        self.storage = storage

    def export_loaded_images_to_text_files(self) -> tuple[int, list[Path]]:
        """
        Write the loaded images' tags and descriptions to caption files next
        to the images (used when switching to text-file storage with the
        override checked). Only images with non-empty tags or a description
        are written, so no empty caption files are created. Return the number
        of written caption files and the paths of the images that failed to
        write.
        """
        images = [image for image in self.images
                  if image.tags or image.description]
        failed_paths = export_images_to_text_files(images, self.tag_separator)
        return len(images) - len(failed_paths), failed_paths

    def import_loaded_images_to_database(self) -> int:
        """
        Import the loaded images' tags and descriptions into the database
        (used when switching to database storage with the override checked).
        Only images with non-empty tags or a description are imported, so
        records of images without data are left untouched. Return the number
        of imported images.
        """
        images = [image for image in self.images
                  if image.tags or image.description]
        import_images(images, get_database_path())
        return len(images)

    def restore_history_tags_and_description(self, is_undo: bool):
        if is_undo:
            source_stack = self.undo_stack
            destination_stack = self.redo_stack
        else:
            # Redo.
            source_stack = self.redo_stack
            destination_stack = self.undo_stack
        if not source_stack:
            return
        history_item = source_stack[-1]
        if history_item.should_ask_for_confirmation:
            undo_or_redo_string = 'Undo' if is_undo else 'Redo'
            reply = get_confirmation_dialog_reply(
                title=undo_or_redo_string,
                question=f'{undo_or_redo_string} '
                         f'"{history_item.action_name}"?')
            if reply != QMessageBox.StandardButton.Yes:
                return
        source_stack.pop()
        self._last_undo_push = None
        tags = [image.tags.copy() for image in self.images]
        descriptions = [image.description for image in self.images]
        reverse_item = HistoryItem(
            history_item.action_name, tags, descriptions,
            history_item.should_ask_for_confirmation)
        destination_stack.append(reverse_item)
        changed_image_indices = []
        for (image_index, (image, history_image_tags,
                           history_image_description)) in enumerate(
                zip(self.images, history_item.tags,
                    history_item.descriptions)):
            if (image.tags == history_image_tags
                    and image.description == history_image_description):
                continue
            changed_image_indices.append(image_index)
            image.tags = history_image_tags.copy()
            image.description = history_image_description
            self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.is_restoring_history = True
            # Lets the grouped tag panel restore the tag order saved with the
            # item being restored, and save the current order with its reverse.
            self.restoring_history_items = (history_item, reverse_item)
            try:
                self.dataChanged.emit(
                    self.index(changed_image_indices[0]),
                    self.index(changed_image_indices[-1]))
            finally:
                self.is_restoring_history = False
                self.restoring_history_items = None
        self.update_undo_and_redo_actions_requested.emit()

    @Slot()
    def undo(self):
        """Undo the last action."""
        self.restore_history_tags_and_description(is_undo=True)

    @Slot()
    def redo(self):
        """Redo the last undone action."""
        self.restore_history_tags_and_description(is_undo=False)

    def is_image_in_scope(self, scope: Scope | str, image_index: int,
                          image: Image) -> bool:
        if scope == Scope.ALL_IMAGES:
            return True
        if scope == Scope.FILTERED_IMAGES:
            return self.proxy_image_list_model.is_image_in_filtered_images(
                image)
        if scope == Scope.SELECTED_IMAGES:
            proxy_index = self.proxy_image_list_model.mapFromSource(
                self.index(image_index))
            return self.image_list_selection_model.isSelected(proxy_index)

    def get_images_in_scope(self, scope: Scope | str) -> list[Image]:
        """The images in the given scope, in model order."""
        return [image for image_index, image in enumerate(self.images)
                if self.is_image_in_scope(scope, image_index, image)]

    def get_text_match_count(self, text: str, scope: Scope | str,
                             whole_tags_only: bool, use_regex: bool,
                             location: Location | str =
                             Location.CAPTIONS) -> int:
        """Get the number of instances of a text in all captions or
        descriptions."""
        match_count = 0
        for image_index, image in enumerate(self.images):
            if not self.is_image_in_scope(scope, image_index, image):
                continue
            if location == Location.DESCRIPTIONS:
                if use_regex:
                    match_count += len(re.findall(pattern=text,
                                                  string=image.description))
                else:
                    match_count += image.description.count(text)
            elif whole_tags_only:
                if use_regex:
                    match_count += len([
                        tag for tag in image.tags
                        if re.fullmatch(pattern=text, string=tag)
                    ])
                else:
                    match_count += image.tags.count(text)
            else:
                caption = self.tag_separator.join(image.tags)
                if use_regex:
                    match_count += len(re.findall(pattern=text,
                                                  string=caption))
                else:
                    match_count += caption.count(text)
        return match_count

    def find_and_replace(self, find_text: str, replace_text: str,
                         scope: Scope | str, use_regex: bool,
                         location: Location | str = Location.CAPTIONS):
        """
        Find and replace arbitrary text in captions, within and across tag
        boundaries, or in descriptions.
        """
        if not find_text:
            return
        self.add_to_undo_stack(action_name='Find and Replace',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if not self.is_image_in_scope(scope, image_index, image):
                continue
            if location == Location.DESCRIPTIONS:
                text = image.description
            else:
                text = self.tag_separator.join(image.tags)
            if use_regex:
                if not re.search(pattern=find_text, string=text):
                    continue
                text = re.sub(pattern=find_text, repl=replace_text,
                              string=text)
            else:
                if find_text not in text:
                    continue
                text = text.replace(find_text, replace_text)
            changed_image_indices.append(image_index)
            if location == Location.DESCRIPTIONS:
                image.description = text
            else:
                image.tags = text.split(self.tag_separator)
            self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))

    def sort_tags_alphabetically(self, do_not_reorder_first_tag: bool):
        """Sort the tags for each image in alphabetical order."""
        self.add_to_undo_stack(action_name='Sort Tags',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if len(image.tags) < 2:
                continue
            old_caption = self.tag_separator.join(image.tags)
            if do_not_reorder_first_tag:
                first_tag = image.tags[0]
                image.tags = [first_tag] + sorted(image.tags[1:])
            else:
                image.tags.sort()
            new_caption = self.tag_separator.join(image.tags)
            if new_caption != old_caption:
                changed_image_indices.append(image_index)
                self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))

    def sort_tags_by_frequency(self, tag_counter: Counter,
                               do_not_reorder_first_tag: bool):
        """
        Sort the tags for each image by the total number of times a tag appears
        across all images.
        """
        self.add_to_undo_stack(action_name='Sort Tags',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if len(image.tags) < 2:
                continue
            old_caption = self.tag_separator.join(image.tags)
            if do_not_reorder_first_tag:
                first_tag = image.tags[0]
                image.tags = [first_tag] + sorted(
                    image.tags[1:], key=lambda tag: tag_counter[tag],
                    reverse=True)
            else:
                image.tags.sort(key=lambda tag: tag_counter[tag], reverse=True)
            new_caption = self.tag_separator.join(image.tags)
            if new_caption != old_caption:
                changed_image_indices.append(image_index)
                self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))

    def reverse_tags_order(self, do_not_reorder_first_tag: bool):
        """Reverse the order of the tags for each image."""
        self.add_to_undo_stack(action_name='Reverse Order of Tags',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if len(image.tags) < 2:
                continue
            changed_image_indices.append(image_index)
            if do_not_reorder_first_tag:
                image.tags = [image.tags[0]] + list(reversed(image.tags[1:]))
            else:
                image.tags = list(reversed(image.tags))
            self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))

    def shuffle_tags(self, do_not_reorder_first_tag: bool):
        """Shuffle the tags for each image randomly."""
        self.add_to_undo_stack(action_name='Shuffle Tags',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if len(image.tags) < 2:
                continue
            changed_image_indices.append(image_index)
            if do_not_reorder_first_tag:
                first_tag, *remaining_tags = image.tags
                random.shuffle(remaining_tags)
                image.tags = [first_tag] + remaining_tags
            else:
                random.shuffle(image.tags)
            self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))

    def move_tags_to_front(self, tags_to_move: list[str]):
        """
        Move one or more tags to the front of the tags list for each image.
        """
        self.add_to_undo_stack(action_name='Move Tags to Front',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if not any(tag in image.tags for tag in tags_to_move):
                continue
            old_caption = self.tag_separator.join(image.tags)
            moved_tags = []
            for tag in tags_to_move:
                tag_count = image.tags.count(tag)
                moved_tags.extend([tag] * tag_count)
            unmoved_tags = [tag for tag in image.tags if tag not in moved_tags]
            image.tags = moved_tags + unmoved_tags
            new_caption = self.tag_separator.join(image.tags)
            if new_caption != old_caption:
                changed_image_indices.append(image_index)
                self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))

    def remove_duplicate_tags(self) -> int:
        """
        Remove duplicate tags for each image. Return the number of removed
        tags.
        """
        self.add_to_undo_stack(action_name='Remove Duplicate Tags',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        removed_tag_count = 0
        for image_index, image in enumerate(self.images):
            tag_count = len(image.tags)
            unique_tag_count = len(set(image.tags))
            if tag_count == unique_tag_count:
                continue
            changed_image_indices.append(image_index)
            removed_tag_count += tag_count - unique_tag_count
            # Use a dictionary instead of a set to preserve the order.
            image.tags = list(dict.fromkeys(image.tags))
            self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))
        return removed_tag_count

    def remove_empty_tags(self) -> int:
        """
        Remove empty tags (tags that are empty strings or only contain
        whitespace) for each image. Return the number of removed tags.
        """
        self.add_to_undo_stack(action_name='Remove Empty Tags',
                               should_ask_for_confirmation=True)
        changed_image_indices = []
        removed_tag_count = 0
        for image_index, image in enumerate(self.images):
            old_tag_count = len(image.tags)
            image.tags = [tag for tag in image.tags if tag.strip()]
            new_tag_count = len(image.tags)
            if old_tag_count == new_tag_count:
                continue
            changed_image_indices.append(image_index)
            removed_tag_count += old_tag_count - new_tag_count
            self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))
        return removed_tag_count

    def update_image_tags(self, image_index: QModelIndex, tags: list[str]):
        image: Image = self.data(image_index, Qt.ItemDataRole.UserRole)
        if image.tags == tags:
            return
        image.tags = tags
        # Save before emitting `dataChanged`, like every other change:
        # listeners (e.g. the tag suggestions) may read the storage, so it
        # must already reflect the new tags and description.
        self.write_image_tags_to_disk(image)
        self.dataChanged.emit(image_index, image_index)

    def update_image_description(self, image_index: QModelIndex, description: str):
        image: Image = self.data(image_index, Qt.ItemDataRole.UserRole)
        if image.description == description:
            return
        image.description = description
        # Save before emitting `dataChanged` (see `update_image_tags`).
        self.write_image_tags_to_disk(image)
        self.dataChanged.emit(image_index, image_index)

    def _valid_indices(self, image_indices: list) -> list[QModelIndex]:
        """Drop indices that no longer point at an image (e.g. after the
        image was removed), so a stale target can't write elsewhere."""
        return [index for index in image_indices
                if index is not None and index.isValid()
                and 0 <= index.row() < len(self.images)]

    @Slot(list, list)
    def add_tags(self, tags: list[str], image_indices: list[QModelIndex]):
        """Add one or more tags to one or more images."""
        self.add_tags_per_image([(index, tags) for index in image_indices],
                                f'Add {pluralize("Tag", len(tags))}')

    def add_tags_per_image(self, tags_by_index: list[tuple[QModelIndex,
                                                           list[str]]],
                           action_name: str):
        """Add each image's own list of tags to it, as one undo step. Tags
        an image already has are skipped."""
        valid_rows = range(len(self.images))
        tags_by_index = [(index, tags) for index, tags in tags_by_index
                         if tags and index is not None and index.isValid()
                         and index.row() in valid_rows]
        if not tags_by_index:
            return
        should_ask_for_confirmation = len(tags_by_index) > 1
        self.add_to_undo_stack(action_name, should_ask_for_confirmation)
        changed_indices = []
        for image_index, tags in tags_by_index:
            image: Image = self.data(image_index, Qt.ItemDataRole.UserRole)
            # A tag that only differs in letter case from one the image
            # already has counts as a duplicate, like in the Image Tags pane.
            existing = {tag.casefold() for tag in image.tags}
            new_tags = []
            for tag in tags:
                if tag.casefold() not in existing:
                    existing.add(tag.casefold())
                    new_tags.append(tag)
            if not new_tags:
                continue
            image.tags.extend(new_tags)
            self.write_image_tags_to_disk(image)
            changed_indices.append(image_index)
        if not changed_indices:
            self.discard_last_undo_entry()
            return
        min_image_index = min(changed_indices, key=lambda index: index.row())
        max_image_index = max(changed_indices, key=lambda index: index.row())
        self.dataChanged.emit(min_image_index, max_image_index)

    @Slot(list, list)
    def remove_tags_from_images(self, tags: list[str],
                                image_indices: list[QModelIndex]):
        """Remove one or more tags from an explicit set of images.

        Unlike `delete_tags` (which works by scope across the whole model),
        this targets exactly the given image indices, as needed by the grouped
        multi-image tags view ("remove from all selected" / "remove from the
        current image"). Only images that actually contain one of the tags are
        modified and written to disk.
        """
        image_indices = self._valid_indices(image_indices)
        if not image_indices or not tags:
            return
        tags_to_remove = set(tags)
        action_name = f'Remove {pluralize("Tag", len(tags))}'
        should_ask_for_confirmation = len(image_indices) > 1
        self.add_to_undo_stack(action_name, should_ask_for_confirmation)
        changed_indices = []
        for image_index in image_indices:
            image: Image = self.data(image_index, Qt.ItemDataRole.UserRole)
            if not any(tag in tags_to_remove for tag in image.tags):
                continue
            image.tags = [tag for tag in image.tags
                          if tag not in tags_to_remove]
            self.write_image_tags_to_disk(image)
            changed_indices.append(image_index)
        if not changed_indices:
            self.discard_last_undo_entry()
            return
        min_image_index = min(changed_indices, key=lambda index: index.row())
        max_image_index = max(changed_indices, key=lambda index: index.row())
        self.dataChanged.emit(min_image_index, max_image_index)

    @Slot(str, str, list)
    def rename_tag_in_images(self, old_tag: str, new_tag: str,
                             image_indices: list[QModelIndex]):
        """Rename a tag on an explicit set of images (grouped view in-place
        edit).

        Mirrors `remove_tags_from_images` but for an in-place rename: only
        the given images that actually contain `old_tag` are modified, the new
        tag takes the old tag's position, and duplicates are collapsed. No-op
        when the new tag is empty or unchanged. A new name containing the tag
        separator (e.g. "a, b") is split into several tags, like the Add Tag
        box does, so the list matches what the caption file reloads as.
        """
        new_tags = [tag.strip() for tag in new_tag.split(self.tag_separator)
                    if tag.strip()]
        image_indices = self._valid_indices(image_indices)
        if (not image_indices or not old_tag or not new_tags
                or new_tags == [old_tag]):
            return
        targets = [index for index in image_indices
                   if old_tag in self.data(index,
                                           Qt.ItemDataRole.UserRole).tags]
        if not targets:
            return
        self.add_to_undo_stack(
            action_name='Rename Tag',
            should_ask_for_confirmation=len(targets) > 1)
        for image_index in targets:
            image: Image = self.data(image_index, Qt.ItemDataRole.UserRole)
            renamed = []
            for tag in image.tags:
                renamed.extend(new_tags if tag == old_tag else [tag])
            image.tags = list(dict.fromkeys(renamed))
            self.write_image_tags_to_disk(image)
        min_image_index = min(targets, key=lambda index: index.row())
        max_image_index = max(targets, key=lambda index: index.row())
        self._emit_renamed(min_image_index, max_image_index,
                           {old_tag: new_tags})

    def _emit_renamed(self, first_index: QModelIndex, last_index: QModelIndex,
                      renames: dict[str, list[str]]):
        """Emit dataChanged while publishing which tags were renamed to what
        (see `tag_renames`)."""
        self.tag_renames = renames
        try:
            self.dataChanged.emit(first_index, last_index)
        finally:
            self.tag_renames = None

    @staticmethod
    def _note_rename(renames: dict[str, list[str]], old_tag: str,
                     new_tags: list[str]):
        """Record the first replacement seen for `old_tag` (ignoring no-ops)."""
        if new_tags != [old_tag] and old_tag not in renames:
            renames[old_tag] = list(new_tags)

    @Slot(list, str)
    def rename_tags(self, old_tags: list[str], new_tag: str,
                    scope: Scope | str = Scope.ALL_IMAGES,
                    use_regex: bool = False):
        pattern = re.compile(old_tags[0]) if use_regex else None
        self.add_to_undo_stack(
            action_name=f'Rename {pluralize("Tag", len(old_tags))}',
            should_ask_for_confirmation=True)
        changed_image_indices = []
        renames: dict[str, list[str]] = {}
        for image_index, image in enumerate(self.images):
            if not self.is_image_in_scope(scope, image_index, image):
                continue
            if use_regex:
                if not any(pattern.fullmatch(image_tag)
                           for image_tag in image.tags):
                    continue
                for image_tag in image.tags:
                    if pattern.fullmatch(image_tag):
                        self._note_rename(renames, image_tag, [new_tag])
                image.tags = [new_tag if pattern.fullmatch(image_tag)
                              else image_tag for image_tag in image.tags]
            else:
                if not any(old_tag in image.tags for old_tag in old_tags):
                    continue
                image.tags = [new_tag if image_tag in old_tags else image_tag
                              for image_tag in image.tags]
            changed_image_indices.append(image_index)
            self.write_image_tags_to_disk(image)
        if not use_regex:
            for old_tag in old_tags:
                self._note_rename(renames, old_tag, [new_tag])
        if changed_image_indices:
            self._emit_renamed(self.index(changed_image_indices[0]),
                               self.index(changed_image_indices[-1]),
                               renames)
        else:
            self.discard_last_undo_entry()

    @Slot(list)
    def delete_tags(self, tags: list[str],
                    scope: Scope | str = Scope.ALL_IMAGES,
                    use_regex: bool = False):
        self.add_to_undo_stack(
            action_name=f'Delete {pluralize("Tag", len(tags))}',
            should_ask_for_confirmation=True)
        changed_image_indices = []
        for image_index, image in enumerate(self.images):
            if not self.is_image_in_scope(scope, image_index, image):
                continue
            if use_regex:
                pattern = tags[0]
                if not any(re.fullmatch(pattern=pattern, string=image_tag)
                           for image_tag in image.tags):
                    continue
                image.tags = [image_tag for image_tag in image.tags
                              if not re.fullmatch(pattern=pattern,
                                                  string=image_tag)]
            else:
                if not any(tag in image.tags for tag in tags):
                    continue
                image.tags = [image_tag for image_tag in image.tags
                              if image_tag not in tags]
            changed_image_indices.append(image_index)
            self.write_image_tags_to_disk(image)
        if changed_image_indices:
            self.dataChanged.emit(self.index(changed_image_indices[0]),
                                  self.index(changed_image_indices[-1]))
        else:
            self.discard_last_undo_entry()
