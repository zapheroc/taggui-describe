"""
"Find Duplicates" review dialog: exact and near-duplicate detection.

Two modes (the "Mode" combo box at the top):

- 'Duplicates' — the exact-duplicate port (Phase 1) of the dialog in
  taggui-plus (https://github.com/Overtime-Artist/taggui-plus): scans the
  loaded images (all, filtered or selected) for images with
  byte-for-byte identical content — using the content hash the app already
  computes for every loaded image, so no file is re-read or decoded. The
  scan is instant and runs automatically when the dialog opens.
- 'Near Duplicates' — near-duplicate ("look-alike") detection by 256-bit
  perceptual hash (pHash): images join a group when their pHashes differ by
  at most the threshold. The scan decodes every uncached image, so it waits
  for an explicit "Scan" press.

In Duplicates mode the user picks one image to keep per group (the "Keep"
radio, exclusive per group); in Near Duplicates mode the user marks one or
more images to keep per group (the "Keep" check box — the group's
representative is kept by default), and the actions then apply to the
images that are not kept. Acting is non-destructive (delete to Recycle
Bin, move to a folder, or tag), optionally copying the tags of the acted-
on images to the kept ones first. In database storage mode the copies of
an exact duplicate share one tag record, so those tag actions are disabled
in Duplicates mode; near-duplicate images are different content (and
different records) and keep them in both storage modes. The kept picks are
kept in memory for the dialog's lifetime (they survive an in-session
regroup or "Show ignored" toggle) and reset when the dialog is closed or
a new scan runs.

Every pHash is cached in the database, keyed by the image's content hash
(schema version 5, the `phash` table), so a rescan decodes and hashes only
the new images. The threshold slider and the "Match mirrored versions"
option are inert until a "Scan" press: the scan re-groups the cached
neighbor lists in memory when the scanned image set and the mirror option
are unchanged and the threshold is at or below the scanned one, and
re-runs the distance pass on the cached hashes otherwise (either way no
file is re-read as long as the hashes are cached). Ignored groups are
persisted in the same database (the `ignored_similarity_groups` table) and
can be forgotten with "Clear group ignores"; "Clear similarity cache"
drops the cached pHashes.

This module is imported lazily by the main window when the dialog is
opened: the Near Duplicates mode pulls in `numpy` (via `models.similarity`)
and `imagehash`/`scipy` (in the scan worker), which the application must
not load at startup.
"""

import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import (QAbstractListModel, QCoreApplication, QEvent,
                            QModelIndex, QObject, QRunnable, Qt, QThreadPool,
                            QTimer, Signal, Slot)
from PySide6.QtGui import (QFont, QIcon, QKeySequence, QPainter,
                           QPalette, QPen, QShortcut)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox,
                                QComboBox, QDialog, QFileDialog, QFrame,
                                QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                                QMenu, QMessageBox, QProgressDialog,
                                QPushButton, QRadioButton, QScrollArea,
                                QSizePolicy, QVBoxLayout, QWidget)

from models.image_list_model import ImageListModel, Scope
from models.image_storage import (
    buffer_phash_row, clear_group_ignores, clear_similarity_cache,
    flush_phash_cache, load_ignored_groups, load_phashes,
    remove_ignored_group, store_ignored_group)
from models.similarity import (PHASH_BITS, PHASH_VERSION, ScanCancelled,
                               TooManyMatches, compute_neighbors,
                               compute_phash, compute_phash_mirrored,
                               phash_from_hex, phash_to_hex, seed_group)
from utils.image import Image
from utils.settings import DEFAULT_SETTINGS, get_settings
from utils.settings_widgets import (FocusedScrollSettingsComboBox,
                                    FocusedScrollSettingsSlider,
                                    SettingsBigCheckBox)
from utils.utils import get_confirmation_dialog_reply, pluralize
from widgets.image_list import move_images_to_folder, trash_images
from widgets.image_viewer import ImageViewer

# The dialog's modes (the "Mode" combo box's items, and the values the
# `similarity_mode` setting holds).
MODE_DUPLICATES = 'Duplicates'
MODE_NEAR_DUPPLICATES = 'Near Duplicates'

# The mode-dependent description shown at the top of the dialog.
_DESCRIPTIONS = {
    MODE_DUPLICATES: (
        'Finds images with byte-for-byte identical content (the same file '
        'saved more than once). In each group, choose the image to keep, '
        'then act on the others. Click a thumbnail to view it larger. '
        'Deleted images go to the Recycle Bin.'),
    MODE_NEAR_DUPPLICATES: (
        'Finds images that look alike: their 256-bit perceptual hashes '
        '(pHash) differ by at most the threshold below. In each group, the '
        'representative is the image the group is built around; it is kept '
        'by default. Check "Keep" for every image to keep (one or more — '
        'clicking a card works too; uncheck them all to act on the whole '
        'group) and act on the images that are not kept. Grouping is '
        'non-transitive: '
        'an image joins a group only if it is within the threshold of that '
        'group\'s representative, so a chain of similar images is not '
        'collapsed into one giant group. Click a thumbnail to view it '
        'larger. Deleted images go to the Recycle Bin.'),
}

# Thumbnail edge length (pixels) for the cards in a group.
_THUMBNAIL_SIZE = 140
# Rendering caps so a pathological result set (e.g. thousands of identical
# training images merged into giant groups) can never freeze the UI or
# exhaust memory. Groups are shown this many at a time ("Show next ...
# groups"), and only this many members per group until the group is
# expanded.
_MAX_GROUPS_SHOWN = 100
_MAX_MEMBERS_PER_GROUP = 12
# The tag applied by the "Tag as duplicate" action.
_DUPLICATE_TAG = 'duplicate'

# The cap on total neighbor links stored for one near-duplicate scan (one
# link is a (member, distance, mirrored) entry on one side of a pair, so
# this is twice the pair count). A threshold so broad that nearly everything
# matches everything (e.g. random images at a high threshold) would
# otherwise allocate unbounded lists; ~5M links are a few hundred MB.
_MAX_SCAN_LINKS = 5_000_000
# The maximum per-seed neighbor list kept for grouping: a seed with more
# neighbors than this (possible only at high thresholds) is grouped with its
# closest members only.
_MAX_GROUP_SIZE = 20_000


def _worker_count() -> int:
    # The distance pass releases the GIL, so scale with the CPU count,
    # capped (the numpy calls are already multi-threaded per worker).
    return max(1, min(16, os.cpu_count() or 4))


# Rules for picking the image to keep in each group.
KEEP_LARGEST_RESOLUTION = 'Largest resolution'
KEEP_LARGEST_FILE = 'Largest file'
KEEP_NEWEST_FILE = 'Newest file'
KEEP_OLDEST_FILE = 'Oldest file'
KEEP_MOST_TAGS = 'Most tags'
KEEP_RULES = [KEEP_LARGEST_RESOLUTION, KEEP_LARGEST_FILE, KEEP_NEWEST_FILE,
              KEEP_OLDEST_FILE, KEEP_MOST_TAGS]


class _ClickableLabel(QLabel):
    """A QLabel that emits ``clicked`` when pressed with the left mouse
    button."""

    clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


def _apply_indicator_palette(button):
    # Only the Base color is overridden, so the other colors keep following
    # the app theme.
    window_color = button.palette().color(QPalette.ColorRole.Window)
    if button.palette().color(QPalette.ColorRole.Base) != window_color:
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Base, window_color)
        button.setPalette(palette)


class _IndicatorPaletteFilter(QObject):
    """Updates the overridden indicator color when the app theme changes."""

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.PaletteChange:
            _apply_indicator_palette(watched)
        return False


_indicator_palette_filter = None


def _make_indicator_visible(button):
    """Fusion fills check box and radio button indicators with the Base
    color, which would make them invisible on the dark results background.
    Use the window color for them instead."""
    global _indicator_palette_filter
    if _indicator_palette_filter is None:
        _indicator_palette_filter = _IndicatorPaletteFilter()
    _apply_indicator_palette(button)
    button.installEventFilter(_indicator_palette_filter)


class _DuplicateCard(QFrame):
    """A frameless card; a kept one is outlined in the highlight color,
    like a selected image in the Images pane. Clicking anywhere on the card
    (except the thumbnail, which opens the preview) changes the image's
    kept state — picking it to be kept (the "Keep" radio, Duplicates mode)
    or toggling it (the "Keep" check box, Near Duplicates mode) — so the
    click target is the whole card, not just the widget.

    The cards of an ignored group are rendered differently: the group's
    whole card area is disabled, which the Fusion style draws in black and
    white (and makes unclickable), so the group reads as "not to be acted
    on"."""

    clicked = Signal()

    def __init__(self):
        super().__init__()
        self.kept = False
        self._preview_zone: QWidget | None = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_preview_zone(self, widget: QWidget):
        """The child that opens the preview on click; clicks inside it do
        not change the card's image's kept state."""
        self._preview_zone = widget

    def mousePressEvent(self, event):
        if (event.button() == Qt.MouseButton.LeftButton
                and (self._preview_zone is None
                     or not self._preview_zone.rect().contains(
                         event.position().toPoint()))):
            self.clicked.emit()
        super().mousePressEvent(event)

    def set_kept(self, kept: bool):
        if kept != self.kept:
            self.kept = kept
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.kept:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self.palette().color(QPalette.ColorRole.Highlight),
                            2))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 4, 4)


def _format_size(num_bytes) -> str:
    """Return a short human-readable file size such as '1.2 MB'."""
    if num_bytes is None:
        return '? '
    size = float(num_bytes)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            if unit == 'B':
                return f'{int(size)} {unit}'
            return f'{size:.1f} {unit}'
        size /= 1024
    return f'{size:.1f} GB'


def _format_modified(mtime_ns) -> str:
    """Return the file's modified date (what "Newest file" and "Oldest file"
    compare), e.g. 'Modified 2024-05-01 14:32'."""
    if mtime_ns is None:
        return 'Modified ?'
    try:
        stamp = datetime.fromtimestamp(mtime_ns / 1e9)
    except (OverflowError, OSError, ValueError):
        return 'Modified ?'
    return f'Modified {stamp:%Y-%m-%d %H:%M}'


def _elide(text: str, limit: int = 22) -> str:
    if len(text) <= limit:
        return text
    return text[:limit - 1] + '\u2026'


class _VerticalWheelForwarder(QObject):
    """Sends plain vertical mouse-wheel scrolling over a group's row of cards
    (including its horizontal scroll bar) to the results list, so scrolling
    down the results never gets caught by a row that slides under the cursor.
    Shift+wheel and sideways (touchpad) scrolling scroll the row."""

    def __init__(self, target_scroll_area: QScrollArea, row_scroll_area:
                 QScrollArea, parent=None):
        super().__init__(parent)
        self.target_scroll_area = target_scroll_area
        self.row_scroll_area = row_scroll_area

    def eventFilter(self, watched, event):
        if event.type() != QEvent.Type.Wheel:
            return False
        delta = event.angleDelta()
        if delta.x() != 0 or delta.y() == 0:
            return False
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            bar = self.row_scroll_area.horizontalScrollBar()
            steps = (-delta.y() / 120 * QApplication.wheelScrollLines()
                     * bar.singleStep())
            bar.setValue(bar.value() + round(steps))
            return True
        QApplication.sendEvent(self.target_scroll_area.verticalScrollBar(),
                               event)
        return True


class _DuplicateScanTask(QObject, QRunnable):
    """Runs the duplicate scan off the UI thread.

    The scan is an in-memory group-by over `Image.content_hash` (the hash
    the storage layer already resolved for every loaded image) plus one
    `os.stat` per image for the size and modified date shown on the cards —
    no file is re-read or decoded.

    Follows the repo's background-work convention: `setAutoDelete(False)` and
    a strong-reference `registry` set the task removes itself from when it is
    done, so the thread pool can never reference a garbage-collected task.
    The `generation` number lets the dialog discard the result when a newer
    scan superseded it.
    """

    # (groups: list of lists of Image, skipped: int)
    finished = Signal(list, int)

    def __init__(self, images: list, generation: int, registry: set):
        QObject.__init__(self)
        QRunnable.__init__(self)
        self.images = images
        self.generation = generation
        self.registry = registry
        # Image path -> (size, mtime_ns), filled in by `run`.
        self.stats: dict = {}
        self._cancelled = False
        # The thread pool must not delete the C++ object while Python may
        # still reference it.
        self.setAutoDelete(False)
        registry.add(self)

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _member_sort_key(self, image):
        width, height = (image.dimensions or (0, 0))
        pixels = (width or 0) * (height or 0)
        size, mtime_ns = self.stats.get(image.path, (0, 0))
        return (pixels, size or 0, mtime_ns or 0)

    @Slot()
    def run(self):
        try:
            skipped = 0
            groups_by_hash: dict = {}
            for image in self.images:
                if self._cancelled:
                    break
                if image.content_hash is None:
                    # The content hash could not be resolved (for example,
                    # text-file mode with an unusable database); the image
                    # cannot be compared.
                    skipped += 1
                    continue
                try:
                    stat_result = image.path.stat()
                except OSError:
                    # The file no longer exists on disk; the directory
                    # reload will drop it.
                    skipped += 1
                    continue
                self.stats[image.path] = (stat_result.st_size,
                                          stat_result.st_mtime_ns)
                groups_by_hash.setdefault(image.content_hash, []).append(image)
            # Keep only the real duplicate groups, largest first; within a
            # group, sort by resolution, size and modified date (descending)
            # so the most "complete-looking" copy comes first.
            groups = []
            for members in groups_by_hash.values():
                if len(members) < 2:
                    continue
                members.sort(key=self._member_sort_key, reverse=True)
                groups.append(members)
            groups.sort(key=len, reverse=True)
            # Drop the image list so a finished task holds no large data.
            self.images = None
            self._emit_finished(groups, skipped)
        finally:
            self.registry.discard(self)

    def _emit_finished(self, groups: list, skipped: int):
        try:
            self.finished.emit(groups, skipped)
        except RuntimeError:
            # The dialog was closed and destroyed while the scan was
            # running; there is no one to notify.
            pass


# The registry of running exact-duplicate scan tasks (see
# `_DuplicateScanTask`).
_scan_registry: set = set()


def _resolution_key(member, stats: dict) -> tuple:
    width, height = (member.dimensions or (0, 0))
    size, _mtime_ns = stats.get(member.path, (0, 0))
    return ((width or 0) * (height or 0), size or 0)


def choose_keeper(members: list, rule: str, tag_counts: dict,
                  stats: dict):
    """Return the member to keep under `rule`. `members` is a list of
    `Image`. `tag_counts` maps an image path to its number of tags;
    `stats` maps a path to (size, mtime_ns).
    Ties fall back to the largest resolution and then to the earliest
    member."""
    def key(member):
        entry = stats.get(member.path)
        size = entry[0] if entry else 0
        # A missing entry is an unknown date (in practice every grouped
        # image is in `stats`), which counts as the newest.
        mtime_ns = entry[1] if entry else None
        resolution = _resolution_key(member, stats)
        if rule == KEEP_LARGEST_FILE:
            return (size or 0,) + resolution
        if rule == KEEP_NEWEST_FILE:
            return (mtime_ns or 0,) + resolution
        if rule == KEEP_OLDEST_FILE:
            if mtime_ns is None:
                return (-float('inf'),) + resolution
            return (-mtime_ns,) + resolution
        if rule == KEEP_MOST_TAGS:
            return (tag_counts.get(member.path, 0),) + resolution
        return resolution

    best = members[0]
    best_key = key(best)
    for member in members[1:]:
        member_key = key(member)
        if member_key > best_key:
            best, best_key = member, member_key
    return best


def describe_relation(member, keeper) -> str:
    """Return a short label describing how `member` compares with the kept
    image (every group member is byte-for-byte identical to it)."""
    if member is keeper:
        return 'Kept'
    return 'Identical file'


class _PhashScanTask(QObject, QRunnable):
    """
    Scan the in-scope images for near-duplicate groups.

    Runs in a `QThreadPool` worker thread, in three phases:

    1. Resolve the pHash of every distinct content hash in scope (the
       database cache first, then `imagehash` for the misses — with the
       "match mirrored versions" option on, the mirrored hash too).
    2. Compute the neighbor links with `compute_neighbors` (threaded).
    3. (On the GUI thread) group them with `seed_group`.

    Follows the repo's background-work convention: `setAutoDelete(False)` and
    a strong-reference `registry` set the task removes itself from when it is
    done, so the thread pool can never reference a garbage-collected task.
    The `generation` number lets the dialog discard the result when a newer
    scan superseded it.

    The `finished` signal carries one dict with the in-memory scan state the
    dialog keeps for threshold changes and ignore actions (re-grouping
    without re-reading any file):

    - `neighbors`: for every compact position, the (index, distance,
      mirrored) list of the other positions within the scanned threshold
      (the complete matrix; a regroup at a lower threshold re-filters
      this list in memory, in `_filter_neighbors`);
    - `hashes`: the 256-bit pHash (int) of each compact position;
    - `mirrors`: the mirrored pHash of each compact position, or None;
    - `hashes_keys`: the content hash of each compact position;
    - `hash_rows`: content hash -> list of in-scope model rows;
    - `stats`: image path -> (size, mtime_ns);
    - `skipped`: the number of in-scope images that could not be compared
      (no content hash, or an unreadable/unhashable file);
    - `scanned_threshold`: the threshold the distance pass used.

    `failed` carries an error message (cancel included); the real
    cancellation state is read from `cancelled`, never from the progress
    dialog.
    """

    # `object`, not `dict`: PySide6 cannot convert a Python `dict` to the
    # C++ type a queued signal needs (the conversion silently corrupts the
    # payload — even small ints fail — and the 256-bit pHash values raise
    # OverflowError). A queued `Signal(object)` delivers the dict as-is.
    finished = Signal(object)
    failed = Signal(str)
    # (completed, total) of the hash-computation phase (0/0 when everything
    # was already cached).
    progress = Signal(int, int)

    def __init__(self, images: list[Image], model: ImageListModel,
                 generation: int, registry: set, scope: Scope | str,
                 threshold: int, match_mirrored: bool,
                 db_path, parent: QObject = None):
        QObject.__init__(self, parent)
        QRunnable.__init__(self)
        self.images = images
        self.model = model
        self.generation = generation
        self.registry = registry
        self.scope = scope
        self.threshold = threshold
        self.match_mirrored = match_mirrored
        self.db_path = db_path
        self._cancelled = False
        # The thread pool must not delete the C++ object while Python may
        # still reference it.
        self.setAutoDelete(False)
        registry.add(self)

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _scope_rows(self) -> list[int]:
        """The model rows of the in-scope images, in model order."""
        return [i for i, image in enumerate(self.images)
                if self.model.is_image_in_scope(self.scope, i, image)]

    @Slot()
    def run(self):
        try:
            self._run_scan()
        except ScanCancelled:
            self._emit_failed('Scan cancelled.')
        except TooManyMatches:
            self._emit_failed(
                'Too many matches: the threshold is so broad that nearly '
                'everything matches everything. Lower the threshold and '
                'rescan.')
        except Exception as e:
            self._emit_failed(str(e) or repr(e))
        finally:
            self.registry.discard(self)

    def _run_scan(self):
        images = self.images
        rows = self._scope_rows()

        # Group the in-scope rows by content hash: the pHash is a property
        # of the image's bytes, so each distinct hash is decoded and hashed
        # once regardless of how many copies are in scope (every copy is
        # reported).
        by_hash: dict[str, list[int]] = {}
        skipped = 0
        for i in rows:
            content_hash = images[i].content_hash
            if content_hash is None:
                # The content hash could not be resolved (for example,
                # text-file mode with an unusable database); the image
                # cannot be compared.
                skipped += 1
            else:
                by_hash.setdefault(content_hash, []).append(i)
        if len(by_hash) < 2:
            self._emit_finished(self._empty_result(skipped))
            return

        # One stat per in-scope image, for the cards (size and modified
        # date) and the seed ordering.
        stats: dict = {}
        for i in rows:
            try:
                stat_result = images[i].path.stat()
            except OSError:
                # The file no longer exists on disk; the directory reload
                # will drop it.
                continue
            stats[images[i].path] = (stat_result.st_size,
                                     stat_result.st_mtime_ns)

        # Resolve every distinct hash's pHash from the cache; queue the
        # misses for computation.
        cached = load_phashes(self.db_path)
        keys = list(by_hash.keys())
        count = len(keys)
        phashes: list = [None] * count
        mirrors: list = [None] * count
        todo_full = []    # positions whose pHash must be computed
        todo_mirror = []  # positions that only need their mirrored hash
        for k, content_hash in enumerate(keys):
            entry = cached.get(content_hash)
            if entry is not None and entry[2] == PHASH_VERSION:
                phashes[k] = entry[0]
                mirrors[k] = entry[1]
                if (self.match_mirrored and entry[1] is None
                        and entry[0] is not None):
                    todo_mirror.append(k)
            else:
                todo_full.append(k)

        if todo_full or todo_mirror:
            self._compute_missing(keys, phashes, mirrors,
                                  todo_full, todo_mirror, by_hash)
        if self.cancelled:
            raise ScanCancelled()

        present = [k for k in range(count) if phashes[k] is not None]
        skipped += count - len(present)
        if len(present) < 2:
            self._emit_finished(self._empty_result(skipped))
            return

        # Compact, index-aligned arrays for the distance pass.
        compact_hashes = [phash_from_hex(phashes[k]) for k in present]
        compact_mirrors = None
        if self.match_mirrored:
            # Every position needs a mirrored hash for the index-aligned
            # min(direct, mirrored) comparison; when one is missing (the
            # mirrored image could not be hashed), fall back to direct-only
            # matching for the whole run.
            if all(mirrors[k] is not None for k in present):
                compact_mirrors = [phash_from_hex(mirrors[k])
                                   for k in present]

        neighbors = compute_neighbors(
            compact_hashes, self.threshold,
            mirrored_hashes=compact_mirrors,
            worker_count=_worker_count(),
            should_cancel=lambda: self.cancelled,
            max_pairs=_MAX_SCAN_LINKS)
        if self.cancelled:
            raise ScanCancelled()

        # Cap each seed's neighbor list (its far members are never picked
        # by the greedy seeding — see `_MAX_GROUP_SIZE`).
        for n in neighbors:
            if len(n) > _MAX_GROUP_SIZE:
                n.sort(key=lambda t: (t[1], t[0]))
                del n[_MAX_GROUP_SIZE:]

        self._emit_finished({
            'neighbors': neighbors,
            'hashes': compact_hashes,
            'mirrors': compact_mirrors,
            'hashes_keys': [keys[k] for k in present],
            'hash_rows': {keys[k]: by_hash[keys[k]] for k in present},
            'stats': stats,
            'skipped': skipped,
            'scanned_threshold': self.threshold,
        })

    @staticmethod
    def _empty_result(skipped: int) -> dict:
        return {'neighbors': [], 'hashes': [], 'mirrors': None,
                'hashes_keys': [], 'hash_rows': {}, 'stats': {},
                'skipped': skipped, 'scanned_threshold': 0}

    def _compute_missing(self, keys: list, phashes: list, mirrors: list,
                         todo_full: list, todo_mirror: list,
                         by_hash: dict):
        """Compute the pHashes (and, with the option on, mirrored pHashes)
        of the cached-hash misses on a thread pool, buffering the rows for a
        batched write to the database."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _work(path, need_phash, need_mirror):
            if self.cancelled:
                return None
            result = (None, None)
            if need_phash:
                phash = compute_phash(path)
                if phash is None:
                    return None  # unhashable file: stays out of the scan
                result = (phash_to_hex(phash), result[1])
            if need_mirror:
                mirror = compute_phash_mirrored(path)
                if mirror is not None:
                    result = (result[0], phash_to_hex(mirror))
            if result[0] is None and result[1] is None:
                return None
            return result

        # One representative path per content hash (any copy works: the
        # bytes are identical).
        rep_path = {k: self.images[min(by_hash[keys[k]])].path
                    for k in set(todo_full) | set(todo_mirror)}

        futures = {}
        total = len(todo_full) + len(todo_mirror)
        done = 0
        with ThreadPoolExecutor(max_workers=_worker_count()) as executor:
            for k in todo_full:
                futures[executor.submit(
                    _work, rep_path[k], True, self.match_mirrored)] = (
                    k, True)
            for k in todo_mirror:
                futures[executor.submit(
                    _work, rep_path[k], False, True)] = (k, False)
            for future in as_completed(futures):
                if self.cancelled:
                    flush_phash_cache(self.db_path)
                    raise ScanCancelled()
                k, need_phash = futures[future]
                result = future.result()
                if result is not None:
                    phash_hex, mirror_hex = result
                    if need_phash:
                        phashes[k] = phash_hex
                    if mirror_hex is not None:
                        mirrors[k] = mirror_hex
                    buffer_phash_row(self.db_path, keys[k],
                                     phashes[k], PHASH_VERSION,
                                     mirrors[k])
                done += 1
                self.progress.emit(done, total)
        flush_phash_cache(self.db_path)

    def _emit_finished(self, result: dict):
        try:
            self.finished.emit(result)
        except RuntimeError:
            # The dialog was closed and destroyed while the scan was
            # running; there is no one to notify.
            pass

    def _emit_failed(self, message: str):
        try:
            self.failed.emit(message)
        except RuntimeError:
            pass


# The registry of running scan tasks (see `_PhashScanTask`).
_phash_scan_registry: set = set()


class _SimilarGroup:
    """One near-duplicate group: the representative (the seed) first, then
    the other members.

    `members` are dicts:

    - `image`: the `Image` at `row`;
    - `row`: the model row;
    - `hash`: the content hash (all copies of it share one member entry per
      row);
    - `distance`: the pHash difference from the representative, in bits
      (0..256), None for the representative itself;
    - `mirrored`: whether the mirrored version of the image matched the
      representative better than the direct image did (the mirror option).
    """

    def __init__(self, dialog, seed_hash: str, group_key: str,
                 ignored: bool, members: list):
        self.dialog = dialog
        self.seed_hash = seed_hash
        # 'phash:<threshold>:<seed_hash>' — the persisted ignore key.
        self.group_key = group_key
        self.ignored = ignored
        self.members = members


# Group membership is unified across the modes: a group is a list of
# `Image` in Duplicates mode and a `_SimilarGroup` in Near Duplicates
# mode (whose members are dicts). These helpers read either shape.


def _member_image(member) -> Image:
    """The `Image` of a group member (a member is an `Image` in
    Duplicates mode, a member dict with an 'image' key in Near
    Duplicates mode)."""
    return member['image'] if isinstance(member, dict) else member


def _group_members(group) -> list:
    """The members of a group (see `_member_image`)."""
    return group.members if isinstance(group, _SimilarGroup) else group


class FindDuplicatesDialog(QDialog):
    def __init__(self, parent, image_list_model: ImageListModel,
                 reload_callback, select_paths_callback=None):
        super().__init__(parent)
        # The dialog is opened afresh every time, so free it when it closes.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.image_list_model = image_list_model
        self.reload_callback = reload_callback
        self.select_paths_callback = select_paths_callback
        self.settings = get_settings()
        # The similarity accessors are storage-agnostic module functions
        # taking the database path (both backends expose `db_path`; in
        # text-file mode they only ever touch the similarity tables).
        self.db_path = image_list_model.storage.db_path
        self._scanner = None
        self._progress_dialog: QProgressDialog | None = None
        self._scan_generation = 0
        self._scanned_count = 0
        self._scanned_threshold = 0
        # The threshold the current in-memory grouping was computed at
        # (the slider's value at the last scan or fast regroup — the slider
        # is inert until a "Scan" press), the mirror option and the image
        # paths of the last scan (what the neighbor lists were computed
        # over): the conditions of the fast in-memory regroup on the next
        # "Scan" press.
        self._applied_threshold = 0
        self._scanned_mirrored = False
        self._scanned_paths: frozenset = frozenset()
        # The results of the last scan: the groups as scanned (Duplicates
        # mode) and the in-memory pHash state (Near Duplicates mode), kept
        # so that a "Scan" at a lower threshold or an ignore action
        # can re-group without re-reading any file.
        self._scanned_groups: list = []
        self._neighbors: list = []
        self._hashes: list = []
        self._mirrors: list | None = None
        self._hashes_keys: list = []
        self._hash_rows: dict = {}
        self._stats: dict = {}
        self._skipped_count = 0
        # Image path -> number of tags, for the cards.
        self._tag_counts: dict = {}
        # Card thumbnails for the current results, reused as instant
        # placeholders in the preview.
        self._thumbnails: dict = {}
        # The current results: the groups (filtered by "Show ignored") and
        # the cards of the groups currently shown.
        self._groups: list = []
        self._group_cards: list = []
        # The images the user marked to keep (image paths, in memory only
        # for the dialog's lifetime): a group's kept images are exactly its
        # members whose path is in this set. The set always mirrors the
        # cards' state: when a group's cards are first built, its kept
        # state is seeded with the default kept image (the representative
        # — or the keep rule's pick in Duplicates mode), and every toggle
        # records the group's checked widgets. A rebuild (a "Show ignored"
        # toggle, a regroup, "Show next 100 groups") re-applies the set
        # instead of resetting the user's picks. The groups' memberships
        # are disjoint (each image is in at most one group — `seed_group`
        # consumes its unassigned set), so the set is safe to manage per
        # group.
        self._kept_images: set = set()
        # The group keys whose kept state is already in `_kept_images`
        # (seeded or user-toggled): "no recorded state" means "the group's
        # cards are not built yet" — not "the user unchecked everything",
        # which is a recorded empty state that must survive a rebuild.
        # Keys are stable across a "Show ignored" rebuild (same threshold,
        # same seeds) and change with a regroup, where the picks follow
        # the kept images instead.
        self._kept_initialized: set = set()
        # The persisted group dismissals (the dismissed group keys).
        self._ignored_groups: set = set(
            row[0] for row in load_ignored_groups(self.db_path))
        # In database storage mode the copies of an exact duplicate share
        # one tag record, so the tag actions below are disabled there in
        # Duplicates mode (near-duplicate images are different content and
        # different records, and keep them in both storage modes).
        self._shared_records = (
            self.image_list_model.storage.shared_content_records)
        self._actions_enabled = False

        self.setWindowTitle('Find Duplicates')
        # Roomy enough for several groups of cards, but never bigger than the
        # screen.
        screen = QApplication.primaryScreen()
        width, height = 1120, 900
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, int(available.width() * 0.9))
            height = min(height, int(available.height() * 0.9))
        self.resize(width, height)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.description_label = QLabel(_DESCRIPTIONS[MODE_DUPLICATES])
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)

        # --- Scan controls --------------------------------------------------
        scan_grid = QGridLayout()
        scan_grid.setHorizontalSpacing(8)
        scan_grid.addWidget(QLabel('Mode:'), 0, 0)
        self.mode_combo_box = FocusedScrollSettingsComboBox(
            key='similarity_mode', default=MODE_DUPLICATES)
        self.mode_combo_box.addItems([MODE_DUPLICATES,
                                      MODE_NEAR_DUPPLICATES])
        # A stored value that is not one of the items leaves the combo on
        # its first item ('Duplicates'): setCurrentText is a no-op for
        # non-item values, and the setting is rewritten when the user next
        # changes the combo.
        self.mode_combo_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.mode_combo_box.setSizePolicy(QSizePolicy.Policy.Fixed,
                                          QSizePolicy.Policy.Fixed)
        scan_grid.addWidget(self.mode_combo_box, 0, 1)
        scan_grid.addWidget(QLabel('Scope:'), 0, 2)
        scope_row = QHBoxLayout()
        self.scope_combo_box = FocusedScrollSettingsComboBox(
            key='duplicate_scan_scope', default=Scope.ALL_IMAGES.value)
        self.scope_combo_box.addItems([scope.value for scope in Scope])
        if self.scope_combo_box.currentText() \
                not in [scope.value for scope in Scope]:
            self.scope_combo_box.setCurrentText(Scope.ALL_IMAGES.value)
        self.scope_combo_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.scope_combo_box.setSizePolicy(QSizePolicy.Policy.Fixed,
                                           QSizePolicy.Policy.Fixed)
        self.scope_combo_box.currentTextChanged.connect(
            self._update_scope_count_label)
        scope_row.addWidget(self.scope_combo_box)
        self.scope_count_label = QLabel()
        scope_row.addSpacing(6)
        scope_row.addWidget(self.scope_count_label)
        scope_row.addStretch()
        scan_grid.addLayout(scope_row, 0, 3, 1, 2)
        self.scan_button = QPushButton('Scan')
        self.scan_button.clicked.connect(self.start_scan)
        scan_grid.addWidget(self.scan_button, 0, 5)
        scan_grid.setColumnStretch(3, 1)

        # The pHash controls (the second row) are Near Duplicates mode
        # only; the "Show ignored" checkbox is always visible (both modes
        # persist group ignores).
        self.threshold_label = QLabel('Max pHash difference:')
        self.threshold_label.setToolTip(
            'Images join a group when their perceptual hashes differ by at '
            'most this many bits (out of 256). Moving the slider changes '
            'nothing until you press "Scan".')
        scan_grid.addWidget(self.threshold_label, 1, 0)
        self.threshold_slider = FocusedScrollSettingsSlider(
            key='similarity_phash_max_distance', default=6,
            minimum=0, maximum=100)
        self.threshold_slider.setToolTip(
            'The change applies the next time you press "Scan".')
        self.threshold_slider.valueChanged.connect(
            self._on_threshold_changed)
        scan_grid.addWidget(self.threshold_slider, 1, 1)
        self.threshold_value_label = QLabel()
        self.threshold_value_label.setText(
            str(self.threshold_slider.value()))
        scan_grid.addWidget(self.threshold_value_label, 1, 2)
        self.match_mirrored_check_box = SettingsBigCheckBox(
            key='similarity_match_mirrored', default=False,
            text='Match mirrored versions')
        # Toggling only persists the option: it takes effect on the next
        # "Scan" press (the mirror changes the pair distances, so a full
        # distance pass runs then).
        self.match_mirrored_check_box.setToolTip(
            'The change applies the next time you press "Scan".')
        scan_grid.addWidget(self.match_mirrored_check_box, 1, 3, 1, 2)
        self.show_ignored_check_box = SettingsBigCheckBox(
            key='similarity_show_ignored', default=True,
            text='Show ignored')
        # The state is restored from the settings in the constructor (and
        # the `toggled` signal fires there, before this connection
        # exists), so connecting afterwards cannot re-apply a stale
        # grouping.
        self.show_ignored_check_box.toggled.connect(self._apply_grouping)
        scan_grid.addWidget(self.show_ignored_check_box, 1, 5)
        layout.addLayout(scan_grid)
        self._update_scope_count_label()

        # --- Results area ---------------------------------------------------
        # A darker background (the color of input fields) sets the results
        # apart from the controls, and each group sits on it as a panel.
        self.results_scroll_area = QScrollArea()
        self.results_scroll_area.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_container.setObjectName('duplicateResults')
        self.results_container.setBackgroundRole(QPalette.ColorRole.Base)
        self.results_container.setAutoFillBackground(True)
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(10, 10, 10, 10)
        self.results_layout.setSpacing(12)
        self.results_layout.addStretch()
        self.results_scroll_area.setWidget(self.results_container)
        layout.addWidget(self.results_scroll_area, 1)

        status_row = QHBoxLayout()
        self.status_label = QLabel(self._idle_status_text())
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 1)
        self.show_more_button = QPushButton()
        self.show_more_button.clicked.connect(self.show_more_groups)
        self.show_more_button.hide()
        status_row.addWidget(self.show_more_button)
        layout.addLayout(status_row)

        # --- Keeper + actions -----------------------------------------------
        actions_box = QGroupBox('Act on the results')
        actions_layout = QGridLayout(actions_box)
        actions_layout.setColumnStretch(4, 1)

        actions_layout.addWidget(QLabel('Keep in each group:'), 0, 0)
        self.keep_rule_combo_box = FocusedScrollSettingsComboBox(
            key='duplicate_keep_rule', default=KEEP_LARGEST_RESOLUTION)
        self.keep_rule_combo_box.addItems(KEEP_RULES)
        if self.keep_rule_combo_box.currentText() not in KEEP_RULES:
            self.keep_rule_combo_box.setCurrentText(
                KEEP_LARGEST_RESOLUTION)
        actions_layout.addWidget(self.keep_rule_combo_box, 0, 1)
        self.apply_rule_button = QPushButton('Apply to all groups')
        self.apply_rule_button.clicked.connect(self.apply_keep_rule)
        actions_layout.addWidget(self.apply_rule_button, 0, 2)

        self.merge_tags_check_box = SettingsBigCheckBox(
            key='duplicate_merge_tags', default=False,
            text='Copy the tags of the other images to the kept image')
        actions_layout.addWidget(self.merge_tags_check_box, 1, 0, 1, 5)

        actions_layout.addWidget(QLabel('Other images:'), 2, 0)
        self.delete_button = QPushButton('Delete to Recycle Bin')
        self.delete_button.clicked.connect(self.delete_others)
        actions_layout.addWidget(self.delete_button, 2, 1)
        self.move_button = QPushButton('Move to folder...')
        self.move_button.clicked.connect(self.move_others)
        actions_layout.addWidget(self.move_button, 2, 2)
        self.tag_button = QPushButton(f'Tag as "{_DUPLICATE_TAG}"')
        self.tag_button.clicked.connect(self.tag_others)
        actions_layout.addWidget(self.tag_button, 2, 3)

        self.clear_cache_button = QPushButton('Clear similarity cache')
        self.clear_cache_button.setToolTip(
            'Delete the cached perceptual hashes of every image: the next '
            'Near Duplicates scan has to re-decode and re-hash them. '
            'Ignored groups are kept.')
        self.clear_cache_button.clicked.connect(self.clear_cache)
        actions_layout.addWidget(self.clear_cache_button, 3, 0, 1, 3)
        self.clear_ignores_button = QPushButton('Clear group ignores')
        self.clear_ignores_button.setToolTip(
            'Forget every ignored group (both modes): '
            'they are shown again on the next scan or regrouping. The '
            'cached perceptual hashes are kept.')
        self.clear_ignores_button.clicked.connect(self.clear_ignores)
        actions_layout.addWidget(self.clear_ignores_button, 3, 3, 1, 2)
        layout.addWidget(actions_box)

        # --- Mode-dependent state (wired last) -------------------------------
        self._update_mode_widgets()
        self._set_actions_enabled(False)
        # Connect the mode combo last, so the setup above (which sets the
        # initial mode) cannot trigger a mode switch.
        self.mode_combo_box.currentTextChanged.connect(
            self._on_mode_changed)
        # The exact-duplicate scan reads no file (it groups by the content
        # hash the app already computed), so it runs as soon as the dialog
        # is opened and the groups are visible without pressing "Scan".
        # Near duplicates decode every uncached image and always wait for
        # an explicit press. (Started in `showEvent`, not here, so a
        # dialog that is constructed but never shown never scans.)
        self._auto_scan_pending = (self.mode == MODE_DUPLICATES
                                   and bool(self.image_list_model.images))

    def showEvent(self, event):
        super().showEvent(event)
        if self._auto_scan_pending:
            self._auto_scan_pending = False
            # Deferred to the next event-loop iteration: the dialog is
            # still laying itself out while it is being shown.
            QTimer.singleShot(0, self.start_scan)

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self.mode_combo_box.currentText()

    def _update_mode_widgets(self):
        """Show/hide the mode-dependent controls and refresh the texts."""
        near = self.mode == MODE_NEAR_DUPPLICATES
        self.threshold_label.setVisible(near)
        self.threshold_slider.setVisible(near)
        self.threshold_value_label.setVisible(near)
        self.match_mirrored_check_box.setVisible(near)
        self.clear_cache_button.setVisible(near)
        self.description_label.setText(_DESCRIPTIONS[self.mode])
        self._update_tag_action_state()

    @Slot(str)
    def _on_mode_changed(self, mode: str):
        if self._scanner is not None:
            # A scan of the old mode is running; let it finish, the switch
            # takes effect from the next scan on.
            return
        self._update_mode_widgets()
        # The other mode's results are meaningless here.
        self._clear_results()
        self.status_label.setText(self._idle_status_text())
        self._set_actions_enabled(False)
        if mode == MODE_DUPLICATES:
            # The exact scan is instant: run it right away.
            self.start_scan()

    def _idle_status_text(self) -> str:
        if self.mode == MODE_NEAR_DUPPLICATES:
            return 'Click "Scan" to look for similar images.'
        return 'Click "Scan" to look for duplicates.'

    def _group_noun(self) -> str:
        """'duplicate' in Duplicates mode, 'similar' in Near Duplicates
        mode (for the status texts)."""
        return ('similar' if self.mode == MODE_NEAR_DUPPLICATES
                else 'duplicate')

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    @property
    def scope(self) -> str:
        return self.scope_combo_box.currentText()

    def _images_in_scope(self) -> list:
        return self.image_list_model.get_images_in_scope(self.scope)

    @Slot()
    def _update_scope_count_label(self):
        count = len(self._images_in_scope())
        self.scope_count_label.setText(
            f'({count} {pluralize("image", count)})')

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    @Slot()
    def start_scan(self):
        if self._scanner is not None:
            return
        images = self._images_in_scope()
        self._update_scope_count_label()
        if len(images) < 2:
            if not self.image_list_model.images:
                text = 'No images are loaded. Load a directory first.'
            else:
                noun = ('similar images' if self.mode == MODE_NEAR_DUPPLICATES
                        else 'duplicates')
                text = (f'The scope "{self.scope}" contains '
                        f'{len(images)} {pluralize("image", len(images))}. '
                        f'At least 2 are needed to look for {noun}.')
            self.status_label.setText(text)
            return
        self._scanned_count = len(images)
        if (self.mode == MODE_NEAR_DUPPLICATES
                and self._fast_regroup_available(images)):
            # The scanned image set and the mirror option are unchanged,
            # and the slider is at or below the scanned threshold: re-group
            # the cached neighbor lists in memory at the slider's value
            # (no file is re-read, no database row is touched, no progress
            # dialog).
            self._applied_threshold = self.threshold_slider.value()
            self._apply_grouping()
            return
        self._clear_results()
        self._set_actions_enabled(False)
        self.scan_button.setEnabled(False)
        self.clear_cache_button.setEnabled(False)
        self._scan_generation += 1
        if self.mode == MODE_NEAR_DUPPLICATES:
            self._start_near_scan(images)
        else:
            self._start_exact_scan(images)

    def _fast_regroup_available(self, images: list) -> bool:
        """Whether a "Scan" press can re-group the last scan's cached
        neighbor lists in memory instead of running a new scan: the scanned
        image set is unchanged, the mirror option is unchanged, and the
        slider is at or below the threshold the last scan computed
        neighbors for (raising it above that needs the distance pass
        re-run, and a changed mirror option changes the pair distances)."""
        if self._scan_generation == 0 or not self._neighbors:
            return False
        if (self.match_mirrored_check_box.isChecked()
                != self._scanned_mirrored):
            return False
        if self.threshold_slider.value() > self._scanned_threshold:
            return False
        scanned_paths = self._scanned_paths
        if len(scanned_paths) != len(images):
            return False
        return all(image.path in scanned_paths for image in images)

    def _start_near_scan(self, images: list):
        # A busy bar until the task reports how many hashes it has to
        # compute (then it becomes a real progress bar); the cancel button
        # stops the scan.
        progress = QProgressDialog(
            'Scanning images for similar ones\u2026', 'Cancel', 0, 0, self)
        progress.setWindowTitle('Find Duplicates')
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        # Don't let Qt auto-hide the dialog the instant the bar reaches its
        # maximum: it is switched to a "Preparing groups" busy bar and then
        # hidden with reset() at the end.
        progress.setAutoReset(False)
        progress.setValue(0)
        self._progress_dialog = progress

        # What this scan computes (recorded for the fast in-memory regroup
        # path on the next "Scan" press; the in-memory state is cleared for
        # the duration of the scan, so a cancelled scan leaves that path
        # unavailable).
        self._scanned_threshold = self.threshold_slider.value()
        self._scanned_mirrored = self.match_mirrored_check_box.isChecked()
        self._scanned_paths = frozenset(image.path for image in images)

        scanner = _PhashScanTask(
            images, self.image_list_model, self._scan_generation,
            _phash_scan_registry, self.scope, self._scanned_threshold,
            self._scanned_mirrored, self.db_path)
        self._scanner = scanner
        # Bound methods (not lambdas) so the connections end when the dialog
        # is deleted.
        scanner.finished.connect(self._on_near_scan_finished)
        scanner.failed.connect(self._on_scan_failed)
        scanner.progress.connect(self._on_scan_progress)
        progress.canceled.connect(scanner.cancel)
        QThreadPool.globalInstance().start(scanner)

    def _start_exact_scan(self, images: list):
        progress = QProgressDialog('Scanning images for duplicates...',
                                   'Cancel', 0, len(images), self)
        progress.setWindowTitle('Find Duplicates')
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        # Don't let Qt auto-hide the dialog the instant the bar reaches its
        # maximum: it is switched to a "Preparing groups" busy bar and then
        # hidden with reset() at the end.
        progress.setAutoReset(False)
        progress.setValue(0)
        self._progress_dialog = progress

        scanner = _DuplicateScanTask(images, self._scan_generation,
                                     _scan_registry)
        self._scanner = scanner
        scanner.finished.connect(self._on_exact_scan_finished)
        progress.canceled.connect(scanner.cancel)
        QThreadPool.globalInstance().start(scanner)

    @Slot(int, int)
    def _on_scan_progress(self, done: int, total: int):
        progress = self._progress_dialog
        if progress is not None and total > 0:
            progress.setRange(0, total)
            progress.setValue(done)
            progress.setLabelText(
                f'Computing perceptual hashes\u2026 {done} of {total}')

    @Slot(object)
    def _on_near_scan_finished(self, result: dict):
        scanner = self._scanner
        if scanner is None or scanner.generation != self._scan_generation:
            # A newer scan superseded this one (or the dialog was closed);
            # its results are stale.
            return
        progress = self._progress_dialog
        self._scanner = None
        self._progress_dialog = None
        # Read the real cancellation state from the scanner. We must NOT
        # rely on progress.wasCanceled() here, because closing/resetting a
        # QProgressDialog itself counts as a cancel.
        if scanner.cancelled:
            if progress is not None:
                progress.reset()
            self.scan_button.setEnabled(True)
            self.clear_cache_button.setEnabled(True)
            self.status_label.setText('Scan cancelled.')
            return
        # The in-memory scan state (the cards outlive the scanner, which is
        # released from the registry when the task object is garbage
        # collected).
        self._neighbors = result['neighbors']
        self._hashes = result['hashes']
        self._mirrors = result['mirrors']
        self._hashes_keys = result['hashes_keys']
        self._hash_rows = result['hash_rows']
        self._stats = result['stats']
        self._skipped_count = result['skipped']
        self._scanned_threshold = result['scanned_threshold']
        # The current grouping runs at the scanned threshold.
        self._applied_threshold = self._scanned_threshold
        # Building the result cards can take a moment on large sets, so show
        # an indeterminate busy bar (which can't be cancelled) meanwhile.
        if progress is not None:
            progress.setLabelText('Preparing similar groups\u2026')
            progress.setCancelButton(None)
            progress.setRange(0, 0)
            progress.show()
            progress.raise_()
        QApplication.processEvents()
        self._apply_grouping()
        if progress is not None:
            progress.reset()
        self.scan_button.setEnabled(True)
        self.clear_cache_button.setEnabled(True)

    @Slot(list, int)
    def _on_exact_scan_finished(self, groups: list, skipped: int):
        scanner = self._scanner
        if scanner is None or scanner.generation != self._scan_generation:
            # A newer scan superseded this one (or the dialog was closed);
            # its results are stale.
            return
        progress = self._progress_dialog
        self._scanner = None
        self._progress_dialog = None
        # Read the real cancellation state from the scanner. We must NOT
        # rely on progress.wasCanceled() here, because closing/resetting a
        # QProgressDialog itself counts as a cancel.
        if scanner.cancelled:
            if progress is not None:
                progress.reset()
            self.scan_button.setEnabled(True)
            self.clear_cache_button.setEnabled(True)
            self.status_label.setText('Scan cancelled.')
            return
        # The stats outlive the scanner (it is released from the registry
        # when the task object is garbage collected).
        self._stats = dict(scanner.stats)
        self._skipped_count = skipped
        self._scanned_groups = groups
        # Building the result cards can take a moment on large sets, so show
        # an indeterminate busy bar (which can't be cancelled) meanwhile.
        if progress is not None:
            progress.setLabelText('Preparing duplicate groups\u2026')
            progress.setCancelButton(None)
            progress.setRange(0, 0)
            progress.show()
            progress.raise_()
        QApplication.processEvents()
        self._apply_grouping()
        if progress is not None:
            progress.reset()
        self.scan_button.setEnabled(True)
        self.clear_cache_button.setEnabled(True)

    @Slot(str)
    def _on_scan_failed(self, message: str):
        scanner = self._scanner
        if scanner is None or scanner.generation != self._scan_generation:
            return
        progress = self._progress_dialog
        self._scanner = None
        self._progress_dialog = None
        self.scan_button.setEnabled(True)
        self.clear_cache_button.setEnabled(True)
        if progress is not None:
            progress.reset()
        if scanner.cancelled:
            self.status_label.setText(message)
            return
        QMessageBox.critical(self, 'Find Duplicates',
                             f'The scan failed: {message}')

    def done(self, result: int):
        # Stop a running scan when the dialog closes (Close button, Escape or
        # the window's X).
        if self._scanner is not None:
            self._scanner.cancel()
            self._scanner = None
        if self._progress_dialog is not None:
            self._progress_dialog.reset()
            self._progress_dialog = None
        self.scan_button.setEnabled(True)
        self.clear_cache_button.setEnabled(True)
        super().done(result)

    # ------------------------------------------------------------------
    # Grouping (from the last scan's in-memory state; no file is re-read)
    # ------------------------------------------------------------------

    def _clear_results(self):
        self._groups = []
        self._scanned_groups = []
        self._neighbors = []
        self._hashes = []
        self._mirrors = None
        self._hashes_keys = []
        self._hash_rows = {}
        self._stats = {}
        self._tag_counts = {}
        self._skipped_count = 0
        self._scanned_threshold = 0
        self._applied_threshold = 0
        self._scanned_mirrored = False
        self._scanned_paths = frozenset()
        self._thumbnails = {}
        # A fresh scan (or a mode switch, or the post-action reload) builds
        # new groups; the kept picks of the old ones no longer apply.
        self._kept_images.clear()
        self._kept_initialized.clear()
        self.show_more_button.hide()
        self._remove_group_cards()

    def _remove_group_cards(self):
        # The boxes about to be hidden may contain the focused widget
        # (a card menu action leaves the focus where it was): see
        # `_redirect_focus_out_of_results` for why that must not happen
        # while the boxes are hidden.
        self._redirect_focus_out_of_results()
        self._group_cards.clear()
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Hide before the deferred delete: a taken-out widget is
                # still a visible child of the results container at its old
                # position until it is destroyed, and painting it there
                # (alongside the replacement boxes) is what flickers.
                widget.hide()
                widget.deleteLater()
        self.results_layout.addStretch()
        self._sync_results_container_size()

    def _sync_results_container_size(self):
        """Resize the results container now, the way the scroll area would
        on the next event loop pass (viewport width, the layout's height).
        The scroll area defers that resize, so without this the container
        keeps its old height for one frame after boxes are added or
        removed, and the area below the content is painted with the
        container's background instead of the viewport's — one more
        intermediate frame. The scroll area's own pass then computes the
        same size and is a no-op."""
        viewport = self.results_scroll_area.viewport()
        hint = self.results_container.sizeHint()
        self.results_container.resize(max(viewport.width(), hint.width()),
                                      hint.height())

    @Slot(int)
    def _on_threshold_changed(self, value: int):
        # The slider is inert until a "Scan" press (see `start_scan` and
        # `_apply_grouping`): this only updates the value label.
        self.threshold_value_label.setText(str(value))

    def _exact_group_key(self, group: list) -> str:
        """The persisted ignore key of an exact-duplicate group: all of its
        members share one content hash (the group is that hash's copies)."""
        return f'exact:{group[0].content_hash}'

    def _apply_grouping(self):
        """(Re)compute the shown groups from the last scan's in-memory state.

        Duplicates mode: the exact groups are re-filtered by "Show ignored"
        (the scan is instant, so a new scan is never needed).

        Near Duplicates mode: the neighbor lists are filtered at the last
        *applied* threshold (the slider's value at the last scan or fast
        regroup — the slider and the mirror option are inert until a "Scan"
        press, which re-groups in memory or runs a new scan) and
        `seed_group` runs. No file is re-read and no database row is
        touched; a "Scan" at a higher threshold re-runs the distance pass
        on the cached hashes instead.

        The grouping — and therefore the groups' persisted keys, which
        name the seed — never depends on user state: toggling "Show
        ignored" only filters the ignored groups out of the shown list
        (and hides/shows their boxes), so a dismissed group always comes
        back under the same key its dismissal matches."""
        if self._scanner is not None:
            return
        if self.mode == MODE_DUPLICATES:
            if not self._scanned_groups:
                return
            show_ignored = self.show_ignored_check_box.isChecked()
            self._groups = [group for group in self._scanned_groups
                            if show_ignored
                            or self._exact_group_key(group)
                            not in self._ignored_groups]
            self._show_groups()
            return
        if self._scan_generation == 0 or not self._neighbors:
            return
        # The live slider value is not used here: it only takes effect on a
        # "Scan" press (see `start_scan`), and regrouping at a different
        # threshold would change the groups' persisted keys, which embed
        # the applied threshold.
        threshold = self._applied_threshold
        show_ignored = self.show_ignored_check_box.isChecked()
        neighbors = self._filter_neighbors(threshold)
        order_keys = self._order_keys(neighbors)
        groups = seed_group(neighbors, order_keys)
        self._groups = [self._make_group(seed, members, threshold)
                        for seed, members in groups]
        if not show_ignored:
            # Ignored groups are not shown.
            self._groups = [group for group in self._groups
                            if not group.ignored]
        self._show_groups()

    def _filter_neighbors(self, threshold: int) -> list:
        filtered = []
        for i, links in enumerate(self._neighbors):
            kept = [(j, distance, mirrored)
                    for (j, distance, mirrored) in links
                    if distance <= threshold]
            filtered.append(kept)
        return filtered

    def _order_keys(self, neighbors: list) -> list:
        """The seed ordering of `seed_group`: (degree, area, size, path),
        larger first."""
        images = self.image_list_model.images
        keys = []
        for p, content_hash in enumerate(self._hashes_keys):
            image = images[self._hash_rows[content_hash][0]]
            width, height = (image.dimensions or (0, 0))
            size, _mtime_ns = self._stats.get(image.path, (0, 0))
            keys.append((len(neighbors[p]),
                         (width or 0) * (height or 0),
                         size or 0, str(image.path)))
        return keys

    def _make_group(self, seed: int, members: list,
                    threshold: int) -> _SimilarGroup:
        images = self.image_list_model.images
        seed_hash = self._hashes_keys[seed]
        group_key = f'phash:{threshold}:{seed_hash}'
        ignored = group_key in self._ignored_groups
        # The distance (and mirrored flag) of every neighbor of the seed,
        # for the cards' labels.
        info = {j: (distance, mirrored)
                for (j, distance, mirrored) in self._neighbors[seed]}
        group_members = []
        for pos in members:
            content_hash = self._hashes_keys[pos]
            if pos != seed:
                distance, mirrored = info.get(pos, (None, False))
            else:
                distance, mirrored = None, False
            for row_position, row in enumerate(self._hash_rows[content_hash]):
                if pos == seed and row_position > 0:
                    # Another copy of the representative's exact content:
                    # byte-identical, so its pHash difference really is 0
                    # (it is not itself "the representative" — that is the
                    # first copy, the one the group is named after).
                    member_distance = 0
                else:
                    member_distance = distance
                group_members.append({
                    'image': images[row],
                    'row': row,
                    'hash': content_hash,
                    'distance': member_distance,
                    'mirrored': mirrored,
                })
        return _SimilarGroup(self, seed_hash, group_key, ignored,
                             group_members)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _image_lookup(self) -> dict:
        """Map each loaded image's path to its (model row, Image)."""
        return {image.path: (row, image)
                for row, image in enumerate(self.image_list_model.images)}

    def _show_groups(self):
        groups = self._groups
        if not groups:
            self._remove_group_cards()
            self.results_scroll_area.verticalScrollBar().setValue(0)
            self._set_no_groups_status()
            self._set_actions_enabled(False)
            return
        self._tag_counts = {image.path: len(image.tags)
                            for image in self.image_list_model.images}
        # Preserve the user's place in the list across the rebuild (see
        # `_restore_scroll_anchor`): which box the viewport top is in, and
        # where inside it — captured before any box is hidden.
        old_group_cards = list(self._group_cards)
        old_offset = self.results_scroll_area.verticalScrollBar().value()
        anchor_key = None
        anchor_inset = 0
        for group_cards in old_group_cards:
            box = group_cards.box
            if box.y() <= old_offset < box.y() + box.height():
                anchor_key = self._group_key_of(group_cards.group)
                anchor_inset = old_offset - box.y()
                break
        # Build every replacement box before touching the current ones:
        # the boxes are parentless (and therefore invisible) until the
        # swap below, so the build can take its time — with event-loop
        # yields in between — without ever painting an intermediate state.
        new_group_cards = self._build_group_cards(groups[:_MAX_GROUPS_SHOWN])
        # Swap the current boxes for the replacements in one
        # paint-suppressed window: the old boxes are hidden before they are
        # destroyed, no paint can happen between the removal and the
        # insertion, and every replacement box is made visible *before* the
        # window closes (see _insert_group_box) — so the first paint after
        # the swap is the final state (a visible rebuild must never flicker).
        self.results_container.setUpdatesEnabled(False)
        try:
            self._remove_group_cards()
            for group_cards in new_group_cards:
                self._insert_group_box(group_cards)
            self._group_cards = new_group_cards
            # Re-apply the kept images the rebuild reset to the defaults,
            # *before* the layout activation: restoring a pick can expand
            # a group whose kept image was behind the "+N more" tile,
            # which changes the box's height — the scroll anchor restore
            # below must read the post-expansion positions.
            self._restore_keeper_selections()
            # Make the replacement boxes' final positions available now
            # (the scroll anchor restore below reads them; a no-op if the
            # forced Show events already activated the layout).
            self.results_layout.activate()
            self._sync_results_container_size()
        finally:
            self.results_container.setUpdatesEnabled(True)
        self._restore_scroll_anchor(self.results_scroll_area, old_group_cards,
                                    new_group_cards, old_offset, anchor_key,
                                    anchor_inset)
        self._set_actions_enabled(True)
        self._update_status()

    def _group_key_of(self, group) -> str:
        """A stable identity for a group across rebuilds: the same key the
        ignore persistence uses (in Duplicates mode the scanned group
        objects are reused as-is, in Near Duplicates mode the grouping
        re-creates `_SimilarGroup` objects with the same key)."""
        if isinstance(group, _SimilarGroup):
            return group.group_key
        return self._exact_group_key(group)

    def _record_group_kept(self, group, kept: set):
        """Remember a group's kept images (`_kept_images`: the dialog-wide
        set of kept image paths) so a rebuild can restore them: the cards
        are rebuilt with the recorded state (`_populate`), and a "Show
        ignored" toggle or a regroup must not destroy the user's picks.
        The group's contribution to the set is replaced: all of the
        group's members leave the set, then its kept images re-enter it.
        The groups' memberships are disjoint (each image is in at most one
        group), so this only ever touches this group's images."""
        self._kept_images.difference_update(
            _member_image(member).path
            for member in _group_members(group))
        self._kept_images.update(kept)

    def _restore_keeper_selections(self):
        """Re-apply the kept images (see `_record_group_kept`) to the
        currently built cards. A pick follows its kept image, not the
        group: a regroup (a threshold change, a "Show ignored" toggle,
        "Clear group ignores") can re-seed a group under a new key or move
        its kept images, and the picks must survive that. The cards are
        already built with the recorded state in `_populate` (which seeds
        a group's default when it has none recorded), so this only does
        real work for a kept image hidden behind a group's "+N more"
        tile — whose `set_kept` expands the row first."""
        for group_cards in self._group_cards:
            kept = {_member_image(member).path
                    for member in _group_members(group_cards.group)
                    if _member_image(member).path in self._kept_images}
            if kept:
                group_cards.set_kept(kept)

    def _restore_scroll_anchor(self, scroll_area, old_group_cards,
                               new_group_cards, old_offset, anchor_key,
                               anchor_inset):
        """Scroll the rebuilt results list back to the user's place: the
        group whose box the viewport top was in keeps it there, at the
        same inset. If that group's box was not rebuilt — it is hidden,
        e.g. an ignored group while "Show ignored" is off — the first
        rebuilt box after it in the old order takes its place (its top
        comes to the viewport top, exactly as hiding it in place would
        do); if nothing was rebuilt after it, the offset is clamped by
        the scroll area to the smaller maximum (the user was at the
        bottom, and stays at the bottom). `anchor_key` is None when the
        viewport top was above the first box: nothing moved above it, so
        the offset stands (clamped)."""
        if anchor_key is None:
            scroll_area.verticalScrollBar().setValue(old_offset)
            return
        for group_cards in new_group_cards:
            if self._group_key_of(group_cards.group) == anchor_key:
                scroll_area.verticalScrollBar().setValue(
                    group_cards.box.y() + anchor_inset)
                return
        # The anchor group's box is gone: find the first rebuilt box that
        # came after it in the old order. A new box's old position is the
        # index of its key in the old list; groups that did not exist
        # before (a fresh scan) have none and are skipped.
        old_keys = [self._group_key_of(group_cards.group)
                    for group_cards in old_group_cards]
        anchor_index = old_keys.index(anchor_key)
        for group_cards in new_group_cards:
            key = self._group_key_of(group_cards.group)
            if key in old_keys and old_keys.index(key) > anchor_index:
                scroll_area.verticalScrollBar().setValue(group_cards.box.y())
                return
        scroll_area.verticalScrollBar().setValue(old_offset)

    def _insert_group_box(self, group_cards):
        """Insert a built box into the results layout and make it visible
        synchronously. Inserting a parentless box posts its Show event but
        does not deliver it: until the event loop runs, the box is still
        hidden, and the first paint after a rebuild would miss it (one
        frame with the old boxes gone and the new ones not yet there —
        the rebuild flicker). Forcing the Show event to be delivered now
        (inside the caller's paint-suppressed window) makes the box part of
        the very next paint."""
        self.results_layout.insertWidget(
            self.results_layout.count() - 1, group_cards.box)
        group_cards.box.show()
        QCoreApplication.sendPostedEvents(group_cards.box, QEvent.Type.Show)

    def _redirect_focus_out_of_results(self):
        """Move the focus out of the results area, if it is there, before
        boxes are hidden: hiding a focused widget makes Qt move the focus
        to a fallback widget and *synchronously* scroll the scroll area to
        make the newly focused widget visible — that is the invisible hand
        that jumps the list when a group is hidden. The "Show ignored"
        checkbox is always visible (it is in the header, outside the
        scroll area), so focusing it scrolls nothing, and it is the very
        control that brings hidden groups back."""
        focus_widget = QApplication.focusWidget()
        if focus_widget is not None and self.results_container.isAncestorOf(
                focus_widget):
            self.show_ignored_check_box.setFocus()

    def _hide_group_cards(self, group_cards):
        """Hide one group's box in place: this is what ignoring a group
        does while "Show ignored" is off — the group must disappear
        right away, but the other groups' boxes are not touched (rebuilding
        every group just to hide one is what the toggle must not do).

        The group is dropped from `_group_cards` and `_groups` exactly as
        `_apply_grouping` computes them with "Show ignored" off, and the
        status/actions follow. The hidden box stays in the layout (a hidden
        item takes no space) and its `_GroupCards` stays alive until the
        next rebuild, where `_remove_group_cards` clears it. The user's
        place in the list is preserved (see the scroll offset math
        below): whatever part of the hidden box was above the viewport
        top is subtracted from the scroll offset, so the group that was
        at the top of the viewport stays there (and hiding the first
        group while at the top simply brings the next group to the top
        instead of jumping down)."""
        if group_cards in self._group_cards:
            self._group_cards.remove(group_cards)
        if group_cards.group in self._groups:
            self._groups.remove(group_cards.group)
        # Preserve the user's scroll position: everything below the hidden
        # box shifts up by the box's height plus the layout's spacing, so
        # the scroll offset must shrink by whatever part of the box was
        # above the current offset. Without this, the list jumps down by
        # the hidden box's height (and the top of the list is lost when
        # the first group is hidden).
        scroll_area = self.results_scroll_area
        offset = scroll_area.verticalScrollBar().value()
        box_top = group_cards.box.y()
        box_bottom = (box_top + group_cards.box.height()
                      + self.results_layout.spacing())
        new_offset = offset - max(0, min(offset, box_bottom) - box_top)
        self._redirect_focus_out_of_results()
        group_cards.box.hide()
        # Hiding a layout item only *posts* a layout request: the siblings
        # would move to their collapsed positions on the next event loop
        # pass, so the first paint would still show the old layout with a
        # gap. Force the re-layout (and the container's own resize, which
        # the scroll area likewise defers) now so the very next paint is
        # the final state.
        self.results_layout.activate()
        self._sync_results_container_size()
        # Re-assert the offset *unconditionally* — also when it did not
        # change (hiding the first group while at the top, hiding a group
        # below the viewport): the re-layout may have clamped it, and
        # anything that moved it during the hide is corrected here, so
        # the group at the viewport top stays exactly where it was.
        scroll_area.verticalScrollBar().setValue(new_offset)
        if self._groups:
            self._set_actions_enabled(True)
            self._update_status()
        else:
            self._set_no_groups_status()
            self._set_actions_enabled(False)

    def _build_group_cards(self, groups: list) -> list:
        """Construct the given groups' card boxes without showing them.
        The boxes are parentless until the caller inserts them, so building
        them can never paint an intermediate state; the event loop is
        yielded to now and then so the window stays responsive while many
        cards are built."""
        group_cards = []
        cards_built = 0
        for group in groups:
            if self.mode == MODE_NEAR_DUPPLICATES:
                # The representative (the first member) is the default
                # keeper.
                group_cards.append(_GroupCards(self, group))
            else:
                group_cards.append(_GroupCards(
                    self, group,
                    choose_keeper(group, self.keep_rule_combo_box.currentText(),
                                  self._tag_counts, self._stats)))
            cards_built += group_cards[-1].shown_count
            if cards_built >= 40:
                cards_built = 0
                QApplication.processEvents()
        return group_cards

    def _add_group_cards(self, groups: list):
        # Appending only (the "Show next 100 groups" button): nothing is
        # removed, so every state along the way is a consistent one.
        for group_cards in self._build_group_cards(groups):
            self._group_cards.append(group_cards)
            self._insert_group_box(group_cards)
        # Re-apply the "Keep" picks of the appended groups.
        self._restore_keeper_selections()
        self._sync_results_container_size()

    @Slot()
    def show_more_groups(self):
        start = len(self._group_cards)
        self._add_group_cards(self._groups[start:start + _MAX_GROUPS_SHOWN])
        self._update_status()

    def _skipped_note(self) -> str:
        if not self._skipped_count:
            return ''
        return (f' {self._skipped_count} '
                f'{pluralize("image", self._skipped_count)} were skipped '
                'because their content could not be checked.')

    def _set_no_groups_status(self):
        """The status line for "no groups to show" (no results at all, or
        every group hidden because it is ignored and "Show ignored" is
        off)."""
        noun = ('similar images' if self.mode == MODE_NEAR_DUPPLICATES
                else 'duplicate files')
        self.status_label.setText(
            f'No {noun} found among {self._scanned_count} '
            f'{pluralize("image", self._scanned_count)}.'
            + self._skipped_note())
        self.show_more_button.hide()

    def _visible_group_members(self, group) -> list:
        """The group's members as the cards show them (all of them: the
        group's title and the results status count the images the user
        can see, which is the group's full membership)."""
        return _group_members(group)

    def _update_status(self):
        groups = self._groups
        shown = len(self._group_cards)
        image_total = sum(len(self._visible_group_members(group))
                          for group in groups)
        message = (
            f'Found {len(groups)} {self._group_noun()} '
            f'{pluralize("group", len(groups))} ({image_total} images) '
            f'among {self._scanned_count} scanned.')
        message += self._skipped_note()
        remaining = len(groups) - shown
        if remaining > 0:
            message += f' Showing {shown}.'
            next_count = min(_MAX_GROUPS_SHOWN, remaining)
            self.show_more_button.setText(
                f'Show next {next_count} {pluralize("group", next_count)}')
            self.show_more_button.show()
        else:
            self.show_more_button.hide()
        if max(len(self._visible_group_members(group)) for group in groups) \
                > _MAX_MEMBERS_PER_GROUP:
            message += (' Some groups are large; their extra images are '
                        'hidden behind a "+N more" tile.')
        self.status_label.setText(message)

    def _card_info_text(self, image: Image) -> str:
        if image.dimensions:
            width, height = image.dimensions
            dimensions_text = f'{width}\u00d7{height}'
        else:
            dimensions_text = 'unknown size'
        tag_count = self._tag_counts.get(image.path)
        tags_text = ('? tags' if tag_count is None
                     else f'{tag_count} {pluralize("tag", tag_count)}')
        size, mtime_ns = self._stats.get(image.path, (None, None))
        return (f'{_elide(image.path.name)}\n'
                f'{dimensions_text}  \u2022  {_format_size(size)}\n'
                f'{_format_modified(mtime_ns)}\n'
                f'{tags_text}')

    def _card_relation_text(self, member: dict) -> tuple:
        """The (label, tooltip) of a card's relation line: the real metric,
        the pHash difference from the group's representative (Near
        Duplicates mode; in Duplicates mode the cards show the "Kept" /
        "Identical file" labels instead, computed by `describe_relation`).
        """
        if member['distance'] is None:
            return ('Representative',
                    'This image is the group\'s representative: every other '
                    'member of the group is within the threshold of its '
                    'pHash.')
        distance = member['distance']
        if member['mirrored']:
            text = f'mirrored: {distance} / {PHASH_BITS}'
            tip = (f'The mirrored version of this image differs from the '
                   f'group\'s representative by {distance} bits (out of '
                   f'{PHASH_BITS}) \u2014 a better match than the direct '
                   'image.')
        else:
            text = f'pHash difference: {distance} / {PHASH_BITS}'
            tip = (f'This image\'s pHash differs from the group\'s '
                   f'representative by {distance} bits (out of '
                   f'{PHASH_BITS}).')
        return text, tip

    def _build_card(self, member, group_cards) -> tuple:
        """Return (card, keep widget, relation label, info label).

        The keep widget is the "Keep" radio (Duplicates mode, exclusive
        per group — exactly one image is kept) or the "Keep" check box
        (Near Duplicates mode — one or more images are kept, the
        representative by default). An ignored group's cards keep their
        widgets (like taggui-plus): the group's card area is disabled, so
        the widgets are grayed out and unclickable, and toggling the
        group's "Ignore" checkbox only enables/disables the area in place
        (no rebuild — the toggle must not flicker)."""
        image = _member_image(member)
        card = _DuplicateCard()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(6, 6, 6, 6)
        card_layout.setSpacing(4)

        thumbnail_label = _ClickableLabel()
        thumbnail_label.setFixedSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE)
        thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumbnail_label.setCursor(Qt.CursorShape.PointingHandCursor)
        # Clicking the thumbnail opens the preview instead of picking the
        # keeper, so it is the card's one non-keep click zone.
        card.set_preview_zone(thumbnail_label)
        self._set_card_thumbnail(image, thumbnail_label)
        thumbnail_label.clicked.connect(
            lambda m=member, gc=group_cards: self._show_preview(gc, m))
        card_layout.addWidget(thumbnail_label)

        info_label = QLabel(self._card_info_text(image))
        info_label.setToolTip(str(image.path))
        card_layout.addWidget(info_label)

        # A space keeps the line's height before the real text is set.
        relation_label = QLabel(' ')
        card_layout.addWidget(relation_label)

        if self.mode == MODE_NEAR_DUPPLICATES:
            # One or more images can be kept (the representative by
            # default): a "Keep" check box that toggles this image's kept
            # state.
            keep_widget = QCheckBox('Keep')
            # A click anywhere on the card (the name, the details, the
            # check box's text, the background) toggles the check box. A
            # click on the check box itself toggles it as usual, and one
            # on the thumbnail opens the preview. The isEnabled guard
            # matters: a programmatic setChecked works on a disabled
            # check box (an ignored group's card area is disabled), so
            # without it a card click would toggle it anyway.
            card.clicked.connect(
                lambda w=keep_widget: w.setChecked(not w.isChecked())
                if w.isEnabled() else None)
        else:
            # The exclusive "Keep" radio (one keeper per group): a click
            # anywhere on the card checks the radio, selecting this image
            # to be kept. A click on the radio itself checks it as usual,
            # and one on the thumbnail opens the preview.
            keep_widget = QRadioButton('Keep')
            card.clicked.connect(
                lambda w=keep_widget: w.setChecked(True)
                if w.isEnabled() else None)
        _make_indicator_visible(keep_widget)
        card_layout.addWidget(keep_widget)

        # The per-member actions (open in the image list).
        card.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        card.customContextMenuRequested.connect(
            lambda pos, m=member, gc=group_cards: self._show_card_menu(
                m, gc, card.mapToGlobal(pos)))
        return card, keep_widget, relation_label, info_label

    # ------------------------------------------------------------------
    # Thumbnails
    # ------------------------------------------------------------------

    def _set_card_thumbnail(self, image: Image, label: QLabel):
        """Show the list's thumbnail on the card (smoothly scaled down to
        the card size) and remember it as the preview's instant placeholder.
        The list thumbnails (default 200 px) are larger than the cards (140
        px), so no separate decode is needed."""
        icon = image.thumbnail
        pixmap = None
        if icon is not None:
            sizes = icon.availableSizes()
            if sizes:
                pixmap = icon.pixmap(sizes[0])
        if pixmap is not None and not pixmap.isNull():
            pixmap = pixmap.scaled(
                _THUMBNAIL_SIZE, _THUMBNAIL_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(pixmap)
        else:
            pixmap = None
            label.setText('no preview')
        self._thumbnails[str(image.path)] = pixmap

    def _show_preview(self, group_cards, member):
        """Open a large preview that can cycle through the group's images
        and let the user change which ones are kept."""
        if group_cards is not None and group_cards.keep_widgets:
            keep_widgets = [(widget, _member_image(m))
                             for (widget, m) in group_cards.keep_widgets]
        else:
            # Fallback: a standalone image with no group/keep context.
            keep_widgets = [(None, _member_image(member))]

        index = 0
        for position, (_widget, current) in enumerate(keep_widgets):
            if current is _member_image(member):
                index = position
                break

        relations = None
        if (self.mode == MODE_NEAR_DUPPLICATES and group_cards is not None
                and isinstance(group_cards.group, _SimilarGroup)):
            # In Near Duplicates mode the preview shows the real metric
            # (each image's pHash difference from the group's
            # representative) instead of the "Kept" / "Identical file"
            # labels, which only make sense for exact duplicates.
            relations = {m['image'].path: self._card_relation_text(m)[0]
                         for m in group_cards.group.members}
        dialog = _PreviewDialog(self, keep_widgets, index,
                                self._thumbnails, self._stats, relations)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.exec()

    # ------------------------------------------------------------------
    # Card menu and group-ignore actions
    # ------------------------------------------------------------------

    def _show_card_menu(self, member, group_cards, global_pos):
        menu = QMenu(self)
        menu.addAction('Open in image list',
                       lambda: self.open_in_image_list(_member_image(member)))
        menu.exec(global_pos)

    def set_group_ignored(self, group, group_cards, ignored: bool):
        """Persistently ignore (or un-ignore) a group and apply the state
        to the group's cards in place: the card area is disabled/enabled,
        which renders the group in black and white (and grays out its
        "Keep" widgets and the "+N more" tile) in a single repaint.

        No rebuild: rebuilding would destroy and re-create every group's
        boxes, and the intermediate states (the old boxes are only
        deleteLater'd, so they linger at their old positions while the
        replacements are inserted one by one) are what makes the toggle
        flicker. While "Show ignored" is on the group is still shown in
        black and white; while it is off the group's box is hidden in
        place right away (`_hide_group_cards`) — same end state as the
        rebuild `_apply_grouping` would produce, without touching the
        other groups' boxes. The action buttons never act on it."""
        if isinstance(group, _SimilarGroup):
            group_key = group.group_key
            seed_hash = group.seed_hash
            # The applied threshold (not the live slider value): the
            # group's key embeds the threshold its grouping was computed
            # at.
            threshold_key = str(self._applied_threshold)
            mode = MODE_NEAR_DUPPLICATES
        else:
            group_key = self._exact_group_key(group)
            seed_hash = group[0].content_hash
            threshold_key = ''
            mode = MODE_DUPLICATES
        if ignored:
            self._ignored_groups.add(group_key)
            store_ignored_group(self.db_path, group_key, seed_hash, mode,
                                threshold_key)
        else:
            self._ignored_groups.discard(group_key)
            remove_ignored_group(self.db_path, group_key)
        group_cards.ignored = ignored
        group_cards.cards_scroll.setEnabled(not ignored)
        group_cards.update_relations()
        if ignored and not self.show_ignored_check_box.isChecked():
            # "Show ignored" is off: the group is not shown at all, so it
            # is hidden in place right away (see `_hide_group_cards`).
            self._hide_group_cards(group_cards)

    # ------------------------------------------------------------------
    # Opening results in the main window
    # ------------------------------------------------------------------

    def open_group_as_selection(self, group):
        """Select every image of the group in the main window's image list
        (driving its multi-image grid preview and the grouped tag view)."""
        if self.select_paths_callback is not None:
            self.select_paths_callback(
                [_member_image(member).path
                 for member in _group_members(group)])

    def open_in_image_list(self, image: Image):
        if self.select_paths_callback is not None:
            self.select_paths_callback([image.path])

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    @Slot()
    def apply_keep_rule(self):
        # Duplicates mode: pick the keep rule's best image per group (as
        # before). Near Duplicates mode: reset the group to the single best
        # image under the rule — unchecking every other kept image.
        rule = self.keep_rule_combo_box.currentText()
        for group_cards in self._group_cards:
            if group_cards.ignored:
                continue
            members = _group_members(group_cards.group)
            keeper = choose_keeper([_member_image(member)
                                    for member in members], rule,
                                   self._tag_counts, self._stats)
            group_cards.set_kept({keeper.path})

    def _keeper_targets(self) -> list:
        """Return (kept images, [non-kept images]) for every shown group
        that is not ignored. Includes images hidden behind a group's "+N
        more" tile."""
        pairs = []
        for group_cards in self._group_cards:
            if group_cards.ignored:
                continue
            kept = {_member_image(member).path
                    for member in _group_members(group_cards.group)
                    if _member_image(member).path in self._kept_images}
            kept_images = []
            others = []
            for member in _group_members(group_cards.group):
                image = _member_image(member)
                if image.path in kept:
                    kept_images.append(image)
                else:
                    others.append(image)
            if others:
                pairs.append((kept_images, others))
        return pairs

    def _tag_actions_allowed(self) -> bool:
        """Whether the tag button and the merge checkbox may be enabled: in
        database storage mode the copies of an exact duplicate share one tag
        record, so tagging (or merging into) "the other copies" cannot exist
        there. Near-duplicate images are different content and different
        records, so the tag actions work in both storage modes there."""
        return not (self._shared_records and self.mode == MODE_DUPLICATES)

    def _update_tag_action_state(self):
        self.tag_button.setEnabled(
            self._actions_enabled and self._tag_actions_allowed())
        self.merge_tags_check_box.setEnabled(self._tag_actions_allowed())

    def _set_actions_enabled(self, enabled: bool):
        self._actions_enabled = enabled
        self.delete_button.setEnabled(enabled)
        self.move_button.setEnabled(enabled)
        self.apply_rule_button.setEnabled(enabled)
        # The tag button (and the merge checkbox) stay disabled in database
        # storage mode in Duplicates mode (the copies share one tag record;
        # see __init__).
        self._update_tag_action_state()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _merge_tags_by_row(self, pairs, lookup, acted_on=None) -> dict:
        """Return {model row: tags} adding the non-kept images' tags (that
        a kept image lacks) to each kept image — to every kept one, when
        the group keeps several (Near Duplicates mode). Only others in
        `acted_on` (a set of paths) count, if given. (`add_tags_per_image`
        later skips the tags an image already has.)"""
        tags_by_row = {}
        for kept_images, others in pairs:
            for kept in kept_images:
                kept_entry = lookup.get(kept.path)
                if kept_entry is None:
                    continue
                tags = []
                for member in others:
                    if acted_on is not None and member.path not in acted_on:
                        continue
                    entry = lookup.get(member.path)
                    if entry is not None:
                        tags.extend(entry[1].tags)
                if tags:
                    tags_by_row.setdefault(kept_entry[0], []).extend(tags)
        return tags_by_row

    def _add_tags_by_row(self, tags_by_row: dict, action_name: str):
        if not tags_by_row:
            return
        model = self.image_list_model
        # One undo step and one combined dataChanged for the whole action
        # (`add_tags_per_image` skips the tags an image already has).
        model.add_tags_per_image(
            [(model.index(row, 0), tags)
             for row, tags in sorted(tags_by_row.items())], action_name)
        self._refresh_tag_counts()

    def _refresh_tag_counts(self):
        self._tag_counts = {image.path: len(image.tags)
                            for image in self.image_list_model.images}
        for group_cards in self._group_cards:
            group_cards.refresh_info()

    def _merge_enabled(self) -> bool:
        return (self.merge_tags_check_box.isChecked()
                and self._tag_actions_allowed())

    @Slot()
    def delete_others(self):
        pairs = self._keeper_targets()
        targets = [member for _kept, others in pairs for member in others]
        if not targets:
            self._warn_no_targets()
            return
        count = len(targets)
        merge = self._merge_enabled()
        if self.mode == MODE_NEAR_DUPPLICATES:
            title = 'Delete Similar Images'
        else:
            title = 'Delete Duplicates'
        question = (f'Move {count} {self._group_noun()} '
                    f'{pluralize("image", count)} and '
                    f'{"its" if count == 1 else "their"} '
                    f'{pluralize("caption", count)} to the Recycle Bin?')
        if self.mode == MODE_NEAR_DUPPLICATES:
            question += ('\n\nThis deletes the image files themselves '
                         '(their tags and descriptions stay with any other '
                         'copies).')
        if merge:
            question += ('\n\nTheir tags will be copied to the kept '
                         'images first.')
        reply = get_confirmation_dialog_reply(title, question)
        if reply != QMessageBox.StandardButton.Yes:
            return
        lookup = self._image_lookup()
        deleted = trash_images(self, [member.path for member in targets],
                               list(lookup))
        if merge and deleted:
            self._add_tags_by_row(
                self._merge_tags_by_row(pairs, lookup, set(deleted)),
                'Merge Duplicate Tags')
        if not deleted:
            self.status_label.setText('Nothing was deleted.')
            return
        self._finish_destructive_action(
            f'Moved {len(deleted)} {pluralize("image", len(deleted))} to '
            f'the Recycle Bin.')

    @Slot()
    def move_others(self):
        pairs = self._keeper_targets()
        targets = [member for _kept, others in pairs for member in others]
        if not targets:
            self._warn_no_targets()
            return
        if self.mode == MODE_NEAR_DUPPLICATES:
            folder_word = 'similar images'
        else:
            folder_word = 'duplicates'
        destination = QFileDialog.getExistingDirectory(
            self, f'Select a folder to move the {folder_word} into',
            self.settings.value('directory_path', type=str) or '')
        if not destination:
            return
        lookup = self._image_lookup()
        moved_pairs = move_images_to_folder(
            self, [member.path for member in targets], Path(destination),
            list(lookup))
        moved = {source for source, _target in moved_pairs}
        if self._merge_enabled() and moved:
            self._add_tags_by_row(
                self._merge_tags_by_row(pairs, lookup, moved),
                'Merge Duplicate Tags')
        if not moved:
            self.status_label.setText('Nothing was moved.')
            return
        self._finish_destructive_action(
            f'Moved {len(moved)} {pluralize("image", len(moved))} to '
            f'{destination}.')

    @Slot()
    def tag_others(self):
        pairs = self._keeper_targets()
        if not pairs:
            self._warn_no_targets()
            return
        lookup = self._image_lookup()
        tags_by_row = (self._merge_tags_by_row(pairs, lookup)
                       if self._merge_enabled() else {})
        tagged_count = 0
        for _kept, others in pairs:
            for member in others:
                entry = lookup.get(member.path)
                if entry is not None:
                    tags_by_row.setdefault(entry[0], []).append(
                        _DUPLICATE_TAG)
                    tagged_count += 1
        if not tags_by_row:
            self.status_label.setText(
                'None of these images are loaded any more. Scan again.')
            return
        self._add_tags_by_row(tags_by_row, 'Tag Duplicates')
        self.status_label.setText(
            f'Tagged {tagged_count} {self._group_noun()} '
            f'{pluralize("image", tagged_count)} as "{_DUPLICATE_TAG}". '
            f'Nothing was deleted.')

    @Slot()
    def clear_cache(self):
        reply = get_confirmation_dialog_reply(
            'Clear Similarity Cache',
            'Delete the cached perceptual hashes of every image? The next '
            'scan will have to re-decode and re-hash them all.\n\n'
            'Ignored groups are kept.')
        if reply != QMessageBox.StandardButton.Yes:
            return
        clear_similarity_cache(self.db_path)
        # The in-memory scan state (the current groups) is still valid; the
        # next scan rebuilds the cache.
        self.status_label.setText(
            'Similarity cache cleared. Click "Scan" to rescan; the current '
            'results are kept until then.')

    @Slot()
    def clear_ignores(self):
        reply = get_confirmation_dialog_reply(
            'Clear Group Ignores',
            'Forget every ignored group (both modes)? '
            'They will be shown again in the current results and on the '
            'next scan or regrouping.\n\n'
            'The cached perceptual hashes are kept.')
        if reply != QMessageBox.StandardButton.Yes:
            return
        clear_group_ignores(self.db_path)
        self._ignored_groups.clear()
        if (self.mode == MODE_DUPLICATES and self._scanned_groups
                or self.mode == MODE_NEAR_DUPPLICATES and self._neighbors):
            self._apply_grouping()
        self.status_label.setText(
            'Group ignores cleared. They are shown again in the current '
            'results; the cached perceptual hashes are kept.')

    def _finish_destructive_action(self, summary: str):
        # Files on disk changed. Refresh the main image list (this runs an
        # asynchronous re-scan of the directory), then clear our results and
        # ask the user to scan again. We deliberately don't auto-rescan here
        # because the model's image list is repopulated in the background and
        # may still be stale at this exact moment.
        self._clear_results()
        self._set_actions_enabled(False)
        self.scan_button.setEnabled(True)
        self.clear_cache_button.setEnabled(True)
        if callable(self.reload_callback):
            self.reload_callback()
        if self.mode == MODE_NEAR_DUPPLICATES:
            word = 'similar images'
        else:
            word = 'duplicates'
        self.status_label.setText(
            f'{summary} The directory is reloading \u2014 click "Scan" '
            f'again to check for any remaining {word}.')

    def _warn_no_targets(self):
        if self.mode == MODE_NEAR_DUPPLICATES:
            text = ('There is nothing to act on: every shown group is '
                    'ignored or keeps all its images.')
        else:
            text = ('There is nothing to act on: every shown group is '
                    'ignored. Un-ignore a group first.')
        QMessageBox.information(self, 'Find Duplicates', text)


class _GroupCards:
    """Renders and manages one group's row of image cards (both modes).

    Only the first ``_MAX_MEMBERS_PER_GROUP`` images are shown initially; a
    clickable "+N more" tile expands the row to show every image in the
    group. ``keep_member`` is the group's default kept image, even when it
    is hidden: in Duplicates mode it is picked by the keep rule when the
    cards are built, in Near Duplicates mode the representative (the first
    member). In Duplicates mode exactly one image is kept (the "Keep"
    radio, exclusive per group); in Near Duplicates mode one or more are
    (the "Keep" check boxes), and the kept images live in the dialog's
    kept set (`FindDuplicatesDialog._kept_images`), which the widgets
    mirror. An ignored group's card area is disabled, so its cards are
    rendered in black and white (the Fusion style desaturates disabled
    widgets) and are unclickable — the group reads as "not to be acted
    on" — while the group's header stays normal and usable.
    """

    def __init__(self, dialog, group, keep_member=None):
        self.dialog = dialog
        self.group = group
        members = _group_members(group)
        # In Duplicates mode the caller passes the keeper picked by the keep
        # rule; in Near Duplicates mode the representative (the first
        # member) is the default kept image.
        self.keep_member = (keep_member if keep_member is not None
                            else _member_image(members[0]))
        self.expanded = False
        # list of (QCheckBox | QRadioButton, member) for the currently
        # shown cards (a check box in Near Duplicates mode — one or more
        # kept — and the exclusive "Keep" radio in Duplicates mode); the
        # preview dialog shares these widgets to (un)keep an image from
        # there.
        self.keep_widgets: list = []
        # list of (relation QLabel, info QLabel, member) for the shown cards.
        self.card_labels: list = []
        # list of (_DuplicateCard, member) for the shown cards.
        self.cards: list = []
        self.shown_count = 0
        # Whether the group is ignored (persisted, both modes): ignored
        # groups are shown in black and white (their card area disabled),
        # and the action buttons never act on them. The group's "Ignore"
        # checkbox toggles this in place (see
        # `FindDuplicatesDialog.set_group_ignored`) — no rebuild.
        self.ignored = (group.ignored if isinstance(group, _SimilarGroup)
                        else f'exact:{group[0].content_hash}'
                        in dialog._ignored_groups)

        title = ('Near duplicates' if dialog.mode == MODE_NEAR_DUPPLICATES
                 else 'Exact duplicates')
        # Count the images the user can see (the group's full membership).
        self.box = QGroupBox(
            f'{title} \u2014 '
            f'{len(dialog._visible_group_members(group))} images')
        # Only a title and a divider line, so the images sit directly on the
        # dark results background like in the Images pane. (The header is
        # never disabled: an ignored group's cards are rendered in black
        # and white, but the group stays fully usable.)
        self.box.setFlat(True)
        title_font = self.box.font()
        title_font.setBold(True)
        self.box.setFont(title_font)
        box_layout = QVBoxLayout(self.box)
        box_layout.setContentsMargins(10, 8, 10, 10)

        # A header row with the group's actions (only the group's title is
        # bold). "Ignore this group" is a checkbox, persisted in both
        # modes: while "Show ignored" is on the group is shown in black
        # and white (its cards disabled), while it is off the group is
        # hidden; either way the action buttons never act on it.
        normal_font = QFont(title_font)
        normal_font.setBold(False)
        header = QHBoxLayout()
        self.ignore_check_box = QCheckBox(
            'Ignore this group (leave all its images untouched)')
        self.ignore_check_box.setFont(normal_font)
        # The persisted state first, then the connection: setting the check
        # during construction must not ignore (or un-ignore) the group.
        self.ignore_check_box.setChecked(self.ignored)
        self.ignore_check_box.toggled.connect(self._on_ignore_toggled)
        _make_indicator_visible(self.ignore_check_box)
        self.open_button = QPushButton('Open as selection')
        self.open_button.setFont(normal_font)
        self.open_button.clicked.connect(self._on_open_as_selection)
        header.addWidget(self.ignore_check_box)
        header.addStretch()
        header.addWidget(self.open_button)
        box_layout.addLayout(header)

        self.cards_scroll = QScrollArea()
        self.cards_scroll.setFont(normal_font)
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.cards_scroll.viewport().setAutoFillBackground(False)
        self.cards_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._wheel_forwarder = _VerticalWheelForwarder(
            dialog.results_scroll_area, self.cards_scroll, self.box)
        self.cards_scroll.viewport().installEventFilter(self._wheel_forwarder)
        self.cards_scroll.horizontalScrollBar().installEventFilter(
            self._wheel_forwarder)
        box_layout.addWidget(self.cards_scroll)
        # An ignored group's cards are rendered in black and white (the
        # Fusion style desaturates disabled widgets) and are unclickable,
        # like taggui-plus: the whole group reads as "not to be acted on".
        # The header above stays enabled, so the group can still be
        # un-ignored or opened as a selection.
        self.cards_scroll.setEnabled(not self.ignored)
        self._populate()

    def _on_ignore_toggled(self, checked: bool):
        # Persist the state and apply it in place (the card area is
        # enabled/disabled, which in one paint renders the group in black
        # and white and grays out its "Keep" widgets) — like taggui-plus,
        # without rebuilding the groups, so the toggle flickers nothing.
        # The checkbox is set before this connection is made in `__init__`,
        # so this cannot fire during construction.
        self.dialog.set_group_ignored(self.group, self, checked)

    def _on_open_as_selection(self):
        self.dialog.open_group_as_selection(self.group)

    def _populate(self):
        button_group = QButtonGroup(self.box)
        button_group.setExclusive(True)
        self.button_group = button_group
        self.keep_widgets = []
        self.card_labels = []
        self.cards = []

        cards_widget = QWidget()
        cards_layout = QHBoxLayout(cards_widget)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(10)

        members = self.dialog._visible_group_members(self.group)
        shown = members if self.expanded else members[:_MAX_MEMBERS_PER_GROUP]
        # The images the group keeps (the dialog's kept set; a rebuild or
        # the "+N more" expansion must not reset the user's picks). When
        # the group's state is not recorded yet — the first build of its
        # cards — it is seeded with the default kept image, so the set
        # always mirrors the widgets' state (a later "the user unchecked
        # everything" empty state must not be mistaken for this).
        group_paths = {_member_image(member).path for member in members}
        group_key = self.dialog._group_key_of(self.group)
        kept = self.dialog._kept_images & group_paths
        if group_key not in self.dialog._kept_initialized:
            if not kept:
                kept = {self.keep_member.path}
            self.dialog._record_group_kept(self.group, kept)
            self.dialog._kept_initialized.add(group_key)
        for position, member in enumerate(shown):
            card, keep_widget, relation_label, info_label = \
                self.dialog._build_card(member, self)
            if isinstance(keep_widget, QRadioButton):
                # The exclusive "Keep" radio (Duplicates mode); the check
                # boxes (Near Duplicates mode) never join the group.
                button_group.addButton(keep_widget)
            # The state first, then the connection: setting it during
            # construction must not be recorded twice.
            keep_widget.setChecked(_member_image(member).path in kept)
            keep_widget.toggled.connect(
                lambda checked, m=member: self._on_keep_toggled(
                    checked, m))
            self.keep_widgets.append((keep_widget, member))
            self.card_labels.append((relation_label, info_label, member))
            self.cards.append((card, member))
            cards_layout.addWidget(card)
            if self.expanded and position % 20 == 19:
                QApplication.processEvents()

        hidden = len(members) - len(shown)
        if hidden > 0:
            more_tile = _ClickableLabel(f'+{hidden} more\n\nClick to show')
            more_tile.setFixedSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE)
            more_tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
            more_tile.setFrameShape(QFrame.Shape.StyledPanel)
            more_tile.setCursor(Qt.CursorShape.PointingHandCursor)
            more_tile.clicked.connect(self.expand)
            # In an ignored group the tile is rendered in black and white
            # by the disabled card area, like the cards.
            cards_layout.addWidget(more_tile)
        cards_layout.addStretch()

        # Fit the row's height to the cards, leaving room for the horizontal
        # scroll bar.
        self.cards_scroll.setFixedHeight(
            cards_widget.sizeHint().height()
            + self.cards_scroll.horizontalScrollBar().sizeHint().height() + 4)
        # setWidget takes ownership and deletes the previously set widget.
        self.cards_scroll.setWidget(cards_widget)
        self.shown_count = len(shown)
        self.update_relations()

    def _on_keep_toggled(self, checked: bool, member):
        # A keep widget changed state (the user toggled a check box, or the
        # exclusive radio group re-checked the radios). The group's kept
        # images are now exactly its checked widgets, so the dialog's kept
        # set is rebuilt from them.
        kept = {_member_image(m).path
                for widget, m in self.keep_widgets if widget.isChecked()}
        self.dialog._record_group_kept(self.group, kept)
        if checked:
            # The checked widget's image is remembered as the group's
            # keeper for the relation labels (Duplicates mode).
            self.keep_member = _member_image(member)
        self.update_relations()

    def update_relations(self):
        """Refresh the relation label on every shown card (the "Kept" /
        "Identical file" labels in Duplicates mode, the pHash difference in
        Near Duplicates mode), and the kept-image outline."""
        near = self.dialog.mode == MODE_NEAR_DUPPLICATES
        for relation_label, _info_label, member in self.card_labels:
            if near:
                text, tip = self.dialog._card_relation_text(member)
                if member is _group_members(self.group)[0]:
                    # The representative's label is bold, whatever the
                    # kept images are.
                    relation_label.setText(f'<b>{text}</b>')
                else:
                    relation_label.setText(text)
                relation_label.setToolTip(tip)
            else:
                image = _member_image(member)
                text = describe_relation(image, self.keep_member)
                relation_label.setText(
                    f'<b>{text}</b>' if image is self.keep_member else text)
        for (keep_widget, _member), (card, _card_member) in zip(
                self.keep_widgets, self.cards):
            # An ignored group's cards are rendered in black and white by
            # the disabled card area (and show no kept outline there
            # either).
            card.set_kept(keep_widget.isChecked() and not self.ignored)

    def refresh_info(self):
        """Refresh each shown card's details (e.g. its tag count)."""
        for _relation_label, info_label, member in self.card_labels:
            info_label.setText(self.dialog._card_info_text(
                _member_image(member)))

    def set_kept(self, kept: set):
        """Make the given image paths (a set) the group's kept images: the
        matching widgets are checked, the rest unchecked, and the kept set
        is recorded on the dialog (`_record_group_kept`) so a rebuild can
        restore it. A group with none of its images in `kept` is left
        alone. A kept image hidden behind the "+N more" tile expands the
        row first (the re-populate re-applies the recorded state)."""
        group_paths = {_member_image(member).path
                       for member in _group_members(self.group)}
        kept = set(kept) & group_paths
        if not kept:
            return
        self.dialog._record_group_kept(self.group, kept)
        shown_paths = {_member_image(member).path
                       for _widget, member in self.keep_widgets}
        if not kept <= shown_paths:
            self.expand()
            return
        for widget, member in self.keep_widgets:
            widget.setChecked(_member_image(member).path in kept)

    def expand(self):
        if self.expanded:
            return
        self.expanded = True
        self._populate()


class _GroupImagesModel(QAbstractListModel):
    """The images of one duplicate group, in the shape the main ImageViewer
    reads: the image record (path and dimensions) for UserRole and the
    already-loaded card thumbnail, used as an instant placeholder, for
    DecorationRole."""

    def __init__(self, members, thumbnails: dict, parent=None):
        super().__init__(parent)
        self.members = members
        self.thumbnails = thumbnails

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.members)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.members):
            return None
        member = self.members[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return member
        if role == Qt.ItemDataRole.DecorationRole:
            pixmap = self.thumbnails.get(str(member.path))
            if pixmap is not None and not pixmap.isNull():
                return QIcon(pixmap)
        return None


_OVERLAY_STYLE = (
    'QFrame#previewOverlay { background-color: rgba(0, 0, 0, 160);'
    ' border-radius: 6px; }'
    'QFrame#previewOverlay QLabel { color: rgba(255, 255, 255, 230);'
    ' background: transparent; }'
    'QFrame#previewOverlay QPushButton { color: white;'
    ' background-color: rgba(255, 255, 255, 28);'
    ' border: 1px solid rgba(255, 255, 255, 70); border-radius: 4px;'
    ' padding: 4px 10px; }'
    'QFrame#previewOverlay QPushButton:hover {'
    ' background-color: rgba(255, 255, 255, 60); }'
    'QFrame#previewOverlay QPushButton:pressed {'
    ' background-color: rgba(255, 255, 255, 90); }'
    'QFrame#previewOverlay QPushButton:disabled {'
    ' color: rgba(255, 255, 255, 110);'
    ' border-color: rgba(255, 255, 255, 35); }')
# Hide the overlaid controls after this long without mouse movement.
_OVERLAY_HIDE_DELAY_MS = 2500
_OVERLAY_MARGIN = 10


class _PreviewDialog(QDialog):
    """Large viewer for a duplicate group, built on the main image viewer:
    zoom (mouse wheel, + / -), pan (drag) and double-click to reset, a
    single-image view that cycles through the group and a grid view of the
    whole group, with controls overlaid on the image and a full-screen
    toggle.

    ``keep_widgets`` is a list of ``(QCheckBox | QRadioButton | None,
    member)`` tuples shared with the main dialog, so "Keep this image"
    here checks (or un-checks, Near Duplicates mode) the matching widget
    back in the results list.

    ``relations``, when given, maps a member's path to the text to
    show for its relation to the rest of the group (Near
    Duplicates mode: the pHash difference from the group's
    representative); otherwise the relation to the kept member
    (the "Kept" / "Identical file" labels) is shown.
    """

    def __init__(self, parent, keep_widgets, index, thumbnails=None,
                 stats=None, relations=None):
        super().__init__(parent)
        # The widgets are shared with the results list (a keep toggle here
        # toggles the matching card's widget); the member is normalized to
        # its `Image` — the viewer and the info labels read `.path` and
        # `.dimensions` directly, which a Near Duplicates member dict
        # would not have.
        self.keep_widgets = [(widget, _member_image(member))
                             for widget, member in keep_widgets]
        self.index = index
        self._grid = False
        self._was_maximized = False
        self._zoomed_at_double_click = False
        members = [member for _widget, member in self.keep_widgets]
        self.model = _GroupImagesModel(members, thumbnails or {}, self)
        self._stats = stats or {}
        self._relations = relations or {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.viewer = ImageViewer(self.model)
        layout.addWidget(self.viewer)
        viewport = self.viewer.scroll_area.viewport()

        # Top left: what is shown.
        self.info_overlay = self._make_overlay(viewport)
        info_layout = QVBoxLayout(self.info_overlay)
        info_layout.setContentsMargins(10, 6, 10, 6)
        info_layout.setSpacing(2)
        self.name_label = QLabel()
        name_font = QFont(self.font())
        name_font.setBold(True)
        self.name_label.setFont(name_font)
        self.details_label = QLabel()
        info_layout.addWidget(self.name_label)
        info_layout.addWidget(self.details_label)

        # Top right: view controls.
        self.view_overlay = self._make_overlay(viewport)
        view_layout = QHBoxLayout(self.view_overlay)
        view_layout.setContentsMargins(6, 6, 6, 6)
        self.grid_button = self._make_button('Grid view (G)')
        self.grid_button.clicked.connect(self.toggle_grid)
        self.full_screen_button = self._make_button('Full screen (F11)')
        self.full_screen_button.clicked.connect(self.toggle_full_screen)
        self.close_button = self._make_button('Close (Esc)')
        self.close_button.clicked.connect(self.reject)
        for button in (self.grid_button, self.full_screen_button,
                       self.close_button):
            view_layout.addWidget(button)

        # Bottom centre: paging and the keep decision.
        self.nav_overlay = self._make_overlay(viewport)
        nav_layout = QHBoxLayout(self.nav_overlay)
        nav_layout.setContentsMargins(6, 6, 6, 6)
        self.prev_button = self._make_button('\u2190 Previous')
        self.prev_button.clicked.connect(lambda: self._step(-1))
        self.keep_button = self._make_button('Keep this image (K)')
        self.keep_button.clicked.connect(self._keep_current)
        self.next_button = self._make_button('Next \u2192')
        self.next_button.clicked.connect(lambda: self._step(1))
        for button in (self.prev_button, self.keep_button, self.next_button):
            nav_layout.addWidget(button)
        self.overlays = (self.info_overlay, self.view_overlay,
                         self.nav_overlay)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(_OVERLAY_HIDE_DELAY_MS)
        self._hide_timer.timeout.connect(self._hide_overlays_if_idle)
        # Watch mouse movement anywhere over the image to bring the controls
        # back, and viewport resizes to keep them in their corners.
        for widget in (viewport, self.viewer.image_label):
            widget.setMouseTracking(True)
            widget.installEventFilter(self)
        for overlay in self.overlays:
            overlay.installEventFilter(self)

        self.viewer.grid_cell_clicked.connect(self._on_grid_cell_clicked)
        # Double-clicking a grid cell opens it in the single-image view.
        self.viewer.scroll_area.reset_requested.connect(
            self._on_double_click)

        own_keys = []
        for key, slot in (
                (Qt.Key.Key_Left, lambda: self._step(-1)),
                (Qt.Key.Key_Right, lambda: self._step(1)),
                (Qt.Key.Key_K, self._keep_current),
                (Qt.Key.Key_G, self.toggle_grid),
                (Qt.Key.Key_F11, self.toggle_full_screen),
                (Qt.Key.Key_Return, self._on_enter),
                (Qt.Key.Key_Enter, self._on_enter),
                (Qt.Key.Key_Plus, self.viewer.zoom_in),
                (Qt.Key.Key_Equal, self.viewer.zoom_in),
                (Qt.Key.Key_Minus, self.viewer.zoom_out),
                (Qt.Key.Key_0, self.viewer.reset_view),
                (Qt.Key.Key_Escape, None)):
            sequence = QKeySequence(key)
            own_keys.append(sequence.toString())
            if slot is not None:
                QShortcut(sequence, self).activated.connect(slot)
        # The main window's zoom keys (hard-coded in MainWindow) work here
        # too. A key is never bound twice, since Qt ignores a key that two
        # shortcuts share.
        for sequence, slot in (
                (QKeySequence('Ctrl+0'), self.viewer.reset_view),
                (QKeySequence('Ctrl++'), self.viewer.zoom_in),
                (QKeySequence('Ctrl+='), self.viewer.zoom_in),
                (QKeySequence('Ctrl+-'), self.viewer.zoom_out)):
            text = sequence.toString()
            if sequence.isEmpty() or text in own_keys:
                continue
            own_keys.append(text)
            QShortcut(sequence, self).activated.connect(slot)

        screen = self.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry()
        self.resize(int(available.width() * 0.85),
                    int(available.height() * 0.85))
        self._show_current()

    @staticmethod
    def _make_overlay(parent) -> QFrame:
        overlay = QFrame(parent)
        overlay.setObjectName('previewOverlay')
        overlay.setStyleSheet(_OVERLAY_STYLE)
        overlay.setCursor(Qt.CursorShape.ArrowCursor)
        # Clicks on the panel itself must not pan the image or pick a cell.
        overlay.setAttribute(Qt.WidgetAttribute.WA_NoMousePropagation)
        return overlay

    @staticmethod
    def _make_button(text: str) -> QPushButton:
        button = QPushButton(text)
        # Keep keyboard focus off the buttons so the arrow keys, Enter and
        # Esc always reach the viewer's shortcuts.
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setAutoDefault(False)
        return button

    # --- Showing images -----------------------------------------------------

    def _indices(self) -> list[QModelIndex]:
        return [self.model.index(row) for row in range(self.model.rowCount())]

    def _grid_cell_cap(self) -> int:
        return max(1, get_settings().value(
            'variant_grid_cell_cap',
            defaultValue=DEFAULT_SETTINGS['variant_grid_cell_cap'], type=int))

    def _kept_member(self):
        for widget, member in self.keep_widgets:
            if widget is not None and widget.isChecked():
                return member
        return None

    def _show_current(self, keep_zoom: bool = False):
        """Show the current image, in the single-image or the grid view.
        With ``keep_zoom`` a zoomed-in view keeps its zoom and position, so
        the same detail can be compared across the group."""
        if self._grid:
            self.viewer.update_grid_current(self._indices(), self.index,
                                            self._grid_cell_cap())
        else:
            self.viewer.exit_grid()
            index = self.model.index(self.index)
            if keep_zoom and not self._at_fit_zoom():
                # Keep the current zoom and scroll position (the viewer
                # would otherwise reset both for a new image).
                self.viewer.load_image(index, preserve_view=True)
            else:
                self.viewer.load_image(index)
        self._update_info()

    def _at_fit_zoom(self) -> bool:
        return abs(self.viewer.zoom_factor - 1.0) < 1e-9

    def _update_info(self):
        _widget, member = self.keep_widgets[self.index]
        total = len(self.keep_widgets)
        self.setWindowTitle(
            f'{member.path}  \u2014  {self.index + 1} of {total}')
        self.name_label.setText(
            f'{member.path.name}  \u2014  {self.index + 1} of {total}')
        parts = []
        if member.dimensions:
            width, height = member.dimensions
            parts.append(f'{width}\u00d7{height}')
        size, mtime_ns = self._stats.get(member.path, (None, None))
        if size is not None:
            parts.append(_format_size(size))
        parts.append(_format_modified(mtime_ns))
        # In Near Duplicates mode the caller passes the real
        # metric (each image's pHash difference from the group
        # representative); the "Kept" / "Identical file" labels
        # only make sense for exact duplicates.
        relation = self._relations.get(member.path)
        if relation is None:
            kept = self._kept_member()
            if kept is not None:
                relation = describe_relation(member, kept)
        if relation:
            parts.append(relation)
        self.details_label.setText('  \u2022  '.join(parts))

        multiple = total > 1
        self.prev_button.setEnabled(multiple)
        self.next_button.setEnabled(multiple)
        widget = self.keep_widgets[self.index][0]
        if widget is None or not widget.isEnabled():
            # No keep widget (a standalone image) or a disabled one (the
            # group's card area is disabled — an ignored group): not
            # keepable here.
            self.keep_button.setText('Keep this image (K)')
            self.keep_button.setEnabled(False)
        elif isinstance(widget, QCheckBox) and widget.isChecked():
            # Near Duplicates mode: the image is kept; the button un-keeps
            # it (the same toggle as the card's check box).
            self.keep_button.setText('\u2713 Keeping \u2014 click to un-keep')
            self.keep_button.setEnabled(True)
        elif widget.isChecked():
            # Duplicates mode: this is the group's single keeper.
            self.keep_button.setText('\u2713 Keeping this image')
            self.keep_button.setEnabled(False)
        else:
            self.keep_button.setText('Keep this image (K)')
            self.keep_button.setEnabled(True)
        self.grid_button.setText('Single view (G)' if self._grid
                                 else 'Grid view (G)')
        self.grid_button.setEnabled(multiple or self._grid)
        self.full_screen_button.setText(
            'Exit full screen (F11)' if self.isFullScreen()
            else 'Full screen (F11)')
        self._position_overlays()

    def _step(self, delta):
        if len(self.keep_widgets) < 2:
            return
        self.index = (self.index + delta) % len(self.keep_widgets)
        self._show_current(keep_zoom=True)
        self._show_overlays()

    def _keep_current(self):
        widget, _member = self.keep_widgets[self.index]
        # A disabled widget (the group's card area is disabled — an
        # ignored group) cannot be toggled.
        if widget is None or not widget.isEnabled():
            return
        if isinstance(widget, QCheckBox):
            # Near Duplicates mode: toggle the kept state (the same as the
            # card's check box).
            widget.setChecked(not widget.isChecked())
        elif not widget.isChecked():
            # Duplicates mode: the exclusive radio — check it (a group
            # always keeps exactly one image).
            widget.setChecked(True)
        self._update_info()

    def toggle_grid(self):
        if not self._grid and len(self.keep_widgets) < 2:
            return
        self._grid = not self._grid
        if self._grid:
            self.viewer.show_grid(self._indices(), self.index,
                                  self._grid_cell_cap())
            self._update_info()
        else:
            self._show_current()
        self._show_overlays()

    def _open_in_single_view(self, row: int):
        self.index = row
        self._grid = False
        self._show_current()

    @Slot(QModelIndex)
    def _on_grid_cell_clicked(self, proxy_index):
        # Like the main grid: a click makes the cell current.
        if self._grid and proxy_index.row() != self.index:
            self.index = proxy_index.row()
            self._show_current()

    def _on_double_click(self):
        # In a zoomed grid, a double-click only resets the zoom (the viewer
        # does that itself); once the whole grid fits, it opens the cell.
        if self._grid and not self._zoomed_at_double_click:
            # The first click of the double-click already made the cell
            # under the cursor current.
            QTimer.singleShot(0, lambda: self._open_in_single_view(
                self.index))

    def _on_enter(self):
        if self._grid:
            self._open_in_single_view(self.index)
        else:
            self._keep_current()

    # --- Full screen ----------------------------------------------------------

    def toggle_full_screen(self):
        if self.isFullScreen():
            if self._was_maximized:
                self.showMaximized()
            else:
                self.showNormal()
        else:
            self._was_maximized = self.isMaximized()
            # Without this, QDialog re-centres itself when shown and lands a
            # few pixels off the monitor, so the Windows taskbar stays on top.
            self.setAttribute(Qt.WidgetAttribute.WA_Moved, True)
            self.showFullScreen()
        self._update_info()
        self._show_overlays()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.toggle_full_screen()
            event.accept()
            return
        super().keyPressEvent(event)

    # --- Overlaid controls ----------------------------------------------------

    def _position_overlays(self):
        viewport = self.viewer.scroll_area.viewport()
        width = viewport.width()
        height = viewport.height()
        for overlay in self.overlays:
            overlay.adjustSize()
        self.info_overlay.move(_OVERLAY_MARGIN, _OVERLAY_MARGIN)
        view_x = width - self.view_overlay.width() - _OVERLAY_MARGIN
        view_y = _OVERLAY_MARGIN
        # In a narrow window, put the buttons under the info box.
        if view_x < self.info_overlay.geometry().right() + _OVERLAY_MARGIN:
            view_x = max(_OVERLAY_MARGIN, view_x)
            view_y = self.info_overlay.geometry().bottom() + _OVERLAY_MARGIN
        self.view_overlay.move(view_x, view_y)
        self.nav_overlay.move(
            (width - self.nav_overlay.width()) // 2,
            height - self.nav_overlay.height() - _OVERLAY_MARGIN)
        for overlay in self.overlays:
            overlay.raise_()

    def _show_overlays(self):
        for overlay in self.overlays:
            overlay.show()
        self._position_overlays()
        self.viewer.scroll_area.viewport().unsetCursor()
        self._hide_timer.start()

    def _hide_overlays_if_idle(self):
        # Keep the controls while the cursor is on them.
        if any(overlay.underMouse() for overlay in self.overlays):
            self._hide_timer.start()
            return
        for overlay in self.overlays:
            overlay.hide()
        if self.isFullScreen() and not self.viewer.scroll_area.is_dragging:
            self.viewer.scroll_area.viewport().setCursor(
                Qt.CursorShape.BlankCursor)

    def eventFilter(self, watched, event):
        event_type = event.type()
        if event_type == QEvent.Type.MouseButtonDblClick:
            # Seen before the viewer resets the zoom for the same event.
            self._zoomed_at_double_click = not self._at_fit_zoom()
        elif event_type == QEvent.Type.MouseMove:
            if not all(overlay.isVisible() for overlay in self.overlays):
                self._show_overlays()
            else:
                self._hide_timer.start()
        elif (event_type == QEvent.Type.Resize
              and watched is self.viewer.scroll_area.viewport()):
            self._position_overlays()
        return False

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._show_overlays)

