"""
"Find Similar Images" review dialog: near-duplicate detection by 256-bit
perceptual hash (pHash).

Scans the loaded images (all, filtered or selected) for images whose pHashes
differ by at most a user-set number of bits (the "pHash difference"), shows
each group side by side, and lets the user act on the group's non-kept images
(delete to Recycle Bin, move to a folder, or copy tags). Grouping is
seed-based (non-transitive): an image joins a group only if it is within the
threshold of that group's representative, so a chain of similar images is not
collapsed into one giant group.

Every pHash is cached in the database, keyed by the image's content hash
(schema version 4, the `phash` table), so a rescan decodes and hashes only
the new images; changing the threshold re-groups the cached hashes without
re-reading any file (and only re-runs the distance pass when it is raised
above the scanned threshold). The optional "match mirrored versions" caches
the mirrored image's hash alongside (`phash_mirror`).

This module (and its heavy dependencies, `numpy` in `models.similarity` and
`imagehash`/`scipy` in the scan worker) is imported lazily by the main window
when the dialog is opened, so the application still starts without loading
them.
"""

import os
from pathlib import Path

from PySide6.QtCore import (QObject, QRunnable, Qt, QThreadPool, Signal,
                            Slot)
from PySide6.QtGui import QColor, QFont, QPainter, QPalette
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox,
                               QComboBox, QDialog, QFileDialog, QFrame,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMenu, QMessageBox, QProgressDialog,
                               QPushButton, QRadioButton, QScrollArea,
                               QSizePolicy, QVBoxLayout, QWidget)

from dialogs.find_duplicates_dialog import (
    _ClickableLabel, _DuplicateCard, _PreviewDialog, _VerticalWheelForwarder,
    _elide, _format_modified, _format_size, _make_indicator_visible,
    _MAX_GROUPS_SHOWN, _MAX_MEMBERS_PER_GROUP, _THUMBNAIL_SIZE)
from models.image_storage import (
    clear_similarity_cache, flush_phash_cache, load_ignored_groups,
    load_ignored_pairs, load_phashes, remove_ignored_group,
    remove_ignored_pair, store_ignored_group, store_ignored_pair,
    buffer_phash_row)
from models.similarity import (PHASH_BITS, PHASH_VERSION, ScanCancelled,
                               TooManyMatches, compute_neighbors,
                               compute_phash, compute_phash_mirrored,
                               phash_from_hex, phash_to_hex, seed_group)
from models.image_list_model import ImageListModel, Scope
from utils.image import Image
from utils.settings import get_settings
from utils.settings_widgets import (SettingsBigCheckBox,
                                    FocusedScrollSettingsComboBox,
                                    FocusedScrollSettingsSpinBox)
from utils.utils import get_confirmation_dialog_reply, pluralize
from widgets.image_list import move_images_to_folder, trash_images

# The active similarity mode (the dialog's mode combo box; only one mode
# exists for now — the combo keeps room for more).
SIMILAR_MODE_NEAR_DUPPLICATES = 'Near duplicates'

# The cap on total neighbor links stored for one scan (one link is a
# (member, distance, mirrored) entry on one side of a pair, so this is
# twice the pair count). A threshold so broad that nearly everything
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
    dialog keeps for threshold changes and split/ignore actions (re-grouping
    without re-reading any file):

    - `neighbors`: for every compact position, the (index, distance,
      mirrored) list of the other positions within the scanned threshold
      (ignored and split pair links already removed);
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
                 generation: int, registry: set, scope: Scope,
                 threshold: int, match_mirrored: bool,
                 ignored_pairs: set, db_path, parent: QObject = None):
        QObject.__init__(self, parent)
        QRunnable.__init__(self)
        self.images = images
        self.model = model
        self.generation = generation
        self.registry = registry
        self.scope = scope
        self.threshold = threshold
        self.match_mirrored = match_mirrored
        # (content_hash_a, content_hash_b) pairs, a < b, whose links are
        # dropped (persisted-ignored and session-split pairs).
        self.ignored_pairs = ignored_pairs
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

        # Map the ignored content-hash pairs to index pairs over the
        # compact arrays.
        excluded: set = set()
        if self.ignored_pairs:
            pos_of = {keys[k]: p for p, k in enumerate(present)}
            for a, b in self.ignored_pairs:
                pa, pb = pos_of.get(a), pos_of.get(b)
                if pa is not None and pb is not None:
                    excluded.add((min(pa, pb), max(pa, pb)))

        neighbors = compute_neighbors(
            compact_hashes, self.threshold,
            mirrored_hashes=compact_mirrors,
            excluded=excluded,
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


class _SimilarCard(_DuplicateCard):
    """A similar-group card (the duplicates dialog's card, plus a dimmed
    state: a translucent gray overlay for a member whose link to the group's
    representative is an ignored pair, shown by "Show ignored")."""

    def __init__(self):
        super().__init__()
        self.dimmed = False

    def set_dimmed(self, dimmed: bool):
        if dimmed != self.dimmed:
            self.dimmed = dimmed
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.dimmed:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(128, 128, 128, 110))
            painter.end()


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
      representative better than the direct image did (the mirror option);
    - `dimmed`: the link to the representative is an ignored pair (only
      ever true with "Show ignored" on).
    """

    def __init__(self, dialog, seed_hash: str, group_key: str,
                 ignored: bool, members: list):
        self.dialog = dialog
        self.seed_hash = seed_hash
        # 'phash:<threshold>:<seed_hash>' — the persisted ignore key.
        self.group_key = group_key
        self.ignored = ignored
        self.members = members


class _SimilarGroupCards:
    """Renders and manages one similar group's row of image cards.

    Only the first ``_MAX_MEMBERS_PER_GROUP`` images are shown initially; a
    clickable "+N more" tile expands the row to show every image in the
    group. ``keep_member`` is the image to keep (by default the
    representative), even when it is hidden. Dimmed members (ignored
    pairs) have no "Keep" radio and are never acted on.
    """

    def __init__(self, dialog, group: _SimilarGroup):
        self.dialog = dialog
        self.group = group
        self.keep_member = group.members[0]['image']
        self.expanded = False
        # list of (QRadioButton, member dict) for the shown cards.
        self.radio_members: list = []
        # list of (relation QLabel, info QLabel, member dict) for the shown
        # cards.
        self.card_labels: list = []
        self.shown_count = 0

        self.box = QGroupBox(
            f'Near duplicates \u2014 {len(group.members)} images')
        # Only a title and a divider line, so the images sit directly on the
        # dark results background like in the Images pane.
        self.box.setFlat(True)
        title_font = self.box.font()
        title_font.setBold(True)
        self.box.setFont(title_font)
        box_layout = QVBoxLayout(self.box)
        box_layout.setContentsMargins(10, 8, 10, 10)

        # A header row with the group's actions (only the group's title is
        # bold).
        normal_font = QFont(title_font)
        normal_font.setBold(False)
        header = QHBoxLayout()
        self.ignore_button = QPushButton(
            'Un-ignore group' if group.ignored else 'Ignore this group')
        self.ignore_button.setFont(normal_font)
        self.ignore_button.clicked.connect(self._on_ignore_button)
        self.open_button = QPushButton('Open as selection')
        self.open_button.setFont(normal_font)
        self.open_button.clicked.connect(self._on_open_as_selection)
        header.addWidget(self.ignore_button)
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
        self._populate()

    def _on_ignore_button(self):
        if self.group.ignored:
            self.dialog.unignore_group(self.group)
        else:
            self.dialog.ignore_group(self.group)

    def _on_open_as_selection(self):
        self.dialog.open_group_as_selection(self.group)

    def _populate(self):
        button_group = QButtonGroup(self.box)
        button_group.setExclusive(True)
        self.button_group = button_group
        self.radio_members = []
        self.card_labels = []
        self.cards = []

        cards_widget = QWidget()
        cards_layout = QHBoxLayout(cards_widget)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(10)

        members = self.group.members
        shown = members if self.expanded else members[:_MAX_MEMBERS_PER_GROUP]
        for position, member in enumerate(shown):
            card, radio, relation_label, info_label = \
                self.dialog._build_card(member, self)
            if radio is not None:
                radio.setChecked(member['image'] is self.keep_member)
                radio.toggled.connect(
                    lambda checked, m=member: self._on_keep_toggled(
                        checked, m))
                button_group.addButton(radio)
                self.radio_members.append((radio, member))
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
        if checked:
            self.keep_member = member['image']
            self.update_relations()

    def update_relations(self):
        """Refresh the pHash-difference label on every shown card."""
        for relation_label, _info_label, member in self.card_labels:
            text, tip = self.dialog._card_relation_text(member)
            if member is self.group.members[0]:
                relation_label.setText(f'<b>{text}</b>')
            else:
                relation_label.setText(text)
            relation_label.setToolTip(tip)
        for card, member in self.cards:
            card.set_kept(not member['dimmed']
                          and member['image'] is self.keep_member)

    def refresh_info(self):
        """Refresh each shown card's details (e.g. its tag count)."""
        for _relation_label, info_label, member in self.card_labels:
            info_label.setText(self.dialog._card_info_text(member['image']))

    def expand(self):
        if self.expanded:
            return
        self.expanded = True
        self._populate()


class _SimilarPreviewDialog(_PreviewDialog):
    """The duplicates dialog's preview for a similar group: the "Identical
    file" relation (wrong for near-duplicates) is replaced by the real
    metric, each image's pHash difference from the group's representative.
    `relations` maps a member's path to that relation text.

    Keep in sync with `_PreviewDialog._update_info` in
    `find_duplicates_dialog.py` (only the relation part differs).
    """

    def __init__(self, parent, radio_members, index, thumbnails=None,
                 stats=None, relations=None):
        super().__init__(parent, radio_members, index, thumbnails, stats)
        self._relations = relations or {}

    def _update_info(self):
        _radio, member = self.radio_members[self.index]
        total = len(self.radio_members)
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
        relation = self._relations.get(member.path)
        if relation:
            parts.append(relation)
        self.details_label.setText('  \u2022  '.join(parts))

        multiple = total > 1
        self.prev_button.setEnabled(multiple)
        self.next_button.setEnabled(multiple)
        radio = self.radio_members[self.index][0]
        if radio is None:
            self.keep_button.setText('Keep this image (K)')
            self.keep_button.setEnabled(False)
        elif radio.isChecked():
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


# The registry of running scan tasks (see `_PhashScanTask`).
_scan_registry: set = set()


class FindSimilarDialog(QDialog):
    def __init__(self, parent, image_list_model: ImageListModel,
                 reload_callback, select_paths_callback):
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
        self._scanner: _PhashScanTask | None = None
        self._progress_dialog: QProgressDialog | None = None
        self._scan_generation = 0
        self._scanned_count = 0
        self._scanned_threshold = 0
        # The in-memory scan state (see `_PhashScanTask.finished`), kept so
        # that a threshold change or a split/ignore action can re-group
        # without re-reading any file.
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
        # The current results: the groups, and the cards of the groups
        # currently shown.
        self._groups: list = []
        self._group_cards: list = []
        # The persisted suppressions (content-hash pairs, a < b, and group
        # keys) plus the session-only split pairs.
        self._ignored_pairs: set = set(load_ignored_pairs(self.db_path))
        self._split_pairs: set = set()
        self._ignored_groups: set = set(
            row[0] for row in load_ignored_groups(self.db_path))

        self.setWindowTitle('Find Similar Images')
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

        description = QLabel(
            'Finds images that look alike: their 256-bit perceptual hashes '
            '(pHash) differ by at most the threshold below. In each group, '
            'the representative is the image the group is built around; '
            'pick the image to keep (by default the representative) and act '
            'on the others. Grouping is non-transitive: an image joins a '
            'group only if it is within the threshold of that group\'s '
            'representative, so a chain of similar images is not collapsed '
            'into one giant group. Click a thumbnail to view it larger. '
            'Deleted images go to the Recycle Bin.')
        description.setWordWrap(True)
        layout.addWidget(description)

        # --- Scan controls --------------------------------------------------
        scan_grid = QGridLayout()
        scan_grid.setHorizontalSpacing(8)
        scan_grid.addWidget(QLabel('Mode:'), 0, 0)
        self.mode_combo_box = FocusedScrollSettingsComboBox(
            key='similarity_mode', default=SIMILAR_MODE_NEAR_DUPPLICATES)
        self.mode_combo_box.addItems([SIMILAR_MODE_NEAR_DUPPLICATES])
        self.mode_combo_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.mode_combo_box.setSizePolicy(QSizePolicy.Policy.Fixed,
                                          QSizePolicy.Policy.Fixed)
        scan_grid.addWidget(self.mode_combo_box, 0, 1)
        scan_grid.addWidget(QLabel('Scope:'), 0, 2)
        scope_row = QHBoxLayout()
        self.scope_combo_box = FocusedScrollSettingsComboBox(
            key='similarity_scan_scope', default=Scope.ALL_IMAGES.value)
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

        scan_grid.addWidget(QLabel('Max pHash difference:'), 1, 0)
        self.threshold_spin_box = FocusedScrollSettingsSpinBox(
            key='similarity_phash_max_distance', default=16,
            minimum=0, maximum=PHASH_BITS)
        self.threshold_spin_box.valueChanged.connect(
            self._on_threshold_changed)
        scan_grid.addWidget(self.threshold_spin_box, 1, 1)
        self.match_mirrored_check_box = SettingsBigCheckBox(
            key='similarity_match_mirrored', default=False,
            text='Match mirrored versions')
        self.match_mirrored_check_box.stateChanged.connect(
            self._on_mirror_toggled)
        scan_grid.addWidget(self.match_mirrored_check_box, 1, 2, 1, 2)
        self.show_ignored_check_box = QCheckBox('Show ignored')
        self.show_ignored_check_box.toggled.connect(self._apply_grouping)
        _make_indicator_visible(self.show_ignored_check_box)
        scan_grid.addWidget(self.show_ignored_check_box, 1, 4)
        layout.addLayout(scan_grid)
        self._update_scope_count_label()

        # --- Results area ---------------------------------------------------
        # A darker background (the color of input fields) sets the results
        # apart from the controls, and each group sits on it as a panel.
        self.results_scroll_area = QScrollArea()
        self.results_scroll_area.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_container.setObjectName('similarResults')
        self.results_container.setBackgroundRole(
            QPalette.ColorRole.Base)
        self.results_container.setAutoFillBackground(True)
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(10, 10, 10, 10)
        self.results_layout.setSpacing(12)
        self.results_layout.addStretch()
        self.results_scroll_area.setWidget(self.results_container)
        layout.addWidget(self.results_scroll_area, 1)

        status_row = QHBoxLayout()
        self.status_label = QLabel(
            'Click "Scan" to look for similar images.')
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 1)
        self.show_more_button = QPushButton()
        self.show_more_button.clicked.connect(self.show_more_groups)
        self.show_more_button.hide()
        status_row.addWidget(self.show_more_button)
        layout.addLayout(status_row)

        # --- Actions ----------------------------------------------------------
        actions_box = QGroupBox('Act on the results')
        actions_layout = QGridLayout(actions_box)
        actions_layout.setColumnStretch(4, 1)

        actions_layout.addWidget(QLabel('Other images:'), 0, 0)
        self.delete_button = QPushButton('Delete to Recycle Bin')
        self.delete_button.clicked.connect(self.delete_others)
        actions_layout.addWidget(self.delete_button, 0, 1)
        self.move_button = QPushButton('Move to folder...')
        self.move_button.clicked.connect(self.move_others)
        actions_layout.addWidget(self.move_button, 0, 2)

        actions_layout.addWidget(QLabel('Tags:'), 1, 0)
        self.copy_tags_button = QPushButton("Copy rep's tags to all")
        self.copy_tags_button.setToolTip(
            "Add the kept image's tags to every other image of its group "
            '(the tags are never removed from any image).')
        self.copy_tags_button.clicked.connect(self.copy_rep_tags_to_all)
        actions_layout.addWidget(self.copy_tags_button, 1, 1)
        self.merge_tags_button = QPushButton('Merge all tags into rep')
        self.merge_tags_button.setToolTip(
            'Add every image\'s tags of a group to its kept image (the tags '
            'are never removed from any image).')
        self.merge_tags_button.clicked.connect(self.merge_all_tags_into_rep)
        actions_layout.addWidget(self.merge_tags_button, 1, 2)

        self.clear_cache_button = QPushButton('Clear similarity cache')
        self.clear_cache_button.setToolTip(
            'Delete the cached perceptual hashes of every image: the next '
            'scan has to re-decode and re-hash them. Ignored pairs and '
            'groups are kept.')
        self.clear_cache_button.clicked.connect(self.clear_cache)
        actions_layout.addWidget(self.clear_cache_button, 2, 0, 1, 3)
        layout.addWidget(actions_box)

        self._set_actions_enabled(False)

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    @property
    def scope(self) -> Scope:
        return Scope(self.scope_combo_box.currentText())

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
        images = self.image_list_model.images
        scope = self.scope
        in_scope = [image for i, image in enumerate(images)
                    if self.image_list_model.is_image_in_scope(scope, i,
                                                               image)]
        if len(in_scope) < 2:
            if not images:
                text = 'No images are loaded. Load a directory first.'
            else:
                text = (f'The scope "{scope.value}" contains '
                        f'{len(in_scope)} {pluralize("image", len(in_scope))}. '
                        f'At least 2 are needed to look for similar images.')
            self.status_label.setText(text)
            return
        self._clear_results()
        self._set_actions_enabled(False)
        self.scan_button.setEnabled(False)
        self.clear_cache_button.setEnabled(False)
        self._scanned_count = len(in_scope)
        self._scan_generation += 1
        self._scanned_threshold = 0

        # A busy bar until the task reports how many hashes it has to
        # compute (then it becomes a real progress bar); the cancel button
        # stops the scan.
        progress = QProgressDialog(
            'Scanning images for similar ones\u2026', 'Cancel', 0, 0, self)
        progress.setWindowTitle('Find Similar Images')
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        # Don't let Qt auto-hide the dialog the instant the bar reaches its
        # maximum: it is switched to a "Preparing groups" busy bar and then
        # hidden with reset() at the end.
        progress.setAutoReset(False)
        progress.setValue(0)
        self._progress_dialog = progress

        scanner = _PhashScanTask(
            images, self.image_list_model, self._scan_generation,
            _scan_registry, scope, self.threshold_spin_box.value(),
            self.match_mirrored_check_box.isChecked(),
            self._ignored_pairs | self._split_pairs, self.db_path)
        self._scanner = scanner
        # Bound methods (not lambdas) so the connections end when the dialog
        # is deleted.
        scanner.finished.connect(self._on_scan_finished)
        scanner.failed.connect(self._on_scan_failed)
        scanner.progress.connect(self._on_scan_progress)
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
    def _on_scan_finished(self, result: dict):
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
        QMessageBox.critical(self, 'Find Similar Images',
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
    # Grouping (from the in-memory neighbor lists; no file is re-read)
    # ------------------------------------------------------------------

    def _clear_results(self):
        self._groups = []
        self._neighbors = []
        self._hashes = []
        self._mirrors = None
        self._hashes_keys = []
        self._hash_rows = {}
        self._stats = {}
        self._tag_counts = {}
        self._skipped_count = 0
        self._thumbnails = {}
        self._scanned_threshold = 0
        self.show_more_button.hide()
        self._remove_group_cards()

    def _remove_group_cards(self):
        self._group_cards.clear()
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.results_layout.addStretch()

    @Slot(int)
    def _on_threshold_changed(self, value: int):
        if self._scanner is not None:
            return
        if self._scan_generation == 0 or not self._neighbors:
            return
        self._apply_grouping()

    @Slot(int)
    def _on_mirror_toggled(self, state):
        if self._scanner is not None:
            return
        if self._scan_generation == 0:
            return
        # The mirrored hashes change the pair distances: a full rescan (the
        # pHashes themselves are cached, so only the misses and the distance
        # pass run).
        self.start_scan()

    def _apply_grouping(self):
        """(Re)compute the groups from the in-memory neighbor lists: filter
        them by the current threshold and the excluded pairs, and run
        `seed_group`. No file is re-read and no database row is touched;
        raising the threshold above the scanned one re-runs the distance
        pass on the cached hashes instead.

        Ignored and split pairs are *always* excluded from the graph, so
        the grouping — and therefore the groups' persisted keys, which
        name the seed — stays stable while the user toggles "Show
        ignored": the toggle only shows the ignored groups and adds the
        dimmed display members of the ignored pairs (see `_make_group`).
        Re-linking the ignored pairs for display would re-seed the
        groups, and an ignored group would come back under a different
        key that its dismissal no longer matches."""
        if self._scanner is not None:
            return
        if self._scan_generation == 0 or not self._neighbors:
            return
        threshold = self.threshold_spin_box.value()
        if threshold > self._scanned_threshold:
            self.start_scan()
            return
        show_ignored = self.show_ignored_check_box.isChecked()
        excluded = self._excluded_index_pairs()
        neighbors = self._filter_neighbors(threshold, excluded)
        order_keys = self._order_keys(neighbors)
        groups = seed_group(neighbors, order_keys)
        self._groups = [self._make_group(seed, members, threshold,
                                         show_ignored)
                        for seed, members in groups]
        if not show_ignored:
            # Ignored groups are not shown (only their members, when
            # absorbed into another group, can still appear).
            self._groups = [group for group in self._groups
                            if not group.ignored]
        self._show_groups()

    def _excluded_index_pairs(self) -> set:
        """The ignored and split content-hash pairs as (min, max) pairs of
        compact positions."""
        pos_of = {content_hash: p for p, content_hash in
                  enumerate(self._hashes_keys)}
        excluded = set()
        for a, b in self._ignored_pairs | self._split_pairs:
            pa, pb = pos_of.get(a), pos_of.get(b)
            if pa is not None and pb is not None:
                excluded.add((min(pa, pb), max(pa, pb)))
        return excluded

    def _filter_neighbors(self, threshold: int, excluded: set) -> list:
        filtered = []
        for i, links in enumerate(self._neighbors):
            kept = [(j, distance, mirrored)
                    for (j, distance, mirrored) in links
                    if distance <= threshold
                    and (min(i, j), max(i, j)) not in excluded]
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

    def _make_group(self, seed: int, members: list, threshold: int,
                    show_ignored: bool) -> _SimilarGroup:
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
                    'dimmed': False,
                })
        if show_ignored:
            # The dimmed display members: the images whose link to the
            # representative is an ignored pair. Those links are excluded
            # from the graph (they never seed or join through it), so the
            # members are added here, display-only, by "Show ignored": the
            # keep/tag/removal actions skip dimmed members, and the card
            # offers "Un-ignore pair". (Split pairs stay hidden — they are
            # a session-only removal, not an "ignored" state.)
            member_positions = set(members)
            for j, (distance, mirrored) in info.items():
                if j in member_positions or distance > threshold:
                    continue
                j_hash = self._hashes_keys[j]
                a, b = (seed_hash, j_hash) if seed_hash < j_hash \
                    else (j_hash, seed_hash)
                if (a, b) not in self._ignored_pairs:
                    continue
                for row in self._hash_rows[j_hash]:
                    group_members.append({
                        'image': images[row],
                        'row': row,
                        'hash': j_hash,
                        'distance': distance,
                        'mirrored': mirrored,
                        'dimmed': True,
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
        self._remove_group_cards()
        if not groups:
            self.status_label.setText(
                f'No similar images found among {self._scanned_count} '
                f'{pluralize("image", self._scanned_count)}.'
                + self._skipped_note())
            self._set_actions_enabled(False)
            return
        self._tag_counts = {image.path: len(image.tags)
                            for image in self.image_list_model.images}
        self._add_group_cards(groups[:_MAX_GROUPS_SHOWN])
        self._set_actions_enabled(True)
        self._update_status()

    def _add_group_cards(self, groups: list):
        # Yield to the event loop now and then so the window stays
        # responsive while many cards are built.
        cards_built = 0
        for group in groups:
            group_cards = _SimilarGroupCards(self, group)
            self._group_cards.append(group_cards)
            self.results_layout.insertWidget(
                self.results_layout.count() - 1, group_cards.box)
            cards_built += group_cards.shown_count
            if cards_built >= 40:
                cards_built = 0
                QApplication.processEvents()

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

    def _update_status(self):
        groups = self._groups
        shown = len(self._group_cards)
        image_total = sum(len(group.members) for group in groups)
        message = (
            f'Found {len(groups)} similar '
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
        if max(len(group.members) for group in groups) \
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
        the pHash difference from the group's representative."""
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
        if member['dimmed']:
            text += ' (ignored pair)'
            tip += (' This link is an ignored pair; with "Show ignored" off '
                    'the images are not grouped together.')
        return text, tip

    def _build_card(self, member: dict, group_cards) -> tuple:
        """Return (card, keep radio or None, relation label, info label)."""
        image = member['image']
        card = _SimilarCard()
        card.set_dimmed(member['dimmed'])
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

        radio = None
        if not member['dimmed']:
            radio = QRadioButton('Keep')
            _make_indicator_visible(radio)
            card_layout.addWidget(radio)
            # A click anywhere on the card (the name, the details, the
            # radio's text, the background) checks the radio, selecting
            # this image to be kept. A click on the radio button itself
            # toggles it as usual, and one on the thumbnail opens the
            # preview.
            card.clicked.connect(lambda: radio.setChecked(True))

        # The per-member actions (split out / ignore this pair / un-ignore
        # / open in the image list).
        card.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        card.customContextMenuRequested.connect(
            lambda pos, m=member, gc=group_cards: self._show_card_menu(
                m, gc, card.mapToGlobal(pos)))
        return card, radio, relation_label, info_label

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

    def _show_preview(self, group_cards, member: dict):
        """Open a large preview that can cycle through the group's images
        and let the user pick which one to keep."""
        if group_cards is not None and group_cards.radio_members:
            radio_members = [(radio, m['image'])
                             for (radio, m) in group_cards.radio_members]
        else:
            # Fallback: a standalone image with no group/keep context.
            radio_members = [(None, member['image'])]

        index = 0
        for position, (_radio, current) in enumerate(radio_members):
            if current is member['image']:
                index = position
                break

        relations = {m['image'].path: self._card_relation_text(m)[0]
                     for m in group_cards.group.members}
        dialog = _SimilarPreviewDialog(self, radio_members, index,
                                       self._thumbnails, self._stats,
                                       relations)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.exec()

    # ------------------------------------------------------------------
    # Card menu and group actions (split / ignore / un-ignore)
    # ------------------------------------------------------------------

    def _show_card_menu(self, member: dict, group_cards: _SimilarGroupCards,
                        global_pos):
        group = group_cards.group
        menu = QMenu(self)
        if member['dimmed']:
            menu.addAction('Un-ignore pair',
                           lambda: self.unignore_pair(member, group))
        else:
            if member['distance'] is not None:
                # The representative cannot be split or ignored (it is the
                # group itself).
                menu.addAction('Split out of this group',
                               lambda: self.split_out(member, group))
                menu.addAction('Ignore this pair',
                               lambda: self.ignore_pair(member, group))
        menu.addAction('Open in image list',
                       lambda: self.open_in_image_list(member['image']))
        menu.exec(global_pos)

    @staticmethod
    def _pair_of(hash_a: str, hash_b: str) -> tuple:
        return (hash_a, hash_b) if hash_a < hash_b else (hash_b, hash_a)

    def split_out(self, member: dict, group: _SimilarGroup):
        """Drop `member`'s link to its group's representative for this
        session only (until the next scan)."""
        pair = self._pair_of(group.seed_hash, member['hash'])
        if pair in self._ignored_pairs:
            return
        self._split_pairs.add(pair)
        self._apply_grouping()

    def ignore_pair(self, member: dict, group: _SimilarGroup):
        """Persistently ignore `member`'s pair with its group's
        representative (the images are never grouped together again)."""
        pair = self._pair_of(group.seed_hash, member['hash'])
        self._split_pairs.discard(pair)
        self._ignored_pairs.add(pair)
        store_ignored_pair(self.db_path, *pair)
        self._apply_grouping()

    def unignore_pair(self, member: dict, group: _SimilarGroup):
        pair = self._pair_of(group.seed_hash, member['hash'])
        self._ignored_pairs.discard(pair)
        self._split_pairs.discard(pair)
        remove_ignored_pair(self.db_path, *pair)
        self._apply_grouping()

    def ignore_group(self, group: _SimilarGroup):
        """Persistently ignore a group at the current threshold (the group
        is not shown again at this threshold)."""
        self._ignored_groups.add(group.group_key)
        store_ignored_group(self.db_path, group.group_key, group.seed_hash,
                            self.mode_combo_box.currentText(),
                            str(self.threshold_spin_box.value()))
        self._apply_grouping()

    def unignore_group(self, group: _SimilarGroup):
        self._ignored_groups.discard(group.group_key)
        remove_ignored_group(self.db_path, group.group_key)
        self._apply_grouping()

    # ------------------------------------------------------------------
    # Opening results in the main window
    # ------------------------------------------------------------------

    def open_group_as_selection(self, group: _SimilarGroup):
        """Select every image of the group in the main window's image list
        (driving its multi-image grid preview and the grouped tag view)."""
        if self.select_paths_callback is not None:
            self.select_paths_callback(
                [member['image'].path for member in group.members])

    def open_in_image_list(self, image: Image):
        if self.select_paths_callback is not None:
            self.select_paths_callback([image.path])

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _keeper_targets(self) -> list:
        """Return (kept image, [other images]) for every shown group that is
        not ignored. Dimmed members (ignored pairs, visible only with
        "Show ignored" on) are not acted on. Includes images hidden behind a
        group's "+N more" tile."""
        pairs = []
        for group_cards in self._group_cards:
            group = group_cards.group
            if group.ignored:
                continue
            keeper = group_cards.keep_member
            others = [member['image'] for member in group.members
                      if member['image'] is not keeper and not member['dimmed']]
            if others:
                pairs.append((keeper, others))
        return pairs

    def _set_actions_enabled(self, enabled: bool):
        self.delete_button.setEnabled(enabled)
        self.move_button.setEnabled(enabled)
        self.copy_tags_button.setEnabled(enabled)
        self.merge_tags_button.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

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

    @Slot()
    def delete_others(self):
        pairs = self._keeper_targets()
        targets = [image for _keeper, others in pairs for image in others]
        if not targets:
            self._warn_no_targets()
            return
        count = len(targets)
        question = (f'Move {count} {pluralize("image", count)} and '
                    f'{"its" if count == 1 else "their"} '
                    f'{pluralize("caption", count)} to the Recycle Bin?\n\n'
                    'This deletes the image files themselves (their tags '
                    'and descriptions stay with any other copies).')
        reply = get_confirmation_dialog_reply('Delete Similar Images',
                                              question)
        if reply != QMessageBox.StandardButton.Yes:
            return
        lookup = self._image_lookup()
        deleted = trash_images(self, [image.path for image in targets],
                               list(lookup))
        if not deleted:
            self.status_label.setText('Nothing was deleted.')
            return
        self._finish_destructive_action(
            f'Moved {len(deleted)} {pluralize("image", len(deleted))} to '
            f'the Recycle Bin.')

    @Slot()
    def move_others(self):
        pairs = self._keeper_targets()
        targets = [image for _keeper, others in pairs for image in others]
        if not targets:
            self._warn_no_targets()
            return
        destination = QFileDialog.getExistingDirectory(
            self, 'Select a folder to move the similar images into',
            self._move_default_directory())
        if not destination:
            return
        lookup = self._image_lookup()
        moved_pairs = move_images_to_folder(
            self, [image.path for image in targets], Path(destination),
            list(lookup))
        moved = {source for source, _target in moved_pairs}
        if not moved:
            self.status_label.setText('Nothing was moved.')
            return
        self._finish_destructive_action(
            f'Moved {len(moved)} {pluralize("image", len(moved))} to '
            f'{destination}.')

    def _move_default_directory(self) -> str:
        # The app loads directories recursively, so an in-dataset
        # "_removed" folder would be reloaded; start the picker at the
        # dataset's parent instead.
        dataset = self.settings.value('directory_path', type=str) or ''
        if dataset:
            return str(Path(dataset).parent)
        return ''

    @Slot()
    def copy_rep_tags_to_all(self):
        """Add the kept image's tags to every other image of its group."""
        lookup = self._image_lookup()
        tags_by_row = {}
        target_count = 0
        for group_cards in self._group_cards:
            group = group_cards.group
            if group.ignored:
                continue
            keeper = group_cards.keep_member
            keeper_entry = lookup.get(keeper.path)
            if keeper_entry is None or not keeper_entry[1].tags:
                continue
            keeper_tags = list(keeper_entry[1].tags)
            for member in group.members:
                if member['image'] is keeper or member['dimmed']:
                    continue
                entry = lookup.get(member['image'].path)
                if entry is not None:
                    tags_by_row.setdefault(entry[0], []).extend(keeper_tags)
                    target_count += 1
        if not tags_by_row:
            self.status_label.setText(
                'Nothing to copy: no kept image has tags.')
            return
        self._add_tags_by_row(tags_by_row, 'Copy Representative Tags')
        self.status_label.setText(
            f"Copied the kept images' tags to {target_count} "
            f'{pluralize("image", target_count)}. Nothing was deleted.')

    @Slot()
    def merge_all_tags_into_rep(self):
        """Add every image's tags of a group to its kept image."""
        lookup = self._image_lookup()
        tags_by_row = {}
        source_count = 0
        for group_cards in self._group_cards:
            group = group_cards.group
            if group.ignored:
                continue
            keeper = group_cards.keep_member
            keeper_entry = lookup.get(keeper.path)
            if keeper_entry is None:
                continue
            for member in group.members:
                if member['image'] is keeper or member['dimmed']:
                    continue
                entry = lookup.get(member['image'].path)
                if entry is None or not entry[1].tags:
                    continue
                tags_by_row.setdefault(keeper_entry[0], []).extend(
                    entry[1].tags)
                source_count += 1
        if not tags_by_row:
            self.status_label.setText('Nothing to merge.')
            return
        self._add_tags_by_row(tags_by_row, 'Merge Similar Tags')
        self.status_label.setText(
            f'Merged the tags of {source_count} '
            f'{pluralize("image", source_count)} into their kept images. '
            'Nothing was deleted.')

    @Slot()
    def clear_cache(self):
        reply = get_confirmation_dialog_reply(
            'Clear Similarity Cache',
            'Delete the cached perceptual hashes of every image? The next '
            'scan will have to re-decode and re-hash them all.\n\n'
            'Ignored pairs and groups are kept.')
        if reply != QMessageBox.StandardButton.Yes:
            return
        clear_similarity_cache(self.db_path)
        # The in-memory scan state (the current groups) is still valid; the
        # next scan rebuilds the cache.
        self.status_label.setText(
            'Similarity cache cleared. Click "Scan" to rescan; the current '
            'results are kept until then.')

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
        self.status_label.setText(
            f'{summary} The directory is reloading \u2014 click "Scan" again '
            f'to check for any remaining similar images.')

    def _warn_no_targets(self):
        QMessageBox.information(
            self, 'Find Similar Images',
            'There is nothing to act on: every shown group is ignored or '
            'has only its kept image.')
