from typing import Optional

from PySide6.QtCore import (QEvent, QItemSelectionModel, QModelIndex,
                            QPersistentModelIndex, QPoint, QPropertyAnimation,
                            QStringListModel, Qt, QTimer, Signal, Slot)
from PySide6.QtGui import QFocusEvent, QKeyEvent
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCompleter,
                               QDockWidget, QHBoxLayout, QLabel, QLineEdit,
                               QListView, QMessageBox, QPushButton, QSplitter,
                               QTextEdit, QVBoxLayout, QWidget)
from transformers import PreTrainedTokenizerBase

from models.image_list_model import ImageListModel
from models.proxy_image_list_model import ProxyImageListModel
from models.tag_counter_model import TagCounterModel
from utils.image import Image
from utils.settings import (DEFAULT_SETTINGS, get_settings,
                            get_tag_separator)
from utils.text_edit_item_delegate import TextEditItemDelegate
from utils.utils import get_confirmation_dialog_reply
from widgets.group_tags_panel import GroupTagsPanel
from widgets.image_list import ImageList

MAX_TOKEN_COUNT = 75
# The held key that makes the grid preview brighten the current (highlighted)
# cell while navigating the grouped view. Observed, not consumed, so
# Ctrl+key shortcuts still work.
GRID_PEEK_KEY = Qt.Key.Key_Control


def _should_redirect_typing_to_tag_input(event: QKeyEvent) -> bool:
    """Whether a keystroke should start a new tag in the Add Tag box.

    Typing a printable character while the tags list (or the Images pane) has
    keyboard focus jumps to the Add Tag box, so the user can start a new tag
    without clicking the box first.

    Keyboard shortcuts such as Ctrl+C or Alt+... are never hijacked. Keys like
    Enter, Tab, the arrows, Delete and Backspace have an empty or
    control-character `text()` and are excluded, so tag deletion and
    navigation keep working.
    """
    blocking_modifiers = (Qt.KeyboardModifier.ControlModifier
                          | Qt.KeyboardModifier.AltModifier
                          | Qt.KeyboardModifier.MetaModifier)
    if ((event.modifiers() & blocking_modifiers)
            != Qt.KeyboardModifier.NoModifier):
        return False
    text = event.text()
    return len(text) == 1 and text.isprintable()


class TagInputBox(QLineEdit):
    # Duration in milliseconds of the border flash / shake feedback.
    FLASH_DURATION_MS = 350

    tags_addition_requested = Signal(list, list)

    def __init__(self, image_tag_list_model: QStringListModel,
                 tag_counter_model: TagCounterModel, image_list: ImageList,
                 tag_separator: str):
        super().__init__()
        self.image_tag_list_model = image_tag_list_model
        self.image_list = image_list
        self.tag_separator = tag_separator

        self.setPlaceholderText('Add Tag')
        self.setStyleSheet('padding: 8px;')
        settings = get_settings()
        autocomplete_tags = settings.value(
            'autocomplete_tags',
            defaultValue=DEFAULT_SETTINGS['autocomplete_tags'], type=bool)
        if autocomplete_tags:
            self.completer = QCompleter(tag_counter_model)
            self.setCompleter(self.completer)

            def add_completed_tag(text: str):
                self.add_tag(text)
                self._return_focus_to_tags_list_if_needed()

            self.completer.activated.connect(add_completed_tag)
            # Clear the input box after the completer inserts the tag into it.
            self.completer.activated.connect(
                lambda: QTimer.singleShot(0, self.clear))
        else:
            self.completer = None
        # State for the border-flash / shake feedback animations.
        self._shake_animation: Optional[QPropertyAnimation] = None
        self._flash_timer: Optional[QTimer] = None
        self._scope_feedback_timer: Optional[QTimer] = None
        # Set by ImageTagsEditor once the tags list exists. Used to return
        # focus to the tags list after a tag is added.
        self.image_tags_list = None
        # True when this box was auto-focused by typing in the tags list (see
        # ImageTagsList.keyPressEvent and ImageTagsEditor.eventFilter). When
        # set, focus returns to the tags list after the tag is added. Cleared
        # once focus leaves the box for any other reason, so a later manual
        # edit doesn't wrongly jump focus.
        self._return_focus_to_tags_list_after_add = False
        # When set, focus (and its selection) returns to this specific list
        # view after a tag is added, instead of the default `image_tags_list`.
        # Used by the grouped Common/Differences list.
        self._return_focus_target = None
        # Set by ImageTagsEditor when the grouped multi-image view is active.
        # When active, added tags go to `target_indices_provider()` (which
        # honors the "All selected / Current image" scope selector), the
        # single-image fast path is skipped, and the per-add confirmation
        # dialog is suppressed (the grouped pane is the review surface).
        self.group_mode_active = False
        self.target_indices_provider = None
        # Whether the pending commit targets only the current image. Kept in
        # sync with the editor's "Current image only" scope toggle.
        self.pending_scope_current = False
        # Placeholder shown in group mode. Restored after a transient
        # post-commit confirmation.
        self._group_placeholder = ''

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() not in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            super().keyPressEvent(event)
            return
        # If Ctrl+Enter is pressed and the completer is visible, add the first
        # tag in the completer popup.
        if (event.modifiers() == Qt.KeyboardModifier.ControlModifier
                and self.completer is not None
                and self.completer.popup().isVisible()):
            first_tag = self.completer.popup().model().data(
                self.completer.model().index(0, 0), Qt.ItemDataRole.EditRole)
            self.add_tag(first_tag)
        # Otherwise, add the tag in the input box.
        else:
            self.add_tag(self.text())
        self.clear()
        if self.completer is not None:
            self.completer.popup().hide()
        self._return_focus_to_tags_list_if_needed()

    def focusOutEvent(self, event: QFocusEvent):
        super().focusOutEvent(event)
        # Drop the auto-focus flag whenever focus genuinely leaves the box, so
        # a later manual edit won't jump focus back to the tags list. The
        # autocomplete popup does not count: it steals focus with
        # PopupFocusReason, and we still want the flag when a suggestion there
        # is what adds the tag.
        if event.reason() != Qt.FocusReason.PopupFocusReason:
            self._return_focus_to_tags_list_after_add = False
            self._return_focus_target = None

    def _return_focus_to_tags_list_if_needed(self):
        """Return keyboard focus to the tags list when this box was auto-focused
        by typing there. The flag is read and cleared before moving focus."""
        should_return = self._return_focus_to_tags_list_after_add
        target = self._return_focus_target
        self._return_focus_to_tags_list_after_add = False
        self._return_focus_target = None
        if not should_return:
            return
        if target is not None:
            target.setFocus()
        elif self.image_tags_list is not None:
            self.image_tags_list.setFocus()

    def add_tag(self, tag: str):
        tag = tag.strip()
        if not tag:
            return
        tags = [t.strip() for t in tag.split(self.tag_separator) if t.strip()]
        if not tags:
            return
        if self.target_indices_provider is not None:
            selected_image_indices = self.target_indices_provider()
        else:
            selected_image_indices = self.image_list.get_selected_image_indices()
        selected_image_count = len(selected_image_indices)
        if selected_image_count == 0:
            # No image to add to: keep the typed text and give feedback
            # instead of silently doing nothing.
            self._flash_duplicate()
            return
        if (not self.group_mode_active and len(tags) == 1
                and selected_image_count == 1):
            # The tag (in any letter case) is already on the image: reject it
            # and give the user visual feedback instead of silently adding a
            # duplicate.
            existing = {existing_tag.casefold()
                        for existing_tag
                        in self.image_tag_list_model.stringList()}
            if tags[0].casefold() in existing:
                self._flash_duplicate()
                self.clear()
                return
            # Add an empty tag and set it to the new tag.
            self.image_tag_list_model.insertRow(
                self.image_tag_list_model.rowCount())
            new_tag_index = self.image_tag_list_model.index(
                self.image_tag_list_model.rowCount() - 1)
            self.image_tag_list_model.setData(new_tag_index, tags[0])
            return
        # Reject a no-op add (every target image already contains every tag)
        # with the same red-border shake as the single-image path, instead of
        # silently doing nothing or asking to confirm adding nothing. Covers
        # both the grouped view and a plain multi-image selection.
        if self._all_target_images_have_tags(tags, selected_image_indices):
            self._flash_duplicate()
            self.clear()
            return
        if selected_image_count > 1 and not self.group_mode_active:
            if len(tags) > 1:
                question = (f'Add tags to {selected_image_count} selected '
                            f'images?')
            else:
                question = (f'Add tag "{tags[0]}" to {selected_image_count} '
                            f'selected images?')
            reply = get_confirmation_dialog_reply(title='Add Tag',
                                                  question=question)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.tags_addition_requested.emit(tags, selected_image_indices)
        if self.group_mode_active:
            # Confirm which scope the tag(s) went to, so the (now hidden)
            # persistent selector isn't missed.
            self._show_scope_feedback(self.pending_scope_current,
                                      selected_image_count)

    def _all_target_images_have_tags(self, tags: list[str],
                                     image_indices: list) -> bool:
        """Whether every target image already contains every tag, making the
        add a no-op. Used to reject duplicate adds with shake feedback in the
        multi-image and grouped views, mirroring the single-image path."""
        if not image_indices:
            return False
        wanted = {tag.casefold() for tag in tags}
        for index in image_indices:
            image = index.model().data(index, Qt.ItemDataRole.UserRole)
            if image is None:
                return False
            if not wanted.issubset(tag.casefold() for tag in image.tags):
                return False
        return True

    def _flash_duplicate(self):
        """Signal a rejected duplicate tag by briefly turning the input box
        border red and shaking it, then restoring the normal border."""
        # A red border for the duration of the shake gives clear "rejected"
        # feedback. Restoring an empty stylesheet brings back the native
        # (Fusion) border afterwards.
        self._flash_border('#e03c3c')
        self._start_shake()

    def _flash_border(self, color: str):
        """Turn the input box border `color` for `FLASH_DURATION_MS`, then
        restore the normal border.
        """
        self.setStyleSheet(f'QLineEdit {{ border: 1px solid {color}; }}')
        if self._flash_timer is not None:
            self._flash_timer.stop()
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._restore_border)
        self._flash_timer.start(self.FLASH_DURATION_MS)

    def _restore_border(self):
        """Remove the temporary colored border and force the widget's style to
        repaint, so the native (Fusion) border returns immediately.
        """
        self.setStyleSheet('')
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _start_shake(self):
        """Nudge the input box left and right a few times to draw attention."""
        if self._shake_animation is not None:
            self._shake_animation.stop()
        original_pos = self.pos()
        animation = QPropertyAnimation(self, b'pos', self)
        animation.setDuration(self.FLASH_DURATION_MS)
        amplitude = 6
        offsets = [0, amplitude, -amplitude, amplitude, -amplitude,
                   amplitude // 2, -amplitude // 2, 0]
        last_index = len(offsets) - 1
        for index, offset in enumerate(offsets):
            animation.setKeyValueAt(index / last_index,
                                    original_pos + QPoint(offset, 0))
        # Ensure the box ends up exactly where it started even if the layout
        # shifted it mid-animation.
        animation.finished.connect(lambda: self.move(original_pos))
        self._shake_animation = animation
        animation.start()

    def set_group_placeholder(self, text: str):
        """Set the Add Tag box placeholder shown in group mode. Applied
        immediately unless a transient post-commit confirmation is currently
        showing."""
        self._group_placeholder = text
        if self._scope_feedback_timer is None:
            self.setPlaceholderText(text)

    def reset_group_placeholder(self):
        """Return to the plain single-image placeholder when leaving group mode."""
        if self._scope_feedback_timer is not None:
            self._scope_feedback_timer.stop()
            self._scope_feedback_timer = None
        self._group_placeholder = ''
        self.setPlaceholderText('Add Tag')

    def _show_scope_feedback(self, scope_current: bool, count: int):
        """Briefly replace the placeholder with a confirmation naming the scope
        the just-added tag went to, then restore the group placeholder."""
        if scope_current:
            message = '\u2713 Added to current image'
        else:
            message = f'\u2713 Added to all {count} images'
        self.setPlaceholderText(message)
        if self._scope_feedback_timer is not None:
            self._scope_feedback_timer.stop()
        self._scope_feedback_timer = QTimer(self)
        self._scope_feedback_timer.setSingleShot(True)
        self._scope_feedback_timer.timeout.connect(
            self._restore_group_placeholder)
        self._scope_feedback_timer.start(1600)

    def _restore_group_placeholder(self):
        self._scope_feedback_timer = None
        self.setPlaceholderText(self._group_placeholder or 'Add Tag')


class ImageTagsList(QListView):
    def __init__(self, image_tag_list_model: QStringListModel):
        super().__init__()
        self.image_tag_list_model = image_tag_list_model
        # Set by ImageTagsEditor: the Add Tag box that typing should jump to,
        # and the Images pane list view used to cycle the selected images with
        # Up/Down at the ends of this list.
        self.tag_input_box = None
        self.image_list_view = None
        self.setModel(self.image_tag_list_model)
        self.setItemDelegate(TextEditItemDelegate(self))
        self.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setWordWrap(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)

    def keyPressEvent(self, event: QKeyEvent):
        """
        Redirect typing to the Add Tag box (only when no tag is selected here),
        cycle the selected images with Up/Down at the ends of this list (when
        2+ images are selected), or delete the selected tags when the delete
        key or backspace key is pressed.
        """
        tag_input_box = self.tag_input_box
        # Only auto-jump to the Add Tag box when no tag is already selected in
        # this list. With a tag selected, typing should type-ahead through the
        # tags (the default QListView behavior) so the user can quickly find a
        # tag to remove instead of starting a new one.
        if (tag_input_box is not None and not tag_input_box.isHidden()
                and not self.selectedIndexes()
                and _should_redirect_typing_to_tag_input(event)):
            # Remember to return focus here after the tag is added, since the
            # box is being auto-focused only because the user typed in this
            # list.
            tag_input_box._return_focus_to_tags_list_after_add = True
            tag_input_box.raise_()
            tag_input_box.setFocus()
            tag_input_box.insert(event.text())
            return
        if (event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down)
                and self._handle_multi_select_navigation(event)):
            return
        if event.key() not in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            super().keyPressEvent(event)
            return
        rows_to_remove = [index.row() for index in self.selectedIndexes()]
        if not rows_to_remove:
            return
        remaining_tags = [tag for i, tag
                          in enumerate(self.image_tag_list_model.stringList())
                          if i not in rows_to_remove]
        self.image_tag_list_model.setStringList(remaining_tags)
        min_removed_row = min(rows_to_remove)
        remaining_row_count = self.image_tag_list_model.rowCount()
        if min_removed_row < remaining_row_count:
            self.select_tag(min_removed_row)
        elif remaining_row_count:
            # Select the last tag.
            self.select_tag(remaining_row_count - 1)

    def _handle_multi_select_navigation(self, event: QKeyEvent) -> bool:
        """Cycle selected images with Up/Down when 2+ images are selected.

        When multiple images are selected, plain Up/Down at the top/bottom of
        the tag list cycles to the previous/next selected image (keeping the
        selection intact). Between the ends, Up/Down navigate tags as usual.
        Ctrl+Up/Down is the deliberate way to select a single image just
        outside the selection; it is handled by the window-wide Ctrl+Up/Down
        shortcuts (go_to_previous_image / go_to_next_image), so here the key is
        only consumed to stop the default list navigation from also running.
        With a single image selected this does nothing (returns False).
        """
        view = self.image_list_view
        if view is None or len(view.selectedIndexes()) < 2:
            return False
        direction = -1 if event.key() == Qt.Key.Key_Up else 1
        modifiers = event.modifiers()
        if (modifiers & Qt.KeyboardModifier.ControlModifier
                and not (modifiers & Qt.KeyboardModifier.ShiftModifier)):
            return True
        if modifiers != Qt.KeyboardModifier.NoModifier:
            return False
        count = self.image_tag_list_model.rowCount()
        row = self.currentIndex().row()
        at_edge = (count == 0
                   or (row <= 0 if direction < 0 else row >= count - 1))
        if at_edge:
            view.switch_between_selected_images(direction)
            return True
        return False

    def select_tag(self, row: int):
        # If the current index is not set, using the arrow keys to navigate
        # through the tags after selecting the tag will not work.
        self.setCurrentIndex(self.image_tag_list_model.index(row))
        self.selectionModel().select(
            self.image_tag_list_model.index(row),
            QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def copy_selected_tags_to_clipboard(self) -> bool:
        """Copy the selected tags to the clipboard as separator-joined text.

        Returns True if at least one tag was copied, so the caller knows the
        copy was handled here instead of falling back to copying every tag of
        the selected image(s).
        """
        selected_rows = sorted(index.row()
                               for index in self.selectedIndexes())
        if not selected_rows:
            return False
        tags = self.image_tag_list_model.stringList()
        selected_tags = [tags[row] for row in selected_rows
                         if 0 <= row < len(tags)]
        if not selected_tags:
            return False
        QApplication.clipboard().setText(
            get_tag_separator().join(selected_tags))
        return True


class ImageTagsEditor(QDockWidget):
    image_data_save_requested = Signal(Image)
    # The set of image file paths whose cells should be marked in the grid
    # preview (the images containing the currently-focused Differences tag).
    grid_mark_paths_changed = Signal(list)
    # Emitted while the keyboard "peek" key is held in the grouped view, so
    # the grid can reveal the current (highlighted) cell at full brightness.
    # True on press, False on release.
    grid_peek_changed = Signal(bool)

    # The images the pane edits are stored as persistent indices, which Qt
    # keeps pointing at the same image when rows above it are added or removed
    # (e.g. by a refresh). A plain index would silently point at a different
    # image. The properties below hand out plain indices, or None when the
    # image is gone.
    @staticmethod
    def _persist(index) -> Optional[QPersistentModelIndex]:
        if index is None or not index.isValid():
            return None
        return QPersistentModelIndex(index)

    @staticmethod
    def _resolve(persistent) -> Optional[QModelIndex]:
        if persistent is None or not persistent.isValid():
            return None
        return persistent.model().index(persistent.row(), persistent.column())

    @property
    def image_index(self) -> Optional[QModelIndex]:
        return self._resolve(getattr(self, '_image_index_ref', None))

    @image_index.setter
    def image_index(self, index):
        self._image_index_ref = self._persist(index)

    @property
    def _group_current_index(self) -> Optional[QModelIndex]:
        return self._resolve(getattr(self, '_group_current_ref', None))

    @_group_current_index.setter
    def _group_current_index(self, index):
        self._group_current_ref = self._persist(index)

    @property
    def _group_source_indices(self) -> list[QModelIndex]:
        resolved = (self._resolve(ref)
                    for ref in getattr(self, '_group_source_refs', []))
        return [index for index in resolved if index is not None]

    @_group_source_indices.setter
    def _group_source_indices(self, indices):
        self._group_source_refs = [ref for ref in map(self._persist, indices)
                                   if ref is not None]

    def __init__(self, proxy_image_list_model: ProxyImageListModel,
                 tag_counter_model: TagCounterModel,
                 image_tag_list_model: QStringListModel, image_list: ImageList,
                 tokenizer: PreTrainedTokenizerBase, tag_separator: str):
        super().__init__()
        self.image_list_model: ImageListModel = \
            proxy_image_list_model.sourceModel()
        self.proxy_image_list_model = proxy_image_list_model
        self.image_tag_list_model = image_tag_list_model
        self.image_list = image_list
        self.tokenizer = tokenizer
        self.tag_separator = tag_separator
        self.image_index = None

        # Each `QDockWidget` needs a unique object name for saving its state.
        self.setObjectName('image_tags_editor')
        self.setWindowTitle('Image Tags')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)
        self.tag_input_box = TagInputBox(self.image_tag_list_model,
                                         tag_counter_model, image_list,
                                         tag_separator)
        # Grouped-view scope toggle: when checked, the Add Tag box adds only
        # to the current (highlighted) image; unchecked adds to all selected.
        # Sits above the Add Tag box and is hidden until grouped mode is
        # entered. The common/differences summary auto-sets it (differences ->
        # on, common -> off); manually toggling it never changes the filter.
        self.scope_button = QPushButton('Current image only')
        self.scope_button.setCheckable(True)
        self.scope_button.setAutoDefault(False)
        self.scope_button.setDefault(False)
        self.scope_button.setVisible(False)
        self.scope_button.toggled.connect(self._on_scope_toggled)
        self.image_tags_list = ImageTagsList(self.image_tag_list_model)
        # Let the tag list redirect typing to the Add Tag box and cycle the
        # selected images with Up/Down at its ends.
        self.image_tags_list.tag_input_box = self.tag_input_box
        self.image_tags_list.image_list_view = self.image_list.list_view
        # Let the Add Tag box return focus to the tag list after an
        # auto-focused add.
        self.tag_input_box.image_tags_list = self.image_tags_list
        # Group (multi-image) mode: the unified grouped-tags panel, hidden
        # until group mode is entered.
        self.group_tags_panel = GroupTagsPanel()
        self.group_tags_panel.setVisible(False)
        # The context menu only offers the "current image" actions when they
        # would change something.
        self.group_tags_panel.current_image_tags_provider = \
            self._current_group_image_tags
        self.group_tags_panel.remove_from_all_requested.connect(
            self._remove_group_tags_from_all)
        self.group_tags_panel.add_to_all_requested.connect(
            self._add_group_tags_to_all)
        self.group_tags_panel.remove_from_current_requested.connect(
            self._remove_group_tags_from_current)
        self.group_tags_panel.add_to_current_requested.connect(
            self._add_group_tags_to_current)
        self.group_tags_panel.partial_focus_changed.connect(
            self._on_partial_focus_changed)
        self.group_tags_panel.cycle_image_requested.connect(
            self._cycle_selected_image)
        self.group_tags_panel.rename_tag_requested.connect(
            self._rename_group_tag)
        self.group_tags_panel.filter_mode_changed.connect(
            self._on_filter_mode_changed)
        # Group-mode state.
        self._group_mode = False
        self._group_source_indices = []
        self._group_current_index = None
        self.tag_input_box.target_indices_provider = \
            self._group_add_target_indices

        self.token_count_label = QLabel()
        # Add description box
        self.description_label = QLabel('Image Description')
        self.description_editor = QTextEdit()
        # Add a debounced timer to auto save the description
        self.description_save_timer = QTimer(self)
        self.description_save_timer.setSingleShot(True)
        self.description_save_timer.setInterval(600)  # ms of silence before saving
        self.description_save_timer.timeout.connect(self.save_image_description)
        self.description_editor.textChanged.connect(self.on_description_changed)

        # A container widget is required to use a layout with a `QDockWidget`.
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.scope_button)
        layout.addWidget(self.tag_input_box)
        # Small container for the tags list
        tags_widget = QWidget()
        tags_layout = QVBoxLayout(tags_widget)
        tags_layout.setContentsMargins(0, 0, 0, 0)
        tags_layout.addWidget(self.image_tags_list)
        tags_layout.addWidget(self.group_tags_panel)
        # Bottom status row: the token count for a single image, or the
        # clickable common/differences summary in the grouped view (the two
        # are mutually exclusive).
        self.token_row = QWidget()
        token_row_layout = QHBoxLayout(self.token_row)
        token_row_layout.setContentsMargins(0, 0, 0, 0)
        self.group_tags_panel.summary_label.setVisible(False)
        token_row_layout.addWidget(self.group_tags_panel.summary_label)
        token_row_layout.addWidget(self.token_count_label)
        token_row_layout.addStretch()
        tags_layout.addWidget(self.token_row)
        # Small container for the description box
        self.description_widget = QWidget()
        description_layout = QVBoxLayout(self.description_widget)
        description_layout.setContentsMargins(0, 0, 0, 0)
        description_layout.addWidget(self.description_label)
        description_layout.addWidget(self.description_editor)
        # Put both sections in a resizable splitter.
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(tags_widget)
        splitter.addWidget(self.description_widget)
        # Prevent a section from being collapsed to zero.
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter)
        self.setWidget(container)

        # When a tag is added, select it and scroll to the bottom of the list.
        self.image_tag_list_model.rowsInserted.connect(
            lambda _, __, last_index:
            self.image_tags_list.selectionModel().select(
                self.image_tag_list_model.index(last_index),
                QItemSelectionModel.SelectionFlag.ClearAndSelect))
        self.image_tag_list_model.rowsInserted.connect(
            self.image_tags_list.scrollToBottom)
        # `rowsInserted` does not have to be connected because `dataChanged`
        # is emitted when a tag is added.
        self.image_tag_list_model.modelReset.connect(self.count_tokens)
        self.image_tag_list_model.dataChanged.connect(self.count_tokens)
        # Keep Left/Right variant cycling working while the user is interacting
        # with the group-mode controls: intercept those arrows and forward them
        # to the Images pane. Installed on the widgets that would otherwise
        # swallow the arrows (the tag list, the scope selector). The Add Tag
        # box is included too, but only cycles when it is empty so typed text
        # can still be edited with the arrows.
        for group_widget in (self.group_tags_panel,
                             self.group_tags_panel.tag_list,
                             self.scope_button,
                             self.tag_input_box):
            group_widget.installEventFilter(self)

    @Slot()
    def count_tokens(self):
        caption = self.tag_separator.join(
            self.image_tag_list_model.stringList())
        # Subtract 2 for the BOS and newline tokens.
        caption_token_count = len(self.tokenizer(caption).input_ids) - 2
        if caption_token_count > MAX_TOKEN_COUNT:
            self.token_count_label.setStyleSheet('color: red;')
        else:
            self.token_count_label.setStyleSheet('')
        self.token_count_label.setText(f'{caption_token_count} / '
                                       f'{MAX_TOKEN_COUNT} Tokens')

    @Slot()
    def select_first_tag(self):
        if self.image_tag_list_model.rowCount() == 0:
            return
        self.image_tags_list.select_tag(0)

    def select_last_tag(self):
        tag_count = self.image_tag_list_model.rowCount()
        if tag_count == 0:
            return
        self.image_tags_list.select_tag(tag_count - 1)

    def focus_tags_list(self):
        """Focus the tag list for the "Focus Image Tags List" shortcut.

        In the single-image view this focuses the Image Tags list and selects
        its first tag. In the grouped view that list is hidden, so focus the
        Common/Differences panel instead.
        """
        self.raise_()
        if self._group_mode:
            self.group_tags_panel.focus_first_tag()
            return
        self.image_tags_list.setFocus()
        self.select_first_tag()

    @Slot()
    def load_image_tags(self, proxy_image_index: QModelIndex):
        if self._group_mode:
            # In grouped multi-image mode the tag view reflects the whole
            # selection, not a single image. Arrowing between the selected
            # images only moves which one is "current" (the target of the
            # "Current image only" scope); it must not reload a single-image
            # view. Just track the current image and keep the group view.
            self._group_current_index = self.proxy_image_list_model.mapToSource(
                proxy_image_index)
            self.image_index = self._group_current_index
            return
        self.image_index = self.proxy_image_list_model.mapToSource(
            proxy_image_index)
        # On image load save the image reference for later use
        self.image: Image = self.proxy_image_list_model.data(
            proxy_image_index, Qt.ItemDataRole.UserRole)
        # The image is None when the index is invalid (for example, when the
        # model is reset while a directory is loading).
        if self.image is None:
            return
        # The image list can be on screen before the background load of this
        # image's tags and description has finished (it shows empty ones
        # until then). Load them now, so the editors always display — and the
        # user always edits against — the image's real data. No-op when the
        # load has already finished (the common case).
        self.image_list_model.ensure_row_loaded(self.image_index.row())
        self.set_image_tags()
        self.set_image_description()

    def set_image_tags(self):
        # If the string list already contains the image's tags, do not reload
        # them. This is the case when the tags are edited directly through the
        # image tags editor. Removing this check breaks the functionality of
        # reordering multiple tags at the same time because it gets interrupted
        # after one tag is moved.
        current_string_list = self.image_tag_list_model.stringList()
        if current_string_list == self.image.tags:
            return
        self.image_tag_list_model.setStringList(self.image.tags)
        self.count_tokens()
        if self.image_tags_list.hasFocus():
            self.select_first_tag()

    def set_image_description(self):
        current_description = self.description_editor.toPlainText()
        if current_description == self.image.description:
            return
        self.description_editor.setPlainText(self.image.description)
        # TODO: Add token counter for description?

    @Slot()
    def on_description_changed(self):
        if self.image is None:
            return
        current_description = self.description_editor.toPlainText()
        # Populate the editor programmatically (e.g. loading a different image,
        # or a storage-mode switch followed by a directory reload) sets the text
        # to exactly the image's current description. Do not treat that as an
        # edit and schedule a save; only react when the text actually differs
        # from the image's description. This mirrors the equality guard in
        # MainWindow.update_image_tags for the tag list.
        if current_description == self.image.description:
            return
        self.image.description = current_description
        self.description_save_timer.start()

    @Slot()
    def save_image_description(self):
        if self.image is None:
            return
        self.image.description = self.description_editor.toPlainText()
        self.image_data_save_requested.emit(self.image)

    # ------------------------------------------------------------------
    # Grouped (multi-image) mode
    # ------------------------------------------------------------------
    def enter_group_mode(self, source_indices: list[QModelIndex],
                         current_index: Optional[QModelIndex]):
        """Switch the pane into the grouped Common/Differences view."""
        self._group_mode = True
        self._group_source_indices = list(source_indices)
        self._group_current_index = current_index
        self.image_index = current_index
        self.tag_input_box.group_mode_active = True
        # Start each grouped session from a clean slate: no filter, scope off
        # (add to all). The filter can then drive the scope as the user clicks.
        self.group_tags_panel.reset_filter()
        self.scope_button.setChecked(False)
        # setChecked only emits `toggled` when the state actually changes, so
        # keep the label in sync explicitly (it should read "All images").
        self._sync_scope_button_text()
        self._refresh_group_panel()
        self._apply_group_visibility()

    def update_group_selection(self, source_indices: list[QModelIndex],
                               current_index: Optional[QModelIndex]):
        """Update the grouped view for a changed multi-image selection.

        When only the current image moved (e.g. arrowing between the same
        selected images), the Common/Differences contents are unchanged, so the
        panel is left untouched — this keeps any selected tag highlighted
        instead of resetting it on every arrow press.
        """
        same_selection = (
            [index.row() for index in source_indices]
            == [index.row() for index in self._group_source_indices])
        self._group_source_indices = list(source_indices)
        self._group_current_index = current_index
        self.image_index = current_index
        if not same_selection:
            self._refresh_group_panel()
        self._update_group_placeholder()

    def exit_group_mode(self):
        """Leave grouped mode and restore the single-image view."""
        if not self._group_mode:
            return
        self._group_mode = False
        self._group_source_indices = []
        self._group_current_index = None
        self.tag_input_box.group_mode_active = False
        self.tag_input_box.reset_group_placeholder()
        # Make sure a held peek doesn't linger after leaving grouped mode.
        self.grid_peek_changed.emit(False)
        self._apply_single_visibility()

    def is_group_mode(self) -> bool:
        return self._group_mode

    def current_group_image_index(self) -> Optional[QModelIndex]:
        """Source-model index of the current image in the grouped view.

        Returns None when not in group mode or no current image is set."""
        if not self._group_mode:
            return None
        return self._group_current_index

    def _current_group_image_tags(self) -> Optional[set[str]]:
        """Tags of the current image in grouped mode, or None if unknown."""
        index = self.current_group_image_index()
        if index is None:
            return None
        image = self.image_list_model.data(index, Qt.ItemDataRole.UserRole)
        return set(image.tags) if image is not None else None

    def _group_images(self) -> list[Image]:
        images = []
        for index in self._group_source_indices:
            image = self.image_list_model.data(index, Qt.ItemDataRole.UserRole)
            if image is not None:
                images.append(image)
        return images

    def selected_tags(self) -> list[str]:
        """The tags selected in the shown tag list (grouped or single image)."""
        if self._group_mode:
            return self.group_tags_panel.tag_list.selected_tags()
        return [index.data(Qt.ItemDataRole.DisplayRole)
                for index in self.image_tags_list.selectedIndexes()]

    def selected_group_tags(self) -> list[str]:
        """Tags selected in the grouped (grid) Image Tags pane.

        Used by the grid cell context menu to add/remove the currently
        selected tag(s) to/from a single image. Returns an empty list when not
        in group mode or nothing is selected.
        """
        if not self._group_mode:
            return []
        return self.group_tags_panel.currently_selected_tags()

    def remember_group_tag_anchor(self):
        """Preserve the grouped pane's cursor position across the next edit.

        Called before a grid cell context-menu add/remove so the Image Tags
        pane keeps its scroll/selection position, matching the panel's own
        Remove/Add actions.
        """
        if self._group_mode:
            self.group_tags_panel.remember_selected_tags_anchor()

    def _shown_tags(self) -> set[str]:
        if self._group_mode:
            return {tag for tag, _ in self.group_tags_panel._rows_full}
        return set(self.image_tag_list_model.stringList())

    def _refresh_group_panel(self, restore_positions: bool = False,
                             renames: dict[str, list[str]] | None = None,
                             saved_order=None):
        self.group_tags_panel.set_images(self._group_images(),
                                         restore_positions=restore_positions,
                                         renames=renames,
                                         saved_order=saved_order)

    def _refresh_group_panel_after_change(self):
        """Refresh the grouped tag list after the image tags changed, saving
        the tag order with the undo entry for this change (or restoring the
        one saved with the entry being undone/redone)."""
        model = self.image_list_model
        panel = self.group_tags_panel
        if model.is_restoring_history:
            saved_order = None
            if model.restoring_history_items is not None:
                restored_item, reverse_item = model.restoring_history_items
                reverse_item.group_order_state = panel.order_state()
                saved_order = restored_item.group_order_state
            self._refresh_group_panel(restore_positions=True,
                                      saved_order=saved_order)
            return
        newest_item = model.newest_history_item
        if newest_item is not None:
            model.newest_history_item = None
            if model.undo_stack and model.undo_stack[-1] is newest_item:
                newest_item.group_order_state = panel.order_state()
        self._refresh_group_panel(renames=model.tag_renames)

    def _update_group_placeholder(self):
        """Set the Add Tag box placeholder for grouped mode. Scope is chosen
        by the persistent "Current image only" toggle, so the placeholder is
        just the plain prompt."""
        self.tag_input_box.set_group_placeholder('Add Tag')

    def _group_add_target_indices(self) -> list[QModelIndex]:
        """Target images for the Add Tag box, honoring the scope toggle
        ("Current image only" = current image, off = all selected)."""
        if self._group_mode:
            if self.tag_input_box.pending_scope_current:
                # Scope toggle on: add to just the current (highlighted) image.
                if self._group_current_index is not None:
                    return [self._group_current_index]
                return []
            return list(self._group_source_indices)
        return self.image_list.get_selected_image_indices()

    @Slot(list)
    def _remove_group_tags_from_all(self, tags: list[str]):
        if not self._group_source_indices or not tags:
            return
        self.image_list_model.remove_tags_from_images(
            tags, list(self._group_source_indices))

    @Slot(list)
    def _add_group_tags_to_all(self, tags: list[str]):
        if not self._group_source_indices or not tags:
            return
        self.image_list_model.add_tags(tags, list(self._group_source_indices))

    @Slot(list)
    def _remove_group_tags_from_current(self, tags: list[str]):
        if self._group_current_index is None or not tags:
            return
        self.image_list_model.remove_tags_from_images(
            tags, [self._group_current_index])

    @Slot(list)
    def _add_group_tags_to_current(self, tags: list[str]):
        if self._group_current_index is None or not tags:
            return
        self.image_list_model.add_tags(tags, [self._group_current_index])

    @Slot(str, str)
    def _rename_group_tag(self, old_tag: str, new_tag: str):
        if not self._group_source_indices or not old_tag or not new_tag:
            return
        self.image_list_model.rename_tag_in_images(
            old_tag, new_tag, list(self._group_source_indices))

    @Slot(list)
    def _on_partial_focus_changed(self, tags: list[str]):
        """Report which selected images contain the focused Differences tag(s).

        Emits the file paths of the matching images so the grid preview can
        mark them. Emits an empty list to clear the marks.
        """
        paths: list[str] = []
        if tags and self._group_mode:
            wanted = set(tags)
            for index in self._group_source_indices:
                image = self.image_list_model.data(index,
                                                   Qt.ItemDataRole.UserRole)
                if image is not None and any(tag in wanted
                                             for tag in image.tags):
                    paths.append(str(image.path))
        self.grid_mark_paths_changed.emit(paths)

    @Slot(int)
    def _cycle_selected_image(self, direction: int):
        """Move the current image to the previous/next selected image.

        Used when the user arrows past the top/bottom of a tag list, so the
        selection is kept intact instead of navigating out of it.
        """
        self.image_list.list_view.switch_between_selected_images(direction)

    def _sync_scope_button_text(self):
        """Show the active scope in the button text.

        A checkable button's color change alone was hard to read, so the label
        names the mode that is currently in effect instead: "Current image
        only" when the scope is the current image, "All images" otherwise."""
        self.scope_button.setText(
            'Current image only' if self.scope_button.isChecked()
            else 'All images')

    @Slot(bool)
    def _on_scope_toggled(self, checked: bool):
        """Persist the Add Tag scope. When checked, the Add Tag box (Enter or
        Ctrl+Enter) targets only the current image; unchecked targets all
        selected. Does not touch the tag-list filter."""
        self.tag_input_box.pending_scope_current = bool(checked)
        self._sync_scope_button_text()

    @Slot(str)
    def _on_filter_mode_changed(self, mode: str):
        """Auto-set the scope toggle from the common/differences filter:
        'differences' -> current image only, 'common' -> all selected. 'all'
        leaves the toggle as the user last set it."""
        if mode == 'differences':
            self.scope_button.setChecked(True)
        elif mode == 'common':
            self.scope_button.setChecked(False)

    def _apply_group_visibility(self):
        """Show the grouped multi-image controls and hide the per-image ones."""
        # Hiding the per-image tag list while it holds keyboard focus makes Qt
        # punt focus to an unrelated widget (the Images filter box), which
        # silently breaks type-to-add-tag and arrow cycling. Remember whether
        # it was focused so we can put focus somewhere sensible afterwards.
        tag_list_had_focus = self.image_tags_list.hasFocus()
        self.image_tags_list.setVisible(False)
        self.token_count_label.setVisible(False)
        self.description_widget.setVisible(False)
        # Grouped controls: the Add Tag scope toggle up top and the clickable
        # common/differences summary in the bottom status row (in place of the
        # per-image token count).
        self.scope_button.setVisible(True)
        self.group_tags_panel.summary_label.setVisible(True)
        self.group_tags_panel.setVisible(True)
        self.tag_input_box.setVisible(True)
        self._update_group_placeholder()
        if tag_list_had_focus:
            # Keep focus on the thumbnails so arrow cycling and typing a new
            # tag keep working after entering the grouped view.
            self.image_list.list_view.setFocus()

    def _apply_single_visibility(self):
        """Show the per-image controls and hide the grouped ones."""
        self.scope_button.setVisible(False)
        self.image_tags_list.setVisible(True)
        self.token_count_label.setVisible(True)
        self.description_widget.setVisible(True)
        self.group_tags_panel.setVisible(False)
        self.group_tags_panel.summary_label.setVisible(False)
        self.tag_input_box.setVisible(True)

    def eventFilter(self, watched, event):
        """Keep arrow cycling and type-to-add-tag working in the grouped view.

        Clicking into the scope selector or the Common/Differences list moves
        keyboard focus there. Without this filter the arrow keys would stop
        cycling between the selected images, and typing a printable character
        would be swallowed by that control instead of starting a new tag in
        the Add Tag box (as it does from the Images pane).
        """
        if self._group_mode and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if (key == GRID_PEEK_KEY
                    and watched is self.group_tags_panel.tag_list
                    and not event.isAutoRepeat()):
                # Hold-to-peek: brighten the current grid cell while Ctrl is
                # held. Observed, not consumed, so Ctrl+key shortcuts still
                # work.
                self.set_grid_peek(True)
            if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                # In the Add Tag box, only cycle when there's no text to edit.
                if watched is self.tag_input_box and self.tag_input_box.text():
                    return super().eventFilter(watched, event)
                direction = -1 if key == Qt.Key.Key_Left else 1
                if self.image_list.list_view.switch_between_selected_images(
                        direction):
                    return True
            elif watched is not self.tag_input_box:
                # Typing a tag character on any grouped-view control jumps to
                # the Add Tag box, mirroring the behavior in the Images pane.
                if _should_redirect_typing_to_tag_input(event):
                    self.raise_()
                    panel = self.group_tags_panel
                    if watched is panel.tag_list:
                        # Match the normal Image Tags list: after the tag is
                        # added, return focus to this list and reselect the tag
                        # that was highlighted when typing began (the model
                        # reset on refresh would otherwise clear the selection).
                        panel.remember_anchor_for_add(watched)
                        self.tag_input_box._return_focus_to_tags_list_after_add \
                            = True
                        self.tag_input_box._return_focus_target = watched
                    self.tag_input_box.setFocus()
                    self.tag_input_box.insert(event.text())
                    return True
        if (self._group_mode and event.type() == QEvent.Type.KeyRelease
                and event.key() == GRID_PEEK_KEY
                and watched is self.group_tags_panel.tag_list
                and not event.isAutoRepeat()):
            self.set_grid_peek(False)
        return super().eventFilter(watched, event)

    def set_grid_peek(self, enabled: bool):
        """Signal the grid to peek (True) or stop peeking (False) the current
        cell. No-op outside grouped mode."""
        if not self._group_mode and enabled:
            return
        self.grid_peek_changed.emit(bool(enabled))

    @Slot()
    def reload_image_tags_if_changed(self, first_changed_index: QModelIndex,
                                     last_changed_index: QModelIndex):
        """
        Reload the tags for the current image if its index is in the range of
        changed indices.
        """
        if self._group_mode:
            # A group edit (or undo/redo) changed tags: recompute the
            # Common/Differences view from the current selection. An undo/redo
            # restores the tag order saved with its history entry, so every tag
            # returns to where it was; a freshly added tag lands at the end
            # and a renamed one keeps its position.
            self._refresh_group_panel_after_change()
            return
        # Not shown in the grouped view, so there's no tag order to save.
        self.image_list_model.newest_history_item = None
        if self.image_index is None:
            return
        if (first_changed_index.row() <= self.image_index.row()
                <= last_changed_index.row()):
            proxy_image_index = self.proxy_image_list_model.mapFromSource(
                self.image_index)
            self.load_image_tags(proxy_image_index)
