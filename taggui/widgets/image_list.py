import shutil
from enum import Enum
from functools import reduce
from operator import or_
from pathlib import Path

from PySide6.QtCore import (QFile, QItemSelection, QItemSelectionModel,
                            QItemSelectionRange, QModelIndex, QRect, QSize, QUrl,
                            Qt, Signal, Slot)
from PySide6.QtGui import (QColor, QContextMenuEvent, QDesktopServices,
                           QFont, QFontMetrics, QIcon, QPainter, QKeyEvent,
                           QPalette, QPen)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDockWidget,
                               QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QListView, QMenu, QMessageBox, QPushButton,
                               QStyle, QStyleOptionViewItem,
                               QStyledItemDelegate, QVBoxLayout, QWidget)
from pyparsing import (CaselessKeyword, CaselessLiteral, Group, OpAssoc,
                       ParseException, QuotedString, Suppress, Word,
                       infix_notation, nums, one_of, printables)

from dialogs.filter_help_dialog import FilterHelpDialog
from models.image_storage import get_storage_backend
from models.proxy_image_list_model import ProxyImageListModel
from utils.image import Image
from utils.settings import DEFAULT_SETTINGS, get_settings
from utils.settings_widgets import SettingsComboBox
from utils.utils import get_confirmation_dialog_reply, pluralize


def replace_filter_wildcards(filter_: str | list) -> str | list:
    """
    Replace escaped wildcard characters to make them compatible with the
    `fnmatch` module.
    """
    if isinstance(filter_, str):
        filter_ = filter_.replace(r'\*', '[*]').replace(r'\?', '[?]')
        return filter_
    replaced_filter = []
    for element in filter_:
        replaced_element = replace_filter_wildcards(element)
        replaced_filter.append(replaced_element)
    return replaced_filter


class FilterLineEdit(QLineEdit):
    def __init__(self):
        super().__init__()
        self.setPlaceholderText('Filter Images')
        self.setToolTip(
            'Filter images by tags, captions, descriptions, names, or paths.\n'
            '  tree          partial match on the tags or file path\n'
            '  tag:tree      match a tag\n'
            '  caption:...   match the joined tags\n'
            '  desc:...      match the description\n'
            '  name:...      match the file name\n'
            '  path:...      match the file path\n'
            '  tags:=n       tag count\n'
            '  chars:>=n     caption length in characters\n'
            '  tokens:<=n    token count\n'
            '  AND / OR / NOT combine filters\n'
            '* and ? are wildcards. Click the ? button for full syntax help.')
        self.setStyleSheet('padding: 8px;')
        self.setClearButtonEnabled(True)
        optionally_quoted_string = (QuotedString(quote_char='"', esc_char='\\')
                                    | QuotedString(quote_char="'",
                                                   esc_char='\\')
                                    | Word(printables, exclude_chars='()'))
        string_filter_keys = ['tag', 'caption', 'desc', 'name', 'path']
        string_filter_expressions = [Group(CaselessLiteral(key) + Suppress(':')
                                           + optionally_quoted_string)
                                     for key in string_filter_keys]
        comparison_operator = one_of('= == != < > <= >=')
        number_filter_keys = ['tags', 'chars', 'tokens']
        number_filter_expressions = [Group(CaselessLiteral(key) + Suppress(':')
                                           + comparison_operator + Word(nums))
                                     for key in number_filter_keys]
        string_filter_expressions = reduce(or_, string_filter_expressions)
        number_filter_expressions = reduce(or_, number_filter_expressions)
        filter_expressions = (string_filter_expressions
                              | number_filter_expressions
                              | optionally_quoted_string)
        self.filter_text_parser = infix_notation(
            filter_expressions,
            # Operator, number of operands, associativity.
            [(CaselessKeyword('NOT'), 1, OpAssoc.RIGHT),
             (CaselessKeyword('AND'), 2, OpAssoc.LEFT),
             (CaselessKeyword('OR'), 2, OpAssoc.LEFT)])

    def parse_filter_text(self) -> list | str | None:
        filter_text = self.text()
        if not filter_text:
            self.setStyleSheet('padding: 8px;')
            return None
        try:
            filter_ = self.filter_text_parser.parse_string(
                filter_text, parse_all=True).as_list()[0]
            filter_ = replace_filter_wildcards(filter_)
            self.setStyleSheet('padding: 8px;')
            return filter_
        except ParseException:
            # Change the background color when the filter text is invalid.
            if self.palette().color(self.backgroundRole()).lightness() < 128:
                # Dark red for dark mode.
                self.setStyleSheet('padding: 8px; background-color: #442222;')
            else:
                # Light red for light mode.
                self.setStyleSheet('padding: 8px; background-color: #ffdddd;')
            return None


class SelectionMode(str, Enum):
    DEFAULT = 'Default'
    TOGGLE = 'Toggle'


class ImageListItemDelegate(QStyledItemDelegate):
    """
    Draw the rows of the image list: the file name on the first line in a
    smaller, dimmer font and the tag preview on the following one or two
    lines. Tags that do not fit are truncated with an ellipsis; the full
    text is shown as a tooltip (see the `ToolTipRole` data in
    `ImageListModel`). Status indicators are drawn on the right side of the
    row: an amber 'n/k' indicator marks an image that has at least one
    duplicate file (n is this copy's position among the k files that share
    one record and the same tags), and a blue dot marks an untagged image.
    """

    # Number of lines the tag preview may occupy (the file name has its own
    # line above it).
    MAX_TAG_LINES = 2
    # Minimum width (in pixels) reserved on the right of the text for the
    # status indicators, so that the text never runs into them.
    STATUS_COLUMN_WIDTH = 16
    UNTAGGED_DOT_DIAMETER = 7
    UNTAGGED_DOT_COLOR = QColor('#4da3ff')
    DUPLICATE_INDICATOR_COLOR = QColor('#f5a623')

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

    def _name_font(self, font: QFont) -> QFont:
        """The file name font: slightly smaller than the tag font."""
        name_font = QFont(font)
        if font.pointSize() > 0:
            name_font.setPointSizeF(font.pointSizeF() * 0.85)
        else:
            name_font.setPixelSize(max(1, int(font.pixelSize() * 0.85)))
        return name_font

    def _icon_size(self, widget: QWidget | None, icon: QIcon) -> QSize:
        if isinstance(widget, QListView):
            return icon.actualSize(widget.iconSize())
        sizes = icon.availableSizes()
        return sizes[0] if sizes else QSize(0, 0)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem,
              index: QModelIndex) -> None:
        widget: QWidget | None = self.parent()
        style = widget.style() if widget is not None else QApplication.style()
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        text_color = (option.palette.color(QPalette.ColorRole.HighlightedText)
                      if selected
                      else option.palette.color(QPalette.ColorRole.Text))

        # Let the style draw the item background (including the selection
        # highlight and focus frame); the file name and tags are drawn below.
        background_option = QStyleOptionViewItem(option)
        background_option.text = ''
        background_option.icon = QIcon()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem,
                          background_option, painter, widget)

        # Draw the thumbnail, then the text to its right.
        text_rect = option.rect
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is not None and not icon.isNull():
            icon_size = self._icon_size(widget, icon)
            icon_rect = QRect(option.rect.left(),
                              option.rect.top() +
                              (option.rect.height() - icon_size.height()) // 2,
                              icon_size.width(), icon_size.height())
            icon.paint(painter, icon_rect, mode=QIcon.Mode.Normal)
            text_rect = option.rect.adjusted(icon_size.width() + 8, 0, 0, 0)

        display_text = index.data(Qt.ItemDataRole.DisplayRole) or ''
        file_name, _, tags_text = display_text.partition('\n')
        image = index.data(Qt.ItemDataRole.UserRole)
        has_untagged = image is not None and not image.tags
        has_duplicate = image is not None and image.duplicate_count > 1
        duplicate_indicator = (f'{image.duplicate_position}'
                               f'/{image.duplicate_count}'
                               if has_duplicate else '')
        name_font = self._name_font(option.font)
        name_metrics = QFontMetrics(name_font)
        tag_metrics = QFontMetrics(option.font)
        # Reserve the status column so the text never overlaps the 'n/k'
        # duplicate indicator and the untagged dot.
        reserved_width = 4
        if has_untagged:
            reserved_width += self.UNTAGGED_DOT_DIAMETER + 4
        if has_duplicate:
            reserved_width += (name_metrics.horizontalAdvance(
                                   duplicate_indicator) + 4)
        text_rect = text_rect.adjusted(
            0, 0, -max(self.STATUS_COLUMN_WIDTH, reserved_width), 0)
        name_height = name_metrics.height()
        tag_height = tag_metrics.height()
        tag_lines = (self._wrap_and_elide(tag_metrics, tags_text,
                                          text_rect.width(), self.MAX_TAG_LINES)
                     if tags_text else [])
        block_height = name_height + len(tag_lines) * tag_height
        y = text_rect.top() + max(0, (text_rect.height() - block_height) // 2)

        draw_flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        name_color = QColor(text_color)
        name_color.setAlphaF(0.6)
        painter.setFont(name_font)
        painter.setPen(name_color)
        painter.drawText(QRect(text_rect.left(), y, text_rect.width(),
                               name_height), draw_flags,
                         name_metrics.elidedText(file_name,
                                                  Qt.TextElideMode.ElideRight,
                                                  text_rect.width()))
        painter.setFont(option.font)
        painter.setPen(text_color)
        line_y = y + name_height
        for line in tag_lines:
            painter.drawText(QRect(text_rect.left(), line_y,
                                   text_rect.width(), tag_height), draw_flags,
                             line)
            line_y += tag_height

        # Draw the status indicators on the right, rightmost to leftmost:
        # a blue dot for an untagged image and the amber 'n/k' duplicate
        # indicator (n = this copy's position among the k duplicate files).
        right_edge = option.rect.right() - 4
        if has_untagged:
            dot_diameter = self.UNTAGGED_DOT_DIAMETER
            dot_x = right_edge - dot_diameter
            dot_y = option.rect.top() + (option.rect.height()
                                         - dot_diameter) // 2
            painter.setPen(QPen(QColor('#ffffff'), 1))
            painter.setBrush(self.UNTAGGED_DOT_COLOR)
            painter.drawEllipse(QRect(dot_x, dot_y, dot_diameter,
                                      dot_diameter))
            right_edge = dot_x - 4
        if has_duplicate:
            indicator_width = name_metrics.horizontalAdvance(
                duplicate_indicator)
            indicator_height = name_metrics.height()
            indicator_y = option.rect.top() + (option.rect.height()
                                               - indicator_height) // 2
            painter.setFont(name_font)
            painter.setPen(self.DUPLICATE_INDICATOR_COLOR)
            painter.drawText(QRect(right_edge - indicator_width,
                                   indicator_y, indicator_width,
                                   indicator_height),
                             Qt.AlignmentFlag.AlignRight
                             | Qt.AlignmentFlag.AlignVCenter,
                             duplicate_indicator)

    def sizeHint(self, option: QStyleOptionViewItem,
                 index: QModelIndex) -> QSize:
        widget: QWidget | None = self.parent()
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        icon_height = 0
        if icon is not None and not icon.isNull():
            icon_height = self._icon_size(widget, icon).height()
        display_text = index.data(Qt.ItemDataRole.DisplayRole) or ''
        name_height = QFontMetrics(self._name_font(option.font)).height()
        tag_height = (QFontMetrics(option.font).height() * self.MAX_TAG_LINES
                      if '\n' in display_text else 0)
        text_height = name_height + tag_height + 8
        width = option.rect.width() if option.rect.width() > 0 else 250
        return QSize(width, max(icon_height, text_height))

    def _wrap_and_elide(self, metrics: QFontMetrics, text: str, width: int,
                        max_lines: int) -> list[str]:
        """
        Wrap `text` into at most `max_lines` lines, each at most `width`
        pixels wide. When the text does not fit, the last line is
        ellipsized.
        """
        words = text.split(' ')
        lines: list[str] = []
        current = ''
        for word in words:
            if not current:
                if metrics.horizontalAdvance(word) <= width:
                    current = word
                else:
                    # A single word longer than the line: elide it.
                    lines.append(metrics.elidedText(
                        word, Qt.TextElideMode.ElideRight, width))
                    return lines[:max_lines]
            elif metrics.horizontalAdvance(f'{current} {word}') <= width:
                current = f'{current} {word}'
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        if len(lines) <= max_lines:
            return lines
        # Keep the first `max_lines - 1` lines and ellipsize the last one
        # from the words that did not fit.
        used_word_count = sum(len(line.split(' '))
                              for line in lines[:max_lines - 1])
        last_line = ''
        for word in words[used_word_count:]:
            candidate = word if not last_line else f'{last_line} {word}'
            if metrics.horizontalAdvance(candidate) <= width:
                last_line = candidate
            else:
                last_line = metrics.elidedText(
                    candidate, Qt.TextElideMode.ElideRight, width)
                break
        lines[max_lines - 1] = last_line
        return lines[:max_lines]


class ImageListView(QListView):
    tags_paste_requested = Signal(list, list)
    directory_reload_requested = Signal()

    def __init__(self, parent, proxy_image_list_model: ProxyImageListModel,
                 tag_separator: str, image_width: int):
        super().__init__(parent)
        self.proxy_image_list_model = proxy_image_list_model
        self.tag_separator = tag_separator
        # The image the context menu was last opened on (set in
        # `contextMenuEvent`). `None` until the first context menu.
        #
        # Store the `Image` object, not a `QModelIndex`: the next model
        # reset (e.g. the directory reload after a deletion) leaves a
        # stored index stale, yet `QModelIndex.isValid()` does not notice
        # that — `data()` on such an index segfaults the app. A Python
        # object reference survives the reset.
        self._context_menu_image = None
        self.setModel(proxy_image_list_model)
        self.setWordWrap(True)
        self.setItemDelegate(ImageListItemDelegate(self))
        # If the actual height of the image is greater than 3 times the width,
        # the image will be scaled down to fit.
        self.setIconSize(QSize(image_width, image_width * 3))

        invert_selection_action = self.addAction('Invert Selection')
        invert_selection_action.setShortcut('Ctrl+I')
        invert_selection_action.triggered.connect(self.invert_selection)
        copy_tags_action = self.addAction('Copy Tags')
        copy_tags_action.setShortcut('Ctrl+C')
        copy_tags_action.triggered.connect(
            self.copy_selected_image_tags)
        paste_tags_action = self.addAction('Paste Tags')
        paste_tags_action.setShortcut('Ctrl+V')
        paste_tags_action.triggered.connect(
            self.paste_tags)
        self.copy_file_names_action = self.addAction('Copy File Name')
        self.copy_file_names_action.setShortcut('Ctrl+Alt+C')
        self.copy_file_names_action.triggered.connect(
            self.copy_selected_image_file_names)
        self.copy_paths_action = self.addAction('Copy Path')
        self.copy_paths_action.setShortcut('Ctrl+Shift+C')
        self.copy_paths_action.triggered.connect(
            self.copy_selected_image_paths)
        self.move_images_action = self.addAction('Move Images to...')
        self.move_images_action.setShortcut('Ctrl+M')
        self.move_images_action.triggered.connect(
            self.move_selected_images)
        self.copy_images_action = self.addAction('Copy Images to...')
        self.copy_images_action.setShortcut('Ctrl+Shift+M')
        self.copy_images_action.triggered.connect(
            self.copy_selected_images)
        self.delete_images_action = self.addAction('Delete Images')
        # Setting the shortcut to `Del` creates a conflict with tag deletion.
        self.delete_images_action.setShortcut('Ctrl+Del')
        self.delete_images_action.triggered.connect(
            self.delete_selected_images)
        self.select_duplicates_action = self.addAction('Select Duplicates')
        self.select_duplicates_action.triggered.connect(
            self.select_duplicate_images)
        self.delete_duplicates_action = self.addAction(
            'Delete Duplicates...')
        self.delete_duplicates_action.triggered.connect(
            self.delete_duplicate_images)
        self.open_image_action = self.addAction('Open Image in Default App')
        self.open_image_action.setShortcut('Ctrl+O')
        self.open_image_action.triggered.connect(self.open_image)

        self.context_menu = QMenu(self)
        self.context_menu.addAction('Select All Images', self.selectAll,
                                    shortcut='Ctrl+A')
        self.context_menu.addAction(invert_selection_action)
        self.context_menu.addSeparator()
        self.context_menu.addAction(copy_tags_action)
        self.context_menu.addAction(paste_tags_action)
        self.context_menu.addAction(self.copy_file_names_action)
        self.context_menu.addAction(self.copy_paths_action)
        self.context_menu.addSeparator()
        self.context_menu.addAction(self.move_images_action)
        self.context_menu.addAction(self.copy_images_action)
        self.context_menu.addAction(self.delete_images_action)
        self.context_menu.addAction(self.select_duplicates_action)
        self.context_menu.addAction(self.delete_duplicates_action)
        self.context_menu.addAction(self.open_image_action)
        self.selectionModel().selectionChanged.connect(
            self.update_context_menu_actions)
        # 'Select Duplicates' also depends on the current image, and moving
        # the current index (arrow keys, right-clicking another row) emits
        # `currentChanged` without changing the selection, so update the
        # enabled state on that too.
        self.selectionModel().currentChanged.connect(
            self.update_context_menu_actions)

    def contextMenuEvent(self, event):
        # Remember the image the menu was opened on so the menu actions
        # can act on it — 'Delete Duplicates...' keeps the copy the menu
        # was opened on. For a right-click that is the image under the
        # cursor; for a keyboard-triggered menu the current image: the
        # event position of a keyboard event is synthetic (`pos()` may be
        # null, per the Qt docs), so `indexAt(event.pos())` would pick an
        # arbitrary row — typically the first — instead of the one the
        # user is working with.
        if event.reason() == QContextMenuEvent.Reason.Mouse:
            context_index = self.indexAt(event.pos())
        else:
            context_index = self.currentIndex()
        self._context_menu_image = (
            context_index.data(Qt.ItemDataRole.UserRole)
            if context_index.isValid() else None)
        # Recompute the enabled states right before showing the menu, in
        # case the current index changed since the last
        # `selectionChanged`/`currentChanged` (e.g. right-clicking the row
        # that is already current emits no signal at all).
        self.update_context_menu_actions()
        self.context_menu.exec_(event.globalPos())

    @Slot()
    def invert_selection(self):
        selected_proxy_rows = {index.row() for index in self.selectedIndexes()}
        all_proxy_rows = set(range(self.proxy_image_list_model.rowCount()))
        unselected_proxy_rows = all_proxy_rows - selected_proxy_rows
        first_unselected_proxy_row = min(unselected_proxy_rows, default=0)
        item_selection = QItemSelection()
        for row in unselected_proxy_rows:
            item_selection.append(
                QItemSelectionRange(self.proxy_image_list_model.index(row, 0)))
        self.setCurrentIndex(self.model().index(first_unselected_proxy_row, 0))
        self.selectionModel().select(
            item_selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def get_selected_images(self) -> list[Image]:
        selected_image_proxy_indices = self.selectedIndexes()
        selected_images = [index.data(Qt.ItemDataRole.UserRole)
                           for index in selected_image_proxy_indices]
        return selected_images

    @Slot()
    def copy_selected_image_tags(self):
        # "Copy Tags" (Ctrl+C) is a window-wide shortcut, so it fires no
        # matter which pane is focused. If the focused widget is a tag list
        # (the Image Tags or All Tags pane) that knows how to copy just its
        # selected tags, let it handle the copy instead of copying every tag
        # of the selected images.
        focus_widget = QApplication.focusWidget()
        while focus_widget is not None:
            copy_selected_tags = getattr(
                focus_widget, 'copy_selected_tags_to_clipboard', None)
            if callable(copy_selected_tags) and copy_selected_tags():
                return
            focus_widget = focus_widget.parentWidget()
        selected_images = self.get_selected_images()
        selected_image_captions = [self.tag_separator.join(image.tags)
                                   for image in selected_images]
        QApplication.clipboard().setText('\n'.join(selected_image_captions))

    def get_selected_image_indices(self) -> list[QModelIndex]:
        selected_image_proxy_indices = self.selectedIndexes()
        selected_image_indices = [
            self.proxy_image_list_model.mapToSource(proxy_index)
            for proxy_index in selected_image_proxy_indices]
        return selected_image_indices

    def switch_between_selected_images(self, direction: int) -> bool:
        """Move the current image to the previous (-1) / next (1) selected
        image, keeping the multi-image selection intact.

        Used by the Image Tags pane when the user arrows past the end of the
        tag list (or presses Left/Right in the grouped view). Returns False
        when there is nothing to cycle (fewer than two selected images, the
        current image is not one of them, or the edge is reached).
        """
        selected_proxy_indices = sorted(self.selectedIndexes(),
                                        key=lambda index: index.row())
        if len(selected_proxy_indices) < 2:
            return False
        selected_rows = [index.row() for index in selected_proxy_indices]
        current_row = self.selectionModel().currentIndex().row()
        if current_row not in selected_rows:
            return False
        current_selected_row = selected_rows.index(current_row)
        next_selected_row = current_selected_row + direction
        if next_selected_row < 0 or next_selected_row >= len(selected_rows):
            return False
        next_proxy_index = selected_proxy_indices[next_selected_row]
        self.selectionModel().setCurrentIndex(
            next_proxy_index, QItemSelectionModel.SelectionFlag.NoUpdate)
        self.scrollTo(next_proxy_index,
                      QAbstractItemView.ScrollHint.EnsureVisible)
        return True

    def move_selection_beyond(self, direction: int) -> bool:
        """Select a single image just outside the current multi-selection.

        Used by Ctrl+Up/Down so the arrow keys can deliberately leave a
        multi-image selection, while plain arrows stay within it. Ctrl+Down
        moves to the image after the last selected one, Ctrl+Up to the image
        before the first selected one. Returns False at the end of the list.
        """
        selected_rows = sorted(index.row() for index in self.selectedIndexes())
        if not selected_rows:
            return False
        target_row = (selected_rows[0] - 1 if direction < 0
                      else selected_rows[-1] + 1)
        if target_row < 0 or target_row >= self.model().rowCount():
            return False
        target_index = self.model().index(target_row, 0)
        self.selectionModel().setCurrentIndex(
            target_index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        self.scrollTo(target_index,
                      QAbstractItemView.ScrollHint.EnsureVisible)
        return True

    def set_current_image_index(self, index: QModelIndex):
        """Make `index` the current image for a programmatic move (filter
        change, go to next image, etc.). In extended selection mode only that
        image is selected, regardless of any held modifier keys (e.g. Ctrl
        still held from the Ctrl+Up/Ctrl+Down shortcut would otherwise add it
        to the selection instead of replacing it)."""
        self._programmatic_current_change = True
        try:
            self.setCurrentIndex(index)
        finally:
            self._programmatic_current_change = False

    def selectionCommand(self, index, event=None):
        if (event is None
                and getattr(self, '_programmatic_current_change', False)
                and self.selectionMode()
                == QAbstractItemView.SelectionMode.ExtendedSelection):
            return QItemSelectionModel.SelectionFlag.ClearAndSelect
        return super().selectionCommand(index, event)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        modifiers = event.modifiers()
        # Hold-to-peek: while the thumbnails have focus in the grouped (grid)
        # view, holding Ctrl brightens the current grid cell (keyboard
        # equivalent of hovering it). Ctrl is observed but NOT consumed, so
        # Ctrl+key shortcuts keep working. Kept in sync with GRID_PEEK_KEY in
        # image_tags_editor.
        editor = getattr(self, 'image_tags_editor', None)
        if (editor is not None and editor.is_group_mode()
                and key == Qt.Key.Key_Control and not event.isAutoRepeat()):
            editor.set_grid_peek(True)
        arrow_keys = (Qt.Key.Key_Left, Qt.Key.Key_Right,
                      Qt.Key.Key_Up, Qt.Key.Key_Down)
        # With a multi-image selection, keep the arrow keys "inside" the
        # selection so it can't be lost by accident. Plain Left/Right/Up/Down
        # cycle the current image among the selected ones (stopping at the
        # ends). Ctrl+Up/Down is the deliberate way to select a new image
        # outside the selection; that is handled by the window-wide Ctrl+Up/
        # Down shortcuts (go_to_previous_image / go_to_next_image), so here we
        # only consume the key to stop the default list navigation (which
        # would toggle the selection) from also running. Other combinations
        # (Shift+arrow to extend, etc.) keep their default behavior.
        if len(self.selectedIndexes()) >= 2 and key in arrow_keys:
            control_pressed = bool(
                modifiers & Qt.KeyboardModifier.ControlModifier)
            shift_pressed = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            if control_pressed and not shift_pressed:
                return
            if modifiers == Qt.KeyboardModifier.NoModifier:
                self.switch_between_selected_images(
                    -1 if key in (Qt.Key.Key_Left, Qt.Key.Key_Up) else 1)
                return
        if self._should_redirect_typing_to_tag_input(event):
            editor = self.image_tags_editor
            editor.raise_()
            editor.tag_input_box.setFocus()
            editor.tag_input_box.insert(event.text())
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        # Release the hold-to-peek started in keyPressEvent (see there).
        editor = getattr(self, 'image_tags_editor', None)
        if (editor is not None and editor.is_group_mode()
                and event.key() == Qt.Key.Key_Control
                and not event.isAutoRepeat()):
            editor.set_grid_peek(False)
        super().keyReleaseEvent(event)

    def _should_redirect_typing_to_tag_input(self, event: QKeyEvent) -> bool:
        """Whether a keystroke in the thumbnails should start a new tag.

        When the "auto-focus Add Tag box when typing" setting is enabled,
        typing a printable character while the thumbnails have focus should
        jump to the Add Tag input in the Image Tags pane instead of doing
        nothing. The Filter Images box is a separate widget, so its typing is
        never affected by this.
        """
        editor = getattr(self, 'image_tags_editor', None)
        # Nowhere to redirect to (e.g. before the editor is wired up, or the
        # box is hidden).
        if editor is None or editor.tag_input_box.isHidden():
            return False
        if not get_settings().value(
                'image_list_auto_focus_add_tag_box',
                defaultValue=DEFAULT_SETTINGS['image_list_auto_focus_add_tag_box'],
                type=bool):
            return False
        # Shared with the Image Tags pane. Imported here to avoid a circular
        # import at module load time (image_tags_editor imports ImageList).
        from widgets.image_tags_editor import (
            _should_redirect_typing_to_tag_input as typing_redirects)
        return typing_redirects(event)

    @Slot()
    def paste_tags(self):
        selected_image_count = len(self.selectedIndexes())
        if selected_image_count > 1:
            reply = get_confirmation_dialog_reply(
                title='Paste Tags',
                question=f'Paste tags to {selected_image_count} selected '
                         f'images?')
            if reply != QMessageBox.StandardButton.Yes:
                return
        tags = QApplication.clipboard().text().split(self.tag_separator)
        selected_image_indices = self.get_selected_image_indices()
        self.tags_paste_requested.emit(tags, selected_image_indices)

    @Slot()
    def copy_selected_image_file_names(self):
        selected_images = self.get_selected_images()
        selected_image_file_names = [image.path.name
                                     for image in selected_images]
        QApplication.clipboard().setText('\n'.join(selected_image_file_names))

    @Slot()
    def copy_selected_image_paths(self):
        selected_images = self.get_selected_images()
        selected_image_paths = [str(image.path) for image in selected_images]
        QApplication.clipboard().setText('\n'.join(selected_image_paths))

    @Slot()
    def move_selected_images(self):
        selected_images = self.get_selected_images()
        selected_image_count = len(selected_images)
        caption = (f'Select directory to move {selected_image_count} selected '
                   f'{pluralize("Image", selected_image_count)} and '
                   f'{pluralize("caption", selected_image_count)} to')
        settings = get_settings()
        move_directory_path = QFileDialog.getExistingDirectory(
            parent=self, caption=caption,
            dir=settings.value('directory_path', type=str))
        if not move_directory_path:
            return
        move_directory_path = Path(move_directory_path)
        for image in selected_images:
            try:
                image.path.replace(move_directory_path / image.path.name)
                caption_file_path = image.path.with_suffix('.txt')
                if caption_file_path.exists():
                    caption_file_path.replace(
                        move_directory_path / caption_file_path.name)
            except OSError:
                QMessageBox.critical(self, 'Error',
                                     f'Failed to move {image.path} to '
                                     f'{move_directory_path}.')
        self.directory_reload_requested.emit()

    @Slot()
    def copy_selected_images(self):
        selected_images = self.get_selected_images()
        selected_image_count = len(selected_images)
        caption = (f'Select directory to copy {selected_image_count} selected '
                   f'{pluralize("Image", selected_image_count)} and '
                   f'{pluralize("caption", selected_image_count)} to')
        settings = get_settings()
        copy_directory_path = QFileDialog.getExistingDirectory(
            parent=self, caption=caption,
            dir=settings.value('directory_path', type=str))
        if not copy_directory_path:
            return
        copy_directory_path = Path(copy_directory_path)
        for image in selected_images:
            try:
                shutil.copy(image.path, copy_directory_path)
                caption_file_path = image.path.with_suffix('.txt')
                if caption_file_path.exists():
                    shutil.copy(caption_file_path, copy_directory_path)
            except OSError:
                QMessageBox.critical(self, 'Error',
                                     f'Failed to copy {image.path} to '
                                     f'{copy_directory_path}.')

    @Slot()
    def delete_selected_images(self):
        selected_images = self.get_selected_images()
        selected_image_count = len(selected_images)
        title = f'Delete {pluralize("Image", selected_image_count)}'
        question = (f'Delete {selected_image_count} selected '
                    f'{pluralize("image", selected_image_count)} and '
                    f'{"its" if selected_image_count == 1 else "their"} '
                    f'{pluralize("caption", selected_image_count)}?')
        reply = get_confirmation_dialog_reply(title, question)
        if reply != QMessageBox.StandardButton.Yes:
            return
        for image in selected_images:
            image_file = QFile(image.path)
            if not image_file.moveToTrash():
                QMessageBox.critical(self, 'Error',
                                     f'Failed to delete {image.path}.')
            caption_file_path = image.path.with_suffix('.txt')
            caption_file = QFile(caption_file_path)
            if caption_file.exists():
                if not caption_file.moveToTrash():
                    QMessageBox.critical(self, 'Error',
                                         f'Failed to delete '
                                         f'{caption_file_path}.')
        self.directory_reload_requested.emit()

    @Slot()
    def select_duplicate_images(self):
        # Select every visible row of an image whose content is the same as
        # the current image's (the copies share one record and the same tags).
        current_index = self.currentIndex()
        if not current_index.isValid():
            return
        current_image = current_index.data(Qt.ItemDataRole.UserRole)
        if (current_image is None or not current_image.content_hash
                or current_image.duplicate_count < 2):
            return
        content_hash = current_image.content_hash
        item_selection = QItemSelection()
        found_count = 0
        for row in range(self.proxy_image_list_model.rowCount()):
            index = self.proxy_image_list_model.index(row, 0)
            image = index.data(Qt.ItemDataRole.UserRole)
            if image is not None and image.content_hash == content_hash:
                if found_count == 0:
                    self.setCurrentIndex(index)
                item_selection.append(QItemSelectionRange(index))
                found_count += 1
        self.selectionModel().select(
            item_selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    @Slot()
    def delete_duplicate_images(self):
        # Keep the image the menu was opened on and move every other loaded
        # copy of the same file content — and any `.txt` caption file next
        # to it — to the trash. The selection is not involved: no
        # pre-selection is needed, and copies hidden by a filter are
        # deleted too (they are looked up among the loaded images, not the
        # visible ones).
        keeper = self._context_menu_image
        if keeper is None:
            current_index = self.currentIndex()
            if current_index.isValid():
                keeper = current_index.data(Qt.ItemDataRole.UserRole)
        if (keeper is None or not keeper.content_hash
                or keeper.duplicate_count < 2):
            return
        content_hash = keeper.content_hash
        source_model = self.proxy_image_list_model.sourceModel()
        deleted_paths = []
        for row in range(source_model.rowCount()):
            image = source_model.index(row, 0).data(Qt.ItemDataRole.UserRole)
            if (image is not None and image is not keeper
                    and image.content_hash == content_hash):
                deleted_paths.append(image.path)
        if not deleted_paths:
            return
        kept_paths = [keeper.path]
        title = 'Delete Duplicate Files'
        question = (f'Delete {len(deleted_paths)} duplicate '
                    f'{pluralize("file", len(deleted_paths))} '
                    f'{"and its" if len(deleted_paths) == 1 else "and their"} '
                    f'caption file(s)?\n\n'
                    f'Kept:\n'
                    f'{"\n".join(f"  {path}" for path in kept_paths)}\n\n'
                    f'Deleted:\n'
                    f'{"\n".join(f"  {path}" for path in deleted_paths)}')
        reply = get_confirmation_dialog_reply(title, question)
        if reply != QMessageBox.StandardButton.Yes:
            return
        for image_path in deleted_paths:
            image_file = QFile(image_path)
            if not image_file.moveToTrash():
                QMessageBox.critical(self, 'Error',
                                     f'Failed to delete {image_path}.')
            caption_file_path = image_path.with_suffix('.txt')
            caption_file = QFile(caption_file_path)
            if caption_file.exists():
                if not caption_file.moveToTrash():
                    QMessageBox.critical(self, 'Error',
                                         f'Failed to delete '
                                         f'{caption_file_path}.')
        self.directory_reload_requested.emit()

    @Slot()
    def open_image(self):
        selected_images = self.get_selected_images()
        image_path = selected_images[0].path
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(image_path)))

    @Slot()
    def update_context_menu_actions(self):
        selected_image_count = len(self.selectedIndexes())
        copy_file_names_action_name = (
            f'Copy File {pluralize("Name", selected_image_count)}')
        copy_paths_action_name = (f'Copy '
                                  f'{pluralize("Path", selected_image_count)}')
        move_images_action_name = (
            f'Move {pluralize("Image", selected_image_count)} to...')
        copy_images_action_name = (
            f'Copy {pluralize("Image", selected_image_count)} to...')
        delete_images_action_name = (
            f'Delete {pluralize("Image", selected_image_count)}')
        self.copy_file_names_action.setText(copy_file_names_action_name)
        self.copy_paths_action.setText(copy_paths_action_name)
        self.move_images_action.setText(move_images_action_name)
        self.copy_images_action.setText(copy_images_action_name)
        self.delete_images_action.setText(delete_images_action_name)
        self.open_image_action.setVisible(selected_image_count == 1)
        current_index = self.currentIndex()
        current_image = (current_index.data(Qt.ItemDataRole.UserRole)
                         if current_index.isValid() else None)
        self.select_duplicates_action.setEnabled(
            bool(current_image is not None
                 and current_image.content_hash
                 and current_image.duplicate_count > 1))
        # Enable "Delete Duplicates..." when the menu was opened on a
        # duplicate image: it keeps that copy and deletes the other loaded
        # copies of the same file content, so the selection is not involved.
        context_image = self._context_menu_image
        if context_image is None:
            context_image = current_image
        self.delete_duplicates_action.setEnabled(
            bool(context_image is not None
                 and context_image.content_hash
                 and context_image.duplicate_count > 1))


class ImageList(QDockWidget):
    def __init__(self, proxy_image_list_model: ProxyImageListModel,
                 tag_separator: str, image_width: int):
        super().__init__()
        self.proxy_image_list_model = proxy_image_list_model
        # Each `QDockWidget` needs a unique object name for saving its state.
        self.setObjectName('image_list')
        self.setWindowTitle('Images')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)

        self.filter_line_edit = FilterLineEdit()
        self.filter_help_button = QPushButton('?')
        self.filter_help_button.setFixedWidth(34)
        self.filter_help_button.setToolTip('Filter syntax help')
        self.filter_help_button.clicked.connect(self.show_filter_help_dialog)
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(self.filter_line_edit, stretch=1)
        filter_layout.addWidget(self.filter_help_button)
        selection_mode_layout = QHBoxLayout()
        selection_mode_label = QLabel('Selection mode')
        self.selection_mode_combo_box = SettingsComboBox(
            key='image_list_selection_mode')
        self.selection_mode_combo_box.addItems(list(SelectionMode))
        selection_mode_layout.addWidget(selection_mode_label)
        selection_mode_layout.addWidget(self.selection_mode_combo_box,
                                        stretch=1)
        self.filter_duplicates_button = QPushButton('Duplicates')
        self.filter_duplicates_button.setCheckable(True)
        self.filter_duplicates_button.setToolTip(
            'Show only images that have at least one other loaded file\n'
            'with identical content.')
        # Duplicates can only be detected in database storage mode (by
        # content hash), so the button starts hidden in text-file mode and
        # the main window keeps it in sync when the backend changes.
        self.filter_duplicates_button.setVisible(
            get_storage_backend() == 'sqlite')
        selection_mode_layout.addWidget(self.filter_duplicates_button)
        self.list_view = ImageListView(self, proxy_image_list_model,
                                       tag_separator, image_width)
        self.image_index_label = QLabel()
        # A container widget is required to use a layout with a `QDockWidget`.
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addLayout(filter_layout)
        layout.addLayout(selection_mode_layout)
        layout.addWidget(self.list_view)
        layout.addWidget(self.image_index_label)
        self.setWidget(container)

        self.selection_mode_combo_box.currentTextChanged.connect(
            self.set_selection_mode)
        self.set_selection_mode(self.selection_mode_combo_box.currentText())

    def set_selection_mode(self, selection_mode: str):
        if selection_mode == SelectionMode.DEFAULT:
            self.list_view.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection)
        elif selection_mode == SelectionMode.TOGGLE:
            self.list_view.setSelectionMode(
                QAbstractItemView.SelectionMode.MultiSelection)

    @Slot()
    def show_filter_help_dialog(self):
        dialog = FilterHelpDialog(self)
        dialog.exec()

    @Slot()
    def update_image_index_label(self, proxy_image_index: QModelIndex):
        image_count = self.proxy_image_list_model.rowCount()
        unfiltered_image_count = (self.proxy_image_list_model.sourceModel()
                                  .rowCount())
        if proxy_image_index.isValid():
            label_text = f'Image {proxy_image_index.row() + 1} / {image_count}'
        else:
            # No image is selected (for example, while a directory is
            # loading). Use a dash rather than '0' so it does not read as
            # 'first image selected'.
            label_text = f'– / {image_count}'
        if image_count != unfiltered_image_count:
            label_text += f' ({unfiltered_image_count} total)'
        self.image_index_label.setText(label_text)

    @Slot()
    def go_to_previous_image(self):
        # With a multi-image selection, Ctrl+Up is the deliberate way to leave
        # it: select the single image just before the first selected one (see
        # move_selection_beyond). These slots are the sole handlers of the
        # window-wide Ctrl+Up/Down shortcuts (registered in MainWindow); the
        # individual panes only consume the key event, so the navigation
        # happens exactly once.
        if len(self.list_view.selectedIndexes()) >= 2:
            self.list_view.move_selection_beyond(-1)
            return
        if self.list_view.selectionModel().currentIndex().row() == 0:
            return
        self.list_view.clearSelection()
        previous_image_index = self.proxy_image_list_model.index(
            self.list_view.selectionModel().currentIndex().row() - 1, 0)
        self.list_view.set_current_image_index(previous_image_index)

    @Slot()
    def go_to_next_image(self):
        # See go_to_previous_image: with a multi-image selection, Ctrl+Down
        # selects the single image just after the last selected one.
        if len(self.list_view.selectedIndexes()) >= 2:
            self.list_view.move_selection_beyond(1)
            return
        if (self.list_view.selectionModel().currentIndex().row()
                == self.proxy_image_list_model.rowCount() - 1):
            return
        self.list_view.clearSelection()
        next_image_index = self.proxy_image_list_model.index(
            self.list_view.selectionModel().currentIndex().row() + 1, 0)
        self.list_view.set_current_image_index(next_image_index)

    @Slot()
    def jump_to_first_untagged_image(self):
        """
        Select the first image that has no tags, or the last image if all
        images are tagged.
        """
        proxy_image_index = None
        for proxy_image_index in range(self.proxy_image_list_model.rowCount()):
            image: Image = self.proxy_image_list_model.data(
                self.proxy_image_list_model.index(proxy_image_index, 0),
                Qt.ItemDataRole.UserRole)
            if not image.tags:
                break
        if proxy_image_index is None:
            return
        self.list_view.clearSelection()
        self.list_view.set_current_image_index(
            self.proxy_image_list_model.index(proxy_image_index, 0))

    def get_selected_image_indices(self) -> list[QModelIndex]:
        return self.list_view.get_selected_image_indices()
