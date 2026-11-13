from pathlib import Path

from PySide6.QtCore import (QEventLoop, QItemSelectionModel, QKeyCombination,
                            QModelIndex, QPoint, QUrl, Qt, Slot, QTimer)
from PySide6.QtGui import (QAction, QCloseEvent, QDesktopServices, QIcon, QFont,
                           QKeySequence, QPixmap, QShortcut)
from PySide6.QtWidgets import (QApplication, QCheckBox, QFileDialog,
                               QLabel, QMainWindow, QMenu, QMessageBox,
                               QStackedWidget, QVBoxLayout, QWidget)
from transformers import AutoTokenizer

from dialogs.batch_reorder_tags_dialog import BatchReorderTagsDialog
from dialogs.find_and_replace_dialog import FindAndReplaceDialog
from dialogs.settings_dialog import SettingsDialog
from models.image_list_model import ImageListModel
from models.image_storage import get_image_storage
from models.image_tag_list_model import ImageTagListModel
from models.proxy_image_list_model import ProxyImageListModel
from models.tag_counter_model import TagCounterModel
from utils.big_widgets import BigPushButton
from utils.image import Image
from utils.key_press_forwarder import KeyPressForwarder
from utils.settings import DEFAULT_SETTINGS, get_settings, get_tag_separator
from utils.shortcut_remover import ShortcutRemover
from utils.utils import (get_resource_path, pluralize,
                         restart_application)
from widgets.all_tags_editor import AllTagsEditor
from widgets.auto_captioner import AutoCaptioner
from widgets.image_list import ImageList
from widgets.image_tags_editor import ImageTagsEditor
from widgets.image_viewer import ImageViewer

ICON_PATH = Path('images/icon.ico')
GITHUB_REPOSITORY_URL = 'https://github.com/zapheroc/taggui-describe'
TOKENIZER_DIRECTORY_PATH = Path('clip-vit-base-patch32')


class MainWindow(QMainWindow):
    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.settings = get_settings()
        # The path of the currently loaded directory. This is set later when a
        # directory is loaded.
        self.directory_path = None
        # The in-flight background text→database import (if any): whether one
        # is running, the settings dialog that requested it, and the backend
        # to switch to when it finishes.
        self._storage_import_in_progress = False
        self._storage_import_dialog = None
        self._storage_import_mode = None
        image_list_image_width = self.settings.value(
            'image_list_image_width',
            defaultValue=DEFAULT_SETTINGS['image_list_image_width'], type=int)
        tag_separator = get_tag_separator()
        self.image_list_model = ImageListModel(image_list_image_width,
                                               tag_separator)
        tokenizer = AutoTokenizer.from_pretrained(
            get_resource_path(TOKENIZER_DIRECTORY_PATH))
        self.proxy_image_list_model = ProxyImageListModel(
            self.image_list_model, tokenizer, tag_separator)
        self.image_list_model.proxy_image_list_model = (
            self.proxy_image_list_model)
        self.tag_counter_model = TagCounterModel()
        self.image_tag_list_model = ImageTagListModel()

        self.setWindowIcon(QIcon(QPixmap(get_resource_path(ICON_PATH))))
        # Not setting this results in some ugly colors.
        self.setPalette(self.app.style().standardPalette())
        # The font size must be set before creating the widgets to ensure that
        # everything has the correct font size.
        self.set_font_size()
        self.image_viewer = ImageViewer(self.proxy_image_list_model)
        self.create_central_widget()
        self.image_list = ImageList(self.proxy_image_list_model,
                                    tag_separator, image_list_image_width)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self.image_list)
        self.image_tags_editor = ImageTagsEditor(
            self.proxy_image_list_model, self.tag_counter_model,
            self.image_tag_list_model, self.image_list, tokenizer,
            tag_separator)
        # The thumbnails need the editor to redirect typing to the Add Tag
        # box (see ImageListView.keyPressEvent and its
        # _should_redirect_typing_to_tag_input) and to support hold-Ctrl grid
        # peeking. Without this wiring both silently do nothing.
        self.image_list.list_view.image_tags_editor = self.image_tags_editor
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self.image_tags_editor)
        self.all_tags_editor = AllTagsEditor(self.tag_counter_model)
        self.tag_counter_model.all_tags_list = (self.all_tags_editor
                                                .all_tags_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self.all_tags_editor)
        self.auto_captioner = AutoCaptioner(self.image_list_model,
                                            self.image_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea,
                           self.auto_captioner)
        self.tabifyDockWidget(self.all_tags_editor, self.auto_captioner)
        self.all_tags_editor.raise_()
        # Set default widths for the dock widgets.
        # Temporarily set a size for the window so that the dock widgets can be
        # expanded to their default widths. If the window geometry was
        # previously saved, it will be restored later.
        self.resize(image_list_image_width * 8,
                    int(image_list_image_width * 4.5))
        self.resizeDocks([self.image_list, self.image_tags_editor,
                          self.all_tags_editor],
                         [int(image_list_image_width * 2.5)] * 3,
                         Qt.Orientation.Horizontal)
        # Disable some widgets until a directory is loaded.
        self.image_tags_editor.tag_input_box.setDisabled(True)
        self.auto_captioner.start_cancel_button.setDisabled(True)
        self.reload_directory_action = QAction('Reload Directory', parent=self)
        self.reload_directory_action.setDisabled(True)
        self.undo_action = QAction('Undo', parent=self)
        self.redo_action = QAction('Redo', parent=self)
        self.toggle_image_list_action = QAction('Images', parent=self)
        self.toggle_image_tags_editor_action = QAction('Image Tags',
                                                       parent=self)
        self.toggle_all_tags_editor_action = QAction('All Tags', parent=self)
        self.toggle_auto_captioner_action = QAction('Auto-Captioner',
                                                    parent=self)
        self.create_menus()

        self.image_list_selection_model = (self.image_list.list_view
                                           .selectionModel())
        self.image_list_model.image_list_selection_model = (
            self.image_list_selection_model)
        self.connect_image_list_signals()
        self.connect_image_tags_editor_signals()
        self.connect_all_tags_editor_signals()
        self.connect_auto_captioner_signals()
        # Forward any unhandled image changing key presses to the image list.
        key_press_forwarder = KeyPressForwarder(
            parent=self, target=self.image_list.list_view,
            keys_to_forward=(Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_PageUp,
                             Qt.Key.Key_PageDown, Qt.Key.Key_Home,
                             Qt.Key.Key_End))
        self.installEventFilter(key_press_forwarder)
        # Remove the Ctrl+Z shortcut from text input boxes to prevent it from
        # conflicting with the undo action.
        ctrl_z = QKeyCombination(Qt.KeyboardModifier.ControlModifier,
                                 key=Qt.Key.Key_Z)
        ctrl_y = QKeyCombination(Qt.KeyboardModifier.ControlModifier,
                                 key=Qt.Key.Key_Y)
        shortcut_remover = ShortcutRemover(parent=self,
                                           shortcuts=(ctrl_z, ctrl_y))
        self.image_list.filter_line_edit.installEventFilter(shortcut_remover)
        self.image_tags_editor.tag_input_box.installEventFilter(
            shortcut_remover)
        self.all_tags_editor.filter_line_edit.installEventFilter(
            shortcut_remover)
        # Set keyboard shortcuts.
        focus_filter_images_box_shortcut = QShortcut(
            QKeySequence('Alt+F'), self)
        focus_filter_images_box_shortcut.activated.connect(
            self.image_list.raise_)
        focus_filter_images_box_shortcut.activated.connect(
            self.image_list.filter_line_edit.setFocus)
        focus_add_tag_box_shortcut = QShortcut(QKeySequence('Alt+A'), self)
        focus_add_tag_box_shortcut.activated.connect(
            self.image_tags_editor.raise_)
        focus_add_tag_box_shortcut.activated.connect(
            self.image_tags_editor.tag_input_box.setFocus)
        # In the grouped (grid) view the single-image tag list is hidden, so
        # use focus_tags_list to reach the right list in either mode.
        focus_image_tags_list_shortcut = QShortcut(QKeySequence('Alt+I'), self)
        focus_image_tags_list_shortcut.activated.connect(
            self.image_tags_editor.focus_tags_list)
        focus_search_tags_box_shortcut = QShortcut(QKeySequence('Alt+S'), self)
        focus_search_tags_box_shortcut.activated.connect(
            self.all_tags_editor.raise_)
        focus_search_tags_box_shortcut.activated.connect(
            self.all_tags_editor.filter_line_edit.setFocus)
        focus_caption_button_shortcut = QShortcut(QKeySequence('Alt+C'), self)
        focus_caption_button_shortcut.activated.connect(
            self.auto_captioner.raise_)
        focus_caption_button_shortcut.activated.connect(
            self.auto_captioner.start_cancel_button.setFocus)
        go_to_previous_image_shortcut = QShortcut(QKeySequence('Ctrl+Up'),
                                                  self)
        go_to_previous_image_shortcut.activated.connect(
            self.image_list.go_to_previous_image)
        go_to_next_image_shortcut = QShortcut(QKeySequence('Ctrl+Down'), self)
        go_to_next_image_shortcut.activated.connect(
            self.image_list.go_to_next_image)
        jump_to_first_untagged_image_shortcut = QShortcut(
            QKeySequence('Ctrl+J'), self)
        jump_to_first_untagged_image_shortcut.activated.connect(
            self.image_list.jump_to_first_untagged_image)
        # Zoom and pan shortcuts for the preview (single image or grid).
        self.reset_image_zoom_shortcut = QShortcut(
            QKeySequence('Ctrl+0'), self)
        self.reset_image_zoom_shortcut.activated.connect(
            self.image_viewer.reset_view)
        # `+` is the shifted `=` key, so register both `Ctrl++` and `Ctrl+=`
        # for zoom in; otherwise the user must hold Shift to zoom in.
        self.zoom_in_shortcut = QShortcut(QKeySequence('Ctrl++'), self)
        self.zoom_in_shortcut.setKeys([QKeySequence('Ctrl++'),
                                       QKeySequence('Ctrl+=')])
        self.zoom_in_shortcut.activated.connect(self.image_viewer.zoom_in)
        self.zoom_out_shortcut = QShortcut(QKeySequence('Ctrl+-'), self)
        self.zoom_out_shortcut.activated.connect(self.image_viewer.zoom_out)
        self.pan_image_left_shortcut = QShortcut(
            QKeySequence('Alt+Left'), self)
        self.pan_image_left_shortcut.activated.connect(
            self.image_viewer.pan_left)
        self.pan_image_right_shortcut = QShortcut(
            QKeySequence('Alt+Right'), self)
        self.pan_image_right_shortcut.activated.connect(
            self.image_viewer.pan_right)
        self.pan_image_up_shortcut = QShortcut(QKeySequence('Alt+Up'), self)
        self.pan_image_up_shortcut.activated.connect(
            self.image_viewer.pan_up)
        self.pan_image_down_shortcut = QShortcut(
            QKeySequence('Alt+Down'), self)
        self.pan_image_down_shortcut.activated.connect(
            self.image_viewer.pan_down)
        self.restore()
        # Show the window once the first image has been rendered.  Fires via
        # image_viewer.first_image_rendered after the initial pixmap.scaled()
        # completes.  _show_window() is idempotent, so it is safe for both the
        # signal and the timer below to call it.
        self.image_viewer.first_image_rendered.connect(self._show_window)
        # Safety net: guarantees the window becomes visible even if no image is
        # ever rendered (e.g. an empty directory, a load error, or a path that
        # never emits first_image_rendered).  5s is a generous upper bound that
        # a normal first render finishes well within.
        QTimer.singleShot(5000, self._show_window)
        self.image_tags_editor.tag_input_box.setFocus()

    def closeEvent(self, event: QCloseEvent):
        """Close any captioning subprocess and save the window geometry and state before closing."""
        self.auto_captioner.model_manager.shutdown()
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('window_state', self.saveState())
        super().closeEvent(event)

    def set_font_size(self):
        font = self.app.font()
        font_size = self.settings.value(
            'font_size', defaultValue=DEFAULT_SETTINGS['font_size'], type=int)
        font.setPointSize(font_size)
        self.app.setFont(font)

    def create_central_widget(self):
        central_widget = QStackedWidget()
        # Put the button inside a widget so that it will not fill up the entire
        # space.
        load_directory_widget = QWidget()
        load_directory_button = BigPushButton('Load Directory...')
        load_directory_button.clicked.connect(self.select_and_load_directory)
        QVBoxLayout(load_directory_widget).addWidget(
            load_directory_button, alignment=Qt.AlignmentFlag.AlignCenter)
        central_widget.addWidget(load_directory_widget)
        central_widget.addWidget(self.image_viewer)
        self.setCentralWidget(central_widget)

    def load_directory(self, path: Path, select_index: int = 0,
                       save_path_to_settings: bool = False):
        self.directory_path = path.resolve()
        if save_path_to_settings:
            self.settings.setValue('directory_path', str(self.directory_path))
        self.setWindowTitle(path.name)
        # The index to select when the directory has finished loading.
        self._pending_select_index = select_index
        self.image_list.filter_line_edit.clear()
        self.all_tags_editor.filter_line_edit.clear()
        # Clear the current index first to make sure that the `currentChanged`
        # signal is emitted even if the image at the index is already selected.
        self.image_list_selection_model.clearCurrentIndex()
        self.centralWidget().setCurrentWidget(self.image_viewer)
        self.reload_directory_action.setDisabled(False)
        self.image_tags_editor.tag_input_box.setDisabled(False)
        self.auto_captioner.start_cancel_button.setDisabled(False)
        # The images are loaded on background threads. When the load is
        # complete, `directory_loaded` is emitted and `on_directory_loaded`
        # selects the requested image.
        self.image_list_model.load_directory(path)

    @Slot()
    def on_directory_loaded(self):
        """Select the requested image once the directory has loaded."""
        row_count = self.proxy_image_list_model.rowCount()
        select_index = self._pending_select_index
        # If the selected image index is out of bounds due to images being
        # deleted, select the last image.
        if select_index >= row_count:
            select_index = row_count - 1
        if row_count > 0:
            self.image_list.list_view.set_current_image_index(
                self.proxy_image_list_model.index(select_index, 0))
        # If no images are present there will be no currentChanged signal and
        # therefore no first_image_rendered signal, so show the window now.
        if not self.image_list_model.images:
            self._show_window()

    @Slot()
    def select_and_load_directory(self):
        initial_directory = (str(self.directory_path)
                             if self.directory_path else '')
        load_directory_path = QFileDialog.getExistingDirectory(
            parent=self, caption='Select directory to load images from',
            dir=initial_directory)
        if not load_directory_path:
            return
        self.load_directory(Path(load_directory_path),
                            save_path_to_settings=True)

    @Slot()
    def reload_directory(self):
        # Save the filter text and the index of the selected image to restore
        # them after reloading the directory.
        filter_text = self.image_list.filter_line_edit.text()
        select_index_key = ('image_index'
                            if self.proxy_image_list_model.filter is None
                            else 'filtered_image_index')
        select_index = self.settings.value(select_index_key, type=int) or 0
        # Selecting the image is deferred to `on_directory_loaded` because
        # the images are loaded on background threads.
        self.load_directory(self.directory_path, select_index=select_index)
        self.image_list.filter_line_edit.setText(filter_text)

    @Slot()
    def show_settings_dialog(self):
        settings_dialog = SettingsDialog(parent=self)
        settings_dialog.variant_grid_overlay_settings_changed.connect(
            self.image_viewer.refresh_grid_overlay_style)
        settings_dialog.storage_backend_change_requested.connect(
            lambda new_mode: self.handle_storage_backend_change(
                settings_dialog, new_mode))
        settings_dialog.database_changed_requested.connect(
            self.on_database_changed)
        settings_dialog.exec()

    @Slot(object, str)
    def handle_storage_backend_change(self, settings_dialog, new_mode: str):
        """
        Switch the tags/descriptions storage mode (from the Settings dialog).
        Switching moves no data unless the user checks the override box:
        then the tags and descriptions of the loaded images are written to
        caption files (when switching to text files) or imported into the
        database (when switching to the database).
        """
        old_mode = self.settings.value(
            'storage_backend',
            defaultValue=DEFAULT_SETTINGS['storage_backend'], type=str)
        if new_mode == old_mode:
            return
        if self._storage_import_in_progress:
            # A text→database import from a previous switch is still running
            # in the background. Switching now would race it, so keep the
            # current mode and ask the user to try again in a moment.
            settings_dialog.revert_storage_backend(old_mode)
            QMessageBox.information(self, 'Storage',
                                    'An import into the database is still '
                                    'running. Please try again in a moment.')
            return
        if not self.image_list_model.images:
            # There are no loaded images, so there is no data to move.
            self._apply_storage_backend(new_mode)
            return
        is_to_sqlite = new_mode == 'sqlite'
        image_count = len(self.image_list_model.images)
        message_box = QMessageBox()
        message_box.setIcon(QMessageBox.Icon.Question)
        if is_to_sqlite:
            message_box.setWindowTitle('Switch to Database Storage')
            message_box.setText('TagGUI will store tags and descriptions in '
                                'the SQLite database from now on.')
            override_text = 'Migrate and override database values'
            override_note = (
                f'If enabled, the tags and descriptions of the {image_count} '
                'currently loaded images will be imported from their caption '
                'files into the database, overwriting existing database '
                'records for those images. Images without tags or a '
                'description are skipped.')
        else:
            message_box.setWindowTitle('Switch to Text File Storage')
            message_box.setText('TagGUI will store tags and descriptions in '
                                'caption files (.txt) next to the images '
                                'from now on.')
            override_text = 'Migrate and override text files'
            override_note = (
                f'If enabled, the tags and descriptions of the {image_count} '
                'currently loaded images will be written to caption files, '
                'overwriting existing caption files. Images without tags or '
                'a description are skipped.')
        # The check box must be a child of the message box.
        override_check_box = QCheckBox(override_text, message_box)
        message_box.setCheckBox(override_check_box)
        # Nest the check box together with a description of what it does, so
        # that the description is clearly associated with the option.
        # (QMessageBox places the check box at row 2, column 2 of its grid
        # layout.)
        grid = message_box.layout()
        grid.removeWidget(override_check_box)
        override_group = QWidget(message_box)
        group_layout = QVBoxLayout(override_group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.addWidget(override_check_box)
        override_note_label = QLabel(override_note, override_group)
        override_note_label.setWordWrap(True)
        override_note_label.setIndent(20)
        # Use the application font (i.e. the configured font size) three
        # point sizes smaller, clamped to a minimum of 1 so that the note
        # never ends up with a zero or negative size.
        override_note_font = QFont(QApplication.instance().font())
        override_note_font.setPointSize(
            max(1, override_note_font.pointSize() - 3))
        override_note_font.setItalic(True)
        override_note_label.setFont(override_note_font)
        group_layout.addWidget(override_note_label)
        grid.addWidget(override_group, 2, 2)
        message_box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        # The default action moves no data.
        message_box.setDefaultButton(QMessageBox.StandardButton.Yes)
        if message_box.exec() != QMessageBox.StandardButton.Yes:
            # The user canceled. Keep the current mode, and restore the
            # combo to it (nothing was written).
            settings_dialog.revert_storage_backend(old_mode)
            return
        if not override_check_box.isChecked():
            # Switch without moving any data.
            self._apply_storage_backend(new_mode)
            return
        # Move the loaded images' data to the new storage, then switch.
        if is_to_sqlite:
            # The import can take a while (it hashes every image file in
            # full), so it runs in a background thread; the switch is applied
            # when it finishes (see `on_storage_import_finished`).
            self._storage_import_in_progress = True
            self._storage_import_dialog = settings_dialog
            self._storage_import_mode = new_mode
            self.image_list_model.start_import_loaded_images_to_database()
            return
        else:
            written_count, failed_paths = (
                self.image_list_model.export_loaded_images_to_text_files())
            if failed_paths:
                self._report_storage_switch_failure(
                    settings_dialog,
                    f'Failed to write caption files for {len(failed_paths)} '
                    f'{pluralize("image", len(failed_paths))} '
                    f'(first: {failed_paths[0]}).')
                return
            self._apply_storage_backend(new_mode,
                                       migrated_count=written_count)

    def _report_storage_switch_failure(self, settings_dialog, message: str):
        """
        Report a failed storage mode switch and keep the current mode (the
        Settings dialog's combo is restored to it; the setting itself was
        not yet changed).
        """
        current_mode = self.settings.value(
            'storage_backend',
            defaultValue=DEFAULT_SETTINGS['storage_backend'], type=str)
        settings_dialog.revert_storage_backend(current_mode)
        QMessageBox.critical(self, 'Error', message)

    @Slot(bool, int, str)
    def on_storage_import_finished(self, succeeded: bool,
                                   imported_count: int, error_message: str):
        """
        A background text→database import (started by
        `handle_storage_backend_change`) has finished. Runs on the GUI
        thread.
        """
        self._storage_import_in_progress = False
        settings_dialog = self._storage_import_dialog
        new_mode = self._storage_import_mode
        self._storage_import_dialog = None
        self._storage_import_mode = None
        if not succeeded:
            self._report_storage_switch_failure(
                settings_dialog,
                f'Failed to import the loaded images into the database: '
                f'{error_message}.')
            return
        self._apply_storage_backend(new_mode,
                                   migrated_count=imported_count)

    def _apply_storage_backend(self, new_mode: str, migrated_count: int = 0):
        """
        Persist a successful storage mode switch, switch the active storage,
        and report the result.
        """
        self.settings.setValue('storage_backend', new_mode)
        self.image_list_model.set_storage(get_image_storage())
        # Reload the directory (if one is loaded) so that the in-memory
        # tags and descriptions come from the new storage. Without this, the
        # data that was loaded from the old storage would keep being
        # displayed — and, the next time an image is edited, written into
        # the new storage — even though the switch moved no data. (The
        # reload also clears the undo/redo history, which captured states
        # of the old storage.)
        if self.directory_path is not None and self.directory_path.is_dir():
            self.reload_directory()
        if migrated_count:
            if new_mode == 'sqlite':
                count_text = (f'{migrated_count} '
                              f'{pluralize("image", migrated_count)}')
                message = (f'Switched to database storage — imported '
                           f'{count_text} into the database.')
            else:
                count_text = (f'{migrated_count} '
                              f'{pluralize("caption file", migrated_count)}')
                message = (f'Switched to text-file storage — wrote {count_text}.')
        else:
            if new_mode == 'sqlite':
                message = 'Switched to database storage. No data was moved.'
            else:
                message = 'Switched to text-file storage. No data was moved.'
        QMessageBox.information(self, 'Storage', message)

    @Slot()
    def on_database_changed(self):
        """
        React to the active database having changed from the Settings dialog
        (it was cleared, or a database at a new location was adopted instead
        of the one in use). Re-point the active storage to the database that
        is now configured, then reload the directory (if one is loaded) so
        that the in-memory tags and descriptions come from that database
        rather than from the stale state loaded from the previous one.
        """
        self.image_list_model.set_storage(get_image_storage())
        if self.directory_path is not None and self.directory_path.is_dir():
            self.reload_directory()

    @Slot()
    def reset_to_defaults(self):
        message_box = QMessageBox()
        message_box.setWindowTitle('Reset to Defaults')
        message_box.setIcon(QMessageBox.Icon.Warning)
        message_box.setText(
            'Reset all settings to their defaults, including the window '
            'layout, prompts, and all configuration options?')
        message_box.setInformativeText(
            'This cannot be undone. Your images and their tags and '
            'descriptions are not affected. The application will restart.')
        message_box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        # A destructive action, so the safe button is the default.
        message_box.setDefaultButton(QMessageBox.StandardButton.No)
        if message_box.exec() != QMessageBox.StandardButton.Yes:
            return
        # Shut down the captioning subprocess so that it is not left running
        # (and holding on to the model in memory) after the restart.
        self.auto_captioner.model_manager.shutdown()
        self.settings.clear()
        self.settings.sync()
        restart_application()

    @Slot()
    def show_find_and_replace_dialog(self):
        find_and_replace_dialog = FindAndReplaceDialog(
            parent=self, image_list_model=self.image_list_model)
        find_and_replace_dialog.exec()

    @Slot()
    def show_batch_reorder_tags_dialog(self):
        batch_reorder_tags_dialog = BatchReorderTagsDialog(
            parent=self, image_list_model=self.image_list_model,
            tag_counter_model=self.tag_counter_model)
        batch_reorder_tags_dialog.exec()

    @Slot()
    def remove_duplicate_tags(self):
        removed_tag_count = self.image_list_model.remove_duplicate_tags()
        message_box = QMessageBox()
        message_box.setWindowTitle('Remove Duplicate Tags')
        message_box.setIcon(QMessageBox.Icon.Information)
        if not removed_tag_count:
            text = 'No duplicate tags were found.'
        else:
            text = (f'Removed {removed_tag_count} duplicate '
                    f'{pluralize("tag", removed_tag_count)}.')
        message_box.setText(text)
        message_box.exec()

    @Slot()
    def remove_empty_tags(self):
        removed_tag_count = self.image_list_model.remove_empty_tags()
        message_box = QMessageBox()
        message_box.setWindowTitle('Remove Empty Tags')
        message_box.setIcon(QMessageBox.Icon.Information)
        if not removed_tag_count:
            text = 'No empty tags were found.'
        else:
            text = (f'Removed {removed_tag_count} empty '
                    f'{pluralize("tag", removed_tag_count)}.')
        message_box.setText(text)
        message_box.exec()

    def create_menus(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu('File')
        load_directory_action = QAction('Load Directory...', parent=self)
        load_directory_action.setShortcut(QKeySequence('Ctrl+L'))
        load_directory_action.triggered.connect(self.select_and_load_directory)
        file_menu.addAction(load_directory_action)
        self.reload_directory_action.setShortcuts(
            [QKeySequence('Ctrl+Shift+L'), QKeySequence('F5')])
        self.reload_directory_action.triggered.connect(self.reload_directory)
        file_menu.addAction(self.reload_directory_action)
        settings_action = QAction('Settings...', parent=self)
        settings_action.setShortcut(QKeySequence('Ctrl+Alt+S'))
        settings_action.triggered.connect(self.show_settings_dialog)
        file_menu.addAction(settings_action)
        exit_action = QAction('Exit', parent=self)
        exit_action.setShortcut(QKeySequence('Ctrl+W'))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menu_bar.addMenu('Edit')
        self.undo_action.setShortcut(QKeySequence('Ctrl+Z'))
        self.undo_action.triggered.connect(self.image_list_model.undo)
        self.undo_action.setDisabled(True)
        edit_menu.addAction(self.undo_action)
        self.redo_action.setShortcut(QKeySequence('Ctrl+Y'))
        self.redo_action.triggered.connect(self.image_list_model.redo)
        self.redo_action.setDisabled(True)
        edit_menu.addAction(self.redo_action)
        find_and_replace_action = QAction('Find and Replace...', parent=self)
        find_and_replace_action.setShortcut(QKeySequence('Ctrl+R'))
        find_and_replace_action.triggered.connect(
            self.show_find_and_replace_dialog)
        edit_menu.addAction(find_and_replace_action)
        batch_reorder_tags_action = QAction('Batch Reorder Tags...',
                                            parent=self)
        batch_reorder_tags_action.setShortcut(QKeySequence('Ctrl+B'))
        batch_reorder_tags_action.triggered.connect(
            self.show_batch_reorder_tags_dialog)
        edit_menu.addAction(batch_reorder_tags_action)
        remove_duplicate_tags_action = QAction('Remove Duplicate Tags',
                                               parent=self)
        remove_duplicate_tags_action.setShortcut(QKeySequence('Ctrl+D'))
        remove_duplicate_tags_action.triggered.connect(
            self.remove_duplicate_tags)
        edit_menu.addAction(remove_duplicate_tags_action)
        remove_empty_tags_action = QAction('Remove Empty Tags', parent=self)
        remove_empty_tags_action.setShortcut(QKeySequence('Ctrl+E'))
        remove_empty_tags_action.triggered.connect(
            self.remove_empty_tags)
        edit_menu.addAction(remove_empty_tags_action)

        view_menu = menu_bar.addMenu('View')
        self.toggle_image_list_action.setCheckable(True)
        self.toggle_image_tags_editor_action.setCheckable(True)
        self.toggle_all_tags_editor_action.setCheckable(True)
        self.toggle_auto_captioner_action.setCheckable(True)
        self.toggle_image_list_action.triggered.connect(
            lambda is_checked: self.image_list.setVisible(is_checked))
        self.toggle_image_tags_editor_action.triggered.connect(
            lambda is_checked: self.image_tags_editor.setVisible(is_checked))
        self.toggle_all_tags_editor_action.triggered.connect(
            lambda is_checked: self.all_tags_editor.setVisible(is_checked))
        self.toggle_auto_captioner_action.triggered.connect(
            lambda is_checked: self.auto_captioner.setVisible(is_checked))
        view_menu.addAction(self.toggle_image_list_action)
        view_menu.addAction(self.toggle_image_tags_editor_action)
        view_menu.addAction(self.toggle_all_tags_editor_action)
        view_menu.addAction(self.toggle_auto_captioner_action)
        view_menu.addSeparator()
        reset_to_defaults_action = QAction('Reset to Defaults...',
                                           parent=self)
        reset_to_defaults_action.triggered.connect(self.reset_to_defaults)
        view_menu.addAction(reset_to_defaults_action)

        help_menu = menu_bar.addMenu('Help')
        open_github_repository_action = QAction('GitHub', parent=self)
        open_github_repository_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(GITHUB_REPOSITORY_URL)))
        help_menu.addAction(open_github_repository_action)

    @Slot()
    def update_undo_and_redo_actions(self):
        if self.image_list_model.undo_stack:
            undo_action_name = self.image_list_model.undo_stack[-1].action_name
            self.undo_action.setText(f'Undo "{undo_action_name}"')
            self.undo_action.setDisabled(False)
        else:
            self.undo_action.setText('Undo')
            self.undo_action.setDisabled(True)
        if self.image_list_model.redo_stack:
            redo_action_name = self.image_list_model.redo_stack[-1].action_name
            self.redo_action.setText(f'Redo "{redo_action_name}"')
            self.redo_action.setDisabled(False)
        else:
            self.redo_action.setText('Redo')
            self.redo_action.setDisabled(True)

    @Slot()
    def set_image_list_filter(self):
        filter_ = self.image_list.filter_line_edit.parse_filter_text()
        self.proxy_image_list_model.filter = filter_
        # Apply the new filter.
        self.proxy_image_list_model.invalidateFilter()
        if filter_ is None:
            all_tags_list_selection_model = (self.all_tags_editor
                                             .all_tags_list.selectionModel())
            all_tags_list_selection_model.clearSelection()
            # Clear the current index.
            self.all_tags_editor.all_tags_list.setCurrentIndex(QModelIndex())
            # Select the previously selected image in the unfiltered image
            # list.
            select_index = self.settings.value('image_index', type=int) or 0
            self.image_list.list_view.set_current_image_index(
                self.proxy_image_list_model.index(select_index, 0))
        else:
            # Select the first image.
            self.image_list.list_view.set_current_image_index(
                self.proxy_image_list_model.index(0, 0))

    @Slot()
    def save_image_index(self, proxy_image_index: QModelIndex):
        """Save the index of the currently selected image."""
        # The index is invalid when the model is reset (for example, while a
        # directory is loading).
        if not proxy_image_index.isValid():
            return
        settings_key = ('image_index'
                        if self.proxy_image_list_model.filter is None
                        else 'filtered_image_index')
        self.settings.setValue(settings_key, proxy_image_index.row())

    def connect_image_list_signals(self):
        self.image_list.filter_line_edit.textChanged.connect(
            self.set_image_list_filter)
        self.image_list_selection_model.currentChanged.connect(
            self.save_image_index)
        self.image_list_selection_model.currentChanged.connect(
            self.image_list.update_image_index_label)
        self.image_list_selection_model.currentChanged.connect(
            self.image_viewer.load_image)
        self.image_list_selection_model.currentChanged.connect(
            self.image_tags_editor.load_image_tags)
        # Keep the grouped tag view and synchronized grid preview in sync with
        # the selection. Reconciled on both a changed selection and a moved
        # current image (arrow keys), after the single-image handlers above.
        self.image_list_selection_model.selectionChanged.connect(
            self._sync_variant_group_view)
        self.image_list_selection_model.currentChanged.connect(
            self._sync_variant_group_view)
        # The grid holds row positions, and rows can be removed, added or
        # reordered without the selection signals firing (e.g. a refresh that
        # removes a selected file), so rebuild it whenever the rows change.
        for signal in (self.proxy_image_list_model.rowsRemoved,
                       self.proxy_image_list_model.rowsInserted,
                       self.proxy_image_list_model.layoutChanged,
                       self.proxy_image_list_model.modelReset):
            signal.connect(self._schedule_variant_group_resync)
        self.image_viewer.clicked.connect(self.image_list.raise_)
        self.image_viewer.clicked.connect(
            self._focus_image_list_on_preview_click)
        self.image_viewer.grid_cell_clicked.connect(self._select_grid_cell)
        self.image_viewer.grid_cell_context_menu_requested.connect(
            self._show_grid_cell_tag_menu)
        self.image_list_model.modelReset.connect(
            lambda: self.tag_counter_model.count_tags(
                self.image_list_model.images))
        self.image_list_model.dataChanged.connect(
            self.on_image_data_changed)
        self.image_list_model.update_undo_and_redo_actions_requested.connect(
            self.update_undo_and_redo_actions)
        self.image_list_model.directory_loaded.connect(
            self.on_directory_loaded)
        self.image_list_model.storage_import_finished.connect(
            self.on_storage_import_finished)
        # Rows are inserted or removed from the proxy image list model when the
        # filter is changed.
        self.proxy_image_list_model.rowsInserted.connect(
            lambda: self.image_list.update_image_index_label(
                self.image_list.list_view.currentIndex()))
        self.proxy_image_list_model.rowsRemoved.connect(
            lambda: self.image_list.update_image_index_label(
                self.image_list.list_view.currentIndex()))
        self.image_list.list_view.directory_reload_requested.connect(
            self.reload_directory)
        self.image_list.list_view.tags_paste_requested.connect(
            self.image_list_model.add_tags)
        # Connecting the signal directly without `isVisible()` causes the menu
        # item to be unchecked when the widget is an inactive tab.
        self.image_list.visibilityChanged.connect(
            lambda: self.toggle_image_list_action.setChecked(
                self.image_list.isVisible()))

    @Slot()
    def _show_window(self):
        """Show the main window if it is not already visible."""
        if not self.isVisible():
            # Show the window while fully transparent so its first paint
            # happens invisibly, then force that paint to complete before
            # revealing it.  Without this the OS briefly shows a blank white
            # window before Qt paints the UI for the first time.
            self.setWindowOpacity(0.0)
            self.show()
            # Flush pending layout and paint events so the window is fully
            # rendered, then reveal it in one step.
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            self.setWindowOpacity(1.0)

    @Slot(QModelIndex)
    def _select_grid_cell(self, proxy_index: QModelIndex):
        """Make a clicked grid cell the current image without changing selection.

        Mirrors the arrow-key path (switch_between_selected_images): only the
        current index moves, using NoUpdate so the selection set is untouched.
        Clicks that don't map to a currently-selected image are ignored.
        """
        if not proxy_index.isValid():
            return
        selected_proxy_indices = self.image_list.list_view.selectedIndexes()
        if proxy_index not in selected_proxy_indices:
            return
        if proxy_index == self.image_list_selection_model.currentIndex():
            return
        self.image_list_selection_model.setCurrentIndex(
            proxy_index, QItemSelectionModel.SelectionFlag.NoUpdate)

    @staticmethod
    def _shorten_tag(tag: str, limit: int = 40) -> str:
        """Truncate a tag for display in a menu label."""
        tag = tag.replace('\n', ' ')
        if len(tag) <= limit:
            return tag
        return tag[:limit - 1] + '\u2026'

    @Slot(QModelIndex, QPoint)
    def _show_grid_cell_tag_menu(self, proxy_index: QModelIndex,
                                 global_point: QPoint):
        """Context menu to add/remove the selected tag on a right-clicked cell.

        Shown when a grid cell is right-clicked in the multi-image (grid) view.
        Adds or removes the tag(s) currently selected in the Image Tags pane
        to/from just the image under the cursor.
        """
        if not proxy_index.isValid():
            return
        source_index = self.proxy_image_list_model.mapToSource(proxy_index)
        if not source_index.isValid():
            return
        image: Image = self.image_list_model.data(source_index,
                                                  Qt.ItemDataRole.UserRole)
        if image is None:
            return
        selected_tags = self.image_tags_editor.selected_group_tags()
        menu = QMenu(self)
        if not selected_tags:
            hint = menu.addAction('Select a tag in the Image Tags pane')
            hint.setEnabled(False)
            menu.exec(global_point)
            return
        existing = set(image.tags)
        missing_tags = [tag for tag in selected_tags if tag not in existing]
        present_tags = [tag for tag in selected_tags if tag in existing]
        if len(selected_tags) == 1:
            tag = selected_tags[0]
            label = f"'{self._shorten_tag(tag)}'"
            add_action = menu.addAction(f'Add {label} to this image')
            remove_action = menu.addAction(f'Remove {label} from this image')
        else:
            count = len(selected_tags)
            add_action = menu.addAction(f'Add {count} tags to this image')
            remove_action = menu.addAction(
                f'Remove {count} tags from this image')
        add_action.setEnabled(bool(missing_tags))
        remove_action.setEnabled(bool(present_tags))
        chosen = menu.exec(global_point)
        if chosen is None:
            return
        if chosen == add_action and missing_tags:
            self.image_tags_editor.remember_group_tag_anchor()
            self.image_list_model.add_tags(missing_tags, [source_index])
        elif chosen == remove_action and present_tags:
            self.image_tags_editor.remember_group_tag_anchor()
            self.image_list_model.remove_tags_from_images(present_tags,
                                                          [source_index])

    @Slot()
    def _schedule_variant_group_resync(self, *args):
        if (not self.image_viewer.is_grid_mode()
                or getattr(self, '_is_variant_group_resync_pending', False)):
            return
        self._is_variant_group_resync_pending = True

        def resync():
            self._is_variant_group_resync_pending = False
            if self.image_viewer.is_grid_mode():
                self._sync_variant_group_view()

        QTimer.singleShot(0, resync)

    @Slot()
    def _sync_variant_group_view(self, *args):
        """Reconcile the grouped tag view and grid preview with the selection.

        When the variant grid setting is on and two or more images are
        selected, the Image Tags pane shows the grouped Common/Differences view
        and the center preview shows a synchronized grid of the selection.
        Otherwise the panes fall back to the normal single-image view. This is
        safe to call repeatedly; it fully derives the desired state from the
        current selection each time.
        """
        grid_enabled = get_settings().value(
            'variant_grid_view_enabled',
            defaultValue=DEFAULT_SETTINGS['variant_grid_view_enabled'],
            type=bool)
        list_view = self.image_list.list_view
        selected_proxy_indices = sorted(list_view.selectedIndexes(),
                                        key=lambda index: index.row())
        if grid_enabled and len(selected_proxy_indices) >= 2:
            current_proxy_index = self.image_list_selection_model.currentIndex()
            selected_rows = [index.row() for index in selected_proxy_indices]
            current_row = current_proxy_index.row()
            current_position = (selected_rows.index(current_row)
                                if current_row in selected_rows else 0)
            source_indices = [
                self.proxy_image_list_model.mapToSource(index)
                for index in selected_proxy_indices]
            current_source_index = source_indices[current_position]
            cell_cap = get_settings().value(
                'variant_grid_cell_cap',
                defaultValue=DEFAULT_SETTINGS['variant_grid_cell_cap'],
                type=int)
            if self.image_tags_editor.is_group_mode():
                self.image_tags_editor.update_group_selection(
                    source_indices, current_source_index)
                self.image_viewer.update_grid_current(
                    selected_proxy_indices, current_position, cell_cap)
            else:
                self.image_tags_editor.enter_group_mode(
                    source_indices, current_source_index)
                self.image_viewer.show_grid(
                    selected_proxy_indices, current_position, cell_cap)
            return
        # Single-image (or disabled) view.
        was_grid = self.image_viewer.is_grid_mode()
        was_group = self.image_tags_editor.is_group_mode()
        self.image_tags_editor.exit_group_mode()
        if was_group:
            # currentChanged fires before selectionChanged, so load_image_tags
            # already ran once while group mode was still active and early
            # returned without refreshing the single-image tag list or the
            # green "Complete" indicator. Now that group mode is off, reload the
            # current image so both reflect its real state (or empty the pane
            # if no image is current, e.g. the filter hides everything).
            current_proxy_index = self.image_list_selection_model.currentIndex()
            self.image_tags_editor.load_image_tags(current_proxy_index)
        if was_grid:
            self.image_viewer.exit_grid()
            # Also called with an invalid index (nothing current), which
            # clears the grid instead of leaving it on screen.
            current_proxy_index = self.image_list_selection_model.currentIndex()
            self.image_viewer.load_image(current_proxy_index)

    @Slot()
    def _focus_image_list_on_preview_click(self):
        """Move focus to the thumbnails after a preview click, unless the user
        was working in the Image Tags pane.

        Clicking or dragging in the preview normally focuses the Images pane so
        the arrow keys navigate images. But if focus is already in the Image
        Tags pane (e.g. the Add Tag input or the tag list), keep it there so
        zooming/dragging the preview doesn't interrupt tag entry or up/down
        cycling through tags."""
        focus_widget = QApplication.focusWidget()
        if (focus_widget is not None
                and self.image_tags_editor.isAncestorOf(focus_widget)):
            return
        self.image_list.list_view.setFocus()

    @Slot(QModelIndex, QModelIndex, list)
    def on_image_data_changed(self, first_changed_index: QModelIndex,
                              last_changed_index: QModelIndex,
                              roles: list):
        """
        Handle changes to image data.

        Thumbnail-only changes (only the decoration role changed) do not
        affect the tags, so they do not need to update the tag counter or the
        image tags editor. This matters because a thumbnail is generated for
        every image, which would otherwise recount the tags of every image
        thousands of times.
        """
        if roles == [Qt.ItemDataRole.DecorationRole]:
            return
        self.tag_counter_model.count_tags(self.image_list_model.images)
        self.image_tags_editor.reload_image_tags_if_changed(
            first_changed_index, last_changed_index)

    @Slot()
    def update_image_tags(self):
        image_index = self.image_tags_editor.image_index
        # `image_index` is None (or invalid) before the first image is loaded
        # or when the current image is cleared; the tag list model can still
        # emit in those states (e.g. during a directory load), so bail out
        # rather than dereference a null index.
        if image_index is None or not image_index.isValid():
            return
        image: Image = self.image_list_model.data(image_index,
                                                  Qt.ItemDataRole.UserRole)
        if image is None:
            return
        old_tags = image.tags
        new_tags = self.image_tag_list_model.stringList()
        if old_tags == new_tags:
            return
        old_tags_count = len(old_tags)
        new_tags_count = len(new_tags)
        if new_tags_count > old_tags_count:
            self.image_list_model.add_to_undo_stack(
                action_name='Add Tag', should_ask_for_confirmation=False)
        elif new_tags_count == old_tags_count:
            if set(new_tags) == set(old_tags):
                self.image_list_model.add_to_undo_stack(
                    action_name='Reorder Tags',
                    should_ask_for_confirmation=False)
            else:
                self.image_list_model.add_to_undo_stack(
                    action_name='Rename Tag',
                    should_ask_for_confirmation=False)
        elif old_tags_count - new_tags_count == 1:
            self.image_list_model.add_to_undo_stack(
                action_name='Delete Tag', should_ask_for_confirmation=False)
        else:
            self.image_list_model.add_to_undo_stack(
                action_name='Delete Tags', should_ask_for_confirmation=False)
        self.image_list_model.update_image_tags(image_index, new_tags)

    def connect_image_tags_editor_signals(self):
        # `rowsInserted` does not have to be connected because `dataChanged`
        # is emitted when a tag is added.
        self.image_tag_list_model.modelReset.connect(self.update_image_tags)
        self.image_tag_list_model.dataChanged.connect(self.update_image_tags)
        self.image_tag_list_model.rowsMoved.connect(self.update_image_tags)
        self.image_tags_editor.visibilityChanged.connect(
            lambda: self.toggle_image_tags_editor_action.setChecked(
                self.image_tags_editor.isVisible()))
        self.image_tags_editor.tag_input_box.tags_addition_requested.connect(
            self.image_list_model.add_tags)
        self.image_tags_editor.image_data_save_requested.connect(
            self.image_list_model.write_image_tags_to_disk)
        self.image_tags_editor.grid_mark_paths_changed.connect(
            self.image_viewer.set_marked_paths)
        self.image_tags_editor.grid_peek_changed.connect(
            self.image_viewer.set_grid_peek)

    @Slot()
    def set_image_list_filter_text(self, selected_tag: str):
        """
        Construct and set the image list filter text from the selected tag in
        the all tags list.
        """
        escaped_selected_tag = (selected_tag.replace('\\', '\\\\')
                                .replace('"', r'\"').replace("'", r"\'"))
        self.image_list.filter_line_edit.setText(
            f'tag:"{escaped_selected_tag}"')

    @Slot(str)
    def add_tag_to_selected_images(self, tag: str):
        selected_image_indices = self.image_list.get_selected_image_indices()
        self.image_list_model.add_tags([tag], selected_image_indices)
        self.image_tags_editor.select_last_tag()

    def connect_all_tags_editor_signals(self):
        self.all_tags_editor.clear_filter_button.clicked.connect(
            self.image_list.filter_line_edit.clear)
        self.tag_counter_model.tags_renaming_requested.connect(
            self.image_list_model.rename_tags)
        self.tag_counter_model.tags_renaming_requested.connect(
            self.image_list.filter_line_edit.clear)
        self.all_tags_editor.all_tags_list.image_list_filter_requested.connect(
            self.set_image_list_filter_text)
        self.all_tags_editor.all_tags_list.tag_addition_requested.connect(
            self.add_tag_to_selected_images)
        self.all_tags_editor.all_tags_list.tags_deletion_requested.connect(
            self.image_list_model.delete_tags)
        self.all_tags_editor.all_tags_list.tags_deletion_requested.connect(
            self.image_list.filter_line_edit.clear)
        self.all_tags_editor.visibilityChanged.connect(
            lambda: self.toggle_all_tags_editor_action.setChecked(
                self.all_tags_editor.isVisible()))

    def connect_auto_captioner_signals(self):
        self.auto_captioner.caption_generated.connect(
            lambda image_index, description, tags:
                self.image_list_model.update_image_description(image_index, description)
                if description else
                self.image_list_model.update_image_tags(image_index, tags)
            )
        self.auto_captioner.caption_generated.connect(
            lambda image_index, *_:
            self.image_tags_editor.reload_image_tags_if_changed(image_index,
                                                                image_index))
        self.auto_captioner.visibilityChanged.connect(
            lambda: self.toggle_auto_captioner_action.setChecked(
                self.auto_captioner.isVisible()))

    def restore(self):
        # Restore the window geometry and state.
        if self.settings.contains('geometry'):
            self.restoreGeometry(self.settings.value('geometry', type=bytes))
        else:
            self.setWindowState(Qt.WindowState.WindowMaximized)
        self.restoreState(self.settings.value('window_state', type=bytes))
        # Get the last index of the last selected image.
        if self.settings.contains('image_index'):
            image_index = self.settings.value('image_index', type=int)
        else:
            image_index = 0
        # Load the last loaded directory.
        if self.settings.contains('directory_path'):
            directory_path = Path(self.settings.value('directory_path',
                                                      type=str))
            if directory_path.is_dir():
                self.load_directory(directory_path, select_index=image_index)
