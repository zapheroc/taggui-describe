import sqlite3
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QGridLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton,
                               QVBoxLayout)

from models.image_storage import (clear_database, get_database_path,
                                  get_default_database_path, migrate_database)
from utils.settings import DEFAULT_SETTINGS, get_settings
from utils.settings_widgets import (SettingsBigCheckBox, SettingsLineEdit,
                                    SettingsSpinBox)
from utils.utils import pluralize


class SettingsDialog(QDialog):
    # Emitted when the grid page/count overlay appearance settings change so
    # an open grid preview can re-render immediately.
    variant_grid_overlay_settings_changed = Signal()
    # Emitted when the tags/descriptions storage backend changes. Carries the
    # new backend ('sqlite' or 'text'). The change is not persisted until the
    # main window has confirmed (or migrated) it, so the combo is plain and
    # the main window may revert it with `revert_storage_backend`.
    storage_backend_change_requested = Signal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.settings = get_settings()
        self.setWindowTitle('Settings')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel('Font size (pt)'), 0, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('File types to show in image list'), 1, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('Image width in image list (px)'), 2, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('Tag separator (\\n for newline)'), 3, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('Insert space after tag separator'), 4, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('Show tag autocomplete suggestions'),
                              5, 0, Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('Auto-captioning models directory'), 6, 0,
                              Qt.AlignmentFlag.AlignRight)

        font_size_spin_box = SettingsSpinBox(
            key='font_size', default=DEFAULT_SETTINGS['font_size'],
            minimum=1, maximum=99)
        font_size_spin_box.valueChanged.connect(self.show_restart_warning)
        # Images that are too small cause lag, so set a minimum width.
        image_list_image_width_spin_box = SettingsSpinBox(
            key='image_list_image_width',
            default=DEFAULT_SETTINGS['image_list_image_width'],
            minimum=16, maximum=9999)
        image_list_image_width_spin_box.valueChanged.connect(
            self.show_restart_warning)
        self.insert_space_after_tag_separator_check_box = SettingsBigCheckBox(
            key='insert_space_after_tag_separator',
            default=DEFAULT_SETTINGS['insert_space_after_tag_separator'])
        self.insert_space_after_tag_separator_check_box.stateChanged.connect(
            self.show_restart_warning)
        tag_separator_line_edit = QLineEdit()
        tag_separator = self.settings.value(
            'tag_separator', defaultValue=DEFAULT_SETTINGS['tag_separator'],
            type=str)
        if tag_separator == '\n':
            tag_separator = r'\n'
            self.disable_insert_space_after_tag_separator_check_box()
        tag_separator_line_edit.setMaximumWidth(50)
        tag_separator_line_edit.setText(tag_separator)
        tag_separator_line_edit.textChanged.connect(
            self.handle_tag_separator_change)
        autocomplete_tags_check_box = SettingsBigCheckBox(
            key='autocomplete_tags',
            default=DEFAULT_SETTINGS['autocomplete_tags'])
        autocomplete_tags_check_box.stateChanged.connect(
            self.show_restart_warning)
        self.models_directory_line_edit = SettingsLineEdit(
            key='models_directory_path',
            default=DEFAULT_SETTINGS['models_directory_path'])
        self.models_directory_line_edit.setMinimumWidth(400)
        self.models_directory_line_edit.setClearButtonEnabled(True)
        self.models_directory_line_edit.textChanged.connect(
            self.show_restart_warning)
        models_directory_button = QPushButton('Select Directory...')
        models_directory_button.setFixedWidth(
            int(models_directory_button.sizeHint().width() * 1.3))
        models_directory_button.clicked.connect(self.set_models_directory_path)
        file_types_line_edit = SettingsLineEdit(
            key='image_list_file_formats',
            default=DEFAULT_SETTINGS['image_list_file_formats'])
        file_types_line_edit.setMinimumWidth(400)
        file_types_line_edit.textChanged.connect(self.show_restart_warning)
        max_image_preview_zoom_spin_box = SettingsSpinBox(
            key='max_image_preview_zoom',
            default=DEFAULT_SETTINGS['max_image_preview_zoom'],
            minimum=1, maximum=100)
        type_to_add_tag_check_box = SettingsBigCheckBox(
            key='image_list_auto_focus_add_tag_box',
            default=DEFAULT_SETTINGS['image_list_auto_focus_add_tag_box'])
        variant_grid_view_check_box = SettingsBigCheckBox(
            key='variant_grid_view_enabled',
            default=DEFAULT_SETTINGS['variant_grid_view_enabled'])
        variant_grid_cell_cap_spin_box = SettingsSpinBox(
            key='variant_grid_cell_cap',
            default=DEFAULT_SETTINGS['variant_grid_cell_cap'],
            minimum=4, maximum=64)
        grid_overlay_font_size_spin_box = SettingsSpinBox(
            key='variant_grid_overlay_font_size',
            default=DEFAULT_SETTINGS['variant_grid_overlay_font_size'],
            minimum=1, maximum=99)
        grid_overlay_font_size_spin_box.valueChanged.connect(
            lambda _: self.variant_grid_overlay_settings_changed.emit())
        grid_overlay_transparency_spin_box = SettingsSpinBox(
            key='variant_grid_overlay_transparency',
            default=DEFAULT_SETTINGS['variant_grid_overlay_transparency'],
            minimum=0, maximum=100)
        grid_overlay_transparency_spin_box.valueChanged.connect(
            lambda _: self.variant_grid_overlay_settings_changed.emit())
        show_grid_overlay_check_box = SettingsBigCheckBox(
            key='variant_grid_overlay_show',
            default=DEFAULT_SETTINGS['variant_grid_overlay_show'])
        grid_overlay_font_size_label = QLabel(
            'Grid page/count overlay text size (pt)')
        grid_overlay_transparency_label = QLabel(
            'Grid page/count overlay background transparency (%)')

        def update_grid_overlay_sub_settings_enabled(checked: bool):
            # Grey out the overlay appearance controls when the overlay is off.
            grid_overlay_font_size_spin_box.setEnabled(checked)
            grid_overlay_transparency_spin_box.setEnabled(checked)
            grid_overlay_font_size_label.setEnabled(checked)
            grid_overlay_transparency_label.setEnabled(checked)

        update_grid_overlay_sub_settings_enabled(
            show_grid_overlay_check_box.isChecked())
        show_grid_overlay_check_box.stateChanged.connect(
            lambda _: (
                update_grid_overlay_sub_settings_enabled(
                    show_grid_overlay_check_box.isChecked()),
                self.variant_grid_overlay_settings_changed.emit()))
        self.storage_backend_combo = QComboBox()
        self.storage_backend_combo.addItem('SQLite database', 'sqlite')
        self.storage_backend_combo.addItem('Text files next to images', 'text')
        storage_backend = self.settings.value(
            'storage_backend',
            defaultValue=DEFAULT_SETTINGS['storage_backend'], type=str)
        # Set the initial value before connecting, so that it does not emit
        # the change signal.
        storage_backend_index = self.storage_backend_combo.findData(
            storage_backend)
        if storage_backend_index != -1:
            self.storage_backend_combo.setCurrentIndex(
                storage_backend_index)
        self.storage_backend_combo.currentIndexChanged.connect(
            self.handle_storage_backend_change)
        database_location = self.settings.value(
            'database_path',
            defaultValue=DEFAULT_SETTINGS['database_path'], type=str)
        self.database_location_line_edit = QLineEdit()
        # The line edit shows the active location: the setting, or the
        # platform default when it is empty.
        self.database_location_line_edit.setText(
            database_location if database_location
            else str(get_default_database_path()))
        self.database_location_line_edit.setMinimumWidth(400)
        self.database_location_line_edit.setClearButtonEnabled(True)
        self.database_location_line_edit.editingFinished.connect(
            self.handle_database_location_commit)
        self.database_location_button = QPushButton('Select...')
        self.database_location_button.setFixedWidth(
            int(self.database_location_button.sizeHint().width() * 1.3))
        self.database_location_button.clicked.connect(
            self.set_database_location_path)
        clear_database_button = QPushButton('Clear Database...')
        clear_database_button.setFixedWidth(
            int(clear_database_button.sizeHint().width() * 1.3))
        clear_database_button.clicked.connect(self.handle_clear_database)
        self._update_storage_rows_enabled()
        # The database location that is currently active. Used so that the
        # same location is not migrated again on repeated commits.
        self._last_database_location = self._effective_database_location()

        grid_layout.addWidget(font_size_spin_box, 0, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(file_types_line_edit, 1, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(image_list_image_width_spin_box, 2, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(tag_separator_line_edit, 3, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.insert_space_after_tag_separator_check_box,
                              4, 1, Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(autocomplete_tags_check_box, 5, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.models_directory_line_edit, 6, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(models_directory_button, 7, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(QLabel('Max image preview zoom (x)'), 8, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(max_image_preview_zoom_spin_box, 8, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(
            QLabel('Auto-focus Add Tag box when typing in Images pane'), 9, 0,
            Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(type_to_add_tag_check_box, 9, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(
            QLabel('Group tags & show grid when multiple images are selected'),
            10, 0, Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(variant_grid_view_check_box, 10, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(
            QLabel('Max grid cells for multi-image preview'), 11, 0,
            Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(variant_grid_cell_cap_spin_box, 11, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(QLabel('Show grid page/count overlay'), 12, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(show_grid_overlay_check_box, 12, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(grid_overlay_font_size_label, 13, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(grid_overlay_font_size_spin_box, 13, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(grid_overlay_transparency_label, 14, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(grid_overlay_transparency_spin_box, 14, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(
            QLabel('Tags and descriptions storage'), 15, 0,
            Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(self.storage_backend_combo, 15, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(QLabel('Database location'), 16, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(self.database_location_line_edit, 16, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(self.database_location_button, 17, 1,
                              Qt.AlignmentFlag.AlignLeft)
        grid_layout.addWidget(clear_database_button, 18, 1,
                              Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(grid_layout)

        # Prevent the grid layout from moving to the center when the warning
        # label is hidden.
        layout.addStretch()
        self.restart_warning = ('Restart the application to apply the new '
                                'settings.')
        self.warning_label = QLabel(self.restart_warning)
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.warning_label.setStyleSheet('color: red;')
        layout.addWidget(self.warning_label)
        # Fix the size of the dialog to its size when the warning label is
        # shown.
        self.setFixedSize(self.sizeHint())
        self.warning_label.hide()

    @Slot()
    def show_restart_warning(self):
        self.warning_label.setText(self.restart_warning)
        self.warning_label.show()

    def disable_insert_space_after_tag_separator_check_box(self):
        self.insert_space_after_tag_separator_check_box.setEnabled(False)
        self.insert_space_after_tag_separator_check_box.setChecked(False)

    @Slot(str)
    def handle_tag_separator_change(self, tag_separator: str):
        if not tag_separator:
            self.warning_label.setText('The tag separator cannot be empty.')
            self.warning_label.show()
            return
        if tag_separator == r'\n':
            tag_separator = '\n'
            self.disable_insert_space_after_tag_separator_check_box()
        else:
            self.insert_space_after_tag_separator_check_box.setEnabled(True)
        self.settings.setValue('tag_separator', tag_separator)
        self.show_restart_warning()

    @Slot()
    def set_models_directory_path(self):
        models_directory_path = self.settings.value(
            'models_directory_path',
            defaultValue=DEFAULT_SETTINGS['models_directory_path'], type=str)
        if models_directory_path:
            initial_directory_path = models_directory_path
        elif self.settings.contains('directory_path'):
            initial_directory_path = self.settings.value('directory_path')
        else:
            initial_directory_path = ''
        models_directory_path = QFileDialog.getExistingDirectory(
            parent=self, caption='Select directory containing auto-captioning '
                                 'models',
            dir=initial_directory_path)
        if models_directory_path:
            self.models_directory_line_edit.setText(models_directory_path)

    def _update_storage_rows_enabled(self):
        """
        Enable the database location row only while SQLite storage is
        selected.
        """
        is_sqlite = self.storage_backend_combo.currentData() == 'sqlite'
        self.database_location_line_edit.setEnabled(is_sqlite)
        self.database_location_button.setEnabled(is_sqlite)

    @Slot(int)
    def handle_storage_backend_change(self, index: int):
        self._update_storage_rows_enabled()
        self.storage_backend_change_requested.emit(
            self.storage_backend_combo.currentData())

    def revert_storage_backend(self, value: str):
        """
        Restore the storage backend combo to `value` without re-emitting
        the change signal (used when a switch is canceled or fails).
        """
        index = self.storage_backend_combo.findData(value)
        if index == -1:
            return
        self.storage_backend_combo.blockSignals(True)
        self.storage_backend_combo.setCurrentIndex(index)
        self.storage_backend_combo.blockSignals(False)
        self._update_storage_rows_enabled()

    def _effective_database_location(self) -> Path:
        """
        The database location the line edit currently represents (the text,
        or the platform default when it is empty).
        """
        text = self.database_location_line_edit.text().strip()
        if text:
            return Path(text)
        return get_default_database_path()

    @Slot()
    def set_database_location_path(self):
        """Choose a database location with a file dialog."""
        database_location = self.settings.value(
            'database_path',
            defaultValue=DEFAULT_SETTINGS['database_path'], type=str)
        initial_path = (Path(database_location).parent
                        if database_location
                        else get_default_database_path().parent)
        database_location = QFileDialog.getSaveFileName(
            parent=self, caption='Select location of the TagGUI database',
            dir=str(initial_path),
            filter='SQLite database (*.sqlite3)')[0]
        if database_location:
            self.database_location_line_edit.setText(database_location)
            self.handle_database_location_commit()

    @Slot()
    def handle_database_location_commit(self):
        """
        Persist the committed database location. When it has changed and a
        database exists at the old location, move it to the new location
        (confirming first when a database already exists there).
        """
        new_location = self._effective_database_location()
        old_location = self._last_database_location
        if new_location == old_location:
            return
        self._last_database_location = new_location
        self.settings.setValue(
            'database_path',
            self.database_location_line_edit.text().strip())
        if not old_location.is_file():
            # There is no database to move.
            return
        try:
            if new_location.is_file():
                message_box = QMessageBox()
                message_box.setWindowTitle('Move Database')
                message_box.setIcon(QMessageBox.Icon.Warning)
                message_box.setText(f'A database already exists at '
                                    f'{new_location}.')
                message_box.setInformativeText(
                    f'Overwrite it with the database at {old_location}?')
                message_box.setStandardButtons(
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No)
                # A destructive action, so the safe button is the default.
                message_box.setDefaultButton(QMessageBox.StandardButton.No)
                if message_box.exec() != QMessageBox.StandardButton.Yes:
                    self._revert_database_location(old_location)
                    return
            migrate_database(old_location, new_location)
        except OSError:
            # The move failed; the database at the old location is intact,
            # so restore it.
            self._revert_database_location(old_location)
            QMessageBox.critical(self, 'Error',
                                 f'Failed to move the database to '
                                 f'{new_location}.')
            return
        QMessageBox.information(self, 'Move Database',
                                f'The database has been moved to '
                                f'{new_location}.')

    def _revert_database_location(self, previous_location: Path):
        """
        Restore the database location line edit and setting to
        `previous_location`.
        """
        default_location = get_default_database_path()
        self.settings.setValue(
            'database_path',
            '' if previous_location == default_location
            else str(previous_location))
        self.database_location_line_edit.setText(str(previous_location))
        self._last_database_location = previous_location

    @Slot()
    def handle_clear_database(self):
        """
        Delete all records from the database (the database file itself stays
        usable, and nothing outside it is touched).
        """
        database_path = get_database_path()
        if not database_path.is_file():
            QMessageBox.information(self, 'Clear Database',
                                    f'There is no database at '
                                    f'{database_path}.')
            return
        message_box = QMessageBox()
        message_box.setWindowTitle('Clear Database')
        message_box.setIcon(QMessageBox.Icon.Warning)
        message_box.setText(
            f'Permanently delete all tags and descriptions stored in the '
            f'TagGUI database at {database_path}?')
        message_box.setInformativeText(
            'This cannot be undone. Image files and .txt caption files are '
            'not affected.')
        message_box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        # A destructive action, so the safe button is the default.
        message_box.setDefaultButton(QMessageBox.StandardButton.No)
        if message_box.exec() != QMessageBox.StandardButton.Yes:
            return
        try:
            cleared_count = clear_database(database_path)
        except sqlite3.Error:
            QMessageBox.critical(self, 'Error',
                                 f'Failed to clear the database at '
                                 f'{database_path}.')
            return
        QMessageBox.information(
            self, 'Clear Database',
            f'Cleared {cleared_count} '
            f'{pluralize("image record", cleared_count)} from the database.')
