from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (QDialog, QFileDialog, QGridLayout, QLabel,
                               QLineEdit, QPushButton, QVBoxLayout)

from utils.settings import DEFAULT_SETTINGS, get_settings
from utils.settings_widgets import (SettingsBigCheckBox, SettingsLineEdit,
                                    SettingsSpinBox)


class SettingsDialog(QDialog):
    # Emitted when the grid page/count overlay appearance settings change so
    # an open grid preview can re-render immediately.
    variant_grid_overlay_settings_changed = Signal()

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
