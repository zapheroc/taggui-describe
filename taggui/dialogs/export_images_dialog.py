from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog,
                                QGridLayout, QHBoxLayout, QLabel, QMessageBox,
                                QPushButton, QVBoxLayout)

from models.image_export import (ExportMode, ExportResult, export_images,
                                  plan_export_names)
from models.image_list_model import ImageListModel, Scope
from utils.settings_widgets import SettingsComboBox, SettingsLineEdit
from utils.utils import pluralize

# How many files are listed in the override and summary dialogs before the
# list is truncated.
MAX_LISTED_FILES = 20


class ExportImagesDialog(QDialog):
    def __init__(self, parent, image_list_model: ImageListModel):
        super().__init__(parent)
        self.image_list_model = image_list_model
        self.setWindowTitle('Export Images')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        description_label = QLabel(
            'Export the images in the chosen scope to the output folder, '
            'each with its tags and description in a text file next to it. '
            'Images with the same file name are renamed (cat.jpg, cat_1.jpg, '
            '...) so that nothing is overwritten by the export itself.')
        description_label.setWordWrap(True)
        layout.addWidget(description_label)
        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel('Scope'), 0, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('Image files'), 1, 0,
                              Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(QLabel('Output folder'), 2, 0,
                              Qt.AlignmentFlag.AlignRight)
        self.scope_combo_box = SettingsComboBox(
            key='export_scope', default=Scope.FILTERED_IMAGES.value)
        self.scope_combo_box.addItems(list(Scope))
        self.scope_combo_box.currentTextChanged.connect(
            self.update_export_button)
        grid_layout.addWidget(self.scope_combo_box, 0, 1)
        self.image_mode_combo_box = SettingsComboBox(
            key='export_image_mode', default=ExportMode.COPY.value)
        self.image_mode_combo_box.addItems(list(ExportMode))
        grid_layout.addWidget(self.image_mode_combo_box, 1, 1)
        self.output_folder_line_edit = SettingsLineEdit(
            key='export_directory')
        self.browse_button = QPushButton('Browse...')
        self.browse_button.clicked.connect(self.browse_output_folder)
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.output_folder_line_edit)
        folder_layout.addWidget(self.browse_button)
        grid_layout.addLayout(folder_layout, 2, 1)
        layout.addLayout(grid_layout)
        self.export_button = QPushButton('Export')
        self.export_button.clicked.connect(self.export)
        layout.addWidget(self.export_button)
        self.update_export_button()

    def get_scope(self) -> Scope:
        """
        Get the selected scope. The saved value may be missing or no
        longer valid, so fall back to filtered images in that case.
        """
        try:
            return Scope(self.scope_combo_box.currentText())
        except ValueError:
            return Scope.FILTERED_IMAGES

    def get_image_mode(self) -> ExportMode:
        """
        Get the selected image file mode. The saved value may be missing or
        no longer valid, so fall back to copy in that case.
        """
        try:
            return ExportMode(self.image_mode_combo_box.currentText())
        except ValueError:
            return ExportMode.COPY

    @Slot()
    def update_export_button(self):
        image_count = len(self.image_list_model.get_images_in_scope(
            self.get_scope()))
        self.export_button.setText(f'Export {image_count} '
                                    f'{pluralize("image", image_count)}')

    @Slot()
    def browse_output_folder(self):
        output_folder = QFileDialog.getExistingDirectory(
            parent=self, caption='Select the output folder',
            dir=self.output_folder_line_edit.text())
        if output_folder:
            self.output_folder_line_edit.setText(output_folder)

    def confirm_overrides(self, output_folder: Path,
                          overrides: list[str]) -> bool:
        """
        Warn the user that the listed files in the output folder would be
        overwritten, and ask whether to proceed.
        """
        message = (f'The following files already exist in '
                   f'{output_folder} and will be overwritten:')
        for name in overrides[:MAX_LISTED_FILES]:
            message += f'\n  {name}'
        if len(overrides) > MAX_LISTED_FILES:
            message += (f'\n  ... and '
                        f'{len(overrides) - MAX_LISTED_FILES} more')
        message_box = QMessageBox()
        message_box.setWindowTitle('Export Images')
        message_box.setIcon(QMessageBox.Icon.Warning)
        message_box.setText(message)
        message_box.setStandardButtons(
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No)
        message_box.setDefaultButton(QMessageBox.StandardButton.No)
        return message_box.exec() == QMessageBox.StandardButton.Yes

    def show_export_summary(self, result: ExportResult):
        image_count = len(result.exported_images)
        text_file_count = len(result.written_text_files)
        message = (f'Exported {image_count} {pluralize("image", image_count)} '
                   f'to {self.output_folder_line_edit.text()}.')
        message += (f' Wrote {text_file_count} '
                    f'{pluralize("text file", text_file_count)}.')
        if result.skipped_text_files:
            skipped = result.skipped_text_files
            message += (f' {skipped} {pluralize("image", skipped)} had no '
                        f'tags or description, so no text file was written '
                        f'for {"it" if skipped == 1 else "them"}.')
        if result.failed:
            message += '\n\nThe following images failed to export:'
            for image, error in result.failed[:MAX_LISTED_FILES]:
                message += f'\n  {image.path}: {error}'
            if len(result.failed) > MAX_LISTED_FILES:
                message += (f'\n  ... and '
                            f'{len(result.failed) - MAX_LISTED_FILES} more')
        message_box = QMessageBox()
        message_box.setWindowTitle('Export Images')
        message_box.setIcon(QMessageBox.Icon.Critical if result.failed
                             else QMessageBox.Icon.Information)
        message_box.setText(message)
        message_box.exec()

    @Slot()
    def export(self):
        output_folder_text = self.output_folder_line_edit.text().strip()
        if not output_folder_text:
            QMessageBox.warning(
                self, 'Export Images', 'Please choose an output folder.')
            return
        output_folder = Path(output_folder_text)
        scope = self.get_scope()
        images = self.image_list_model.get_images_in_scope(scope)
        if not images:
            scope_text = {
                Scope.ALL_IMAGES: 'images in the full dataset',
                Scope.FILTERED_IMAGES: 'images in the current view',
                Scope.SELECTED_IMAGES: 'selected images',
            }[scope]
            QMessageBox.information(
                self, 'Export Images',
                f'There are no {scope_text} to export.')
            return
        planned, overrides = plan_export_names(images, output_folder)
        if overrides and not self.confirm_overrides(output_folder,
                                                    overrides):
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = export_images(images, self.image_list_model.tag_separator,
                                   output_folder, self.get_image_mode())
        except OSError as exception:
            QMessageBox.critical(self, 'Export Images', str(exception))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.show_export_summary(result)
        self.accept()
