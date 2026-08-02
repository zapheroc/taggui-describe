import sys
import importlib.util
from pathlib import Path
from enum import StrEnum

from PySide6.QtCore import QModelIndex, Qt, Signal, Slot
from PySide6.QtGui import QFontMetrics, QTextCursor
from PySide6.QtWidgets import (QAbstractScrollArea, QDockWidget, QFormLayout,
                               QFrame, QHBoxLayout, QLabel, QMessageBox,
                               QPlainTextEdit, QProgressBar, QScrollArea,
                               QVBoxLayout, QWidget, QApplication, QPushButton)

from auto_captioning.captioning_thread import CaptioningThread
from auto_captioning.model_subprocess_manager import ModelSubprocessManager
from auto_captioning.models_list import MODELS, get_model_class
from dialogs.caption_multiple_images_dialog import CaptionMultipleImagesDialog
from models.image_list_model import ImageListModel
from utils.big_widgets import TallPushButton
from utils.enums import CaptionDevice, CaptionPosition
from utils.settings import DEFAULT_SETTINGS, get_settings, get_tag_separator
from utils.settings_widgets import (FocusedScrollSettingsComboBox,
                                    FocusedScrollSettingsDoubleSpinBox,
                                    FocusedScrollSettingsSpinBox,
                                    SettingsBigCheckBox, SettingsLineEdit,
                                    SettingsPlainTextEdit)
from utils.utils import pluralize
from widgets.image_list import ImageList
from auto_captioning.settings_group import SettingGroup


BITSANDBYTES_AVAILABLE = importlib.util.find_spec('bitsandbytes') is not None


def set_text_edit_height(text_edit: QPlainTextEdit, line_count: int):
    """
    Set the height of a text edit to the height of a given number of lines.
    """
    # From https://stackoverflow.com/a/46997337.
    document = text_edit.document()
    font_metrics = QFontMetrics(document.defaultFont())
    margins = text_edit.contentsMargins()
    height = int(font_metrics.lineSpacing() * line_count
                 + margins.top() + margins.bottom()
                 + document.documentMargin() * 2
                 + text_edit.frameWidth() * 2)
    text_edit.setFixedHeight(height)


class HorizontalLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Raised)


class LabelPosition(StrEnum):
    # Basic
    ABOVE = 'above'
    BESIDE = 'beside'
    DEFAULT = 'default'


class CaptionSettingsForm(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.settings = get_settings()

        # Registry: (owning form, field widget, groups, is_advanced)
        self.registered_rows: list[
            tuple[QFormLayout, QWidget, set[SettingGroup], bool]] = []
        self.advanced_expanded = False

        # ------------------------------------------------------------------
        # Basic settings
        # ------------------------------------------------------------------
        self.basic_settings_form = QFormLayout()
        self.basic_settings_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        self.basic_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.basic_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.basic_settings_form.setFormAlignment(Qt.AlignmentFlag.AlignRight)

        # Model selection — always visible (empty group set).
        self.model_combo_box = FocusedScrollSettingsComboBox(key='model_id')
        # `setEditable()` must be called before `addItems()` to preserve any
        # custom model that was set.
        # TODO: Make more clear you can use custom models by specifying a model path
        self.model_combo_box.setEditable(True)
        self.model_combo_box.addItems(self.get_local_model_paths())
        self.model_combo_box.addItems(MODELS)
        self._register(self.basic_settings_form, 'Model',
                       self.model_combo_box, groups=set())

        # Caption position — always visible (empty group set).
        self.caption_position_combo_box = FocusedScrollSettingsComboBox(
            key='caption_position')
        self.caption_position_combo_box.addItems(list(CaptionPosition))
        self._register(self.basic_settings_form, 'Caption position',
                       self.caption_position_combo_box, groups=set())

        # Prompt.
        self.prompt_text_edit = SettingsPlainTextEdit(key='prompt')
        set_text_edit_height(self.prompt_text_edit, 4)
        self._register(self.basic_settings_form, 'Prompt',
                       self.prompt_text_edit, {SettingGroup.PROMPT})

        # Caption start.
        self.caption_start_line_edit = SettingsLineEdit(key='caption_start')
        self.caption_start_line_edit.setClearButtonEnabled(True)
        self._register(self.basic_settings_form, 'Start caption with',
                       self.caption_start_line_edit,
                       {SettingGroup.CAPTION_START})

        # Remove tag separators.
        self.remove_tag_separators_check_box = SettingsBigCheckBox(
            key='remove_tag_separators', default=True)
        self._register(self.basic_settings_form,
                       'Remove tag separators in captions',
                       self.remove_tag_separators_check_box,
                       {SettingGroup.REMOVE_TAG_SEPARATORS}, label_position=LabelPosition.BESIDE)
        
        # Add seperator line
        self.basic_settings_form.addRow(HorizontalLine())

        # Device.
        self.device_combo_box = FocusedScrollSettingsComboBox(key='device')
        self.device_combo_box.addItems(list(CaptionDevice))
        self._register(self.basic_settings_form, 'Device',
                       self.device_combo_box, {SettingGroup.DEVICE})

        # Load in 4-bit — arrives as its own layout, so wrap it in a widget.
        self.load_in_4_bit_check_box = SettingsBigCheckBox(
            key='load_in_4_bit', default=True)
        self.load_in_4_bit_container = self._register(self.basic_settings_form, 'Load in 4-bit (requires bitsandbytes)',
                                                      self.load_in_4_bit_check_box, {SettingGroup.LOAD_IN_4_BIT}, label_position=LabelPosition.BESIDE)

        # WD Tagger settings — built as a sub-form, registered as one composite.
        self.wd_tagger_settings_form = QFormLayout()
        self.wd_tagger_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.wd_tagger_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.wd_tagger_settings_form.setContentsMargins(0, 0, 0, 0)

        self.show_probabilities_check_box = SettingsBigCheckBox(
            key='show_probabilities', default=True)
        self.wd_tagger_settings_form.addRow('Show probabilities',
                                             self.show_probabilities_check_box)

        self.min_probability_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='min_probability', default=0.4, minimum=0.01, maximum=1)
        self.min_probability_spin_box.setSingleStep(0.01)
        self.wd_tagger_settings_form.addRow('Minimum probability',
                                            self.min_probability_spin_box)

        self.max_tags_spin_box = FocusedScrollSettingsSpinBox(
            key='max_tags', default=50, minimum=1, maximum=999)
        self.wd_tagger_settings_form.addRow('Maximum tags',
                                            self.max_tags_spin_box)

        self.tags_to_exclude_text_edit = SettingsPlainTextEdit(
            key='tags_to_exclude')
        set_text_edit_height(self.tags_to_exclude_text_edit, 3)
        self.tags_to_exclude_container = self._position_label('Tags to exclude', self.tags_to_exclude_text_edit, label_position=LabelPosition.ABOVE)
        self.wd_tagger_settings_form.addRow(self.tags_to_exclude_container)

        self.wd_tagger_settings_container = QWidget()
        self.wd_tagger_settings_container.setLayout(
            self.wd_tagger_settings_form)
        self._register(self.basic_settings_form, None,
                       self.wd_tagger_settings_container,
                       {SettingGroup.WD_TAGGER})

        self.addLayout(self.basic_settings_form)

        # ------------------------------------------------------------------
        # Advanced settings toggle
        # ------------------------------------------------------------------
        self.toggle_advanced_settings_button = QPushButton(
            'Show Advanced Settings')
        self.toggle_advanced_settings_button.setCheckable(True)
        self.toggle_advanced_settings_button.clicked.connect(
            self.toggle_advanced_settings)
        self.addWidget(self.toggle_advanced_settings_button)

        # ------------------------------------------------------------------
        # Advanced settings
        # ------------------------------------------------------------------
        self.advanced_settings_form = QFormLayout()
        self.advanced_settings_form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft)
        self.advanced_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Banned words
        self.bad_words_line_edit = SettingsLineEdit(key='bad_words')
        self._register(self.advanced_settings_form, 'Discourage from caption',
                       self.bad_words_line_edit, {SettingGroup.DISCOURAGED_WORDS}, label_position=LabelPosition.ABOVE)

        # Forced words
        self.forced_words_line_edit = SettingsLineEdit(key='forced_words')
        self._register(self.advanced_settings_form, 'Include in caption',
                       self.forced_words_line_edit, {SettingGroup.FORCED_WORDS}, label_position=LabelPosition.ABOVE)

        # Add seperator line
        self.advanced_settings_form.addRow(HorizontalLine())
        
        # Min / max new tokens.
        self.min_new_tokens_spin_box = FocusedScrollSettingsSpinBox(
            key='min_new_tokens', default=1, minimum=1, maximum=4096)
        self._register(self.advanced_settings_form, 'Minimum tokens',
                       self.min_new_tokens_spin_box,
                       {SettingGroup.MIN_TOKENS}, is_advanced=True)

        self.max_new_tokens_spin_box = FocusedScrollSettingsSpinBox(
            key='max_new_tokens', default=512, minimum=1, maximum=4096)
        self._register(self.advanced_settings_form, 'Maximum tokens',
                       self.max_new_tokens_spin_box,
                       {SettingGroup.MAX_TOKENS}, is_advanced=True)

        # Number of beams.
        self.num_beams_spin_box = FocusedScrollSettingsSpinBox(
            key='num_beams', default=1, minimum=1, maximum=100)
        self._register(self.advanced_settings_form, 'Number of beams',
                       self.num_beams_spin_box,
                       {SettingGroup.NUM_BEAMS}, is_advanced=True)

        # Length penalty.
        self.length_penalty_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='length_penalty', default=1, minimum=-5, maximum=5)
        self.length_penalty_spin_box.setSingleStep(0.1)
        self._register(self.advanced_settings_form, 'Length penalty',
                       self.length_penalty_spin_box,
                       {SettingGroup.LENGTH_PENALTY}, is_advanced=True)

        # Sampling — shared by transformers and llama.
        self.use_sampling_check_box = SettingsBigCheckBox(
            key='do_sample', default=False)
        self._register(self.advanced_settings_form, 'Use sampling',
                       self.use_sampling_check_box,
                       {SettingGroup.USE_SAMPLING}, is_advanced=True)

        self.temperature_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='temperature', default=1, minimum=0.01, maximum=2)
        self.temperature_spin_box.setSingleStep(0.01)
        self._register(self.advanced_settings_form, 'Temperature',
                       self.temperature_spin_box,
                       {SettingGroup.TEMPERATURE}, is_advanced=True)

        self.top_k_spin_box = FocusedScrollSettingsSpinBox(
            key='top_k', default=64, minimum=0, maximum=200)
        self._register(self.advanced_settings_form, 'Top-k',
                       self.top_k_spin_box,
                       {SettingGroup.TOP_K}, is_advanced=True)

        self.top_p_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='top_p', default=0.95, minimum=0, maximum=1)
        self.top_p_spin_box.setSingleStep(0.01)
        self._register(self.advanced_settings_form, 'Top-p',
                       self.top_p_spin_box,
                       {SettingGroup.TOP_P}, is_advanced=True)

        # Repetition penalty — shared by transformers and llama.
        self.repetition_penalty_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='repetition_penalty', default=1, minimum=1, maximum=2)
        self.repetition_penalty_spin_box.setSingleStep(0.01)
        self._register(self.advanced_settings_form, 'Repetition penalty',
                       self.repetition_penalty_spin_box,
                       {SettingGroup.REPETITION_PENALTY}, is_advanced=True)

        # No-repeat n-gram size.
        self.no_repeat_ngram_size_spin_box = FocusedScrollSettingsSpinBox(
            key='no_repeat_ngram_size', default=3, minimum=0, maximum=100)
        self._register(self.advanced_settings_form, 'No-repeat n-gram size',
                       self.no_repeat_ngram_size_spin_box,
                       {SettingGroup.NO_REPEAT_NGRAM}, is_advanced=True)

        # Add seperator line
        self._register(self.advanced_settings_form, None, HorizontalLine(), {SettingGroup.GPU_INDEX}, is_advanced=True)

        # GPU index.
        self.gpu_index_spin_box = FocusedScrollSettingsSpinBox(
            key='gpu_index', default=0, minimum=0, maximum=100)
        self._register(self.advanced_settings_form, 'GPU index',
                       self.gpu_index_spin_box, {SettingGroup.GPU_INDEX})
        # Add advanced settings
        self.advanced_settings_container = QWidget()
        self.advanced_settings_container.setLayout(self.advanced_settings_form)
        self.addWidget(self.advanced_settings_container)

        # ------------------------------------------------------------------
        # Wiring
        # ------------------------------------------------------------------
        self.device_combo_box.currentTextChanged.connect(
            self.set_load_in_4_bit_visibility)
        self.model_combo_box.currentTextChanged.connect(
            self.show_settings_for_model)

        # Initial state: advanced collapsed, correct groups for current model.
        # show_settings_for_model computes container visibility itself, so no
        # separate setVisible(False) is needed (and it would be immediately
        # overwritten anyway).
        self.show_settings_for_model(self.model_combo_box.currentText())

    @staticmethod
    def _position_label(label_text: str, field: QWidget, label_position: LabelPosition = LabelPosition.BESIDE) -> QWidget:
        """Stack a label besides its field inside a single widget so it can be
        added to a QFormLayout as one spanning row (label-above layout), regardless
        of the form's row-wrap policy."""
        label = QLabel(label_text)
        if label_position is LabelPosition.ABOVE:
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(label)
            layout.addWidget(field)
        if label_position is LabelPosition.BESIDE:
            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(field)
            layout.addWidget(label)
        layout.addStretch()
        container = QWidget()
        container.setLayout(layout)
        return container

    # TODO: Use the label from the enum value
    def _register(self, form: QFormLayout, label: str | None, field: QWidget,
                  groups: set[SettingGroup], is_advanced: bool = False,
                  label_position: LabelPosition = LabelPosition.DEFAULT):
        if label is not None and label_position is not LabelPosition.DEFAULT:
            # Build our own stacked label+field and add it as a spanning row.
            row_widget = self._position_label(label, field, label_position=label_position)
            form.addRow(row_widget)
            registered_field = row_widget
        elif label is None:
            form.addRow(field)
            registered_field = field
        else:
            form.addRow(label, field)
            registered_field = field
        self.registered_rows.append(
            (form, registered_field, groups, is_advanced))
        return registered_field

    def _active_groups(self) -> set[SettingGroup] | None:
        """Return the setting groups for the currently selected model, or
        None if there is no valid model selected yet."""
        model_id = self.model_combo_box.currentText()
        if not model_id:
            return None
        model_class = get_model_class(model_id)
        if model_class is None:
            return None
        return model_class.get_setting_groups()

    def get_local_model_paths(self) -> list[str]:
        models_directory_path = self.settings.value(
            'models_directory_path',
            defaultValue=DEFAULT_SETTINGS['models_directory_path'], type=str)
        if not models_directory_path:
            return []
        models_directory_path = Path(models_directory_path)
        print(f'Loading local auto-captioning model paths under '
              f'{models_directory_path}...')
        # Auto-captioning models have a `config.json` file.
        config_paths = set(models_directory_path.glob('**/config.json'))
        # WD Tagger models have a `selected_tags.csv` file.
        selected_tags_paths = set(
            models_directory_path.glob('**/selected_tags.csv'))
        model_directory_paths = [str(path.parent) for path
                                 in config_paths | selected_tags_paths]
        model_directory_paths.sort()
        print(f'Loaded {len(model_directory_paths)} model '
              f'{pluralize("path", len(model_directory_paths))}.')
        return model_directory_paths

    @Slot(str)
    def show_settings_for_model(self, model_id: str):
        active_groups = self._active_groups()
        if active_groups is None:
            # No valid model yet: hide the advanced affordances and leave the
            # always-visible rows (empty group set) showing.
            self.advanced_expanded = False
            self.toggle_advanced_settings_button.setChecked(False)
            self.toggle_advanced_settings_button.setText(
                'Show Advanced Settings')
            self.toggle_advanced_settings_button.setVisible(False)
            self.advanced_settings_container.setVisible(False)
            for form, field, groups, _is_advanced in self.registered_rows:
                form.setRowVisible(field, not groups)
            return

        # Does this model have any relevant advanced rows?
        has_advanced = any(
            is_advanced and ((not groups) or bool(groups & active_groups))
            for _form, _field, groups, is_advanced in self.registered_rows)

        if not has_advanced:
            self.advanced_expanded = False
            self.toggle_advanced_settings_button.setChecked(False)
            self.toggle_advanced_settings_button.setText(
                'Show Advanced Settings')

        # Toggle button + container only exist when there's something to show.
        self.toggle_advanced_settings_button.setVisible(has_advanced)
        self.advanced_settings_container.setVisible(
            has_advanced and self.advanced_expanded)

        for form, field, groups, _is_advanced in self.registered_rows:
            visible = (not groups) or bool(groups & active_groups)
            form.setRowVisible(field, visible)

        # bitsandbytes / device override for the load-in-4-bit row.
        self.set_load_in_4_bit_visibility(self.device_combo_box.currentText())

    @Slot(str)
    def set_load_in_4_bit_visibility(self, device: str):
        active_groups = self._active_groups()
        if active_groups is None:
            self.basic_settings_form.setRowVisible(
                self.load_in_4_bit_container, False)
            return
        visible = (SettingGroup.LOAD_IN_4_BIT in active_groups
                   and device == CaptionDevice.GPU
                   and BITSANDBYTES_AVAILABLE)
        self.basic_settings_form.setRowVisible(
            self.load_in_4_bit_container, visible)

    @Slot()
    def toggle_advanced_settings(self):
        # Ignore toggles when there is nothing to show (button should be hidden
        # in that case, but stay defensive).
        if not self.toggle_advanced_settings_button.isVisible():
            return
        self.advanced_expanded = not self.advanced_expanded
        self.toggle_advanced_settings_button.setText(
            'Hide Advanced Settings' if self.advanced_expanded
            else 'Show Advanced Settings')
        self.toggle_advanced_settings_button.setChecked(self.advanced_expanded)
        self.advanced_settings_container.setVisible(self.advanced_expanded)

    def get_caption_settings(self) -> dict:
        return {
            'model_id': self.model_combo_box.currentText(),
            'prompt': self.prompt_text_edit.toPlainText(),
            'caption_start': self.caption_start_line_edit.text(),
            'caption_position': self.caption_position_combo_box.currentText(),
            'device': self.device_combo_box.currentText(),
            'gpu_index': self.gpu_index_spin_box.value(),
            'load_in_4_bit': self.load_in_4_bit_check_box.isChecked(),
            'remove_tag_separators':
                self.remove_tag_separators_check_box.isChecked(),
            'bad_words': self.bad_words_line_edit.text(),
            'forced_words': self.forced_words_line_edit.text(),
            'generation_parameters': {
                'min_new_tokens': self.min_new_tokens_spin_box.value(),
                'max_new_tokens': self.max_new_tokens_spin_box.value(),
                'num_beams': self.num_beams_spin_box.value(),
                'length_penalty': self.length_penalty_spin_box.value(),
                'do_sample': self.use_sampling_check_box.isChecked(),
                'temperature': self.temperature_spin_box.value(),
                'top_k': self.top_k_spin_box.value(),
                'top_p': self.top_p_spin_box.value(),
                'repetition_penalty': self.repetition_penalty_spin_box.value(),
                'no_repeat_ngram_size':
                    self.no_repeat_ngram_size_spin_box.value()
            },
            'wd_tagger_settings': {
                'show_probabilities':
                    self.show_probabilities_check_box.isChecked(),
                'min_probability': self.min_probability_spin_box.value(),
                'max_tags': self.max_tags_spin_box.value(),
                'tags_to_exclude':
                    self.tags_to_exclude_text_edit.toPlainText()
            }
        }


@Slot()
def restore_stdout_and_stderr():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


class AutoCaptioner(QDockWidget):
    caption_generated = Signal(QModelIndex, str, list)

    def __init__(self, image_list_model: ImageListModel,
                 image_list: ImageList):
        super().__init__()
        self.image_list_model = image_list_model
        self.image_list = image_list
        self.settings = get_settings()
        self.is_captioning = False
        self.captioning_thread = None
        self.model_manager = ModelSubprocessManager()
        # Ensure there are no orphaned processes on shutdown
        QApplication.instance().aboutToQuit.connect(self.model_manager.shutdown)
        # Whether the last block of text in the console text edit should be
        # replaced with the next block of text that is outputted.
        self.replace_last_console_text_edit_block = False

        # Each `QDockWidget` needs a unique object name for saving its state.
        self.setObjectName('auto_captioner')
        self.setWindowTitle('Auto-Captioner')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)
        # TODO Pin the start button so it's always visible, and make it smaller
        self.start_cancel_button = TallPushButton('Start Auto-Captioning')
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat('%v / %m images captioned (%p%)')
        self.progress_bar.hide()
        self.console_text_edit = QPlainTextEdit()
        set_text_edit_height(self.console_text_edit, 4)
        self.console_text_edit.setReadOnly(True)
        self.console_text_edit.hide()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.start_cancel_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.console_text_edit)
        self.caption_settings_form = CaptionSettingsForm()
        layout.addLayout(self.caption_settings_form)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(container)
        self.setWidget(scroll_area)

        self.start_cancel_button.clicked.connect(
            self.start_or_cancel_captioning)

    @Slot()
    def start_or_cancel_captioning(self):
        if self.is_captioning:
            # Cancel captioning.
            self.captioning_thread.is_canceled = True
            self.model_manager.cancel()
            self.start_cancel_button.setEnabled(False)
            self.start_cancel_button.setText('Canceling Auto-Captioning...')
        else:
            # Start captioning.
            self.generate_captions()

    def set_is_captioning(self, is_captioning: bool):
        self.is_captioning = is_captioning
        button_text = ('Cancel Auto-Captioning' if is_captioning
                       else 'Start Auto-Captioning')
        self.start_cancel_button.setText(button_text)

    @Slot(str)
    def update_console_text_edit(self, text: str):
        # '\x1b[A' is the ANSI escape sequence for moving the cursor up.
        if text == '\x1b[A':
            self.replace_last_console_text_edit_block = True
            return
        text = text.strip()
        if not text:
            return
        if self.console_text_edit.isHidden():
            self.console_text_edit.show()
        if self.replace_last_console_text_edit_block:
            self.replace_last_console_text_edit_block = False
            # Select and remove the last block of text.
            self.console_text_edit.moveCursor(QTextCursor.MoveOperation.End)
            self.console_text_edit.moveCursor(
                QTextCursor.MoveOperation.StartOfBlock,
                QTextCursor.MoveMode.KeepAnchor)
            self.console_text_edit.textCursor().removeSelectedText()
            # Delete the newline.
            self.console_text_edit.textCursor().deletePreviousChar()
        self.console_text_edit.appendPlainText(text)

    @Slot()
    def show_alert(self):
        if self.captioning_thread.is_canceled:
            return
        if self.captioning_thread.is_error:
            icon = QMessageBox.Icon.Critical
            text = ('An error occurred during captioning. See the '
                    'Auto-Captioner console for more information.')
        else:
            icon = QMessageBox.Icon.Information
            text = 'Captioning has finished.'
        alert = QMessageBox()
        alert.setIcon(icon)
        alert.setText(text)
        alert.exec()

    @Slot()
    def generate_captions(self):
        selected_image_indices = self.image_list.get_selected_image_indices()
        selected_image_count = len(selected_image_indices)
        show_alert_when_finished = False
        if selected_image_count > 1:
            confirmation_dialog = CaptionMultipleImagesDialog(
                selected_image_count)
            reply = confirmation_dialog.exec()
            if reply != QMessageBox.StandardButton.Yes:
                return
            show_alert_when_finished = (confirmation_dialog
                                        .show_alert_check_box.isChecked())
        self.set_is_captioning(True)
        caption_settings = self.caption_settings_form.get_caption_settings()
        if caption_settings['caption_position'] != CaptionPosition.DO_NOT_ADD:
            self.image_list_model.add_to_undo_stack(
                action_name=f'Generate '
                            f'{pluralize("Caption", selected_image_count)}',
                should_ask_for_confirmation=selected_image_count > 1)
        if selected_image_count > 1:
            self.progress_bar.setRange(0, selected_image_count)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
        tag_separator = get_tag_separator()
        models_directory_path = self.settings.value(
            'models_directory_path',
            defaultValue=DEFAULT_SETTINGS['models_directory_path'], type=str)
        models_directory_path = (Path(models_directory_path)
                                 if models_directory_path else None)
        self.captioning_thread = CaptioningThread(
            self, self.image_list_model, selected_image_indices,
            caption_settings, tag_separator, models_directory_path)
        self.captioning_thread.text_outputted.connect(
            self.update_console_text_edit)
        self.captioning_thread.clear_console_text_edit_requested.connect(
            self.console_text_edit.clear)
        self.captioning_thread.caption_generated.connect(
            self.caption_generated)
        self.captioning_thread.progress_bar_update_requested.connect(
            self.progress_bar.setValue)
        self.captioning_thread.finished.connect(
            lambda: self.set_is_captioning(False))
        self.captioning_thread.finished.connect(restore_stdout_and_stderr)
        self.captioning_thread.finished.connect(self.progress_bar.hide)
        self.captioning_thread.finished.connect(
            lambda: self.start_cancel_button.setEnabled(True))
        if show_alert_when_finished:
            self.captioning_thread.finished.connect(self.show_alert)
        # Redirect `stdout` and `stderr` so that the outputs are displayed in
        # the console text edit.
        sys.stdout = self.captioning_thread
        sys.stderr = self.captioning_thread
        self.captioning_thread.start()


    def closeEvent(self, event):
        self.model_manager.shutdown()
        super().closeEvent(event)
