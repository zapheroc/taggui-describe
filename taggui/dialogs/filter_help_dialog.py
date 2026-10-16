from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog,
                               QHeaderView, QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)


# (Example filter, what it matches)
EXAMPLES = [
    ('tree', 'Partial match on the tags or the file path'),
    ('"no humans"', 'Phrases with spaces must be quoted'),
    ('tag:tree', 'Matches an image that has the tag "tree"'),
    ('tag:tree*', '* matches any characters, ? matches exactly one'),
    ('caption:watercraft', 'Matches if the joined tags contain "watercraft"'),
    ('desc:blue', 'Matches if the description contains "blue"'),
    ('name:Kirito', 'Matches if the file name contains "Kirito"'),
    ('path:2024', 'Matches if the file path contains "2024"'),
    ('tree AND cat', 'Both filters must match'),
    ('tree OR cat', 'Either filter may match'),
    ('NOT cat', 'Matches images that do not match "cat"'),
    ('(tree OR cat) AND bird', 'Parentheses group sub-filters'),
    ('tags:=3', 'Exactly 3 tags'),
    ('tags:>=3', '3 or more tags'),
    ('chars:<100', 'Tag text shorter than 100 characters'),
    ('tokens:<=75', '75 tokens or fewer (the Stable Diffusion prompt limit)'),
]

DEFAULT_WIDTH = 800


class FilterHelpDialog(QDialog):
    """
    A reference for the filter syntax used in the image list filter field.
    Double-clicking an example copies it to the clipboard.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Filter Syntax')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        description_label = QLabel(
            'Filter the image list by tags, captions, descriptions, file '
            'names, or paths. A plain word is a partial match on the tags '
            'or the file path.')
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        self.examples_table = QTableWidget(len(EXAMPLES), 2)
        self.examples_table.setHorizontalHeaderLabels(['Example',
                                                        'What it does'])
        self.examples_table.verticalHeader().setVisible(False)
        self.examples_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.examples_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.examples_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.examples_table.setShowGrid(False)
        for row, (example, description) in enumerate(EXAMPLES):
            example_item = QTableWidgetItem(example)
            description_item = QTableWidgetItem(description)
            description_item.setFlags(description_item.flags() &
                                       ~Qt.ItemFlag.ItemIsSelectable)
            self.examples_table.setItem(row, 0, example_item)
            self.examples_table.setItem(row, 1, description_item)
        header = self.examples_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        # Show every example without scrolling. The row height is not
        # known until the dialog has been laid out, so this is corrected
        # in `_fit_table_height`.
        self.examples_table.setFixedHeight(
            self.examples_table.rowHeight(0) * len(EXAMPLES)
            + header.height()
            + 2 * self.examples_table.frameWidth())
        self.examples_table.cellDoubleClicked.connect(self.copy_example)
        layout.addWidget(self.examples_table)
        QTimer.singleShot(0, self._fit_table_height)

        self.status_label = QLabel('Double-click an example to copy it.')
        layout.addWidget(self.status_label)

        notes_label = QLabel(
            'Combine filters with AND, OR, and NOT (NOT binds tightest, then '
            'AND, then OR). Use =, ==, !=, <, >, <=, or >= for the tag count '
            '(tags:), caption length in characters (chars:), and token count '
            '(tokens:).')
        notes_label.setWordWrap(True)
        layout.addWidget(notes_label)

        close_button = QPushButton('Close')
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.resize(DEFAULT_WIDTH, 600)

    def _fit_table_height(self):
        """
        Set the table height to fit all the rows once the real row height
        is known (it only exists after the first layout pass). The viewport
        holds the header and the rows, so the widget height is that plus
        the frame.
        """
        self.examples_table.setFixedHeight(
            self.examples_table.rowHeight(0) * len(EXAMPLES)
            + self.examples_table.horizontalHeader().height()
            + 2 * self.examples_table.frameWidth())
        # Size the dialog so that every widget fits. Some platforms do not
        # propagate the layout's minimum size to the window, so set it
        # explicitly once the table height is final.
        self.resize(DEFAULT_WIDTH, self.layout().sizeHint().height())

    def copy_example(self, row: int, column: int):
        example = self.examples_table.item(row, 0).text()
        QApplication.clipboard().setText(example)
        self.status_label.setText(f'Copied "{example}"')
