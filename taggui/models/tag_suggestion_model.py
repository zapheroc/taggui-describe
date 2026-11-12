from PySide6.QtCore import QAbstractListModel, Qt, Slot


class TagSuggestionModel(QAbstractListModel):
    """
    The tag suggestions offered by the Add Tag box's completer.

    Each suggestion is a `(tag, count)` pair, where `count` is the number of
    images that use the tag. When the SQLite database is the active storage
    backend the suggestions are all the tags stored in it, so every tag ever
    used is offered no matter which folder is open; otherwise they are the
    tags of the loaded images.
    """

    def __init__(self):
        super().__init__()
        self.suggestions: list[tuple[str, int]] = []

    def rowCount(self, parent=None) -> int:
        return len(self.suggestions)

    def data(self, index, role=None) -> str:
        tag, count = self.suggestions[index.row()]
        if role == Qt.ItemDataRole.EditRole:
            # The completer matches against and completes from this role
            # (its default completion role), so it must be the bare tag,
            # not the display string.
            return tag
        if role == Qt.ItemDataRole.DisplayRole:
            return f'{tag} ({count})'

    @Slot(list)
    def update_suggestions(self, suggestions: list[tuple[str, int]]):
        self.beginResetModel()
        self.suggestions = list(suggestions)
        self.endResetModel()
