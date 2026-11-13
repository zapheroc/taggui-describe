import hashlib
import os
import platform
import shutil
import sqlite3
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from utils.image import Image
from utils.settings import (DEFAULT_SETTINGS, get_settings,
                            get_tag_separator)

# The version of the database schema currently in use. A database that was
# created (or last upgraded) at version N has `PRAGMA user_version` set to N.
SCHEMA_VERSION = 1

# The size (in bytes) of the chunks read when computing content hashes.
HASH_CHUNK_SIZE = 1024 * 1024

# One entry per schema version: `_MIGRATIONS[n]` is the SQL that upgrades a
# database to version n + 1, applied in its own transaction. A fresh database
# has `PRAGMA user_version` 0. Future schema changes are appended here.
#
# All statements use IF NOT EXISTS: the directory load runs one task per
# image on worker threads, and every task opens its own connection (which
# migrates on open). With `IF NOT EXISTS`, concurrent first connections
# simply skip the objects another connection already created, instead of
# failing. It also repairs a database whose last migration was interrupted.
_MIGRATIONS = [
    # Version 1.
    (
        # Each image's record, keyed by the stable hash of its content.
        ('CREATE TABLE IF NOT EXISTS images ('
         'hash        TEXT PRIMARY KEY, '
         'description TEXT NOT NULL DEFAULT \'\')'),
        # The deduplicated set of all tag names (case-sensitive, as in the
        # caption files).
        ('CREATE TABLE IF NOT EXISTS tags ('
         'id   INTEGER PRIMARY KEY AUTOINCREMENT, '
         'name TEXT NOT NULL UNIQUE)'),
        # Which tags an image has, and in which order.
        ('CREATE TABLE IF NOT EXISTS image_tags ('
         'image_hash TEXT    NOT NULL '
         'REFERENCES images(hash) ON DELETE CASCADE, '
         'tag_id     INTEGER NOT NULL REFERENCES tags(id), '
         'position   INTEGER NOT NULL, '
         'PRIMARY KEY (image_hash, position))'),
        # Tag name -> images, for tag-based lookups and filtering.
        'CREATE INDEX IF NOT EXISTS image_tags_tag_id ON image_tags(tag_id)',
    ),
]


def compute_content_hash(image_path: Path) -> str:
    """
    Compute a stable blake2b hash of the image file's content.

    The same bytes always produce the same hash, regardless of the file's
    name or location, so the database can identify an image by its content
    rather than its path.
    """
    hash_object = hashlib.blake2b()
    with open(image_path, 'rb') as image_file:
        while chunk := image_file.read(HASH_CHUNK_SIZE):
            hash_object.update(chunk)
    return hash_object.hexdigest()


def get_storage_backend() -> str:
    """The active storage backend setting: 'sqlite' or 'text'."""
    settings = get_settings()
    return settings.value('storage_backend',
                          defaultValue=DEFAULT_SETTINGS['storage_backend'],
                          type=str)


def get_default_database_path() -> Path:
    """
    The platform-appropriate default location of the database file.
    """
    if platform.system() == 'Windows':
        local_app_data = os.environ.get('LOCALAPPDATA')
        if local_app_data:
            base_path = Path(local_app_data)
        else:
            base_path = Path.home() / 'AppData' / 'Local'
        return base_path / 'TagGUI-Describe' / 'taggui-describe.sqlite3'
    if platform.system() == 'Darwin':
        return (Path.home() / 'Library' / 'Application Support'
                / 'TagGUI-Describe' / 'taggui-describe.sqlite3')
    # Linux and other Unix systems.
    data_home = os.environ.get('XDG_DATA_HOME')
    if data_home:
        base_path = Path(data_home)
    else:
        base_path = Path.home() / '.local' / 'share'
    return base_path / 'taggui-describe' / 'taggui-describe.sqlite3'


def get_database_path() -> Path:
    """The active database location (the setting, or the default)."""
    settings = get_settings()
    default_path = DEFAULT_SETTINGS['database_path']
    database_path = settings.value('database_path',
                                   defaultValue=default_path, type=str)
    if database_path:
        return Path(database_path)
    return get_default_database_path()


def read_caption_file(image_path: Path,
                      tag_separator: str) -> tuple[list[str], str]:
    """
    Read the caption file `<image>.txt` if it exists, and parse it into tags
    and description.

    Returns empty data when there is no caption file or it cannot be read —
    a failed read must never hide an image from the list.
    """
    text_file_path = image_path.with_suffix('.txt')
    if not text_file_path.is_file():
        return [], ''
    try:
        # `errors='replace'` inserts a replacement marker such as
        # '?' when there is malformed data.
        caption = text_file_path.read_text(encoding='utf-8', errors='replace')
    except OSError as exception:
        # The caption file could not be read (for example, it was deleted
        # after the directory was scanned or it is not readable). Load the
        # image without tags or a description instead of silently hiding it
        # from the list. The user can still tag the image, which (re)writes
        # its storage.
        print(f'Failed to read caption file {text_file_path}: {exception}',
              file=sys.stderr)
        return [], ''
    tags_raw, _, description = caption.partition('\n\n')
    if tags_raw:
        tags = [tag for tag in (part.strip() for part in
                                tags_raw.split(tag_separator)) if tag]
    else:
        tags = []
    return tags, description.strip()


def write_caption_file(image_path: Path, tags: list[str], description: str,
                       tag_separator: str) -> Path:
    """
    Write the tags and description to the caption file next to the image,
    in the standard format (tags joined by the tag separator, then a blank
    line, then the description). Return the path of the caption file.
    """
    text_file_path = image_path.with_suffix('.txt')
    image_text = tag_separator.join(tags) + '\n\n' + description
    text_file_path.write_text(image_text, encoding='utf-8', errors='replace')
    return text_file_path


def _connect(db_path: Path) -> sqlite3.Connection:
    """
    Open a connection to the database at `db_path`, creating the file (and
    its directory) if necessary, and apply any pending schema migrations.

    A fresh connection is used for every operation, so the GUI thread and
    the directory-load worker threads never share one.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.execute('PRAGMA foreign_keys = ON')
    connection.execute('PRAGMA busy_timeout = 5000')
    _apply_migrations(connection)
    return connection


# Serializes schema migrations across the worker threads of a directory
# load: every connection migrates on open, so without the lock the first
# loads of a fresh database would race on the DDL.
_MIGRATION_LOCK = threading.Lock()


def _apply_migrations(connection: sqlite3.Connection):
    """
    Bring the database schema up to `SCHEMA_VERSION`, one step per version,
    each step applied in its own transaction so that a failed step leaves
    the schema exactly as it was.
    """
    with _MIGRATION_LOCK:
        current_version = connection.execute(
            'PRAGMA user_version').fetchone()[0]
        for version in range(current_version + 1, SCHEMA_VERSION + 1):
            connection.execute('BEGIN')
            try:
                for statement in _MIGRATIONS[version - 1]:
                    connection.execute(statement)
                connection.execute(f'PRAGMA user_version = {version}')
            except Exception:
                connection.execute('ROLLBACK')
                raise
            connection.execute('COMMIT')


def _upsert_image(connection: sqlite3.Connection, content_hash: str,
                  tags: list[str], description: str):
    """
    Insert or update one image's record: its description and its ordered
    list of tags (deduplicated by name, with their positions). The caller
    controls the transaction.
    """
    connection.execute(
        'INSERT INTO images (hash, description) VALUES (?, ?) '
        'ON CONFLICT (hash) DO UPDATE SET description = excluded.description',
        (content_hash, description))
    connection.execute('DELETE FROM image_tags WHERE image_hash = ?',
                       (content_hash,))
    for position, tag in enumerate(tags):
        connection.execute('INSERT OR IGNORE INTO tags (name) VALUES (?)',
                           (tag,))
        tag_id = connection.execute('SELECT id FROM tags WHERE name = ?',
                                    (tag,)).fetchone()[0]
        connection.execute(
            'INSERT INTO image_tags (image_hash, tag_id, position) '
            'VALUES (?, ?, ?)', (content_hash, tag_id, position))


class ImageStorage(ABC):
    """Read and write the tags and descriptions of the loaded images."""

    @abstractmethod
    def load_image(self, image_path: Path) -> tuple[list[str], str, str | None]:
        """
        Load the tags, description and (when applicable) content hash of one
        image.
        """

    @abstractmethod
    def save_image(self, image: Image) -> None:
        """Save the tags and description of one image."""


class TextFileStorage(ImageStorage):
    """
    Store each image's tags and description in a caption file
    (`<stem>.txt`) next to the image, the historical TagGUI format.
    """

    def __init__(self, tag_separator: str):
        self.tag_separator = tag_separator

    def load_image(self, image_path: Path) -> tuple[list[str], str, str | None]:
        tags, description = read_caption_file(image_path, self.tag_separator)
        return tags, description, None

    def save_image(self, image: Image) -> None:
        write_caption_file(image.path, image.tags, image.description,
                           self.tag_separator)


class SqliteStorage(ImageStorage):
    """
    Store all images' tags and descriptions in a SQLite database, keyed by
    the image's content hash (the same image content retrieves the same
    tags and description, no matter which folder it is opened from).
    """

    def __init__(self, db_path: Path, tag_separator: str):
        self.db_path = db_path
        self.tag_separator = tag_separator

    def load_image(self, image_path: Path) -> tuple[list[str], str, str | None]:
        try:
            content_hash = compute_content_hash(image_path)
        except OSError as exception:
            # The image file is gone or unreadable (it was deleted or moved
            # after the directory was scanned). Load it without data instead
            # of hiding it from the list.
            print(f'Failed to read {image_path}: {exception}',
                  file=sys.stderr)
            return [], '', None
        try:
            connection = _connect(self.db_path)
            try:
                record = connection.execute(
                    'SELECT description FROM images WHERE hash = ?',
                    (content_hash,)).fetchone()
                if record is not None:
                    tags = [row[0] for row in connection.execute(
                        'SELECT tags.name FROM image_tags '
                        'JOIN tags ON tags.id = image_tags.tag_id '
                        'WHERE image_tags.image_hash = ? '
                        'ORDER BY image_tags.position',
                        (content_hash,))]
                    return tags, record[0], content_hash
                # There is no record yet. If a caption file exists, import
                # its data into the database so that it is not lost. This is
                # additive only: it never overwrites an existing record.
                tags, description = read_caption_file(
                    image_path, self.tag_separator)
                if tags or description:
                    with connection:
                        _upsert_image(connection, content_hash, tags,
                                      description)
                return tags, description, content_hash
            finally:
                connection.close()
        except (OSError, sqlite3.Error) as exception:
            # The database is unavailable (for example, it is corrupted or
            # on an unreadable filesystem). Fall back to the caption file so
            # that the image is still visible. The next save retries against
            # the database, and reports the failure if it persists.
            print(f'Failed to load {image_path} from the database: '
                  f'{exception}', file=sys.stderr)
            tags, description = read_caption_file(image_path,
                                                  self.tag_separator)
            return tags, description, content_hash

    def save_image(self, image: Image) -> None:
        content_hash = (image.content_hash
                        or compute_content_hash(image.path))
        connection = _connect(self.db_path)
        try:
            with connection:
                _upsert_image(connection, content_hash, image.tags,
                              image.description)
        finally:
            connection.close()


def clear_database(db_path: Path) -> int:
    """
    Delete all records from the database at `db_path`. The database file
    and its schema remain and stay usable (nothing else is deleted). Return
    the number of image records that were cleared.
    """
    connection = _connect(db_path)
    try:
        with connection:
            cleared_count = connection.execute(
                'SELECT COUNT(*) FROM images').fetchone()[0]
            # The child tables first, so the foreign keys stay satisfied.
            connection.execute('DELETE FROM image_tags')
            connection.execute('DELETE FROM tags')
            connection.execute('DELETE FROM images')
        return cleared_count
    finally:
        connection.close()


def import_images(images: list[Image], db_path: Path) -> None:
    """
    Import the given images' tags and descriptions into the database at
    `db_path`, overwriting any existing records for those images. The
    caller is expected to pass only images with non-empty tags or a
    description, so records of images without data are left untouched.
    """
    connection = _connect(db_path)
    try:
        with connection:
            for image in images:
                content_hash = (image.content_hash
                                or compute_content_hash(image.path))
                _upsert_image(connection, content_hash, image.tags,
                              image.description)
    finally:
        connection.close()


def export_images_to_text_files(images: list[Image],
                                tag_separator: str) -> list[Path]:
    """
    Write the given images' tags and descriptions to caption files next to
    the images, overwriting any existing caption files. The caller is
    expected to pass only images with non-empty tags or a description, so
    no empty caption files are created. Return the paths of the images that
    failed to write.
    """
    failed_paths = []
    for image in images:
        try:
            write_caption_file(image.path, image.tags, image.description,
                               tag_separator)
        except OSError:
            failed_paths.append(image.path)
    return failed_paths


def migrate_database(old_path: Path, new_path: Path) -> None:
    """
    Move the database file from `old_path` to `new_path` (the caller
    confirms the overwrite when a file already exists at `new_path`).

    The file is copied, the copy is verified to be a byte-for-byte match
    with the original (size and content hash), and only then is the
    original deleted. If the copy cannot be verified, it is removed and
    the original is left in place (an OSError is raised), so a failed
    move never destroys the only copy of the database.
    """
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old_path, new_path)
    # Verify the copy before deleting the original: the size check fails
    # fast, the content hash proves the bytes match.
    if (old_path.stat().st_size != new_path.stat().st_size
            or compute_content_hash(old_path) != compute_content_hash(new_path)):
        # The copy is not a verified match, so remove it and keep the
        # original.
        try:
            new_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise OSError('The copy of the database could not be verified '
                      'against the original')
    old_path.unlink()


def get_image_storage() -> ImageStorage:
    """
    The storage backend that matches the current settings.
    """
    tag_separator = get_tag_separator()
    if get_storage_backend() == 'sqlite':
        return SqliteStorage(get_database_path(), tag_separator)
    return TextFileStorage(tag_separator)
