import os
import platform
import shutil
import sqlite3
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import xxhash

from utils.image import Image
from utils.settings import (DEFAULT_SETTINGS, get_settings,
                            get_tag_separator)
from utils.utils import pluralize

# The version of the database schema currently in use. A database that was
# created (or last upgraded) at version N has `PRAGMA user_version` set to N.
SCHEMA_VERSION = 3

# The size (in bytes) of the chunks read when computing content hashes.
HASH_CHUNK_SIZE = 1024 * 1024

# One entry per schema version: `_MIGRATIONS[n]` is the SQL that upgrades a
# database to version n + 1, applied in its own transaction. A fresh database
# has `PRAGMA user_version` 0. Future schema changes are appended here.
#
# All statements use IF NOT EXISTS: the directory load runs one task per
# image on worker threads, and the first connection each thread opens
# migrates on open. With `IF NOT EXISTS`, concurrent first connections
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
    # Version 2.
    #
    # Historical version number: an earlier build of this app added a
    # `path_index` table at version 2, so databases created by it are at
    # `user_version` 2. The current code does not use that table (it is
    # left in place for such databases), so this step is deliberately
    # empty: it only advances those databases to version 3, where the hash
    # cache is created. Version number 2 must never be reused for new
    # objects: doing so is what made the hash-cache migration silently
    # skip pre-existing version-2 databases ("no such table: hash_cache").
    (),
    # Version 3.
    (
        # A cache of (path, size, mtime_ns, ctime_ns, inode) -> content hash,
        # so re-opening a folder whose files have not changed does not have
        # to hash every file again. The row is trusted only when all five
        # stat fields match together (ctime in particular defeats a byte
        # swap with a restored mtime, which userspace cannot forge); on a
        # miss the file is re-hashed and the row replaced.
        ('CREATE TABLE IF NOT EXISTS hash_cache ('
         'path     TEXT    PRIMARY KEY, '
         'size     INTEGER NOT NULL, '
         'mtime_ns INTEGER NOT NULL, '
         'ctime_ns INTEGER NOT NULL, '
         'inode    INTEGER NOT NULL, '
         'hash     TEXT    NOT NULL)'),
    ),
]


def compute_content_hash(image_path: Path) -> str:
    """
    Compute a stable xxh3-128 hash of the image file's content.

    The same bytes always produce the same hash, regardless of the file's
    name or location, so the database can identify an image by its content
    rather than its path.
    """
    hash_object = xxhash.xxh3_128()
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

    The repeated read path of a directory load reuses one connection per
    thread instead (see `_connect_reusable`); saves and the other
    one-shot operations use a fresh connection, so no operation ever shares
    a connection with a worker thread of a directory load.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.execute('PRAGMA foreign_keys = ON')
    connection.execute('PRAGMA busy_timeout = 5000')
    _apply_migrations(connection)
    return connection


# One reusable connection per thread, for the repeated read path of a
# directory load. It is keyed by the thread's `get_ident()`, which is stable
# across the tasks a QThreadPool worker thread runs. (`threading.local()` is
# deliberately not used: its per-thread state is intermittently lost on Qt's
# native worker threads, which silently defeated the reuse.) The stored key
# (device, inode) lets a replaced or deleted database file be detected and
# the cached connection discarded.
#
# Each thread owns its cached connection, so callers must never close it; it
# is closed only when it goes stale. In practice the set of load threads is
# the (bounded) QThreadPool worker set, so these dicts stay small; an entry
# for a thread that has exited is harmless (its connection is re-validated,
# and at most one open connection per such id lingers until process exit,
# which the operating system reclaims).
_reusable_connections: dict[int, sqlite3.Connection] = {}
_reusable_keys: dict[int, tuple[int, int] | None] = {}
_reusable_lock = threading.Lock()


def _connect_reusable(db_path: Path) -> sqlite3.Connection:
    """
    A per-thread connection to the database at `db_path`, kept open and
    reused by the calling thread.

    Opening a SQLite connection is part of the per-image cost of a
    directory load (it stats and locks the file and re-checks the schema
    every time), so the load path reuses one connection per thread. The
    cached connection is validated on every call: it is discarded and
    re-opened when the database file is missing or has been replaced (new
    device or inode), or no longer answers a trivial query.
    """
    thread_id = threading.get_ident()
    with _reusable_lock:
        connection = _reusable_connections.get(thread_id)
        key = _reusable_keys.get(thread_id)
    try:
        wanted = (db_path.stat().st_dev, db_path.stat().st_ino)
    except OSError:
        wanted = None  # The file is missing; `_connect` (re)creates it.
    if connection is not None and wanted is not None and key == wanted:
        try:
            connection.execute('SELECT 1')
            return connection
        except (sqlite3.Error, OSError):
            pass
    # Stale or absent: drop the cached entry (if it is still ours) and close
    # the connection we own, then open a fresh one.
    if connection is not None:
        with _reusable_lock:
            if _reusable_connections.get(thread_id) is connection:
                del _reusable_connections[thread_id]
                _reusable_keys.pop(thread_id, None)
        try:
            connection.close()
        except sqlite3.Error:
            pass
    connection = _connect(db_path)
    try:
        new_key = (db_path.stat().st_dev, db_path.stat().st_ino)
    except OSError:
        new_key = None
    with _reusable_lock:
        _reusable_connections[thread_id] = connection
        _reusable_keys[thread_id] = new_key
    return connection


# Serializes schema migrations across the worker threads of a directory
# load: every new connection migrates on open, so without the lock the
# first loads of a fresh database would race on the DDL.
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


# The `hash_cache` rows computed during a directory load are written in
# batches rather than one commit per image. A per-image commit makes every
# worker thread contend for the database's single write lock (and, in the
# default rollback journal mode, briefly blocks the other threads' reads),
# which dominated the cold-load cost. The buffer is flushed when it reaches
# `_HASH_CACHE_FLUSH_THRESHOLD` rows, and any remainder is flushed by
# `SqliteStorage.flush_hash_cache()` once the load completes. Losing the
# buffer (for example on a crash) is safe: the rows are derived data, and
# the affected files are simply re-hashed on the next load.
_hash_cache_pending: list[tuple] = []
_hash_cache_pending_lock = threading.Lock()
_HASH_CACHE_FLUSH_THRESHOLD = 256


def _flush_hash_cache_buffer(connection: sqlite3.Connection) -> None:
    """Write the pending `hash_cache` rows in a single transaction.

    Best-effort: on a write failure the rows are re-queued (so they are not
    lost and are retried on the next flush) and the error is not propagated,
    because a load must never fail to produce data over an unwritten cache.
    """
    with _hash_cache_pending_lock:
        if not _hash_cache_pending:
            return
        rows = _hash_cache_pending[:]
        _hash_cache_pending.clear()
    try:
        with connection:
            connection.executemany(
                'INSERT OR REPLACE INTO hash_cache '
                '(path, size, mtime_ns, ctime_ns, inode, hash) '
                'VALUES (?, ?, ?, ?, ?, ?)', rows)
    except (OSError, sqlite3.Error) as exception:
        with _hash_cache_pending_lock:
            _hash_cache_pending[:0] = rows
        print(f'Failed to write the hash cache: {exception}',
              file=sys.stderr)


def _buffer_hash_cache_row(connection: sqlite3.Connection, image_path: Path,
                           file_stat: os.stat_result, content_hash: str) -> None:
    """Queue a computed `hash_cache` row, flushing the buffer when it is full."""
    with _hash_cache_pending_lock:
        _hash_cache_pending.append(
            (str(image_path), file_stat.st_size, file_stat.st_mtime_ns,
             file_stat.st_ctime_ns, file_stat.st_ino, content_hash))
        should_flush = (len(_hash_cache_pending)
                        >= _HASH_CACHE_FLUSH_THRESHOLD)
    if should_flush:
        _flush_hash_cache_buffer(connection)


def _flush_hash_cache(db_path: Path) -> None:
    """
    Write the `hash_cache` rows buffered during a directory load (see
    `_flush_hash_cache_buffer`).

    Runs on the GUI thread after all the worker tasks of a load have
    finished, using a fresh connection so it never shares one with a
    still-live load worker thread. The buffer is module-global, so this
    drains rows from any in-flight load too — safe, because the rows are
    content-addressed and `INSERT OR REPLACE` is idempotent. A failed write
    leaves the rows queued (see `_flush_hash_cache_buffer`) and is not
    raised here: the load has already succeeded, and the rows are derived
    data.
    """
    with _hash_cache_pending_lock:
        pending = len(_hash_cache_pending) > 0
    if not pending:
        return
    connection = _connect(db_path)
    try:
        _flush_hash_cache_buffer(connection)
    finally:
        connection.close()


def _prune_hash_cache(db_path: Path, paths) -> None:
    """
    Drop the `hash_cache` rows of the given file paths (their files were
    removed).

    Best-effort: the cache is derived data, and a stale row is harmless
    (a file re-created at the same path has a new ctime, so it would not
    match a cached row), so a failure is reported but never raised.
    """
    paths = [str(path) for path in paths]
    if not paths:
        return
    connection = _connect(db_path)
    try:
        with connection:
            # `executemany` with one parameter per path avoids the limit on
            # the number of bound parameters of a single `IN (...)` clause.
            connection.executemany(
                'DELETE FROM hash_cache WHERE path = ?',
                [(path,) for path in paths])
    except (OSError, sqlite3.Error) as exception:
        print(f'Failed to prune the hash cache rows of '
              f'{len(paths)} {pluralize("path", len(paths))} '
              f'(first: {paths[0]}): {exception}', file=sys.stderr)
    finally:
        connection.close()


def _resolve_content_hash(connection: sqlite3.Connection, image_path: Path,
                          file_stat: os.stat_result) -> str:
    """
    The content hash of `image_path`, from the `hash_cache` table when the
    file's identity (path, size, mtime_ns, ctime_ns, inode) matches a cached
    row, and from a fresh hash computation otherwise (in which case the
    cache row is queued to be written in a batch — see
    `_buffer_hash_cache_row`).

    All five stat fields must match: the cache trusts them together, and in
    particular ctime cannot be restored from userspace, so a byte swap with
    a restored mtime still re-hashes. (The same class of trust as git's
    index stat cache: a false hit would require matching path + size +
    nanosecond-precision mtime and ctime + inode at the same time.)
    """
    row = connection.execute(
        'SELECT hash FROM hash_cache '
        'WHERE path = ? AND size = ? AND mtime_ns = ? AND ctime_ns = ? '
        'AND inode = ?',
        (str(image_path), file_stat.st_size, file_stat.st_mtime_ns,
         file_stat.st_ctime_ns, file_stat.st_ino)).fetchone()
    if row is not None:
        return row[0]
    content_hash = compute_content_hash(image_path)
    # Queue the cache row for a batched write instead of committing per
    # image (see `_buffer_hash_cache_row` for why).
    _buffer_hash_cache_row(connection, image_path, file_stat, content_hash)
    return content_hash


class ImageStorage(ABC):
    """Read and write the tags and descriptions of the loaded images."""

    # Whether files with identical content share their stored tags and
    # description (database mode: one record per content; text-file mode:
    # each file has its own caption file).
    shared_content_records: bool = False

    @abstractmethod
    def load_image(self, image_path: Path) -> tuple[list[str], str, str | None]:
        """
        Load the tags, description and content hash of one image.

        The content hash is None when it could not be resolved (for example
        when the database holding the hash cache is unavailable); the image
        must still load.
        """

    @abstractmethod
    def save_image(self, image: Image) -> None:
        """Save the tags and description of one image."""

    def save_images(self, images: list[Image]) -> list[Image]:
        """
        Save the tags and descriptions of several images, and return the
        images that could not be saved (an empty list when all were saved).

        The default saves one image at a time and collects the failures, so
        backends keep working when only `save_image` is implemented (the
        text-file backend uses it: one caption file per image is the format).
        `SqliteStorage` overrides it to write all the records in a single
        atomic transaction.
        """
        failed = []
        for image in images:
            try:
                self.save_image(image)
            except (OSError, sqlite3.Error) as exception:
                print(f'Failed to save tags for {image.path}: {exception}',
                      file=sys.stderr)
                failed.append(image)
        return failed

    def flush_hash_cache(self) -> None:
        """
        Write any pending derived data computed during a directory load.

        A no-op for backends without a derived cache; called by the model
        when a directory load completes, on the GUI thread.
        """

    def prune_hash_cache(self, paths) -> None:
        """
        Drop the derived cache entries of the given file paths (their files
        were removed).

        Best-effort and a no-op for backends without such a cache (or when
        there is nothing to prune): the cache is derived data, so failing to
        prune it must never matter. Called by the model after a directory
        refresh removed files, on the GUI thread.
        """


class TextFileStorage(ImageStorage):
    """
    Store each image's tags and description in a caption file
    (`<stem>.txt`) next to the image, the historical TagGUI format.

    The tags and description never touch the database. The content hash of
    each image is still resolved on load (and cached in the database's
    `hash_cache` table), because it keys the on-disk thumbnail cache and
    drives the duplicate-content detection.
    """

    def __init__(self, tag_separator: str, db_path: Path):
        self.tag_separator = tag_separator
        self.db_path = db_path

    def load_image(self, image_path: Path) -> tuple[list[str], str, str | None]:
        # The caption file first: the tags and description must load even
        # when the database (hash cache) is unavailable.
        tags, description = read_caption_file(image_path, self.tag_separator)
        try:
            file_stat = image_path.stat()
        except OSError as exception:
            print(f'Failed to read {image_path}: {exception}',
                  file=sys.stderr)
            return tags, description, None
        try:
            connection = _connect_reusable(self.db_path)
            content_hash = _resolve_content_hash(
                connection, image_path, file_stat)
        except (OSError, sqlite3.Error) as exception:
            # The database is unavailable (corrupted file, unwritable
            # location, ...): the image still loads, and its thumbnail is
            # simply not cached (the cache is keyed by the content hash,
            # which could not be resolved).
            print(f'Failed to resolve the content hash of {image_path}: '
                  f'{exception}', file=sys.stderr)
            content_hash = None
        return tags, description, content_hash

    def save_image(self, image: Image) -> None:
        write_caption_file(image.path, image.tags, image.description,
                           self.tag_separator)

    def flush_hash_cache(self) -> None:
        """
        Write the `hash_cache` rows buffered during the directory load
        (see `_flush_hash_cache`).
        """
        _flush_hash_cache(self.db_path)

    def prune_hash_cache(self, paths) -> None:
        """Drop the `hash_cache` rows of the given (removed) files."""
        _prune_hash_cache(self.db_path, paths)


class SqliteStorage(ImageStorage):
    """
    Store all images' tags and descriptions in a SQLite database, keyed by
    the image's content hash (the same image content retrieves the same
    tags and description, no matter which folder it is opened from).

    Hashing every file on every load is the dominant per-image cost of a
    directory load, so the computed hashes are cached in the database
    (schema v3 `hash_cache` table), keyed by the file's path and stat
    identity; unchanged files are re-identified in a single indexed lookup
    instead of a full read. The load path also reuses one SQLite
    connection per thread (`_connect_reusable`); saves keep using fresh
    connections.
    """

    shared_content_records = True

    def __init__(self, db_path: Path, tag_separator: str):
        self.db_path = db_path
        self.tag_separator = tag_separator

    def load_image(self, image_path: Path) -> tuple[list[str], str, str | None]:
        try:
            file_stat = image_path.stat()
        except OSError as exception:
            # The image file is gone or unreadable (it was deleted or moved
            # after the directory was scanned). Load it without data instead
            # of hiding it from the list.
            print(f'Failed to read {image_path}: {exception}',
                  file=sys.stderr)
            return [], '', None
        content_hash: str | None = None
        try:
            connection = _connect_reusable(self.db_path)
            content_hash = _resolve_content_hash(
                connection, image_path, file_stat)
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
        except (OSError, sqlite3.Error) as exception:
            # The image file or the database became unavailable (for
            # example, the file was deleted after the directory was
            # scanned, or the database is corrupted or on an unreadable
            # filesystem). Fall back to the caption file so that the image
            # is still visible; the hash is returned when it is already
            # known (from the cache or computed just above). The next save
            # retries against the database, and reports the failure if it
            # persists.
            print(f'Failed to load {image_path} from the database: '
                  f'{exception}', file=sys.stderr)
            if content_hash is None:
                try:
                    content_hash = compute_content_hash(image_path)
                except OSError:
                    content_hash = None
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

    def save_images(self, images: list[Image]) -> list[Image]:
        """
        Save several images' records with one connection and one all-or-
        nothing transaction.

        `save_image` pays a connect + commit per image, which dominates a
        batch operation over thousands of images. The statements are the
        same as `_upsert_image`'s, just batched: upsert the records, remove
        the old tag assignments, register the new tag names (rows are never
        deleted — suggestions rely on them), resolve the names' ids in one
        query, insert the assignments. A failure rolls the whole batch back
        and all the input images are reported as failed.

        Loaded images that share a content (duplicate copies) share one
        record: the last one seen wins, as with sequential `save_image`
        calls.
        """
        entries = []
        failed = []
        for image in images:
            # Hashing is file I/O, not part of the atomic write: resolve it
            # before opening the transaction.
            try:
                content_hash = (image.content_hash
                                or compute_content_hash(image.path))
            except OSError:
                failed.append(image)
                continue
            entries.append((content_hash, image.tags, image.description))
        if not entries:
            return failed
        by_hash = {}
        for entry in entries:
            by_hash[entry[0]] = entry
        entries = list(by_hash.values())
        connection = _connect(self.db_path)
        try:
            with connection:
                connection.executemany(
                    'INSERT INTO images (hash, description) VALUES (?, ?) '
                    'ON CONFLICT (hash) DO UPDATE SET '
                    'description = excluded.description',
                    [(content_hash, description)
                     for content_hash, _, description in entries])
                connection.executemany(
                    'DELETE FROM image_tags WHERE image_hash = ?',
                    [(content_hash,) for content_hash, _, _ in entries])
                tag_names = set()
                for _, tags, _ in entries:
                    tag_names.update(tags)
                name_to_id = {}
                if tag_names:
                    connection.executemany(
                        'INSERT OR IGNORE INTO tags (name) VALUES (?)',
                        [(name,) for name in tag_names])
                    # SQLite's bound-parameter limit: chunk the IN list.
                    names = list(tag_names)
                    for start in range(0, len(names), 999):
                        chunk = names[start:start + 999]
                        name_to_id.update(connection.execute(
                            'SELECT name, id FROM tags '
                            'WHERE name IN (' + ','.join('?' * len(chunk))
                            + ')', chunk))
                connection.executemany(
                    'INSERT INTO image_tags (image_hash, tag_id, position) '
                    'VALUES (?, ?, ?)',
                    [(content_hash, name_to_id[tag], position)
                     for content_hash, tags, _ in entries
                     for position, tag in enumerate(tags)])
        except (OSError, sqlite3.Error):
            # The `with connection` context manager rolled the batch back;
            # report every input image as unsaved.
            return list(images)
        finally:
            connection.close()
        return failed

    def flush_hash_cache(self) -> None:
        """
        Write the `hash_cache` rows buffered during the directory load
        (see `_flush_hash_cache`).
        """
        _flush_hash_cache(self.db_path)

    def prune_hash_cache(self, paths) -> None:
        """Drop the `hash_cache` rows of the given (removed) files."""
        _prune_hash_cache(self.db_path, paths)


def clear_database(db_path: Path) -> int:
    """
    Delete all records from the database at `db_path`. The database file
    and its schema remain and stay usable. The `hash_cache` table is kept:
    it is derived data (file identity -> content hash), and clearing the
    records does not change any file's content, so the cached hashes are
    still valid. Return the number of image records that were cleared.
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


def get_database_tag_names(db_path: Path) -> list[str]:
    """
    All the tag names stored in the database at `db_path`, alphabetically.
    Tags that no image currently uses are included too, so that every tag
    ever stored remains available (for tag suggestions).
    """
    connection = _connect(db_path)
    try:
        return [row[0] for row in connection.execute(
            'SELECT name FROM tags ORDER BY name')]
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
    move never destroys the only copy of the database. The hash cache is
    part of the database file, so it moves with it.
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
    database_path = get_database_path()
    if get_storage_backend() == 'sqlite':
        return SqliteStorage(database_path, tag_separator)
    # Text-file mode uses the same database for the content hashes that key
    # the thumbnail cache and drive duplicate detection (never for the tags
    # or description, which stay in the caption files).
    return TextFileStorage(tag_separator, database_path)
