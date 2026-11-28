# AGENTS.md

Guidance for AI agents (and new developers) working on the TagGUI-Describe codebase.

Do not commit changes with git unless explicitly asked to do so.

## What this is

TagGUI-Describe is a **fork** of TagGUI, a cross-platform desktop application
written in Python + PySide6 (Qt 6). It loads a directory of images, reads/writes
their tags and descriptions, and provides fast keyboard-driven manual tagging
(autocomplete, token counting, batch rename/delete/reorder, advanced filtering)
plus automatic caption/tag generation using local AI models. Its target users
are people building LoRA training datasets for image models (Stable Diffusion,
Flux, Anima, etc.).

Fork additions over upstream TagGUI (the parts this doc covers that upstream
does not):
- A **pluggable storage layer** (`taggui/models/image_storage.py`): tags and
  descriptions live either in per-image `.txt` caption files (the historical
  format) or in a **local SQLite database keyed by the image's content hash**
  (the default). The two are switchable in Settings, with migration in both
  directions. See "Storage backends and the database" below.
- **Duplicate-content awareness**: images with identical bytes are detected
  in both storage modes (`n/k` indicator, "Duplicates" filter,
  Select/Delete Duplicates); in database mode they additionally share one
  database record and are kept in sync.
- The **PixAI Tagger** (`pixai-labs/pixai-tagger-v1.0`) alongside the WD Tagger
  as the fast "tagger" models (both vectorized — see "Performance notes").
- A **multi-image grid preview** in the image viewer (variant grid).
- A **Find Duplicates** dialog (`Tools` -> `Find Duplicates...`,
  `taggui/dialogs/find_duplicates_dialog.py`): a single dialog with two
  modes. **Duplicates** (default) reviews groups of byte-identical images
  (the exact-duplicate port of taggui-plus's dialog); scanning is an
  in-memory group-by over `Image.content_hash` plus `os.stat` — no hashing
  or file reads — and runs automatically when the dialog opens. **Near
  Duplicates** reviews groups of visually similar images via 256-bit pHash
  (`taggui/models/similarity.py` + the `phash` database cache);
  the scan decodes every uncached image and always waits for an explicit
  "Scan". In database storage mode exact-duplicate copies share one tag
  record, so the tag actions (tag-as-`duplicate`, tag merge) are disabled
  there; near-duplicate groups are different content and work in both
  storage modes. See "Find Duplicates dialog" below.

There is **no** web backend, CI, or package manifest
(`pyproject.toml`/`setup.py`) — it is a plain script-based app. There is **no
in-repo test suite**; the performance work was verified with a standalone suite
in `/tmp/opencode/taggui-bench/scripts/test_fixes.py` (not part of the repo).

## Running it

- Python 3.12 recommended (3.11 works). The code uses `StrEnum` (3.11+) and
  `QFormLayout.RowWrapPolicy`/`setRowVisible` (PySide6 6.3+), so 3.10 and
  older PySide6 will not work.
- Install `requirements.txt` in a venv. Dependencies are heavy (torch,
  transformers, bitsandbytes, onnxruntime, llama-cpp-python) and take a long
  time. Notes:
  - `llama-cpp-python` is pulled with CUDA wheels via the `--extra-index-url`
    line in `requirements.txt`.
  - `onnxruntime-gpu` vs CPU-only `onnxruntime` is selected by platform
    markers (no GPU build exists for macOS).
  - flash-attention is **not** a requirement (commented out, "confirm it is
    actually needed").
- Run from the repo root:

  ```
  python taggui/run_gui.py
  ```

  **Do not run it as a module** (`python -m taggui.run_gui` fails). Every import
  inside `taggui/` is relative to the `taggui/` directory itself (e.g.
  `from utils.settings import get_settings`, `from widgets.main_window import
  MainWindow`), relying on the script directory being on `sys.path`.
- `TAGGUI_ENVIRONMENT=development` disables the warning/log suppression that
  `run_gui.py` applies otherwise.
- Linux: may need `libxcb-cursor0`.
- Local developer notes live in `dev_local/` (`todo.txt`, `notes.txt`,
  `ui_option_notes.txt`, `architecture.txt`). This directory is **gitignored** —
  it is only present on some machines. Read them if they exist; they track the
  current work-in-progress and known bugs.
- The repo may contain local virtualenvs (`venv-desktop`, `venv-mobile`,
  `venv-test-final`) — gitignored, leave them alone.

## Repository layout

```
taggui/run_gui.py                  Entry point (QApplication, warning suppression,
                                   PyInstaller `compile_worker` re-exec hack)
taggui/widgets/main_window.py      MainWindow: assembles everything, menus,
                                   shortcuts, signal wiring (the app's "glue")
taggui/widgets/image_list.py       Images pane; FilterLineEdit with the pyparsing
                                   filter grammar; image context menu actions;
                                   `trash_images` / `move_images_to_folder`
                                   (shared file-move helpers, also used by the
                                   Find Duplicates dialog)
taggui/widgets/image_tags_editor.py Tag pane: TagInputBox (autocomplete),
                                   tag list, token counter, description editor
taggui/widgets/group_tags_panel.py Grouped tags view for multi-image
                                   (variant) selections: every tag across the selection with a
                                   k/N count badge, stable order, "Only differences" filter
taggui/widgets/all_tags_editor.py  All Tags pane: click action, sort, rename,
                                   delete of tags across images
taggui/widgets/auto_captioner.py   Auto-Captioner dock: settings form (per-model
                                   show/hide), Start/Cancel button, console,
                                   owns ModelSubprocessManager
taggui/widgets/image_viewer.py     Full-resolution image display (background load)
taggui/models/image_list_model.py  Core data model: Image list, directory load,
                                   undo/redo, ALL batch tag operations, disk I/O
                                   (reads/writes go through the active storage
                                   backend; syncs duplicate-content images)
taggui/models/image_storage.py           Storage layer: ImageStorage ABC + TextFileStorage (.txt) and SqliteStorage (SQLite keyed by content hash); schema/migrations, content hashing, DB tag names, import/export, clear/migrate; schema v4 adds the `phash` cache (256-bit pHash + mirror, hex TEXT) and the ignored-similarity group table, v5 drops the pair table an earlier v4 build created, v6 adds the `similarity_kept_images` table the Find Duplicates dialog's Near Duplicates mode remembers its keep picks in
taggui/models/similarity.py        pHash core for the Find Duplicates dialog's
                                    Near Duplicates mode: `compute_phash` /
                                    `compute_phash_mirrored` (ImageHash,
                                    256 bits), hex round-trip, threaded
                                    `compute_neighbors` (LUT Hamming pass),
                                    non-transitive `seed_group`; numpy at
                                    module level, imagehash/scipy only
                                    inside the compute functions (the module
                                    is imported only when the dialog opens)
taggui/models/proxy_image_list_model.py  Filter evaluation (recursive AST walk)
taggui/models/tag_counter_model.py       Tag frequency counter (autocomplete source) + apply_tag_delta (incremental update)
taggui/models/proxy_tag_counter_model.py Sorting/filtering for the All Tags list
taggui/models/image_tag_list_model.py    QStringListModel for one image's tags
taggui/models/image_export.py        Export Images: copy/symlink/hard-link the
                                    image files into an output folder (unique
                                    names) and write their caption files
taggui/models/tag_suggestion_model.py  TagSuggestionModel: the (tag, count)
                                    pairs behind the Add Tag box's completer
taggui/auto_captioning/auto_captioning_model.py        ABC: prompt template vars,
                                   get_image_prompt, abstract hooks
taggui/auto_captioning/transformers_captioning_model.py  Hugging Face backend
                                   (device/dtype/4-bit, bad/forced words,
                                   generation, postprocessing)
taggui/auto_captioning/llama_captioning_model.py       llama-cpp-python (GGUF)
                                   backend: chat handler, data-URI images,
                                   generation params
taggui/auto_captioning/captioning_thread.py   QThread that drives a captioning run
                                   (per-image loop, caption position logic)
taggui/auto_captioning/model_subprocess_manager.py  Spawns/reuses the worker
                                   subprocess, request/response queues
taggui/auto_captioning/model_worker.py   Runs inside the subprocess: load model,
                                   serve caption/update-settings/shutdown requests
taggui/auto_captioning/worker_context.py  Stand-in for the old GUI objects inside
                                   the subprocess
taggui/auto_captioning/models_list.py     MODELS list (combo box) +
                                   get_model_class(model_id) dispatch
taggui/auto_captioning/models/*.py        One small subclass per model family
taggui/auto_captioning/settings_group.py  SettingGroup enum (UI show/hide groups)
taggui/dialogs/*.py                      Find & Replace, Batch Reorder, Settings
                                   (storage backend + database path), Export Images,
                                   Filter Help, Find Duplicates (dual-mode
                                   exact / pHash near-duplicate review
                                   dialog), multi-image caption
                                   confirmation dialogs
taggui/utils/*.py                        Settings helpers, Image dataclass,
                                   enums, thumbnail generation, QSettings-backed
                                   widgets, small Qt utilities
clip-vit-base-patch32/                   Bundled CLIP tokenizer (for the
                                   Stable Diffusion token counter only)
images/                                  App icon and README screenshots
taggui.spec                              PyInstaller build spec
dev_local/                               Gitignored local dev notes (if present)
```

## Core concepts

### Data model

- `Image` (in `taggui/utils/image.py`) is a plain dataclass:
  `path: Path`, `dimensions`, `tags: list[str]`, `description: str`,
  `thumbnail`, plus the fork's `content_hash: str | None` (the stable
  xxh3-128 hash of the file's content — the key of the image's database
  record, set by the storage backend when it loads the image; in text-file
  mode it still keys the on-disk thumbnail cache and drives the duplicate
  detection; `None` only when the hash could not be resolved),
  `duplicate_count: int` (how many of the loaded images share this image's
  content, itself included; set when the directory loads), and
  `duplicate_position: int` (this copy's 1-based position among them).
  `ImageListModel` (`taggui/models/image_list_model.py`) owns the list, the
  undo/redo stacks, and all persistence — routed through the active storage
  backend (see "Storage backends and the database").
- `ImageListModel.data(index, Qt.ItemDataRole.UserRole)` returns the `Image`
  object itself — this is how all code gets image data.
- `ImageListModel.proxy_image_list_model` and `.image_list_selection_model`
  are **back-references assigned in `MainWindow.__init__` after
  construction** (circular wiring; the model needs them for scope checks like
  "Filtered images" / "Selected images").
- The tag separator (default `,` + optional space) is read from QSettings via
  `get_tag_separator()` **once at `ImageListModel` construction** and captured
  by both storage backends. Changing it in Settings requires an app restart
  (the Settings dialog warns about this).
- `Image` field order matters: `content_hash`, `duplicate_count`,
  `duplicate_position` are trailing fields with defaults; keep new fields
  after them if you add any.

### Storage backends and the database

Fork addition: tags and descriptions do not have to live in `.txt` files.
`taggui/models/image_storage.py` is the pluggable storage layer:

- `ImageStorage` (ABC): `load_image(image_path) -> (tags, description,
  content_hash | None)` and `save_image(image)`; the class attribute
  `shared_content_records` (True only on `SqliteStorage`) tells the model
  whether duplicate copies share a storage record.
- `TextFileStorage` — the historical format: a `<stem>.txt` caption file next
  to the image containing `tags joined by the tag separator` + `"\n\n"` +
  `description`. Read/write use `errors='replace'`; a failed read must never
  hide an image (return empty tags/description instead). It also resolves
  each image's content hash on load through the database's `hash_cache`
  machinery (`_connect_reusable` + `_resolve_content_hash`, buffered and
  written by `flush_hash_cache`), so the thumbnail cache and duplicate
  detection work in text-file mode — but it **never reads or writes the
  `images`/`tags`/`image_tags` record tables** (see the D1 gotcha below).
  `database_path` therefore matters in text-file mode too; an unusable
  database degrades to "cache skipped" (`content_hash=None`), never to
  "image hidden".
- `SqliteStorage` (the **default**) — one SQLite database keyed by the image's
  **content hash** (`compute_content_hash`, xxh3-128 over 1 MiB chunks —
  switched from blake2b for speed, `dev_local/perf_hash_thumbnails_report.md`),
  so identical images in different folders share one record and always show
  the same tags. Computed hashes are cached in the database (schema v3
  `hash_cache`, keyed by path + size + mtime_ns + ctime_ns + inode — see
  "Schema"), so re-opening a folder whose files are unchanged does not
  re-hash them. The directory-load path reuses one connection per worker
  thread (`_connect_reusable`, validated per call by file device/inode and a
  `SELECT 1`; the calling thread is the sole owner — callers never close it).
  The cache is a **dict keyed by `threading.get_ident()`** guarded by a lock,
  deliberately *not* `threading.local()`: that state is intermittently lost
  on Qt's QThreadPool native worker threads (measured 16/101 reads lost on
  a heavily-reused thread), which silently defeats the reuse. `save_image`
  and the one-shot helpers keep a fresh `_connect`, so a writer never shares
  a connection with a directory-load worker thread. Computed cache rows are
  **buffered and written in batches** (see the `hash_cache` schema note and
  `SqliteStorage.flush_hash_cache()`), because one commit per image makes
  every worker thread contend for the single write lock.
  `PRAGMA foreign_keys = ON` and `busy_timeout = 5000`; schema migrations run
  on connect, serialized by a module-level lock (fresh first loads race on
  the DDL without it).
- `get_storage_backend()` reads the `storage_backend` setting (`'sqlite'` |
  `'text'`, default `'sqlite'`); `get_image_storage()` constructs the active
  backend; `ImageListModel.set_storage()` switches it (called by
  `MainWindow._apply_storage_backend` after a confirmed switch, which also
  reloads the directory so the in-memory data comes from the new storage).

**Schema** (v6; version tracked in `PRAGMA user_version`, future changes
appended to `_MIGRATIONS`, each step in its own transaction, all DDL
`IF NOT EXISTS` so concurrent first connections don't fail):

- **Never reuse a schema version number with different content.** An earlier
  build of this app used version 2 for a `path_index` table, so real
  databases exist at `user_version` 2 without `hash_cache`. The current
  `_MIGRATIONS` therefore has a deliberately empty v2 step and creates
  `hash_cache` at v3. Databases from the earlier build keep their
  `path_index` table (the current code never reads it; it is left for
  rollback safety).

- `images(hash TEXT PRIMARY KEY, description TEXT)` — one row per distinct
  image content.
- `tags(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)` — the
  de-duplicated set of all tag names ever stored (case-sensitive, as in the
  caption files).
- `image_tags(image_hash REFERENCES images(hash) ON DELETE CASCADE,
  tag_id REFERENCES tags(id), position INTEGER, PRIMARY KEY (image_hash,
  position))` — which tags an image has, in order; index
  `image_tags_tag_id` for tag → images lookups.
- `hash_cache(path TEXT PRIMARY KEY, size, mtime_ns, ctime_ns, inode, hash)`
  (v3) — file identity → content hash, so an unchanged file is re-identified
  in one indexed lookup instead of being re-read and re-hashed. Trusted only
  when **all five** stat fields match together: `ctime` cannot be restored
  from userspace, so a byte-swap with a restored mtime still re-hashes (the
  same class of trust as git's index stat cache). On a miss the file is
  hashed and the row is **queued** (`_buffer_hash_cache_row`) rather than
  committed per image: a per-image commit makes every load worker thread
  contend for the database's single write lock (measured ~+0.5–2.8 ms/image
  under 8 threads), which dominated the cold-load cost. The buffer (module
  global, lock-guarded) flushes itself at `_HASH_CACHE_FLUSH_THRESHOLD` rows
  (one `executemany` + one commit, best-effort — failed rows are re-queued
  and never raised, since a load must never fail over an unwritten cache),
  and the remainder is flushed by `SqliteStorage.flush_hash_cache()`, which
  `ImageListModel.on_directory_loaded` calls on the GUI thread after all
  worker tasks have finished (fresh connection). Losing the buffer (crash)
  is safe: the rows are derived data and the files are simply re-hashed on
  the next load.
- `phash(hash TEXT PRIMARY KEY, phash TEXT NOT NULL, phash_mirror TEXT,
  phash_version INTEGER NOT NULL)` (v4) — the Find Duplicates dialog's
  perceptual-hash cache (Near Duplicates mode), keyed by content hash:
  `phash` is the 64-hex-char 256-bit pHash, `phash_mirror` the mirrored
  image's pHash (NULL until a "Match mirrored versions" scan computes it),
  `phash_version` distinguishes hash algorithms/parameters (old-version rows
  are recomputed). Written in batches like `hash_cache` (the scan runs in
  worker threads; `flush_phash_cache`). "Clear similarity cache" wipes it.
- `ignored_similarity_groups(group_key TEXT PRIMARY KEY, seed_hash TEXT NOT
  NULL, mode TEXT NOT NULL, threshold_key TEXT NOT NULL)` (v4) — the
  dialog's group dismissals (`'phash:<threshold>:<seed_hash>'` /
  `'exact:<content_hash>'`). An earlier build of the v4 step also created
  an `ignored_similarity_pairs` table (the pairwise "never group these two
  together" exclusions, removed with the per-pair ignore feature); the v5
  step drops it again (`DROP TABLE IF EXISTS` — a no-op for databases that
  migrated to the current v4, which never had it).
- `similarity_kept_images(hash TEXT PRIMARY KEY, kept INTEGER NOT NULL)`
  (v6) — the dialog's remembered keep picks (Near Duplicates mode only),
  keyed by the image's content hash. Three states: **no row** = default
  (kept only when the image is its group's representative), `kept = 1` =
  the user explicitly kept it, `kept = 0` = the user explicitly un-kept it
  (an explicit "no" that wins over the representative default). Only
  deviations from the default are stored (one row per image the user
  actually touched); a content back in its default state has its row
  deleted, and the dialog's destructive actions (delete/move) remove the
  rows of the images they act on. The dialog holds an in-memory mirror
  (`_kept_picks`, loaded in `__init__`) and batches each group's new picks
  into one `update_similarity_kept` connection. No foreign key to `images`,
  like the `phash` cache and ignored groups: in text-file storage mode no
  `images` record exists for the content, and orphan rows are bounded by
  the destructive actions.

**Semantics to preserve:**
- `SqliteStorage.load_image` stats the file, resolves its content hash from
  `hash_cache` (or computes it and queues the cache row for a batched write
  on a miss), then looks up the record. **When no record exists, it imports a caption file
  into the database** (additive only — never overwrites an existing record).
  This is the text → database import path that runs on first load of each
  image.
- `save_image` upserts via `_upsert_image`: update the description, replace
  the image's `image_tags` rows, `INSERT OR IGNORE` the tag names. **Old tag
  rows are never deleted** — every tag ever stored stays available as a tag
  suggestion. Do not "clean up" the `tags` table; the suggestions rely on it.
- Database failure (corrupted file, unreadable filesystem) degrades: loads
  fall back to the caption file so images stay visible; saves retry against
  the database and surface an error. A failed save must not sync the
  duplicate copies to unsaved values (see `write_image_tags_to_disk`).
- Helpers: `get_database_tag_names(db_path)` (all tag names, alphabetical —
  feeds the tag suggestions), `clear_database(db_path)` (wipes records, keeps
  the file, schema and the still-valid `hash_cache` — it is derived data and
  clearing records does not change any file's content),
  `import_images(images, db_path)`,
  `export_images_to_text_files(images, tag_separator)` (writes caption files
  next to the images; used by `ImageListModel.export_loaded_images_to_text_files`
  when switching to text-file storage), `migrate_database(old, new)` (moves
  the DB file: copy, verify size + content hash, then delete the original — a
  failed move never destroys the only copy).
- Database location: the `database_path` setting, or the platform default
  (Windows `%LOCALAPPDATA%/TagGUI-Describe/`, macOS
  `~/Library/Application Support/TagGUI-Describe/`, Linux
  `$XDG_DATA_HOME` or `~/.local/share`) with file name
  `taggui-describe.sqlite3`.

**Duplicate-content handling** (both storage modes): loaded images that share
a content hash form a duplicate group; `ImageListModel._duplicate_index` maps
each hash to its rows. The image list shows an amber `n/k` indicator on
duplicates, the proxy model has a `filter_duplicates` toggle (always visible;
it keeps images with `duplicate_count >= 2`), and the context menu offers
"Select Duplicates" and "Delete Duplicates…".
In database mode the copies share one record, so
`_sync_duplicate_images` (called after every successful save via
`write_image_tags_to_disk`) copies the saved tags/description to the other
in-memory copies and emits `dataChanged` for them (suppressed while an
undo/redo restore is in progress; deferred into the batch's combined
`dataChanged` while a batch operation runs — see
`_begin_batch_sync`/`_finish_batch_rows`). In text-file mode the copies each
have their own caption file, so that sync is skipped by the
`shared_content_records` guard at the top of `_sync_duplicate_images` —
"tag all copies" is done via "Select Duplicates" + a batch operation on the
selection.

### Find Duplicates dialog

Fork addition (`taggui/dialogs/find_duplicates_dialog.py`, opened from
`Tools` -> `Find Duplicates...`, no keyboard shortcut). One dialog with two
modes (combo near the top, persisted in `similarity_mode`; a stored value
that is not one of the combo's items leaves the combo on its first item,
`'Duplicates'` — `QComboBox.setCurrentText` is a no-op for non-item
values):

- **Duplicates** (default) — the exact-duplicate port of taggui-plus's
  dialog. It finds loaded images with byte-identical content and shows each
  group side by side.
- **Near Duplicates** — visually similar images via 256-bit pHash
  (`taggui/models/similarity.py`). The group header text switches
  `'Exact duplicates'` <-> `'Near duplicates'` with the mode, and the
  mode-dependent controls (slider, mirror option, clear-cache button) are
  shown only in Near Duplicates mode.

Both modes show each group side by side. The user picks which images to
keep — in Duplicates mode exactly one (an exclusive "Keep" radio, the keep
rule's pick by default); in Near Duplicates mode one or more ("Keep" check
boxes, the group's representative by default). The current results' kept
state is in memory (the dialog's `_kept_images` path set, mirrored by the
cards' widgets) and survives a "Show ignored" toggle and a regroup (picks
follow their kept images). In **Near Duplicates mode only**, the picks are
also **remembered in the database** (`similarity_kept_images`, schema v6 —
the user's per-image decisions, keyed by content hash): they survive a
close/reopen, a mode switch and a rescan (a fresh scan still clears the
in-memory `_kept_images`, but each group's cards are re-seeded from the
persisted picks — kept when the user explicitly kept the content, otherwise
the representative by default, and an explicit `0` wins over the
representative default). Only deviations from the default are stored, and
returning an image to its default state deletes its row. The destructive
actions (delete/move) forget the picks of the images they act on, so a
stale decision does not resurface for a byte-identical file loaded later.
Duplicates mode is deliberately unchanged: its picks are in-memory only
(seeded from the keep rule) and nothing is written. The actions then act
on the non-kept members of each non-ignored group (delete to Recycle Bin,
move to a folder, tag as `duplicate`), optionally copying their tags to
the kept images first — to every kept one, when a group keeps several.
The global "Other images:" buttons act on all groups at once; each
group's header additionally carries its own `Delete (n)` / `Move (n)`
buttons that act on that group alone (see "Per-group Delete/Move" below).

- **Duplicates scan** — an in-memory group-by over `Image.content_hash`
  plus one `os.stat` per image (size + `mtime_ns`, for the cards and the
  keep rules), run on a `QThreadPool` worker following the repo's
  background-work convention (strong-reference registry + generation
  counter, cancellable). No file is re-read or decoded. Because it is
  instant, it **runs automatically when the dialog opens** (deferred to the
  first `showEvent`, so a dialog that is constructed but never shown never
  scans). Images with `content_hash is None` (the hash could not be
  resolved, e.g. text-file mode with an unusable database) are excluded and
  reported as skipped. Groups are sorted largest-first; members by
  `(width*height, size, mtime_ns)` descending.
- **Near Duplicates scan** — `_PhashScanTask` (same QThreadPool pattern,
  its own registry, `Signal(object)` finished payload — a `Signal(dict)`
  corrupts payloads across queued connections) in three phases: (1) resolve
  the pHash of every distinct content hash in scope (database `phash`
  cache first, then ImageHash for the misses — with "Match mirrored
  versions" on, the mirrored hash too; computed rows are buffered like
  `hash_cache`), (2) compute the neighbor links with
  `similarity.compute_neighbors` (threaded LUT Hamming pass over 32-byte
  hashes), (3) group with `similarity.seed_group` (non-transitive: an
  image joins a group only within threshold of the group's *representative*;
  seeds greedy by `(link count, area, size, path)`, counting links in
  both directions). With "Match mirrored versions" the pair distance
  `min(direct(i, j), hamming(i, mirror_j))` is *not* symmetric, so a
  link counts in either direction — for the seed's degree and for
  joining (a flipped copy that links only to the representative joins
  its group instead of being stranded); each image is a member of at
  most one group. `seed_group` still emits a 1-image group for an
  image whose links were all absorbed by earlier groups
  (non-transitivity forbids merging it into another group), and the
  dialog **drops 1-image groups** — every shown group has at least two
  images. The scan decodes
  every uncached image and always waits for an explicit "Scan" press.
  The **Max pHash difference** slider (0–100, persisted in
  `similarity_phash_max_distance`, default 6) and "Match mirrored
  versions" (`similarity_match_mirrored`) are **inert until a "Scan"
  press** — moving/toggling them only persists the value. A "Scan"
  press then re-groups the cached neighbor lists in memory when the
  scanned image set and the mirror option are unchanged and the slider is
  at or below the scanned threshold (no file is re-read, no distance
  pass), and re-runs the distance pass on the cached hashes otherwise
  (raising the slider above the scanned threshold, or a changed mirror
  option, which changes the pair distances). The grouping — and
  therefore the groups' persisted keys — runs at the *applied*
  threshold (the slider's value at the last scan or fast regroup,
  `_applied_threshold`), never at the live slider value. "Clear
  similarity cache" wipes the `phash` rows only.
- **Keep rules** — `Largest resolution`, `Largest file`, `Newest file`,
  `Oldest file` (a missing mtime can never be picked by a file-date rule:
  the "newest" rule maps it to the oldest possible date, the "oldest" rule
  to the newest), `Most tags`; ties fall back to resolution then the
  earliest member. In Duplicates mode the cards are built with the keep rule already
  applied; in Near Duplicates mode the representative (first member) is the
  default kept image. "Apply to all groups" re-picks every non-ignored
  group: in Duplicates mode it re-picks the keep rule's keeper, in Near
  Duplicates mode it resets the group to the single best image under the
  rule (unchecking the other kept images).
- **Settings** (QSettings-backed widgets, persisted): `duplicate_scan_scope`
  (`'All images'` default), `duplicate_keep_rule` (`'Largest resolution'`
  default), `duplicate_merge_tags` (default `False`), `similarity_mode`
  (`'Duplicates'` default), `similarity_phash_max_distance` (6),
  `similarity_match_mirrored` (False), `similarity_show_ignored` (True —
  whether ignored groups are shown in black and white or hidden entirely).
- **Tag actions** go through `add_tags_per_image` (one undo step, one
  combined `dataChanged`) and are **disabled in Duplicates mode when
  `storage.shared_content_records` is True** (database mode): there any tag
  change to one copy syncs to all copies (`_sync_duplicate_images`), so
  "tag only the extras" cannot exist, and the tag merge is a no-op. Near
  Duplicates groups are different content (different records), so the tag
  actions work in both storage modes there; when a Near Duplicates group
  keeps several images, the merge copies the non-kept members' missing
  tags into **every** kept image.
- **Delete** reuses `trash_images` (Recycle Bin; a caption still used by a
  remaining loaded image is kept). **Move** uses `move_images_to_folder`
  (`widgets/image_list.py`): existing files are never overwritten (a
  colliding name gets a numeric suffix), an image is never placed next to a
  caption file it would silently adopt, and a caption shared by two loaded
  images is copied until the last user is moved. Neither action touches the
  database: records are keyed by content (a moved file's stale `hash_cache`
  row is simply re-resolved on the next load). Both destructive actions
  then call the dialog's `reload_callback` (`MainWindow.reload_directory`,
  the fast diff path) and clear their results — no auto-rescan.
- **Ignore / show ignored** — the group's "Ignore this group (leave all
  its images untouched)" **checkbox** (checked when the group is ignored;
  toggling it ignores/un-ignores) persists a dismissal in
  `ignored_similarity_groups`, keyed `'phash:<threshold>:<seed_hash>'` in
  Near Duplicates mode (so it is threshold-specific — the *applied*
  threshold the grouping was computed at) and `'exact:<content_hash>'` in
  Duplicates mode. Toggling is applied **in place** by
  `set_group_ignored`: persist, update `_ignored_groups`, then flip
  `cards_scroll.setEnabled` and run `update_relations()` on the existing
  `_GroupCards` — no box is rebuilt (parity with taggui-plus's
  `setEnabled`-only toggle), so the toggle can never flicker. The "Show
  ignored" checkbox is **persisted** (`similarity_show_ignored`, a
  `SettingsBigCheckBox`, default **On**) and is restored when the dialog
  reopens (and drives the grouping on open): ignored groups stay visible
  with
  their **cards rendered in black and white** — the group's `cards_scroll`
  is disabled (`_GroupCards` sets `setEnabled(not self.ignored)` at
  construction; the toggle flips it in place), and the Fusion style
  desaturates disabled widgets, so the thumbnails, labels, "+N more" tile
  and the *kept but disabled* "Keep" widgets all read as inert, like the
  ignore behavior in taggui-plus (the keep widgets are built for every
  member, ignored groups included — the disabled card area grays them out
  and makes them unclickable); the cards are also
  unclickable (no preview, no keep picking, no card menu). With the
  checkbox **off**, ignoring a group hides it **right away, in place**:
  `set_group_ignored` calls `_hide_group_cards`, which drops the group
  from `_groups`/`_group_cards` (exactly what the rebuild
  `_apply_grouping` would compute with "Show ignored" off), hides the
  box, and — because hiding a layout item only *posts* a layout request —
  forces the re-layout (`results_layout.activate()`) and the container
  resize (`_sync_results_container_size`) synchronously, so the very next
  paint is the final state (no gap, no leftover box, no one-frame
  background difference below the content). The other groups' boxes are
  untouched; hiding the last group leaves the "no groups" status with the
  actions disabled. The user's place in the list is preserved too — by
  two parts, because two things would otherwise move the scrollbar:
  (1) the *focus*: the box being hidden contains the just-clicked Ignore
  checkbox, and hiding a focused widget makes Qt move the focus to a
  fallback widget and *synchronously* scroll the scroll area to make the
  newly focused widget visible (that invisible hand is what jumped the
  list — the "ignore the top item while at the top and the scrollbar
  moves down" report); `_redirect_focus_out_of_results` therefore runs
  **before** `box.hide()`: if `QApplication.focusWidget()` is inside
  `results_container`, the focus goes to the "Show ignored" checkbox
  (always visible in the header, scrolls nothing, and it is the very
  control that brings hidden groups back); (2) the *layout*: everything
  below the hidden box shifts up by the box's height plus the layout's
  spacing, so the scroll offset must shrink by whatever part of the box
  was above the current offset. The re-assertion
  (`verticalScrollBar().setValue(new_offset)`) runs **unconditionally**
  after the synchronous re-layout — *not* guarded by `new_offset !=
  offset`: the re-layout may have clamped the value, and anything that
  moved it during the hide is corrected, which is exactly the cases the
  guard would have skipped (hiding the first group while at the top
  stays at the top, the next group coming to the top; hiding a group
  below the viewport leaves the offset alone). A hidden group's checkbox
  is
  unreachable (the whole
  box, header included, is hidden), so un-ignoring it means toggling
  "Show ignored" back on (rebuild) or "Clear group ignores". While the
  group is shown ignored, its header (title and its controls) stays
  normal and fully usable, so it can still be un-ignored or opened as a
  selection.
  Everything that *structurally* changes the boxes — the "Show ignored"
  toggle, a new scan — goes through
  `_apply_grouping` → `_show_groups`, which rebuilds **atomically** (a
  visible rebuild must never flicker either): all replacement boxes are
  built first, parentless and therefore invisible (`processEvents()`
  every 40 cards), then under one
  `results_container.setUpdatesEnabled(False)` window the old boxes are
  `hide()`d **before** `deleteLater()` (a taken-out widget is still a
  visible child of the container at its old position until it is
  destroyed — painting it there alongside the replacements is what
  flickers) and every replacement is inserted with `_insert_group_box`,
  which also *force-delivers* the box's Show event (`box.show()` +
  `QCoreApplication.sendPostedEvents(box, QEvent.Type.Show)`) —
  inserting a parentless box only *posts* that event, and without
  forcing it the first paint after the swap would miss the new boxes
  (one blank frame of flicker). The first paint after the window closes
  is the final state. The user's place in the list is preserved across
  the rebuild the same way: before the swap, `_show_groups` records the
  scroll offset and which box the viewport top is in, identified by
  *group key*, not object identity (`_group_key_of`: the `group_key` of
  the `_SimilarGroup` in Near Duplicates mode, `exact:<content_hash>` in
  Duplicates mode — the same keys the ignore persistence uses, so they
  are stable across rebuilds even though Near Duplicates mode re-creates
  the `_SimilarGroup` objects); after the swap — inside the paint-
  suppressed window, after a forced `results_layout.activate()` so the
  replacement boxes' `y()` positions are final — `_restore_scroll_anchor`
  scrolls the rebuilt box of the same group back to the viewport top at
  the same inset. If that group's box was not rebuilt (it was ignored
  and "Show ignored" is off, so it is hidden), the first rebuilt box
  after it in the old order takes its place (its top at the viewport
  top, exactly as the in-place hide does); if nothing was rebuilt after
  it, the offset is clamped to the smaller maximum (the user was at the
  bottom, and stays at the bottom); if the viewport top was above the
  first box, the offset stands. `_remove_group_cards` (hides *all* boxes
  during the swap) runs the same focus redirect as `_hide_group_cards`
  first — a card-menu action leaves the focus inside a box about to be
  hidden — and the no-groups branch scrolls to 0. "Show next 100 groups"
  appends via
  `_add_group_cards`, which uses the same `_insert_group_box`. Every
  path that adds or removes boxes also calls
  `_sync_results_container_size()`: the scroll area (widget resizable)
  defers resizing its container to the next event loop pass, so the
  container otherwise keeps its old height for one frame and the area
  below the content is painted with the container's background instead of
  the viewport's — the helper resizes it now (viewport width × layout
  height, the scroll area's own target, which its later pass then
  recomputes as a no-op).
- **Select all** — a button in each group's header, shown **only in Near
  Duplicates mode** (the Duplicates mode keeps exactly one image via the
  exclusive "Keep" radio, so it has no such button). It checks the "Keep"
  check box of **every** image of the group, so all of them are kept —
  via `set_kept`, which records the state for rebuilds and expands the
  "+N more" row when kept images are hidden behind it — and never touches
  the main window's image list. An ignored group's kept state is not
  changed by it (like the action buttons, it never acts on ignored
  groups).
- **Per-group Delete/Move** — each group's header also carries a
  `Delete (n)` and a `Move (n)` button (**both modes**, no per-group
  "Tag as duplicate"), where `n` is the group's current count of
  **non-kept** images (the same targets the global "Other images:"
  buttons use, for this group alone — uncheck all "Keep" boxes to act on
  the whole group). The slots `_GroupCards._on_group_delete` /
  `_on_group_move` compute the group's pair via
  `FindDuplicatesDialog._keeper_pair` (one group's worth of
  `_keeper_targets`, `None` for an ignored group or one that keeps all
  of its images) and call the shared
  `FindDuplicatesDialog._perform_delete` / `_perform_move(pairs,
  group_cards)` — the bodies of the global `delete_others` /
  `move_others` (which are thin wrappers passing `group_cards=None`):
  confirmation, `trash_images` / `move_images_to_folder` on a fresh
  `_image_lookup()`, the optional tag merge (the same "Copy the tags..."
  checkbox; `_merge_tags_by_row` + `_add_tags_by_row`), and
  `_forget_kept_picks` for the acted-on images are identical for both
  scopes. Only the finish differs: `None` runs the existing
  `_finish_destructive_action` (clear all results, reload, "click Scan
  again"); a group runs `_finish_group_action(group_cards, summary)`,
  which (1) drops the group's kept state from the bookkeeping —
  `_record_group_kept(group, set())` **and** `_kept_initialized.discard`
  of the group's key, so a rescan that re-forms the same group re-seeds
  from the remembered picks instead of a stale empty set — (2) removes
  the box in place via `_hide_group_cards` (scroll/focus preserved, the
  other boxes untouched), (3) calls `reload_callback` as the global
  actions do, and (4) sets the status to the summary plus "The remaining
  groups are still shown — click \"Scan\" to refresh." (or "No groups are
  left — click \"Scan\" to look again." for the last group, whose removal
  also disables the global action buttons via `_hide_group_cards`). The
  remaining groups' cards keep showing the old model data until the next
  scan, deliberately: every later action on them rebuilds its paths and
  lookup from the current model, and the picks are content-keyed. The
  button labels, tooltips and enabled state are managed by
  `_GroupCards._update_action_buttons` (enabled iff
  `dialog._actions_enabled and not ignored and n > 0`), called from
  `_populate` (the header is built before the kept state is seeded, so
  the construction-time state is not final), `_on_keep_toggled`,
  `set_kept`, `set_group_ignored`, and the dialog's
  `_set_actions_enabled` (which loops the shown groups).
- **Keep picker** — the whole card is clickable: `_DuplicateCard`
  emits `clicked(bool)` on a left press anywhere on the card except inside
  the thumbnail zone registered via `set_preview_zone` (the argument is
  whether Shift was held), and `_build_card` connects it to
  `_on_keep_card_clicked` (Duplicates mode: checking the exclusive `Keep`
  radio — the per-group `QButtonGroup` does the rest; Near Duplicates mode:
  toggling the `Keep` check box; the card shows a hand cursor over its whole
  area). In Near Duplicates mode a press with Shift held makes that image
  the group's *only* kept image (`group_cards.set_kept({path})` — the
  others' check boxes are unchecked, a no-op when it already is the only
  kept one); in Duplicates mode a plain click already selects exactly one,
  so Shift has no extra effect. A left press on the thumbnail label opens
  the preview and then propagates to the card (a `QLabel` ignores mouse
  presses), where the zone check swallows it — so a thumbnail click never
  changes the kept state.
- **Preview** — clicking a thumbnail opens `_PreviewDialog`, which reuses
  the shared `ImageViewer` with a standalone `_GroupImagesModel` (the
  `Image` via `UserRole`, its `thumbnail` `QIcon` via `DecorationRole` —
  the viewer only ever calls `data(index, role)`) and
  `load_image(index, preserve_view=True)` for zoom-preserving steps through
  the group (the new optional `ImageViewer.load_image` parameter; the
  main-window paths are unchanged). The "Keep this image" button toggles
  the matching card's keep widget (the `(QCheckBox | QRadioButton | None,
  member)` tuples are shared): in Near Duplicates mode a kept image shows
  "✓ Keeping — click to un-keep" and the button un-keeps it, in Duplicates
  mode the checked radio shows "✓ Keeping this image" (disabled). The card
  thumbnails are the list thumbnails (already decoded)
  smoothly scaled to 140 px — no separate decode thread. In Near Duplicates
  mode the cards and the preview also carry the relation text per member
  (`'Representative'`, `'pHash difference: N / 256'`,
  `'mirrored: N / 256'`).
- **Lazy import** — the dialog module (and with it `models.similarity` and
  its numpy import) is imported only in
  `MainWindow.show_find_duplicates_dialog`, never at app startup;
  `imagehash`/scipy are imported only inside the scan worker's compute
  functions, when a cache miss has to be hashed.
- Rendering caps: 100 groups shown at a time ("Show next 100 groups"), 12
  members per group until expanded via a "+N more" tile; cards are built in
  40-card chunks with `processEvents` between.

### Undo/redo and batch operations

- Undo/redo is a **full snapshot** of all images' tags + descriptions
  (`HistoryItem`), capped at 32 entries (`UNDO_STACK_SIZE`). Any batch
  operation must call `add_to_undo_stack(action_name,
  should_ask_for_confirmation)` *before* mutating.
- The canonical batch-operation pattern (see `find_and_replace`,
  `rename_tags`, `delete_tags`, `sort_tags_*`, `move_tags_to_front`, etc. in
  `ImageListModel`):
  1. `add_to_undo_stack(...)` and `_begin_batch_sync()`
  2. Loop over `self.images`, skip images out of scope (`is_image_in_scope`)
  3. Mutate `image.tags` / `image.description`, call `write_image_tags_to_disk`
     for changed images (each save may also refresh the image's duplicate
     copies; while the batch is active those refreshes are deferred instead of
     each emitting their own `dataChanged`)
  4. `_finish_batch_rows(changed_rows)` combines the changed rows with the
     deferred duplicate-sync rows, then emit
     `dataChanged(self.index(min), self.index(max))` **once** over them
- `Scope` (`ALL_IMAGES` / `FILTERED_IMAGES` / `SELECTED_IMAGES`) and `Location`
  (`CAPTIONS` / `DESCRIPTIONS`) are defined in `image_list_model.py` and used
  by the dialogs.
- Single-image edits flow through `update_image_tags` /
  `update_image_description` (both persist to disk) or via the tag editor's
  `QStringListModel` signals wired in `MainWindow.connect_image_tags_editor_signals`.
  **The save happens before `dataChanged` is emitted** — listeners may read
  the storage, so it must already reflect the new values.
- `MainWindow.on_image_data_changed` is the **only** `dataChanged` handler
  (direct, same-thread connection) and consumes the model's pending flags:
  - thumbnail-only change (only `DecorationRole`) → do nothing (thumbnails
    change for every image; keep this check working);
  - `_pending_description_only` (set by `update_image_description`) → neither
    recount nor suggestion refresh;
  - `_pending_tag_delta = (old_tags, new_tags)` (set by `update_image_tags`)
    and/or `_pending_sync_deltas = [(old_tags, new_tags), ...]` (set by
    `on_batch_save_finished` for the copies its completed save synced, and
    by `update_image_tags` for the copies a single-image save refreshed —
    the latter sets **both** flags for the same emission) →
    `TagCounterModel.apply_tag_delta(old, new)` for the delta and for
    **each** pair (incremental) + `update_tag_suggestions()`, with **no**
    full recount and **no** `_cached_database_tag_names` invalidation (the
    names the save wrote are the loaded images' tags, which the counter
    holds);
  - otherwise (multi-row batch/undo, or no flag) → full `count_tags(images)`
    + drop `_cached_database_tag_names` + `update_tag_suggestions()`.
  Then it reloads the tag editor if the current image changed. **Do not add a
  second `dataChanged` handler** — the flags are consumed here exactly once.

### Filtering (Images pane)

- `FilterLineEdit.parse_filter_text()` (`taggui/widgets/image_list.py`) uses a
  pyparsing `infix_notation` grammar to produce a list AST, e.g.
  `['tag', 'cat']`, `['NOT', [...]]`, `[left, 'AND', right]`. Precedence is
  NOT > AND > OR. Quoted strings support `\\` escapes; `\*`/`\?` escape
  wildcards (translated to `[*]`/`[?]` for `fnmatch` in
  `replace_filter_wildcards`).
- Evaluation is the recursive `does_image_match_filter` in
  `taggui/models/proxy_image_list_model.py`. Prefixes: `tag:`, `caption:`,
  `desc:`, `name:`, `path:`, and numeric `tags:=n`, `chars:>=n`,
  `tokens:<n` (operators `= == != < > <= >=`).
- The `tokens:` filter and the tag editor's token counter both use the bundled
  CLIP ViT base patch32 tokenizer (see below) and subtract 2 tokens for BOS +
  newline.

### Token counting

- The Stable Diffusion token counter uses the **CLIP ViT base patch32
  tokenizer** from the `clip-vit-base-patch32/` directory at the repo root,
  loaded in `MainWindow` with
  `AutoTokenizer.from_pretrained(get_resource_path(Path('clip-vit-base-patch32')))`.
  `get_resource_path` (`taggui/utils/utils.py`) handles the PyInstaller
  `sys._MEIPASS` temp dir — keep using it for bundled resources.
- `MAX_TOKEN_COUNT = 75` in `taggui/widgets/image_tags_editor.py`; the label
  turns red above it. This is the same tokenizer the `tokens:` filter uses.

### Settings

- All persistent settings are QSettings with organization/application
  `('taggui', 'taggui')` (`get_settings()` in `taggui/utils/settings.py`).
- `SettingsBigCheckBox`, `SettingsComboBox`, `SettingsSpinBox`,
  `SettingsDoubleSpinBox`, `SettingsLineEdit`, `SettingsPlainTextEdit`
  (`taggui/utils/settings_widgets.py`) auto-persist their value to QSettings
  under a string key. Defaults are declared either inline (widget
  constructor) or in `DEFAULT_SETTINGS` in `utils/settings.py`.
- The auto-captioner's per-model settings (prompt, device, seed, temperature,
  ...) are also persisted this way, keyed by the widget key strings in
  `auto_captioner.py`.
- Storage (fork): `storage_backend` (`'sqlite'` default, or `'text'`) and
  `database_path` (`''` default = the platform default location, see
  "Storage backends and the database"). Both are managed by the Settings
  dialog (`taggui/dialogs/settings_dialog.py`), which emits
  `storage_backend_change_requested` / `database_changed_requested` so
  `MainWindow` can confirm/migrate/reload — the dialog itself only edits
  settings; the switching logic lives in `MainWindow._apply_storage_backend`
  and `on_database_changed`.
- Find Duplicates dialog (fork): `duplicate_scan_scope`,
  `duplicate_keep_rule`, `duplicate_merge_tags`, `similarity_mode`
  (`'Duplicates'` / `'Near Duplicates'`), `similarity_phash_max_distance`
  (6), `similarity_match_mirrored` (False), `similarity_show_ignored`
  (True) — see "Find Duplicates dialog".
- Settings that require a restart (font size, tag separator, image width, file
  formats, autocomplete, models directory) show a warning in the Settings
  dialog; new restart-required settings should too.

## The auto-captioning subsystem

This is the most complex part and was recently refactored to run models in a
separate **subprocess**. Read `dev_local/architecture.txt` if present — it
describes the *older* in-thread design.

### Runtime architecture

1. `AutoCaptioner` (a `QDockWidget`) owns a `ModelSubprocessManager`
   (`taggui/auto_captioning/model_subprocess_manager.py`).
2. On "Start Auto-Captioning", `AutoCaptioner.generate_captions()` builds the
   `caption_settings` dict (see below) and starts a `CaptioningThread` (QThread).
3. `CaptioningThread.run_captioning()` first emits
   `clear_console_text_edit_requested` (the console is cleared *before* the
   possibly long model load, so no stale output from a prior run is visible),
   then calls
   `manager.ensure_model(caption_settings, models_directory_path, tag_separator, ...)`:
   - The manager uses a **`spawn`** multiprocessing context (mandatory for CUDA;
     `fork` after CUDA init is unsafe).
   - If a live worker exists and the config signature
     `(model_id, device, load_in_4_bit, gpu_index)` is unchanged, it reuses it
     and just queues `('update-settings', caption_settings, tag_separator)`.
   - Otherwise it kills the old process (freeing its VRAM) and spawns a new one
     running `run_worker` (`model_worker.py`), then blocks on the response queue
     until `('loaded',)`, `('load_error', ...)`, or the process dies.
4. Per image, the thread queues `('caption', payload)` with
   `payload = {'path': str, 'tags': list, 'description': str}` and waits for
   `('caption', caption, console_output)`.
5. For each result, `CaptioningThread` computes new tags/description based on
   the `caption_position` setting (`add_caption_to_tags` /
   `add_caption_to_description` in `captioning_thread.py`) and emits
   `caption_generated(QModelIndex, str, list)` — **(index, description, tags)** —
   which `MainWindow.connect_auto_captioner_signals` routes to
   `update_image_description` (if description is not None) or
   `update_image_tags`.

**Worker protocol** (queues of tuples, in `model_worker.py` /
`model_subprocess_manager.py`):
- Requests: `('caption', payload)`, `('update-settings', caption_settings, tag_separator)`, `('shutdown',)`.
- Responses: `('log', text)`, `('loaded',)`, `('load_error', msg)`, `('caption', caption, console_output)`, `('caption_error', msg)`, `('fatal', traceback)`, `('exiting',)`.
- The worker redirects its `sys.stdout`/`stderr` into the response queue as
  `('log', ...)` messages, so model-load logs and prints land in the
  Auto-Captioner console. The *main* thread's `CaptioningThread` also acts as a
  stdout/stderr sink (`write`/`flush`/`isatty`) and `AutoCaptioner` redirects
  `sys.stdout` to it while a run is in flight (restored on `finished`).
- `AutoCaptioner.update_console_text_edit` treats a bare `\x1b[A` (ANSI cursor
  up) as "replace the last console block" — this is how progress-style output
  is de-duplicated.

### Designed but unimplemented: cancel that keeps the model warm

Canceling currently terminates the worker, so the next run pays a full model
reload (see "Cancel semantics" in Gotchas). The designed alternative
interrupts an in-flight generation *in-process* and keeps the worker/model
warm. Deferred to a separate branch: it touches the core captioning path, and
the llama streaming path cannot be exercised on a machine without CUDA + a
GGUF model. Findings below were verified against the installed
`transformers` 5.19.0 and `llama-cpp-python` 0.3.36 **sources**:

- **transformers:** `generate()` accepts `stopping_criteria` (polled at every
  decoding step) — a criterion returning `WorkerContext.is_canceled` stops
  generation within one token. Low risk.
- **llama.cpp:** call `create_chat_completion(stream=True)` — it is a lazy
  generator (generation only advances while iterated). Accumulate
  `chunk['choices'][0]['delta']['content']` (first chunk is
  `delta={'role': 'assistant'}`, last chunk has `finish_reason` set and an
  empty `delta`), check the cancel flag per chunk, and `.close()` the
  iterator to deterministically stop the `Llama.generate()` loop. **Avoid
  `llama_set_abort_callback`**: `Llama.eval()` ignores `llama_decode`'s
  abort return value (2), and `Llama.generate` then `sample()`s on a
  half-computed context — undefined behavior.
- **Worker protocol:** add a `('canceled',)` response. `model_worker.py`
  checks `ctx.is_canceled` after `generate_caption` and sends it instead of
  `('caption', ...)` (both backends' `generate_caption` return a partial
  caption on cancel — discard it). `ModelSubprocessManager.caption()` raises
  a dedicated exception; `CaptioningThread` treats it as a clean stop (no
  `is_error`).
- **Event lifecycle:** `cancel()` would then only set the `mp.Event` (drop
  the terminate for the loaded-model case), and the event **must be cleared
  on spawn and on the reuse path** — otherwise the first image of the next
  run is instantly "canceled".
- **Cancel during model load** cannot be interrupted in-process (the worker
  is blocked inside the library load): either accept the wait (the run ends
  right after `('loaded',)` via the existing `is_canceled` check) or keep
  process termination for that phase only.

Files to touch: `llama_captioning_model.py` (streaming + accumulation +
cancel check; keep the `remove_tag_separators` post-processing),
`transformers_captioning_model.py` (`stopping_criteria` on `generate()`),
`model_worker.py`, `model_subprocess_manager.py`, `captioning_thread.py`.
`WorkerContext.is_canceled` already exists and is the worker-side hook.

### Class hierarchy for models

```
AutoCaptioningModel (ABC; auto_captioning_model.py)
├── TransformersCaptioningModel (transformers_captioning_model.py)
│   ├── Florence2 / Florence2Promptgen
│   ├── Joycaption, Kosmos2, Llava1Point5, LlavaLlama3, LlavaNext*
│   ├── Moondream1 / Moondream2
│   ├── Phi3Vision
│   ├── WdTagger   (ONNX Runtime, not actually transformers — see below)
│   └── PixAiTagger (torch ViT-Det classifier — see below)
└── LlamaCaptioningModel (llama_captioning_model.py)
    ├── Gemma4
    └── Qwen38
```

- `get_model_class(model_id)` in `models_list.py` dispatches by **substring
  matching on the lowercased model id**; order of the `if` chain matters
  (e.g. `llava-v1.6-34b` is matched before `llava`). Unknown ids fall back to
  the generic `TransformersCaptioningModel`.
- The `MODELS` list (same file) is what populates the model combo box. For the
  llama models the listed ids are *display* ids; the actual GGUF repos live in
  each class (`get_model_repo_id`), e.g. `llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF`.
- Local models: the combo box is editable; directories found under the
  `models_directory_path` setting (containing `config.json` for transformers
  models or `selected_tags.csv` for WD Tagger) are listed automatically. Inside
  `load_processor_and_model`, if `<models_dir>/<model_id>/config.json` (or
  `selected_tags.csv`) exists, `model_id` is rewritten to the local path.
- `WdTagger` subclasses `TransformersCaptioningModel` for reuse but overrides
  almost everything: `get_processor()` returns `None`, the model is an
  ONNX `InferenceSession` (`WdTaggerModel`), images go through
  RGBA→white-background→square-pad→resize→float32→channel-reversed numpy
  preprocessing, rating tags (category `9` in `selected_tags.csv`) are dropped,
  and `show_probabilities` only changes the console output. GPU use requires
  `onnxruntime-gpu` (CUDAExecutionProvider).
- `PixAiTagger` (fork addition, `models/pixai_tagger.py`) is a plain image
  **classifier** (ViT-Det, loaded via `AutoModel` with `trust_remote_code`),
  not a text-to-text model: no prompt, `generate_caption` runs
  `model(pixel_values).sigmoid()` over all tags. A tag is kept when its
  probability is **strictly greater** than its category's threshold (the
  per-category user setting, else the model's `category_best_threshold`, else
  0.0; categories come from the config's `tags_split` ranges) — see
  `_pixai_thresholds` (cached in `_pixai_thresholds_cache`, rebuilt on
  `update_caption_settings`). Tag names get underscores → spaces (kaomoji in
  the `KAOMOJIS` set kept as-is). `load_model` must call `model.to(device)`
  after `from_pretrained`: the RoPE `freqs_cis` buffers are non-persistent and
  would otherwise stay on the CPU, crashing the forward pass.
- **Both taggers share the vectorized post-processing**
  `select_top_tags(probabilities, keep, tag_names, max_tags)` in
  `wd_tagger.py` (numpy boolean mask + stable `np.argsort` descending, ties in
  tag order). The rating/exclude name→index maps are precomputed in
  `__init__`/`load_model`. This was the "Fix 1" perf work — see
  "Performance notes".

### Lazy startup (per-captioner imports)

The heavy ML libraries are never imported at app startup:

- `models_list.py` (imported by `widgets/auto_captioner.py` to populate the
  combo box) exposes only the string `MODELS` list. `get_model_class`
  imports each model class inside its dispatch branch, and
  `get_setting_groups(model_id)` returns the group-set constants from
  `settings_group.py` **without importing any class** — it must mirror
  `get_model_class`'s dispatch exactly (same precedence).
- `transformers_captioning_model.py` keeps `torch`/`numpy` at module top but
  defers `transformers` into the methods that use it, and resolves
  `transformers_model_class` lazily via
  `getattr(type(self), 'transformers_model_class', None)` — it must never be
  defined (or imported) at class-definition time, or the whole transformers
  stack leaks into the non-transformers backends that subclass this base.
- The token counter does not use `transformers.AutoTokenizer`:
  `utils/clip_tokenizer.py` (`ClipTokenizer`) tokenizes the bundled
  `clip-vit-base-patch32/` vocab + merges with the Rust `tokenizers` library
  (byte-identical ids).
- `llama_captioning_model.py` imports `llama_cpp` at module top — acceptable
  because the module is only imported when a llama model is selected (see
  the `llama-cpp-python` RPATH gotcha below).

Net effect: a browsing/tagging session loads none of `torch`/`transformers`/
`onnxruntime`/`llama_cpp`, and each captioner loads only its own backend
(WD Tagger: `onnxruntime`, not `transformers`; PixAi: `transformers`+`torch`;
llama: `llama_cpp`). Launch went from 3.36 s to 0.22 s
(`dev_local/perf_lazy_launch_report.md`).

**If you add a model:** register the id in `MODELS`, add a branch to
`get_model_class`, and mirror it in `get_setting_groups` — do not import
model classes at module scope in `models_list.py` or `auto_captioner.py`.

### The caption_settings dict contract

`CaptionSettingsForm.get_caption_settings()` in `auto_captioner.py` is the
single canonical settings structure passed to model constructors and to
`update_caption_settings()`. Top-level keys: `model_id`, `prompt`,
`caption_start`, `caption_position`, `device`, `gpu_index`, `load_in_4_bit`,
`remove_tag_separators`, `bad_words`, `forced_words`, `seed`,
`generation_parameters` (nested: `min_new_tokens`, `max_new_tokens`,
`num_beams`, `length_penalty`, `do_sample`, `temperature`, `top_k`, `top_p`,
`repetition_penalty`, `no_repeat_ngram_size`), `wd_tagger_settings` (nested:
`show_probabilities`, `min_probability`, `max_tags`, `tags_to_exclude`).

**If you add a key, update:** the form's `get_caption_settings()`, the
concerned model's `__init__` *and* `update_caption_settings`, and register a
UI row with a `SettingGroup` if it should be user-visible.

### Per-model settings UI (SettingGroups)

- Each model class declares `get_setting_groups() -> set[SettingGroup]`
  (class method). `CaptionSettingsForm.show_settings_for_model` shows/hides
  each registered form row based on whether its `SettingGroup` is in the set.
  Rows with `group=None` (model, caption position) are always visible.
- `SettingGroup` (`settings_group.py`) carries a label and an
  `is_advanced` flag; advanced groups are hidden behind the
  "Show Advanced Settings" button.
- The `load_in_4_bit` row is additionally hidden unless device == GPU and
  bitsandbytes is importable (`BITSANDBYTES_AVAILABLE`).

### Prompt handling

- Template variables `{tags}`, `{name}`, `{directory}`/`{folder}` are
  substituted per image in `AutoCaptioningModel.replace_template_variables`
  (escape braces with `\{` `\}`).
- `get_image_prompt` uses the user prompt if set, else
  `get_default_prompt()`, then passes through `format_prompt()` — the standard
  extension point where each model family applies its chat template (see
  `format_prompt` implementations in `models/*.py`).
- Post-processing: `get_caption_from_generated_tokens` (transformers backend)
  strips the echoed prompt and prepends `caption_start`; both backends then
  apply `remove_tag_separators` (replacing `self.context.tag_separator` with a
  space). The llama backend does this at the end of
  `LlamaCaptioningModel.generate_caption` (it has no `caption_start`).
- `bad_words` / `forced_words` (transformers only): comma-separated (escape
  commas with `\,`); forced words use `|` for alternatives and require
  `num_beams > 1` (enforced by `get_error_message`).

## Performance notes

The captioning performance work was measured, fixed, and re-measured (Nov
2026). All numbers live in `dev_local/perf_captioning_report.md`
(**gitignored** — only present on some machines; read it if it exists), and
the verified test suite is
`/tmp/opencode/taggui-bench/scripts/test_fixes.py` (Parts A–D, 45 checks;
not part of the repo, needs the repo + its venv on `sys.path`).

### Worker side (per image)

- **The model inference is the dominant per-image term** (~45 ms of a ~66 ms
  total for WD Tagger at 1024² on GPU). An earlier draft of the report
  claimed a ~47 ms "Python loop" — that was a measurement artifact:
  `WdTaggerModel.generate_tags` calls `session.run(...)` as its first
  statement, and the stage bench measured `generate_tags`, so inference was
  double-counted (report §0).
- The pure-Python post-processing loop was ~3 ms (WD, 10,861 tags) / ~8.8 ms
  (PixAI, 30,877 tags) per image. After vectorizing to the shared
  `select_top_tags` ("Fix 1", both taggers) it is ~0.3 ms / ~1.5 ms.
- When benchmarking a tagger's "loop", **fake the inference** (monkeypatch
  `InferenceSession.run` / the `model(pixel_values)` call) or you re-measure
  inference.

### GUI side (per caption) — the part that scaled with the dataset

The old per-caption path was `count_tags` (full recount of **all** images) +
`SELECT name FROM tags` over the **whole** database + merge/sort + completer
update. It grew with both dataset and database size (19.9 ms at 20k images;
37.5 ms at 20k images + a 50k-tag database). The current path
("Fixes 2–4"):

- **Fix 2 — incremental tag counter.** `update_image_tags` /
  `update_image_description` set the `_pending_tag_delta` /
  `_pending_description_only` flags before emitting `dataChanged`; the only
  `dataChanged` handler (`MainWindow.on_image_data_changed`) applies
  `TagCounterModel.apply_tag_delta` for single-image changes and falls back to
  a full `count_tags` for batches/undo. Description-only captions do **neither**
  a recount nor a suggestion refresh. The same delta mechanism covers the
  duplicate-copy refresh a save emits (`_pending_sync_deltas`, one pair per
  synced copy) — both the completed-batch-save emission and, since the
  per-caption fix, the single-image save in `update_image_tags`: that
  emission no longer triggers a second full recount or an extra database
  tag-name scan (the scoped-op-with-out-of-scope duplicates case that
  otherwise recounted twice; without this, a caption on an image with
  duplicate copies — the combined change spans several rows — also
  recounted).
- **Fix 3 — cached database tag names.**
  `MainWindow._cached_database_tag_names` is a **`list[str]`** (the
  alphabetical `get_database_tag_names` result) — **a list, not a set**:
  iterating a set of 50k names measured ~3× slower than the equivalent list
  (23.4 ms vs 7.2 ms), and the earlier per-caption
  `cached.update(local_names)` was redundant (the merge starts from the tag
  counter, which holds the local names). The cache must be invalidated
  (`= None`) in `_apply_storage_backend`, `on_database_changed`, and the
  full-recount branch of `on_image_data_changed` — new places that can change
  the stored tag set must do the same.
- **Fix 4 — the All Tags view reset is deferred for the duration of a
  captioning run.** Fixes 2 & 3 removed the *data* cost, but every
  per-caption counter update still ran `beginResetModel`/`endResetModel` on
  `TagCounterModel`, and that reset re-lays out **every** All Tags row: in
  the real GUI at 8,531 images / 3,730 unique tags the reset cycle was 47.6
  ms per caption (`endResetModel` 46.8 ms,
  `all_tags_list.doItemsLayout` 58.4 ms — the data side,
  `Counter.most_common()`, is 0.3 ms). The cost scales with the **tag
  vocabulary**, not the image count, so on larger datasets it is the
  reported "UI completely freezes" during captioning. While a run is in
  flight, `count_tags` / `apply_tag_delta` route the reset through
  `_notify_all_tags_view()`: they update `tag_counter` / `most_common_tags`
  but only set `_view_reset_pending`. `MainWindow.on_is_captioning_changed`
  (wired to `AutoCaptioner.is_captioning_changed`, emitted by
  `set_is_captioning()` at run start and at run end/cancel) sets
  `TagCounterModel.defer_view_reset` for the run's duration and applies one
  consolidated `flush_view_reset()` when the run ends —
  `QSortFilterProxyModel` re-applies its active sort after a source reset
  (verified empirically), so the list comes back correctly sorted. User
  actions must never act on a stale row→tag mapping (a tag whose count
  reached 0 — possible with the tag-removing caption positions — would index
  out of range), so every read of the All Tags pane flushes first via
  `AllTagsEditor.flush_counter_view`: `mousePressEvent`, the Delete-key path
  of `keyPressEvent`, `handle_selection_change` (FILTER_IMAGES),
  `copy_selected_tags_to_clipboard`, `doubleClicked` (before the inline-edit
  delegate reads the row), and `TagCounterModel.setData` (the inline rename
  commit).

Measured per-caption GUI chain (30 tags/image, idle machine): old 1.2–37.5 ms
→ new **~0.2 ms (1k-tag DB) / ~6.8 ms (50k-tag DB)** — that is the data
side, and it is **flat in image count**. The view side (Fix 4) was not in
those numbers: measured in the real GUI at 8,531 images / 3,730 unique tags,
the per-caption All Tags reset alone was 47–58 ms, and after deferral the
full per-caption GUI cost (`update_image_tags_total`) is ~12.4 ms (was
40.5 ms), `apply_tag_delta` 0.65 ms (was 45.4 ms), and `count_tags` (pure
recount, no reset) 13.7 ms (was 56.1 ms) — the GUI side can no longer
outrun the model.

### Bench hygiene on this machine

- The harness's `llama-server` (look up its PID; it holds ~23 GB of the RTX
  3090's VRAM) saturates CPU **and** GPU while it generates — measurements
  taken near it inflate 100–1000×. Bench only when a 1-second `/proc/stat`
  sample shows the whole system at ~0.5% busy; `ps` `%CPU` is a lifetime
  average and cannot be trusted for this.
- Report the **minimum** over iterations: contention only inflates numbers.
- Isolated XDG databases (the bench/test scripts point `XDG_DATA_HOME` at
  `/tmp/opencode/taggui-bench/xdg-data/`) **persist across runs**, and the
  database is keyed by **content hash**: byte-identical test images (e.g.
  solid-color PNGs) collide with prior runs, so "updated" tags silently no-op
  against the stale record. Delete the DB at test start or generate
  `os.urandom` pixels per run.

## Recipe: adding a new captioning model

1. Create `taggui/auto_captioning/models/<name>.py` with a small subclass:
   - For a Hugging Face model: subclass `TransformersCaptioningModel`,
     override `get_default_prompt()` / `format_prompt()` /
     `postprocess_*` as needed (see `phi_3_vision.py` or `llava_next.py`).
   - For a llama.cpp GGUF model: subclass `LlamaCaptioningModel` and implement
     `get_model_repo_id()`, `get_model_file_name()`, `get_vision_file_name()`
     (and `get_llama_kwargs()` / `get_template_variables()` if needed — see
     `gemma_4.py` and `qwen_3_8.py`).
2. Import and register it in `models_list.py`: add the display id to `MODELS`
   and a branch in `get_model_class()` (mind the substring-match ordering).
3. If it needs new settings rows, add a `SettingGroup` in
   `settings_group.py`, register a widget in `CaptionSettingsForm` (with the
   group), read the key in `get_caption_settings()`, and consume it in the
   model's `__init__` + `update_caption_settings`.
4. Test both a fresh model load and switching back and forth between models
   (the subprocess-reuse path sends `update-settings` instead of reloading).

## Gotchas and known issues

- **No in-repo tests.** Verify changes by running the GUI. The heavy ML
  libraries are imported lazily (see "Lazy startup" below): the GUI starts
  without loading `torch`/`transformers`/`onnxruntime`/`llama_cpp`, and each
  captioner loads only its own backend when selected. The performance work
  was verified with standalone suites (e.g.
  `/tmp/opencode/taggui-bench/scripts/test_fixes.py`, not in the repo) —
  re-run them after touching the taggers, the tag counter, or the
  storage/suggestion paths.
- **The database is keyed by content hash, not path.** Copying an image file
  into the folder (or a byte-identical file in another folder) shares the
  original's record and tags — that is a feature, but it means byte-identical
  test images collide with prior runs (updates silently no-op against the
  stale record), and "delete the file" does not remove the record (neither
  should it; the record may belong to another copy).
- **Find Duplicates** (see the "Find Duplicates dialog" section): in
  database storage mode the *Duplicates-mode* tag actions are intentionally
  disabled (the copies share one tag record, so tagging one copy tags all
  of them — Near Duplicates groups are different content and work in both
  storage modes); the
  `move_images_to_folder` helper must stay never-overwrite (a colliding name
  gets a numeric suffix, and an image is never renamed onto an existing
  caption it would silently adopt); moving or deleting never touches the
  database — do not "fix" a moved file's stale `hash_cache` row, it is
  re-resolved on the next load. The dialog's tag actions rely on
  `add_tags_per_image`'s single combined `dataChanged`; do not add a second
  `dataChanged` consumer for them (see the pending-delta flags gotcha).
- **`_upsert_image` never deletes tag rows.** When an image loses a tag, the
  `image_tags` row goes away but the `tags` name stays — by design, so every
  tag ever stored remains a suggestion. Don't "clean up" the `tags` table.
- **D1 — text-file mode must never create record rows.** `TextFileStorage`
  may only read the `hash_cache` table and buffer/flush `hash_cache` rows
  (derived data); it must never touch `images`/`tags`/`image_tags`.
  `SqliteStorage.load_image` imports caption files *additively only* — an
  existing record is never re-imported — so any record created in text mode
  (even an empty one) would silently block the caption import on a later
  switch to database mode, dropping the `.txt` tags from the database view.
  Guarded by the regression suite (`test_text_mode_cache.py` T2/T3/T9/T10).
- **The pending-delta flags are consumed exactly once.**
  `MainWindow.on_image_data_changed` is the only `dataChanged` handler and
  reads/clears `_pending_tag_delta` / `_pending_description_only` /
  `_pending_sync_deltas`; adding a second handler (or emitting an extra
  `dataChanged` for the same edit) will either re-run the full recount or
  drop the delta. Each flag is set right before its `dataChanged` and
  consumed by that emission's slot. `_pending_tag_delta` and
  `_pending_sync_deltas` **are** set together for the same emission by
  `update_image_tags` (a single-image save that refreshed duplicate
  copies); `_pending_description_only` is never set together with the
  others (it implies both are cleared).
- **`_cached_database_tag_names` must stay a `list`, not a `set`.** At 50k
  tag names, iterating a set measured ~3× slower than the equivalent list
  (see "Performance notes"), and any new code path that can change the stored
  tag set must invalidate the cache (`= None`).
- **Storage is captured at construction time.** `ImageListModel` and both
  backends hold the `tag_separator` from when they were built, and switching
  backends only takes effect after `set_storage` + directory reload
  (`MainWindow._apply_storage_backend`); code that reads settings directly
  instead of the active storage will see stale data after a switch.
- **`self.device` only exists on the transformers backend.**
  `TransformersCaptioningModel.__init__` sets `self.device` (a `torch.device`);
  the llama-cpp backend has no such attribute. Never reference `self.device`
  from `AutoCaptioningModel` or code shared by both backends.
- **`llama-cpp-python`'s CUDA wheels have no RPATH to the nvidia pip
  packages.** `libllama.so` links against `libcudart.so.13` /
  `libcublas.so.13`, which live in `<site-packages>/nvidia/cu13/lib/`
  (they arrive as transitive dependencies of torch via `nvidia-cudnn-cu13`
  — in CUDA 13 the core packages are the *unsuffixed* `nvidia-cuda-runtime`
  and `nvidia-cublas`, versioned 13.x). PyTorch's own libs find them via a
  baked-in RPATH, but `libllama.so` has none, so `import llama_cpp` fails
  with `libcudart.so.13: cannot open shared object file` unless
  `LD_LIBRARY_PATH` is set. `utils/cuda_preload.py`
  (`preload_cuda_libs()`) therefore preloads the cudart/cublas/cublasLt
  libs by absolute path (discovered from the `nvidia` namespace package;
  best-effort, no-op when they're on the loader path already or absent),
  and `llama_captioning_model.py` calls it before the `llama_cpp` import.
  If you move the `llama_cpp` import elsewhere, keep that call (or an
  equivalent env fix) before it. `CUDA_MAJOR` in that module must stay in
  sync with the llama-cpp-python wheel in `requirements.txt` (currently
  `cu132`) — the nvidia pins in `requirements.txt` (marked
  `platform_system == "Linux"`, because `nvidia-cublas` ships **no Windows
  wheel**; the `!= darwin` marker would break Windows installs) are the
  other half of that sync. The load failure surfaces as a `RuntimeError`
  from llama-cpp-python 0.3.x's ctypes loader (not `ImportError`), and
  `models_list._import_llama_model` rewords it into an actionable message
  for the GUI console — keep the catch tuple `(RuntimeError, OSError)` so
  older/newer versions' `OSError`/`ImportError` forms still get reworded.
  The Windows branch of `cuda_preload.py` is **unverified**: it is
  reachable (the `cu132` index ships win_amd64 wheels) and needed
  (llama-cpp-python's own loader only adds `CUDA_PATH/bin`+`lib` for a
  *system* toolkit, not the pip-package `nvidia/cu13/bin`), but it has
  never been exercised on a Windows machine.
- **The llama.cpp kwargs are GPU-oriented; CPU testing is constrained.**
  `type_v=8` (quantized V cache) is rejected without `flash_attn`
  ("quantized V cache requires flash_attn to be enabled"); Gemma 4's
  hybrid sliding-window layers fall back to a full-size SWA cache
  (~25-30 GB) whenever FA is off, regardless of `n_ctx` (see `gemma_4.py`).
  In the installed build (llama 0.5.0 / ggml 0.25.3, cu132 wheels) the
  CUDA backend is hard-linked into the ggml base libraries (it cannot be
  removed from the lib dir, and `GGML_CUDA_DEVICES=none` is ignored), so
  even `n_gpu_layers=0` still allocates the compute-graph scratch buffer on
  the GPU — on a busy 3090 that OOMs at context creation. CPU smoke tests
  therefore need: `n_gpu_layers=0`, `flash_attn=False`, `type_k=type_v=0`,
  a small `n_ctx`, and a model small enough that its graph buffers fit in
  the free VRAM (`/tmp/opencode/perf/test_llama_small_cpu.py` uses a ~3B
  multimodal model for exactly this reason).
- **The vision (mmproj) model is pinned to the GPU by default — and a full
  GPU aborts the process.** `MTMDChatHandler.__init__` (llama-cpp-python)
  takes `use_gpu: bool = True`, and its `_init_mtmd_context` sets
  `ctx_params.use_gpu` from it; the app's `LlamaVisionChatHandler` forwards
  `**kwargs` to the base, so it is configurable, but
  `LlamaCaptioningModel.get_chat_handler` never passes it (and
  `get_llama_kwargs` hard-codes `n_gpu_layers=-1`), so the llama backend
  always puts *everything* on the GPU — the UI's CPU/GPU device setting has
  no effect on it. Worse, when the clip buffer allocation fails (GPU out of
  memory), the C code hits `GGML_ASSERT(buffer)` in
  `clip_model_loader::load_tensors` and the process dies with SIGABRT
  (core dump) instead of raising a Python error. For CPU tests pass
  `use_gpu=False` to the chat handler (see `test_llama_small_cpu.py`);
  consider honoring the `device` captioning setting (and degrading to a
  clean error on allocation failure) as a follow-up.
- **Cancel semantics:** canceling sets the QThread's `is_canceled` flag and
  `ModelSubprocessManager.cancel()` terminates the worker process — that is
  what stops an in-flight generation, because a worker blocked inside a model
  will not read the request queue. `CaptioningThread` treats the resulting
  process death as a clean stop (no error alert) when `is_canceled` is set,
  and as an error otherwise; the next run spawns a fresh worker. Trade-off:
  canceling unloads the model from memory. A designed in-process alternative
  that keeps it warm is documented in "Designed but unimplemented: cancel
  that keeps the model warm" in the auto-captioning section.
- `ModelSubprocessManager._config_signature` is transformers-oriented (see the
  TODO in that file); e.g. switching llama model files with the same `model_id`
  would incorrectly reuse the process.
- Closing the Auto-Captioner dock (`closeEvent`) or quitting the app calls
  `model_manager.shutdown()`, killing the worker process — the model is not
  kept warm across dock close/open.
- `run_gui.py` calls `settings.clear()` on an uncaught startup exception —
  beware when debugging crashes.
- `Moondream1.patch_source_code()` **rewrites a file in the Hugging Face
  cache** (`modeling_phi.py`) and reloads — a side effect to know about.
- `dev_local/todo.txt` (if present) tracks open items at the time of writing:
  model configs not fully shared between model types (UI can show wrong rows),
  "Start caption with" appearing at start *and* end of captions, and a fixed
  WD-tagger "show probabilities" reload bug.
- `QImageReader.setAllocationLimit(0)` is set in `run_gui.py` to allow huge
  images; keep it.
- Exif orientation is handled in multiple places (dimension swap in
  `DirectoryLoadTask`, `exif_transpose` in `TransformersCaptioningModel.load_image`,
  `setAutoTransform` in thumbnails/full-image loads). If you add an image
  read path, handle it there too.
- PyInstaller: `taggui.spec` bundles `clip-vit-base-patch32/` and the icon, and
  `run_gui.py` contains a `compile_worker` argv re-exec hack for PyTorch in
  bundles. `get_resource_path` must be used for bundled resources.
- **PySide6 returns non-owning wrappers for reference-typed values.**
  `option.font` (from a `QStyleOptionViewItem` built transiently by Qt in
  `sizeHint`/`paint`) wraps the option's internal C++ `QFont`, which is
  destroyed with the option: keeping the returned wrapper (e.g. as a dict
  key in a long-lived cache) dangles, and the *next* hash/lookup of that key
  raises `RuntimeError: Internal C++ object (QFont) already deleted`.
  `ImageListItemDelegate._text_heights` therefore copies the font
  (`QFont(font)`) before using it as a key. If you cache any other
  reference-typed value returned from a transient C++ object, copy it the
  same way.

## Conventions

- 4-space indent, single quotes, type hints on function signatures,
  `Signal`/`Slot` and `@Slot` decorators for Qt callbacks.
- Qt models are updated with `beginResetModel`/`endResetModel` or
  `dataChanged`; batch operations emit one `dataChanged` spanning min→max row.
- Background work uses `QThreadPool` + `QRunnable` tasks that:
  `setAutoDelete(False)`, register themselves in a strong-reference `set`
  (registry) they remove from in `finally`, and use a generation counter so
  superseded results are discarded. Keep this pattern — it fixes Qt
  worker-thread GC/crash issues documented in the code comments.
- Worker-thread signals are delivered to the GUI thread via queued
  connections; slots must not assume they run where the signal was emitted.
- User-facing strings use plain English; confirmation dialogs use
  `get_confirmation_dialog_reply`.
- There is no linter/formatter config in the repo; match the surrounding
  style rather than reformatting.
