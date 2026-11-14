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
- **Duplicate-content awareness**: images with identical bytes share one
  database record and are kept in sync, and a "Duplicates" filter lists them.
- The **PixAI Tagger** (`pixai-labs/pixai-tagger-v1.0`) alongside the WD Tagger
  as the fast "tagger" models (both vectorized — see "Performance notes").
- A **multi-image grid preview** in the image viewer (variant grid).

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
                                   filter grammar; image context menu actions
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
taggui/models/image_storage.py           Storage layer: ImageStorage ABC + TextFileStorage (.txt) and SqliteStorage (SQLite keyed by content hash); schema/migrations, content hashing, DB tag names, import/export, clear/migrate
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
                                   Filter Help, multi-image caption confirmation dialogs
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
  `thumbnail`, plus the fork's `content_hash: str | None` (the stable blake2b
  hash of the file's content — the key of the image's database record, set by
  the storage backend when it loads the image; `None` in text-file mode),
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
  content_hash | None)` and `save_image(image)`.
- `TextFileStorage` — the historical format: a `<stem>.txt` caption file next
  to the image containing `tags joined by the tag separator` + `"\n\n"` +
  `description`. Read/write use `errors='replace'`; a failed read must never
  hide an image (return empty tags/description instead).
- `SqliteStorage` (the **default**) — one SQLite database keyed by the image's
  **content hash** (`compute_content_hash`, blake2b over 1 MiB chunks), so
  identical images in different folders share one record and always show the
  same tags. A fresh connection is opened per operation (the GUI thread and the
  directory-load worker threads never share one); `PRAGMA foreign_keys = ON`
  and `busy_timeout = 5000`; schema migrations run on connect, serialized by a
  module-level lock (fresh first loads race on the DDL without it).
- `get_storage_backend()` reads the `storage_backend` setting (`'sqlite'` |
  `'text'`, default `'sqlite'`); `get_image_storage()` constructs the active
  backend; `ImageListModel.set_storage()` switches it (called by
  `MainWindow._apply_storage_backend` after a confirmed switch, which also
  reloads the directory so the in-memory data comes from the new storage).

**Schema** (v1; version tracked in `PRAGMA user_version`, future changes
appended to `_MIGRATIONS`, each step in its own transaction, all DDL
`IF NOT EXISTS` so concurrent first connections don't fail):

- `images(hash TEXT PRIMARY KEY, description TEXT)` — one row per distinct
  image content.
- `tags(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)` — the
  de-duplicated set of all tag names ever stored (case-sensitive, as in the
  caption files).
- `image_tags(image_hash REFERENCES images(hash) ON DELETE CASCADE,
  tag_id REFERENCES tags(id), position INTEGER, PRIMARY KEY (image_hash,
  position))` — which tags an image has, in order; index
  `image_tags_tag_id` for tag → images lookups.

**Semantics to preserve:**
- `SqliteStorage.load_image` hashes the file, looks up the record. **When no
  record exists, it imports a caption file into the database** (additive only —
  never overwrites an existing record). This is the text → database import
  path that runs on first load of each image.
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
  the file and schema), `import_images(images, db_path)`,
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

**Duplicate-content handling** (database mode): loaded images that share a
content hash share one record; `ImageListModel._duplicate_index` maps each
hash to its rows, and `_sync_duplicate_images` (called after every successful
save via `write_image_tags_to_disk`) copies the saved tags/description to the
other in-memory copies and emits `dataChanged` for them (suppressed while an
undo/redo restore is in progress; deferred into the batch's combined
`dataChanged` while a batch operation runs — see
`_begin_batch_sync`/`_finish_batch_rows`). The image list shows an amber
`n/k` indicator on duplicates, and the proxy model has a `filter_duplicates`
toggle (shown only in database mode) that keeps images with
`duplicate_count >= 2`; the context menu offers "Select Duplicates" and
"Delete Duplicates…".

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
    and a single-row change → `TagCounterModel.apply_tag_delta(old, new)`
    (incremental) + `update_tag_suggestions()`;
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
37.5 ms at 20k images + a 50k-tag database). The current path ("Fixes 2 & 3"):

- **Fix 2 — incremental tag counter.** `update_image_tags` /
  `update_image_description` set the `_pending_tag_delta` /
  `_pending_description_only` flags before emitting `dataChanged`; the only
  `dataChanged` handler (`MainWindow.on_image_data_changed`) applies
  `TagCounterModel.apply_tag_delta` for single-image changes and falls back to
  a full `count_tags` for batches/undo. Description-only captions do **neither**
  a recount nor a suggestion refresh.
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

Measured per-caption GUI chain (30 tags/image, idle machine): old 1.2–37.5 ms
→ new **~0.2 ms (1k-tag DB) / ~6.8 ms (50k-tag DB)**, and **flat in image
count** — the GUI side can no longer outrun the model.

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

- **No in-repo tests.** Verify changes by running the GUI. The ML import
  chain is loaded eagerly (`transformers`, `torch`, etc. are imported at
  module import time in several files), so the GUI cannot run without the
  heavy deps. The performance work was verified with a standalone suite
  (`/tmp/opencode/taggui-bench/scripts/test_fixes.py`, not in the repo) —
  re-run it after touching the taggers, the tag counter, or the
  storage/suggestion paths.
- **The database is keyed by content hash, not path.** Copying an image file
  into the folder (or a byte-identical file in another folder) shares the
  original's record and tags — that is a feature, but it means byte-identical
  test images collide with prior runs (updates silently no-op against the
  stale record), and "delete the file" does not remove the record (neither
  should it; the record may belong to another copy).
- **`_upsert_image` never deletes tag rows.** When an image loses a tag, the
  `image_tags` row goes away but the `tags` name stays — by design, so every
  tag ever stored remains a suggestion. Don't "clean up" the `tags` table.
- **The pending-delta flags are consumed exactly once.**
  `MainWindow.on_image_data_changed` is the only `dataChanged` handler and
  reads/clears `_pending_tag_delta` / `_pending_description_only`; adding a
  second handler (or emitting an extra `dataChanged` for the same edit) will
  either re-run the full recount or drop the delta.
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
