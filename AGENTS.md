# AGENTS.md

Guidance for AI agents (and new developers) working on the TagGUI codebase.

## What this is

TagGUI is a cross-platform desktop application written in Python + PySide6 (Qt 6).
It loads a directory of images, reads/writes per-image `.txt` caption files, and
provides fast keyboard-driven manual tagging (autocomplete, token counting,
batch rename/delete/reorder, advanced filtering) plus automatic caption/tag
generation using local AI models. Its target users are people building LoRA
training datasets for image models (Stable Diffusion, Flux, Anima, etc.).

There is **no** web backend, database, test suite, CI, or package manifest
(`pyproject.toml`/`setup.py`) — it is a plain script-based app.

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
taggui/widgets/all_tags_editor.py  All Tags pane: click action, sort, rename,
                                   delete of tags across images
taggui/widgets/auto_captioner.py   Auto-Captioner dock: settings form (per-model
                                   show/hide), Start/Cancel button, console,
                                   owns ModelSubprocessManager
taggui/widgets/image_viewer.py     Full-resolution image display (background load)
taggui/models/image_list_model.py  Core data model: Image list, directory load,
                                   undo/redo, ALL batch tag operations, disk I/O
taggui/models/proxy_image_list_model.py  Filter evaluation (recursive AST walk)
taggui/models/tag_counter_model.py       Tag frequency counter (autocomplete source)
taggui/models/proxy_tag_counter_model.py Sorting/filtering for the All Tags list
taggui/models/image_tag_list_model.py    QStringListModel for one image's tags
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
taggui/dialogs/*.py                      Find & Replace, Batch Reorder, Settings,
                                   multi-image caption confirmation dialogs
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

### Data model and on-disk format

- `Image` (in `taggui/utils/image.py`) is a plain dataclass:
  `path: Path`, `dimensions`, `tags: list[str]`, `description: str`,
  `thumbnail`. `ImageListModel` (`taggui/models/image_list_model.py`) owns the
  list, the undo/redo stacks, and all disk I/O.
- Each image's tags live in a sibling `.txt` file with the same stem. Format:
  `tags joined by the tag separator` + `"\n\n"` + `description`.
  Read: `caption.partition('\n\n')` in `DirectoryLoadTask.run()`.
  Write: `ImageListModel.write_image_tags_to_disk()`. Both use
  `errors='replace'`. If you change the file format, update both sides.
- The tag separator (default `,` + optional space) is read from QSettings via
  `get_tag_separator()` **once at `ImageListModel` construction**. Changing it
  in Settings requires an app restart (the Settings dialog warns about this).
- `ImageListModel.data(index, Qt.ItemDataRole.UserRole)` returns the `Image`
  object itself — this is how all code gets image data.
- `ImageListModel.proxy_image_list_model` and `.image_list_selection_model`
  are **back-references assigned in `MainWindow.__init__` after
  construction** (circular wiring; the model needs them for scope checks like
  "Filtered images" / "Selected images").

### Undo/redo and batch operations

- Undo/redo is a **full snapshot** of all images' tags + descriptions
  (`HistoryItem`), capped at 32 entries (`UNDO_STACK_SIZE`). Any batch
  operation must call `add_to_undo_stack(action_name,
  should_ask_for_confirmation)` *before* mutating.
- The canonical batch-operation pattern (see `find_and_replace`,
  `rename_tags`, `delete_tags`, `sort_tags_*`, `move_tags_to_front`, etc. in
  `ImageListModel`):
  1. `add_to_undo_stack(...)`
  2. Loop over `self.images`, skip images out of scope (`is_image_in_scope`)
  3. Mutate `image.tags` / `image.description`, call `write_image_tags_to_disk`
     for changed images
  4. Emit `dataChanged(self.index(min), self.index(max))` once at the end
- `Scope` (`ALL_IMAGES` / `FILTERED_IMAGES` / `SELECTED_IMAGES`) and `Location`
  (`CAPTIONS` / `DESCRIPTIONS`) are defined in `image_list_model.py` and used
  by the dialogs.
- `MainWindow.on_image_data_changed` recomputes the tag counter and reloads the
  tag editor on any `dataChanged`, **except** when only the DecorationRole
  (thumbnail) changed — keep that check working; thumbnails change for every
  image.
- Single-image edits flow through `update_image_tags` /
  `update_image_description` (both persist to disk) or via the tag editor's
  `QStringListModel` signals wired in `MainWindow.connect_image_tags_editor_signals`.

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
3. `CaptioningThread.run_captioning()` calls
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

### Class hierarchy for models

```
AutoCaptioningModel (ABC; auto_captioning_model.py)
├── TransformersCaptioningModel (transformers_captioning_model.py)
│   ├── Florence2 / Florence2Promptgen
│   ├── Joycaption, Kosmos2, Llava1Point5, LlavaLlama3, LlavaNext*
│   ├── Moondream1 / Moondream2
│   ├── Phi3Vision
│   └── WdTagger  (ONNX Runtime, not actually transformers — see below)
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
- Post-processing: `get_caption_from_generated_tokens` strips the echoed
  prompt, prepends `caption_start`, and (transformers backend only) applies
  `remove_tag_separators`.
- `bad_words` / `forced_words` (transformers only): comma-separated (escape
  commas with `\,`); forced words use `|` for alternatives and require
  `num_beams > 1` (enforced by `get_error_message`).

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

- **No tests.** Verify changes by running the GUI. The ML import chain is
  loaded eagerly (`transformers`, `torch`, etc. are imported at module import
  time in several files), so the GUI cannot run without the heavy deps.
- **`LlamaCaptioningModel.generate_caption` does not apply
  `remove_tag_separators`** (the transformers backend does, in
  `get_caption_from_generated_tokens`). The checkbox is shown for llama models
  (it's in the base `get_setting_groups()`), so it is currently a no-op for
  them. Decide whether to implement or hide it.
- `AutoCaptioningModel.get_captioning_message` references `self.device`, which
  only exists on transformers models. It is currently only called in
  commented-out code, but calling it on a llama model would raise.
- **Cancel semantics:** cancel sets the QThread's `is_canceled` flag and the
  worker's `mp.Event`, but the worker loop in `model_worker.py` never checks
  `WorkerContext.is_canceled`. Cancel only takes effect between images in the
  main thread; an in-flight generation runs to completion.
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
