import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from models.image_storage import write_caption_file
from utils.image import Image


class ExportMode(str, Enum):
    COPY = 'Copy'
    SYMLINK = 'Symlink'
    HARDLINK = 'Hard link'


@dataclass
class ExportResult:
    # The destination paths of the image files that were exported.
    exported_images: list[Path] = field(default_factory=list)
    # The caption files that were written.
    written_text_files: list[Path] = field(default_factory=list)
    # The images that had no tags and no description, so no caption file
    # was written for them.
    skipped_text_files: int = 0
    # The (image, error message) pairs for the images that failed to export.
    failed: list[tuple[Image, str]] = field(default_factory=list)


def unique_name(base_name: str, used_names: set[str]) -> str:
    """
    `base_name` if it is not used, otherwise the first of
    `stem_1.suffix`, `stem_2.suffix`, ... that is not used.

    `used_names` contains casefolded names: `Cat.jpg` and `cat.jpg` collide
    because on case-insensitive file systems (Windows, macOS) they would
    overwrite each other.
    """
    if base_name.casefold() not in used_names:
        return base_name
    stem, suffix = os.path.splitext(base_name)
    number = 1
    while f'{stem}_{number}{suffix}'.casefold() in used_names:
        number += 1
    return f'{stem}_{number}{suffix}'


def list_existing_names(destination: Path) -> set[str]:
    """The casefolded names of the files already in `destination`.

    An empty set is returned when the folder does not exist yet (or is not a
    directory): nothing can be overwritten. Other errors (for example a
    permission error) are deliberately not caught: silently skipping the
    check would let a later operation overwrite files without confirmation.
    """
    try:
        return {path.name.casefold() for path in destination.iterdir()}
    except (FileNotFoundError, NotADirectoryError):
        return set()


def plan_export_names(images: list[Image], destination: Path,
                      write_text_files: bool = True
                      ) -> tuple[list[tuple[Image, Path]], list[str]]:
    """
    Plan where each image will be exported to in the flat `destination`
    folder. Images whose file names collide are renamed (`cat.jpg`,
    `cat_1.jpg`, ...) so that no two exports share a name.

    Returns a list of (image, destination image path) pairs in input order,
    and the names of the files already in `destination` that the plan will
    overwrite (image files and, when `write_text_files` is true, the caption
    files).

    Renaming only happens when two images in the export set collide: an
    exported image that matches a file already in `destination` keeps its
    name (it is listed in the overrides, and the user confirms replacing
    it). Name comparisons are case-insensitive so that `Cat.jpg` and
    `cat.jpg` do not collide on case-insensitive file systems.
    """
    existing_names = list_existing_names(destination)
    used_names = set()
    planned = []
    overrides = []
    for image in images:
        name = unique_name(image.path.name, used_names)
        used_names.add(name.casefold())
        destination_path = destination / name
        same_location = image.path.resolve() == destination_path.resolve()
        if not same_location:
            if name.casefold() in existing_names:
                overrides.append(name)
            if write_text_files:
                # The caption file follows the (possibly renamed) image name.
                text_name = f'{destination_path.stem}.txt'
                if text_name.casefold() in existing_names:
                    overrides.append(text_name)
        planned.append((image, destination_path))
    return planned, overrides


def _export_image_file(source: Path, destination: Path,
                       mode: ExportMode) -> None:
    """Copy, symlink, or hard link `source` to `destination`."""
    if source.resolve() == destination.resolve():
        # The image is already at the destination (the export folder is the
        # source folder); copying it onto itself would fail.
        return
    if destination.is_dir():
        # `shutil.copy` would silently copy into a directory that has the
        # same name instead of failing; reject it explicitly so that the
        # image is reported as failed.
        raise OSError(f'{destination} is a directory; expected a file')
    if destination.exists() or destination.is_symlink():
        # Replace an existing file (the user confirmed the override). A
        # symlink or hard link cannot be created over an existing path, so
        # the old file is removed first; `shutil.copy` would replace it
        # either way.
        destination.unlink()
    if mode == ExportMode.COPY:
        shutil.copy(source, destination)
    elif mode == ExportMode.SYMLINK:
        # Always write an absolute link so the link works regardless of
        # where the export folder (or the source) is later moved.
        destination.symlink_to(source.resolve())
    elif mode == ExportMode.HARDLINK:
        os.link(source, destination)
    else:
        raise ValueError(f'Unknown export mode: {mode!r}')


def export_images(images: list[Image], tag_separator: str, destination: Path,
                  mode: ExportMode | str = ExportMode.COPY,
                  write_text_files: bool = True) -> ExportResult:
    """
    Export the given images (in input order) to the flat `destination`
    folder: each image is copied, symlinked, or hard linked according to
    `mode`, and — when `write_text_files` is true — an image with tags or a
    description gets a caption file written next to its exported copy, in
    the standard format.

    The destination folder is created if it does not exist; a failure to
    create it raises `OSError`. Failures on individual images do not stop
    the export: they are recorded in the returned result.
    """
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exception:
        raise OSError(f'Failed to create the output folder '
                      f'{destination}: {exception}') from exception
    mode = ExportMode(mode)
    planned, _ = plan_export_names(images, destination, write_text_files)
    result = ExportResult()
    for image, destination_path in planned:
        try:
            _export_image_file(image.path, destination_path, mode)
            if write_text_files:
                if image.tags or image.description:
                    text_file_path = write_caption_file(
                        destination_path, image.tags, image.description,
                        tag_separator)
                    result.written_text_files.append(text_file_path)
                else:
                    result.skipped_text_files += 1
            # Only an image whose image file *and* caption file (if it has
            # tags or a description) were written counts as exported.
            result.exported_images.append(destination_path)
        except OSError as exception:
            result.failed.append((image, str(exception)))
    return result
