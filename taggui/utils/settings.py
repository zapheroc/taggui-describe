from PySide6.QtCore import QSettings

# Defaults for settings that are accessed from multiple places.
DEFAULT_SETTINGS = {
    'font_size': 16,
    # Common image formats that are supported in PySide6.
    'image_list_file_formats': 'bmp, gif, jpg, jpeg, png, tif, tiff, webp',
    'image_list_image_width': 200,
    # When enabled, typing a printable character while the Images pane
    # thumbnails have focus moves focus to the Add Tag input in the Image
    # Tags pane and starts the tag with that character.
    'image_list_auto_focus_add_tag_box': True,
    # Maximum zoom factor for the single-image preview (1 = native size).
    'max_image_preview_zoom': 6,
    'tag_separator': ',',
    'insert_space_after_tag_separator': True,
    'autocomplete_tags': True,
    # When enabled, selecting two or more images switches the Image Tags pane
    # into a grouped Common/Differences view and the center preview into a
    # synchronized grid of the selected images. Disable to keep the classic
    # single-image behavior regardless of how many images are selected.
    'variant_grid_view_enabled': True,
    # Maximum number of cells shown at once in the multi-image grid preview.
    # When more images than this are selected, the grid shows a sliding window
    # of cells centered on the current image. Kept modest so compositing stays
    # fast and cells stay large enough to be useful.
    'variant_grid_cell_cap': 12,
    # Subtle overlay in the multi-image grid preview showing the current page
    # and the total number of selected images.
    'variant_grid_overlay_show': True,
    'variant_grid_overlay_font_size': 10,
    'variant_grid_overlay_transparency': 55,
    'models_directory_path': '',
    # How the tags and descriptions of the loaded images are stored:
    # 'sqlite' (a local SQLite database, keyed by image content) or 'text'
    # (a .txt caption file next to each image).
    'storage_backend': 'sqlite',
    # Location of the database file. Empty means the platform-appropriate
    # default location.
    'database_path': '',
    # Find Duplicates dialog (Tools -> Find Duplicates...).
    # Which images a scan covers; values are the Scope enum's.
    'duplicate_scan_scope': 'All images',
    # Which image to keep in each group when "Apply to all groups" is used.
    'duplicate_keep_rule': 'Largest resolution',
    # When deleting/moving/tagging the non-kept copies of a group, first copy
    # their tags to the kept image (text-file mode only; in database mode the
    # copies share one tag record).
    'duplicate_merge_tags': False,
    # The Find Duplicates dialog's mode: 'Duplicates' (byte-for-byte
    # identical content; an instant in-memory scan, run automatically when
    # the dialog opens) or 'Near Duplicates' (256-bit perceptual hash; the
    # scan decodes the uncached images, so it waits for a Scan press).
    'similarity_mode': 'Duplicates',
    # Maximum pHash difference (0-100; 6 is the default, 8 conservative,
    # 28 broad) for an image to join a near-duplicate group.
    'similarity_phash_max_distance': 6,
    # When enabled, a second hash of each image's left-right mirrored
    # version is compared too, so mirrored variants are matched (the card
    # then shows "mirrored: N / 256"). pHash is not mirror-invariant, so
    # this is off by default.
    'similarity_match_mirrored': False,
    # Maximum size of the on-disk thumbnail cache (see
    # `utils/thumbnail_cache.py`), in MiB. Entries are regenerated from the
    # images when evicted, so this only bounds disk usage. 4 GiB holds the
    # 200-px thumbnails of tens of thousands of images at once.
    'thumbnail_cache_max_size_mb': 4096
}


def get_settings() -> QSettings:
    settings = QSettings('taggui', 'taggui')
    return settings


def get_tag_separator() -> str:
    settings = get_settings()
    tag_separator = settings.value(
        'tag_separator', defaultValue=DEFAULT_SETTINGS['tag_separator'],
        type=str)
    insert_space_after_tag_separator = settings.value(
        'insert_space_after_tag_separator',
        defaultValue=DEFAULT_SETTINGS['insert_space_after_tag_separator'],
        type=bool)
    if insert_space_after_tag_separator:
        tag_separator += ' '
    return tag_separator
