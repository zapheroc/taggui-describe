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
    'models_directory_path': ''
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
