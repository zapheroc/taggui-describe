# Based on the PixAI Tagger v1.0 Hugging Face repository
# (https://huggingface.co/pixai-labs/pixai-tagger-v1.0) and its
# `tagger_pipeline.py`.
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from auto_captioning.transformers_captioning_model import (
    TransformersCaptioningModel)
from auto_captioning.models.wd_tagger import (KAOMOJIS, get_tags_to_exclude,
                                               select_top_tags)
from auto_captioning.settings_group import SettingGroup
from auto_captioning.worker_context import WorkerContext
from utils.image import Image


class PixAiTagger(TransformersCaptioningModel):
    # The PixAI Tagger is a plain image classifier (a ViT-Det model), not a
    # text-to-text model, so load it with `AutoModel` (which resolves to the
    # repo's `ViTDetCls` class via its `auto_map`).
    transformers_model_class = AutoModel
    # Keep the alpha channel so the model's own `RescalePadProcessor` can
    # composite transparent images onto a white background (matching the WD
    # Tagger). Loading as 'RGB' would drop the alpha first and render
    # transparent regions as their raw stored color (usually black) instead.
    image_mode = 'RGBA'

    def __init__(self,
                 worker_context: WorkerContext,
                 caption_settings: dict):
        super().__init__(worker_context, caption_settings)
        # The `Load in 4-bit` row is hidden for the PixAI Tagger, so the
        # option must not apply either (a hidden row means "not supported",
        # not "no control"). The model is a small ViT classifier whose tag
        # thresholds are calibrated for full-precision probabilities, and
        # 4-bit quantization would silently shift them.
        self.load_in_4_bit = False
        # The shared tagger options (show probabilities, maximum tags, tags
        # to exclude) live in `wd_tagger_settings`; the PixAI per-category
        # thresholds live in `pixai_tagger_settings`.
        self.tagger_settings = caption_settings['wd_tagger_settings']
        self.pixai_tagger_settings = caption_settings['pixai_tagger_settings']
        self.show_probabilities = self.tagger_settings['show_probabilities']

    def update_caption_settings(self, caption_settings: dict):
        super().update_caption_settings(caption_settings)
        self.load_in_4_bit = False
        self.tagger_settings = caption_settings['wd_tagger_settings']
        self.pixai_tagger_settings = caption_settings['pixai_tagger_settings']
        self.show_probabilities = self.tagger_settings['show_probabilities']
        # The per-category thresholds feed `_pixai_thresholds`, so a settings
        # change invalidates the cached arrays (they are rebuilt on next use).
        self._pixai_thresholds_cache = None

    @classmethod
    def get_setting_groups(cls) -> set[SettingGroup]:
        return {SettingGroup.TAGGER, SettingGroup.PIXAI_TAGGER,
                SettingGroup.DEVICE, SettingGroup.GPU_INDEX}

    def get_error_message(self) -> str | None:
        return None

    def get_processor(self):
        # `RescalePadProcessor` (via the repo's `auto_map`): resizes the
        # image, pads it to a square, and normalizes it. It also composites
        # transparent images onto a white background.
        return AutoImageProcessor.from_pretrained(self.model_id,
                                                  trust_remote_code=True)

    def load_model(self, model_load_arguments: dict):
        model = super().load_model(model_load_arguments)
        # The model registers its RoPE `freqs_cis` as non-persistent
        # buffers, so they are not part of the checkpoint's state dict and
        # `from_pretrained` leaves them on the CPU even when the model is
        # loaded onto a GPU via `device_map`. Move them to the model's
        # device, otherwise the forward pass fails with a device mismatch
        # error. Tensors already on the device are untouched, so this is a
        # no-op for the parameters (including 4-bit ones) and a no-op when
        # the model is on the CPU.
        model.to(self.device)
        # Precomputed once here (not per image) for `generate_caption`: the
        # transformed tag names in their original order (kaomoji kept, other
        # underscores turned into spaces) and a name -> index map for fast
        # exclusion. These depend only on the model config, so computing them
        # once avoids a per-caption Python pass over every tag.
        config = model.config
        self.pixai_tags = [tag if tag in KAOMOJIS else tag.replace('_', ' ')
                           for tag in config.tags]
        self.pixai_name_to_index = {tag: index for index, tag in
                                    enumerate(self.pixai_tags)}
        self._pixai_thresholds_cache = None
        return model

    def _pixai_thresholds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        The per-tag `(covered, threshold)` arrays. A tag is covered when it
        falls within a `tags_split` range; its threshold is the user's
        per-category setting when present, else the model's recommended
        threshold, else 0.0. Rebuilt only when the per-category settings
        change (see `update_caption_settings`).
        """
        if self._pixai_thresholds_cache is not None:
            return self._pixai_thresholds_cache
        config = self.model.config
        category_thresholds = config.category_best_threshold or {}
        total = len(config.tags)
        covered = np.zeros(total, dtype=bool)
        thresholds = np.zeros(total, dtype=np.float64)
        start = 0
        for category, count in config.tags_split:
            threshold = self.pixai_tagger_settings.get(
                category, category_thresholds.get(category, 0.0))
            covered[start:start + count] = True
            thresholds[start:start + count] = threshold
            start += count
        self._pixai_thresholds_cache = (covered, thresholds)
        return self._pixai_thresholds_cache

    def get_model_inputs(self, image_prompt: str, image: Image):
        # The processor handles resizing, square padding, and normalization,
        # so no manual preprocessing is needed.
        pil_image = self.load_image(image)
        model_inputs = (self.processor(images=pil_image, return_tensors='pt')
                        .to(self.device, **self.dtype_argument))
        return model_inputs

    def generate_caption(self, model_inputs,
                         image_prompt: str) -> tuple[str, str]:
        with torch.inference_mode():
            probabilities = (self.model(model_inputs['pixel_values'])
                             .sigmoid()[0].cpu().numpy().astype(np.float32))
        # A tag is included only if its probability is strictly greater than
        # its category's threshold (see `_pixai_thresholds`).
        covered, thresholds = self._pixai_thresholds()
        keep = covered & (probabilities > thresholds)
        # Exclude the configured tags (precomputed name -> index map).
        tags_to_exclude = get_tags_to_exclude(
            self.tagger_settings['tags_to_exclude'])
        if tags_to_exclude:
            for name in tags_to_exclude:
                index = self.pixai_name_to_index.get(name)
                if index is not None:
                    keep[index] = False
        # Keep the top tags by probability (descending, ties in tag order).
        tags, probabilities = select_top_tags(
            probabilities, keep, self.pixai_tags,
            self.tagger_settings['max_tags'])
        caption = self.context.tag_separator.join(tags)
        if self.show_probabilities:
            console_output_caption = self.context.tag_separator.join(
                f'{tag} ({probability:.2f})'
                for tag, probability in zip(tags, probabilities)
            )
        else:
            console_output_caption = caption
        return caption, console_output_caption
