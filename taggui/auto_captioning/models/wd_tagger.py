# Based on
# https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py.
import csv
import re
from pathlib import Path

import huggingface_hub
import numpy as np
from PIL import Image as PilImage
import onnxruntime
from onnxruntime import InferenceSession
from auto_captioning.transformers_captioning_model import TransformersCaptioningModel
from utils.image import Image
from auto_captioning.worker_context import WorkerContext
from auto_captioning.settings_group import SettingGroup

KAOMOJIS = ['0_0', '(o)_(o)', '+_+', '+_-', '._.', '<o>_<o>', '<|>_<|>', '=_=',
            '>_<', '3_3', '6_9', '>_o', '@_@', '^_^', 'o_o', 'u_u', 'x_x',
            '|_|', '||_||']


def get_tags_to_exclude(tags_to_exclude_string: str) -> list[str]:
    if not tags_to_exclude_string.strip():
        return []
    tags = re.split(r'(?<!\\),', tags_to_exclude_string)
    tags = [tag.strip().replace(r'\,', ',') for tag in tags]
    return tags


def select_top_tags(probabilities: np.ndarray, keep: np.ndarray,
                    tags: list[str], max_tags: int) -> tuple[tuple, tuple]:
    """
    Return the top `max_tags` tags by probability (descending; ties keep
    their position in `tags`, matching the previous stable descending sort),
    restricted to the positions where `keep` is True.

    `probabilities` and `keep` must be 1-D arrays the same length as
    `tags`. Returns empty tuples when nothing is kept or `max_tags` is 0.

    This replaces the per-tag Python loop that used to filter and sort every
    tag of every image: it is a handful of vectorized NumPy operations instead
    of an O(tag count) Python pass, which is the difference between ~47 ms and
    a fraction of a millisecond for a tagger with thousands of tags.
    """
    if max_tags <= 0 or not keep.any():
        return (), ()
    candidate_positions = np.flatnonzero(keep)
    candidate_probabilities = probabilities[candidate_positions]
    # Descending by probability; `kind='stable'` keeps equal probabilities in
    # their original (tag-list) order, exactly like the old
    # `sort(key=..., reverse=True)`.
    order = np.argsort(-candidate_probabilities, kind='stable')[:max_tags]
    selected_positions = candidate_positions[order]
    selected_tags = [tags[index] for index in selected_positions]
    selected_probabilities = candidate_probabilities[order]
    return tuple(selected_tags), tuple(selected_probabilities)


class WdTaggerModel:
    def __init__(self, model_id: str,
                 providers: list[str | tuple] | None = None):
        model_path = Path(model_id) / 'model.onnx'
        if not model_path.is_file():
            model_path = huggingface_hub.hf_hub_download(model_id,
                                                         filename='model.onnx')
        tags_path = Path(model_id) / 'selected_tags.csv'
        if not tags_path.is_file():
            tags_path = huggingface_hub.hf_hub_download(
                model_id, filename='selected_tags.csv')
        # `providers` orders the execution providers by preference (for
        # example `CUDAExecutionProvider` for the GPU). When omitted,
        # ONNX Runtime uses its default order.
        self.inference_session = InferenceSession(
            model_path, providers=providers)
        self.tags = []
        self.rating_tags_indices = []
        self.general_tags_indices = []
        self.character_tags_indices = []
        with open(tags_path, 'r') as tags_file:
            reader = csv.DictReader(tags_file)
            for index, line in enumerate(reader):
                tag = line['name']
                if tag not in KAOMOJIS:
                    tag = tag.replace('_', ' ')
                self.tags.append(tag)
                category = line['category']
                if category == '9':
                    self.rating_tags_indices.append(index)
                elif category == '0':
                    self.general_tags_indices.append(index)
                elif category == '4':
                    self.character_tags_indices.append(index)
        # Precomputed once here (not per image) for `generate_tags`, which runs
        # for every caption: the non-rating tags in their original order, a
        # boolean mask selecting them in the probability vector, and a
        # name -> index map (among the non-rating tags) for fast exclusion.
        # This turns the old per-image Python pass over every tag into a
        # handful of vectorized NumPy operations.
        rating_set = set(self.rating_tags_indices)
        self.non_rating_tags = [tag for index, tag in enumerate(self.tags)
                                if index not in rating_set]
        self.non_rating_mask = np.ones(len(self.tags), dtype=bool)
        if self.rating_tags_indices:
            self.non_rating_mask[self.rating_tags_indices] = False
        self.tag_name_to_index = {tag: index for index, tag in
                                  enumerate(self.non_rating_tags)}

    def generate_tags(self, image_array: np.ndarray,
                      wd_tagger_settings: dict) -> tuple[tuple, tuple]:
        input_name = self.inference_session.get_inputs()[0].name
        output_name = self.inference_session.get_outputs()[0].name
        probabilities = self.inference_session.run(
            [output_name], {input_name: image_array})[0][0].astype(np.float32)
        # Exclude the rating tags (precomputed in `__init__`).
        probabilities = probabilities[self.non_rating_mask]
        # Keep the tags at or above the minimum probability.
        keep = probabilities >= wd_tagger_settings['min_probability']
        # Exclude the configured tags (precomputed name -> index map).
        tags_to_exclude = get_tags_to_exclude(
            wd_tagger_settings['tags_to_exclude'])
        if tags_to_exclude:
            for name in tags_to_exclude:
                index = self.tag_name_to_index.get(name)
                if index is not None:
                    keep[index] = False
        return select_top_tags(probabilities, keep, self.non_rating_tags,
                               wd_tagger_settings['max_tags'])


class WdTagger(TransformersCaptioningModel):
    image_mode = 'RGBA'

    def __init__(self,
                 worker_context: WorkerContext,
                 caption_settings: dict):
        super().__init__(worker_context, caption_settings)
        self.wd_tagger_settings = caption_settings['wd_tagger_settings']
        self.show_probabilities = self.wd_tagger_settings['show_probabilities']

    def update_caption_settings(self, caption_settings: dict):
        super().update_caption_settings(caption_settings)
        self.wd_tagger_settings = caption_settings['wd_tagger_settings']
        self.show_probabilities = self.wd_tagger_settings['show_probabilities']

    @classmethod
    def get_setting_groups(cls) -> set[SettingGroup]:
        return {SettingGroup.TAGGER, SettingGroup.WD_TAGGER,
                SettingGroup.DEVICE, SettingGroup.GPU_INDEX}

    def get_error_message(self) -> str | None:
        return None

    def get_processor(self):
        return None

    def _get_onnxruntime_providers(self) -> list:
        """Use the GPU (the CUDA execution provider) when the user requested
        the GPU and it is available in this ONNX Runtime build. Otherwise
        fall back to the CPU."""
        if (self.device.type == 'cuda'
                and 'CUDAExecutionProvider'
                in onnxruntime.get_available_providers()):
            print('Using the GPU for the WD Tagger model.')
            gpu_index = self.caption_settings['gpu_index']
            if gpu_index == 0:
                # Use the default GPU. Omit the `device_id` provider option:
                # ONNX Runtime 1.31 logs a spurious "No registered plugin EP
                # device found" warning when `device_id` is present.
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']
            return [('CUDAExecutionProvider',
                     {'device_id': gpu_index}),
                    'CPUExecutionProvider']
        if self.device.type == 'cuda':
            print('The `onnxruntime-gpu` package is not installed, so the '
                  'WD Tagger model will run on the CPU. Install '
                  '`onnxruntime-gpu` to use the GPU.')
        else:
            print('Using the CPU for the WD Tagger model.')
        return ['CPUExecutionProvider']

    def get_model(self):
        return WdTaggerModel(self.model_id,
                             providers=self._get_onnxruntime_providers())

    def get_model_inputs(self, image_prompt: str, image: Image) -> np.ndarray:
        pil_image = self.load_image(image)
        # Add a white background to the image in case it has transparent areas.
        canvas = PilImage.new('RGBA', pil_image.size, (255, 255, 255))
        canvas.alpha_composite(pil_image)
        pil_image = canvas.convert('RGB')
        # Pad the image to make it square.
        max_dimension = max(pil_image.size)
        canvas = PilImage.new('RGB', (max_dimension, max_dimension),
                              (255, 255, 255))
        horizontal_padding = (max_dimension - pil_image.width) // 2
        vertical_padding = (max_dimension - pil_image.height) // 2
        canvas.paste(pil_image, (horizontal_padding, vertical_padding))
        # Resize the image to the model's input dimensions.
        _, input_dimension, *_ = (self.model.inference_session.get_inputs()[0]
                                  .shape)
        if max_dimension != input_dimension:
            input_dimensions = (input_dimension, input_dimension)
            canvas = canvas.resize(input_dimensions,
                                   resample=PilImage.Resampling.BICUBIC)
        # Convert the image to a numpy array.
        image_array = np.array(canvas, dtype=np.float32)
        # Reverse the order of the color channels.
        image_array = image_array[:, :, ::-1]
        # Add a batch dimension.
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def generate_caption(self, model_inputs: np.ndarray,
                         image_prompt: str) -> tuple[str, str]:
        tags, probabilities = self.model.generate_tags(model_inputs,
                                                       self.wd_tagger_settings)
        caption = self.context.tag_separator.join(tags)
        if self.show_probabilities:
            console_output_caption = self.context.tag_separator.join(
                f'{tag} ({probability:.2f})'
                for tag, probability in zip(tags, probabilities)
            )
        else:
            console_output_caption = caption
        return caption, console_output_caption
