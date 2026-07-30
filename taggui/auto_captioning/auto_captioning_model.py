import re
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from auto_captioning.worker_context import WorkerContext
from transformers import BatchFeature
from utils.image import Image


def _replace_template_variable(match: re.Match, image: Image) -> str:
    template_variable = match.group(0)[1:-1].lower()
    if template_variable == 'tags':
        return ', '.join(image.tags)
    if template_variable == 'name':
        return image.path.stem
    if template_variable in ('directory', 'folder'):
        return image.path.parent.name


class AutoCaptioningModel(ABC):

    def __init__(self,
                 worker_context: WorkerContext,
                 caption_settings: dict):
        self.context = worker_context
        self.thread_parent = worker_context.parent()
        self.caption_settings = caption_settings
        self.model_id = caption_settings['model_id']
        self.prompt = caption_settings['prompt']
        self.caption_start = caption_settings['caption_start']
        # TODO: May not need these variables anymore
        self.processor = None
        self.model = None
        self.tokenizer = None
        # Set a default device string that will be overridden by transformers
        self.device = "cuda"

    def update_caption_settings(self, caption_settings: dict):
        self.prompt = caption_settings['prompt']
        self.caption_start = caption_settings['caption_start']

    def get_input_text(self, image_prompt: str) -> str:
        if image_prompt and self.caption_start:
            text = f'{image_prompt} {self.caption_start}'
        else:
            text = image_prompt or self.caption_start
        return text

    @staticmethod
    def replace_template_variables(text: str, image: Image) -> str:
        # Replace template variables inside curly braces that are not escaped.
        text = re.sub(r'(?<!\\){[^{}]+(?<!\\)}',
                        lambda match: _replace_template_variable(match, image), text)
        # Unescape escaped curly braces.
        text = re.sub(r'\\([{}])', r'\1', text)
        return text

    def get_additional_error_message(self) -> str | None:
        return None

    @staticmethod
    def get_captioning_start_datetime_string(
            captioning_start_datetime: datetime) -> str:
        return captioning_start_datetime.strftime('%Y-%m-%d %H:%M:%S')

    def get_captioning_message(self, are_multiple_images_selected: bool,
                               captioning_start_datetime: datetime) -> str:
        if are_multiple_images_selected:
            captioning_start_datetime_string = (
                self.get_captioning_start_datetime_string(
                    captioning_start_datetime))
            return (f'Captioning... (device: {self.device}, start time: '
                    f'{captioning_start_datetime_string})')
        return f'Captioning... (device: {self.device})'

    @staticmethod
    def get_default_prompt() -> str:
        return ''

    @staticmethod
    def format_prompt(prompt: str) -> str:
        return prompt

    def get_image_prompt(self, image: Image) -> str | None:
        if self.prompt:
            image_prompt = self.replace_template_variables(self.prompt, image)
        else:
            self.prompt = self.get_default_prompt()
            image_prompt = self.prompt
        # TODO: Gemma should use format prompt to format the prompt
        image_prompt = self.format_prompt(image_prompt)
        return image_prompt

    @abstractmethod
    def get_error_message(self) -> str | None:
        pass

    @abstractmethod
    def load_processor_and_model(self):
        pass

    @abstractmethod
    def get_model_inputs(self, image_prompt: str,
                         image: Image) -> BatchFeature | dict | np.ndarray:
        pass

    @abstractmethod
    def generate_caption(self, model_inputs: BatchFeature | dict | np.ndarray,
                         image_prompt: str) -> tuple[str, str]:
        pass
