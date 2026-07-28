import gc
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np

from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          BatchFeature, BitsAndBytesConfig)

from utils.image import Image


class AutoCaptioningModel(ABC):


    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        self.thread = captioning_thread_
        self.thread_parent = captioning_thread_.parent()
        self.caption_settings = caption_settings
        self.model_id = caption_settings['model_id']
        self.prompt = caption_settings['prompt']
        self.caption_start = caption_settings['caption_start']
        # self.device_setting: CaptionDevice = caption_settings['device']
        # self.load_in_4_bit = caption_settings['load_in_4_bit']
        self.processor = None
        self.model = None
        self.tokenizer = None

    def clear_model_memory(self):
        processor = self.thread_parent.processor
        model = self.thread_parent.model
        if model:


            # Garbage collect the previous processor and model to free up
            # memory.
            self.thread_parent.processor = None
            self.thread_parent.model = None
            del processor
            del model
            gc.collect()
            print(f'Unloading {self.model_id}...')
        else:
            print(f'Failed Unloading {self.model_id}...')
            
        # self.thread.clear_console_text_edit_requested.emit()
        
        # raise ValueError("Should unload model" + self.model_id)


    def get_additional_error_message(self) -> str | None:
        return None

    @abstractmethod
    def get_error_message(self) -> str | None:
        pass

    @abstractmethod
    def load_processor_and_model(self):
        pass

    @abstractmethod
    def get_captioning_message(self, are_multiple_images_selected: bool, captioning_start_datetime: datetime) -> str:
        pass

    @abstractmethod
    def get_image_prompt(self, image: Image) -> str | None:
        pass

    # TODO: Need to make this abstract to support llama-cpp-python.
    @abstractmethod
    def get_model_inputs(self, image_prompt: str,
                         image: Image) -> BatchFeature | dict | np.ndarray:
        pass

    @abstractmethod
    def generate_caption(self, model_inputs: BatchFeature | dict | np.ndarray,
                         image_prompt: str) -> tuple[str, str]:
        pass