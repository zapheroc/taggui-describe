import base64
from pathlib import Path
import numpy as np
from llama_cpp import Llama
from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image
from auto_captioning.worker_context import WorkerContext
from auto_captioning.settings_group import SettingGroup

# TODO: This belong in gemma 4 or a utlity class
def image_to_data_uri(image_path: Path) -> str:
    """Encode an image file as a base64 data URI for the chat handler."""
    mime = 'image/jpeg'
    suffix = image_path.suffix.lower()
    if suffix == '.png':
        mime = 'image/png'
    elif suffix == '.webp':
        mime = 'image/webp'
    elif suffix == '.gif':
        mime = 'image/gif'
    elif suffix == '.bmp':
        mime = 'image/bmp'
    data = base64.b64encode(image_path.read_bytes()).decode('utf-8')
    return f'data:{mime};base64,{data}'


class LlamaCaptioningModel(AutoCaptioningModel):

    def __init__(self,
                 worker_context: WorkerContext,
                 caption_settings: dict):
        super().__init__(worker_context, caption_settings)
        generation_params = caption_settings['generation_parameters']
        self.max_tokens = generation_params['max_new_tokens']
        self.temperature = generation_params['temperature']
        self.top_k = generation_params['top_k']
        self.top_p = generation_params['top_p']
        self.repeat_penalty = generation_params['repetition_penalty']

    @classmethod
    def get_setting_groups(cls) -> set[SettingGroup]:
        # TODO configure these groups
        return super().get_setting_groups() | {
            SettingGroup.MAX_TOKENS,
            SettingGroup.TEMPERATURE,
            SettingGroup.TOP_K,
            SettingGroup.TOP_P,
            SettingGroup.REPETITION_PENALTY,
        }

    def update_caption_settings(self, caption_settings: dict):
        super().update_caption_settings(caption_settings)
        generation_params = caption_settings['generation_parameters']
        self.max_tokens = generation_params['max_new_tokens']
        self.temperature = generation_params['temperature']
        self.top_k = generation_params['top_k']
        self.top_p = generation_params['top_p']
        self.repeat_penalty = generation_params['repetition_penalty']

    @staticmethod
    def get_model_repo_id() -> str:
        raise NotImplementedError("The model repo id must be overidden in a child class")

    @staticmethod
    def get_model_file_name() -> str:
        raise NotImplementedError("The model file must be overidden in a child class")

    @staticmethod
    def get_vision_file_name() -> str:
        raise NotImplementedError("The vision file must be overidden in a child class")

    def get_chat_handler(self, models_directory_path: str):
        raise NotImplementedError("The chat handler must be overridden in a child class")

    def get_error_message(self):
        pass

    def get_model_inputs(self, image_prompt: str,
                         image: Image) -> dict | np.ndarray:
        # Parse the image input text
        text = self.get_input_text(image_prompt)
        user_content = [
            {
                'type': 'image_url', 'image_url': {'url': image_to_data_uri(Path(image.path))}
            },
            {'type': 'text', 'text': text}
        ]
        # TODO: Allow config of system prompt
        message = [
            {
                'role': 'system', 'content': 'You are a helpful image captioner.'
            },
            {'role': 'user', 'content': user_content},
        ]
        return message

    def load_processor_and_model(self):
        print(f'Loading {self.model_id}...')
        models_directory_path = self.context.models_directory_path
        self.processor = self.get_chat_handler(models_directory_path)
        if models_directory_path:
            model_path = models_directory_path / self.get_model_file_name()
            self.model = Llama(
                model_path=str(model_path),
                chat_handler=self.processor,
                n_ctx=2048,
                n_gpu_layers=-1,
                # logits_all=False,
                flash_attn=True,
                verbose=False,
                type_k=8,
                type_v=8,
            )
        else:
            self.model = Llama.from_pretrained(
                repo_id=self.get_model_repo_id(),
                filename=self.get_model_file_name(),
                chat_handler=self.processor,
                n_ctx=2048,
                n_gpu_layers=-1,
                # logits_all=False,
                flash_attn=True,
                verbose=False,
                type_k=8,
                type_v=8,
            )
        # TODO: Are these relevant with thread parent deprecated?
        self.thread_parent.processor = self.processor
        self.thread_parent.model = self.model
        self.thread_parent.model_id = self.model_id

    def generate_caption(self, model_inputs: dict | np.ndarray,
                         image_prompt: str) -> tuple[str, str]:
        response = self.model.create_chat_completion(
            messages=model_inputs,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty
        )
        caption = response['choices'][0]['message']['content'].strip()
        if self.remove_tag_separators:
            caption = caption.replace(self.context.tag_separator, ' ')
        return caption, caption
