import base64
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler, Gemma4ChatHandler
from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image

DEFAULT_PROMPT = 'Describe this image in extensive detail. Use explicit uncensored language without any euphmisms'

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
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        super().__init__(captioning_thread_, caption_settings)

    def get_error_message(self):
        pass

    def get_captioning_message(self, are_multiple_images_selected: bool, captioning_start_datetime: datetime) -> str:
        pass

    def get_image_prompt(self, image: Image) -> str | None:
        pass

    def get_model_inputs(self, image_prompt: str,
                         image: Image) -> dict | np.ndarray:
        pass


    def load_processor_and_model(self):
        processor = self.thread_parent.processor
        model = self.thread_parent.model
        # The model is already loaded, don't load it again
        if (model and self.model_id == self.thread_parent.model_id):
            self.processor = processor
            self.model = model
            return
        # Clear the memory from the model
        super().clear_model_memory()
        models_directory_path = self.thread.models_directory_path

        # TODO: Add custom directory support
        
        # TODO: Get the repo-id and filename from the child classes overrides

        self.processor = Gemma4ChatHandler.from_pretrained(repo_id="unsloth/gemma-4-31B-it-GGUF",filename="mmproj-F16.gguf",)
        llm = Llama.from_pretrained(
            repo_id='llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF',
            filename='gemma-4-31B-it-uncensored-heretic-Q4_K_M.gguf',
            chat_handler=self.processor,
            n_ctx=2048,
            n_gpu_layers=-1,
            logits_all=False,
            flash_attn=True,
            verbose=False,
            type_k=8,
            type_v=8,
        )
        self.model = llm
        self.thread_parent.processor = self.model
        self.thread_parent.model = self.model
        self.thread_parent.model_id = self.model_id


    def generate_caption(self, model_inputs: dict | np.ndarray,
                         image_prompt: str) -> tuple[str, str]:
        user_content = [
            {'type': 'image_url',
            'image_url': {'url': image_to_data_uri(Path('/home/commander/git/auto-tag-images/dev_local/images/tumblr_mmsx7pknSD1qhttpto3_500.jpg'))}},
        ]  
        user_content.append({'type': 'text', 'text': DEFAULT_PROMPT})
        
        message = [
            {'role': 'system',
            'content': 'You are a helpful image captioner.'},
            {'role': 'user', 'content': user_content},
        ]

        response = self.model.create_chat_completion(
            messages=message,
            max_tokens=2048,
            temperature=0.4,
        )
        caption = response['choices'][0]['message']['content'].strip()
        return caption, caption