import base64
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler
from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image

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

@staticmethod
def get_model_repo_id() -> str:
    raise NotImplementedError("The model repo id must be overidden in a child class")

@staticmethod
def get_model_file_name() -> str:
    raise NotImplementedError("The model file must be overidden in a child class")

@staticmethod
def get_vision_file_name() -> str:
    raise NotImplementedError("The vision file must be overidden in a child class")

class LlamaCaptioningModel(AutoCaptioningModel):

    def __init__(self,
                 captioning_thread_: 'captioning_thread.CaptioningThread',
                 caption_settings: dict):
        super().__init__(captioning_thread_, caption_settings)

    def get_error_message(self):
        pass

    def get_model_inputs(self, image_prompt: str,
                         image: Image) -> dict | np.ndarray:
        # Parse the image input text
        text = self.get_input_text(image_prompt)
        user_content = [
            {'type': 'image_url',
            'image_url': {'url': image_to_data_uri(Path(image.path))}
            },
            {'type': 'text', 'text': text}
        ]  
        # TODO: Allow config of system prompt
        message = [
            {'role': 'system',
            'content': 'You are a helpful image captioner.'},
            {'role': 'user', 'content': user_content},
        ]
        return message


    def load_processor_and_model(self):
        processor = self.thread_parent.processor
        model = self.thread_parent.model
        # TODO: We don't have to check for the model being loaded since subprocess manager handles that
        # The model is already loaded, don't load it again
        if (model and self.model_id == self.thread_parent.model_id):
            self.processor = processor
            self.model = model
            return
        # Clear the memory from the model
        super().clear_model_memory()
        print(f'Loading {self.model_id}...')
        # models_directory_path = self.context.models_directory_path

        # TODO: Add custom directory support
        
        # TODO: Get the repo-id and filename from the child classes overrides

        self.processor = Gemma4ChatHandler.from_pretrained(repo_id=self.get_model_repo_id(),filename=self.get_vision_file_name())        
        llm = Llama.from_pretrained(
            repo_id=self.get_model_repo_id(),
            filename=self.get_model_file_name(),
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
        self.thread_parent.processor = self.processor
        self.thread_parent.model = self.model
        self.thread_parent.model_id = self.model_id


    def generate_caption(self, model_inputs: dict | np.ndarray,
                         image_prompt: str) -> tuple[str, str]:
        response = self.model.create_chat_completion(
            messages=model_inputs,
            max_tokens=2048,
            temperature=0.4,
        )
        caption = response['choices'][0]['message']['content'].strip()
        return caption, caption