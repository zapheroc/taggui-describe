import base64
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
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

    # def load_processor_and_model(self):
    #     mmproj_path = hf_hub_download(
    #         repo_id='llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF',
    #         filename='gemma-4-31B-it-mmproj-BF16.gguf',
    #     )
    #     self.processor = Llava15ChatHandler(clip_model_path=str('/home/commander/.lmstudio/models/llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF/gemma-4-31B-it-mmproj-BF16.gguf'))

    #     llm = Llama.from_pretrained(
    #         repo_id='llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF',
    #         filename='gemma-4-31B-it-uncensored-heretic-Q4_K_M.gguf',
    #         chat_handler=self.processor,
    #         n_ctx=1024,
    #         n_gpu_layers=-1,
    #         logits_all=False,
    #         flash_attn=True,
    #         verbose=False,
    #         type_k=8,
    #         type_v=8,
    #     )
    #     self.model = llm

    def get_captioning_message(self, are_multiple_images_selected: bool, captioning_start_datetime: datetime) -> str:
        pass

    def get_image_prompt(self, image: Image) -> str | None:
        pass

    def get_model_inputs(self, image_prompt: str,
                         image: Image) -> dict | np.ndarray:
        pass


    def load_processor_and_model(self):
        # models_directory_path = self.thread.models_directory_path
        # if models_directory_path:
        #     config_path = models_directory_path / self.model_id / 'config.json'
        #     tags_path = (models_directory_path / self.model_id
        #                  / 'selected_tags.csv')
        #     if config_path.is_file() or tags_path.is_file():
        #         self.model_id = str(models_directory_path / self.model_id)
        # If the processor and model were previously loaded, use them.
        # processor = self.thread_parent.processor
        model = self.thread_parent.model
        # Only GPUs support 4-bit quantization.
        # self.load_in_4_bit = self.load_in_4_bit and self.device.type == 'cuda'
        # if (model and self.thread_parent.model_id == self.model_id):
        #     # self.processor = processor
        #     self.model = model
        #     return
        # Load the new processor and model.
        print("Should unload model")
        super().clear_model_memory()
        self.processor = Llava15ChatHandler(clip_model_path=str('/home/commander/.lmstudio/models/llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF/gemma-4-31B-it-mmproj-BF16.gguf'))
        # self.thread_parent.processor = self.processor
        # self.model = self.get_model()
        self.model = "Gemma4 31B"
        self.thread_parent.model = self.model
        self.thread_parent.model_id = self.model_id
        # self.thread_parent.model_device_type = self.device.type
        # self.thread_parent.is_model_loaded_in_4_bit = self.load_in_4_bit


    def generate_caption(self, model_inputs: dict | np.ndarray,
                         image_prompt: str) -> tuple[str, str]:
        # super().clear_model_memory()
        chat_handler = Llava15ChatHandler(clip_model_path=str('/home/commander/.lmstudio/models/llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF/gemma-4-31B-it-mmproj-BF16.gguf'))
        llm = Llama(
            model_path=str('/home/commander/.lmstudio/models/llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF/gemma-4-31B-it-uncensored-heretic-Q4_K_M.gguf'),
            chat_handler=chat_handler,
            n_ctx=1024,
            n_gpu_layers=-1,
            logits_all=False,
            # flash_attn=True,
            # verbose=False,
            # type_k=8,   # Q8_0
            # type_v=8,   # Q8_0

      )
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

        response = llm.create_chat_completion(
            messages=message,
            max_tokens=2048,
            temperature=0.4,
        )
        caption = response['choices'][0]['message']['content'].strip()
        return caption, caption