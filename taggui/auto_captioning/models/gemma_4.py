# import torch
# from transformers import AutoModelForImageTextToText, AutoProcessor

from auto_captioning.llama_captioning_model import LlamaCaptioningModel
# from utils.image import Image


class Gemma4(LlamaCaptioningModel):
    
    @staticmethod
    def get_default_prompt() -> str:
        return 'Describe this image in extensive detail. Use explicit uncensored language without any euphmisms.'

    @staticmethod
    def get_model_repo_id() -> str:
        return 'llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF'
    
    @staticmethod
    def get_model_file_name() -> str:
        return 'gemma-4-31B-it-uncensored-heretic-Q4_K_M.gguf'

    @staticmethod
    def get_vision_file_name() -> str:
        return 'gemma-4-31B-it-mmproj-BF16.gguf'