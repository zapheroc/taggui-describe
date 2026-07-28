# import torch
# from transformers import AutoModelForImageTextToText, AutoProcessor

from auto_captioning.llama_captioning_model import LlamaCaptioningModel
# from utils.image import Image


class Gemma4(LlamaCaptioningModel):
    
    @staticmethod
    def get_default_prompt() -> str:
        return 'Describe this image in extensive detail. Use explicit uncensored language without any euphmisms.'