from auto_captioning.llama_captioning_model import LlamaCaptioningModel
from llama_cpp.llama_chat_format import Gemma4ChatHandler
from auto_captioning.worker_context import WorkerContext

class Gemma4(LlamaCaptioningModel):

    def __init__(self,
                 worker_context: WorkerContext,
                 caption_settings: dict):
        super().__init__(worker_context, caption_settings)

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

    def get_chat_handler(self, models_directory_path):
        if models_directory_path:
            vision_path = models_directory_path / self.get_vision_file_name()
            return Gemma4ChatHandler(clip_model_path=str(vision_path))
        return Gemma4ChatHandler.from_pretrained(repo_id=self.get_model_repo_id(), filename=self.get_vision_file_name())
