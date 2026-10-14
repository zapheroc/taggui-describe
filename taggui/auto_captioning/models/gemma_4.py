from auto_captioning.llama_captioning_model import LlamaCaptioningModel


class Gemma4(LlamaCaptioningModel):
    """Gemma 4 31B uncensored (Heretic) image captioning model."""

    @staticmethod
    def get_default_prompt() -> str:
        return 'Describe this image in extensive detail. Use explicit uncensored language without any euphmisms. Output only the image description.'

    @staticmethod
    def get_model_repo_id() -> str:
        return 'llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF'

    @staticmethod
    def get_model_file_name() -> str:
        return 'gemma-4-31B-it-uncensored-heretic-Q4_K_M.gguf'

    @staticmethod
    def get_vision_file_name() -> str:
        return 'gemma-4-31B-it-mmproj-BF16.gguf'

    @staticmethod
    def get_llama_kwargs() -> dict:
        kwargs = LlamaCaptioningModel.get_llama_kwargs()
        # Gemma 4 uses hybrid sliding-window attention. At the default
        # context size its sliding-window layers fall back to a full-size
        # KV cache and context creation fails with "Failed to create
        # llama_context" (not enough memory), so load it with the smaller
        # context size that fits.
        kwargs['n_ctx'] = 4096
        return kwargs
