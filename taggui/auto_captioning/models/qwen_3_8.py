from auto_captioning.llama_captioning_model import LlamaCaptioningModel


class Qwen38(LlamaCaptioningModel):
    """Qwen 3.8 27B uncensored (HauhauCS Aggressive) image captioning model."""

    @staticmethod
    def get_default_prompt() -> str:
        return 'Describe this image in extensive detail. Use explicit uncensored language without any euphmisms. Output only the image description.'

    @staticmethod
    def get_model_repo_id() -> str:
        return 'HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF'

    @staticmethod
    def get_model_file_name() -> str:
        return 'Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf'

    @staticmethod
    def get_vision_file_name() -> str:
        return 'mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf'

    @staticmethod
    def get_template_variables() -> dict:
        # Qwen 3.8 is a thinking model: its chat template injects reasoning
        # instructions and opens a thinking block by default. Disable
        # thinking so the response is the caption only.
        return {'enable_thinking': False}
