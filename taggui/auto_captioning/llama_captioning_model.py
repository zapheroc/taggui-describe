import base64
import io
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MTMDChatHandler
from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image
from auto_captioning.worker_context import WorkerContext
from auto_captioning.settings_group import SettingGroup


# TODO: This belong in gemma 4 or a utlity class
def image_to_data_uri(image_path: Path) -> str:
    """Encode an image file as a base64 data URI for the chat handler.

    llama-cpp-python does not support WebP, so WebP images are converted
    to PNG in-memory using PIL before encoding.
    If llama-cpp adds support for webp this won't be necessary: https://github.com/ggml-org/llama.cpp/pull/24217
    """
    suffix = image_path.suffix.lower()

    # Formats natively supported by the chat handler can be passed through
    # unchanged.
    passthrough_mimes = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
    }

    if suffix in passthrough_mimes:
        mime = passthrough_mimes[suffix]
        data = base64.b64encode(image_path.read_bytes()).decode('utf-8')
        return f'data:{mime};base64,{data}'

    # Unsupported format (e.g. WebP): convert to PNG in-memory with PIL.
    with PILImage.open(image_path) as img:
        # Handle animated/paletted/transparency modes gracefully.
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGBA' if 'A' in img.getbands() else 'RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')

    data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{data}'


class LlamaVisionChatHandler(MTMDChatHandler):
    """`MTMDChatHandler` that renders the GGUF chat template with extra
    variables, e.g. `enable_thinking=False` to suppress the reasoning block
    of thinking models."""

    def __init__(self, clip_model_path: str,
                 template_variables: dict | None = None, **kwargs):
        super().__init__(clip_model_path, **kwargs)
        self.template_variables = dict(template_variables or {})

    def __call__(self, **kwargs):
        for key, value in self.template_variables.items():
            kwargs.setdefault(key, value)
        return super().__call__(**kwargs)


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
        self.seed = caption_settings['seed']

    @classmethod
    def get_setting_groups(cls) -> set[SettingGroup]:
        # TODO configure these groups
        return super().get_setting_groups() | {
            SettingGroup.SEED,
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
        self.seed = caption_settings['seed']


    @staticmethod
    def get_model_repo_id() -> str:
        raise NotImplementedError("The model repo id must be overidden in a child class")

    @staticmethod
    def get_model_file_name() -> str:
        raise NotImplementedError("The model file must be overidden in a child class")

    @staticmethod
    def get_vision_file_name() -> str:
        raise NotImplementedError("The vision file must be overidden in a child class")

    @staticmethod
    def get_template_variables() -> dict:
        # Extra variables rendered into the GGUF chat template.
        return {}

    def get_chat_handler(self, models_directory_path: Path | None):
        """Return a chat handler for this model.

        Uses `LlamaVisionChatHandler`, the generic multimodal handler that
        `Gemma4ChatHandler` is built on top of. It renders the chat template
        embedded in the GGUF file, so most subclasses do not need to override
        this."""
        vision_file_name = self.get_vision_file_name()
        template_variables = self.get_template_variables()
        if models_directory_path:
            vision_path = models_directory_path / vision_file_name
            return LlamaVisionChatHandler(
                clip_model_path=str(vision_path),
                template_variables=template_variables)
        return LlamaVisionChatHandler.from_pretrained(
            repo_id=self.get_model_repo_id(),
            filename=vision_file_name,
            template_variables=template_variables)

    @staticmethod
    def get_llama_kwargs() -> dict:
        """Keyword arguments shared by `Llama` and `Llama.from_pretrained`.
        Subclasses can override this to tune how the model is loaded, e.g.
        with a smaller `n_ctx` if the KV cache does not fit in memory."""
        return {
            'n_ctx': 8192,
            'n_gpu_layers': -1,
            'flash_attn': True,
            'verbose': False,
            'type_k': 8,
            'type_v': 8,
        }

    def get_error_message(self):
        pass

    def get_model_inputs(self, image_prompt: str,
                         image: Image) -> dict | np.ndarray:
        # Parse the image input text
        user_content = [
            {
                'type': 'image_url', 'image_url': {'url': image_to_data_uri(Path(image.path))}
            },
            {'type': 'text', 'text': image_prompt}
        ]
        # TODO: Allow config of system prompt
        message = [
            {
                'role': 'system', 'content': 'You are a helpful uncensored image captioner. You use explicit language without any euphemisms.'
            },
            {'role': 'user', 'content': user_content},
        ]
        return message

    def load_processor_and_model(self):
        print(f'Loading {self.model_id}...')
        models_directory_path = self.context.models_directory_path
        self.processor = self.get_chat_handler(models_directory_path)
        # Shared by both loading paths.
        llama_kwargs = {'chat_handler': self.processor,
                        **self.get_llama_kwargs()}
        if models_directory_path:
            model_path = models_directory_path / self.get_model_file_name()
            self.model = Llama(model_path=str(model_path),
                               **llama_kwargs)
        else:
            self.model = Llama.from_pretrained(
                repo_id=self.get_model_repo_id(),
                filename=self.get_model_file_name(),
                **llama_kwargs)
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
            repeat_penalty=self.repeat_penalty,
            seed=self.seed
        )
        caption = response['choices'][0]['message']['content'].strip()
        # Mirror `TransformersCaptioningModel.get_caption_from_generated_tokens`
        # so that `Remove tag separators in caption` is honored: replace the
        # tag separator with a space so the caption is kept as a single
        # free-form tag.
        if self.remove_tag_separators:
            caption = caption.replace(self.context.tag_separator, ' ').strip()
        return caption, caption
