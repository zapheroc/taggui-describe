import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from auto_captioning.auto_captioning_model import AutoCaptioningModel
from utils.image import Image


class Gemma4(AutoCaptioningModel):
    dtype = torch.bfloat16
    transformers_model_class = AutoModelForImageTextToText
    use_safetensors = True
    image_mode = 'RGB'

    # Gemma 4 visual token budget: 70, 140, 280 (default), 560, 1120.
    # Lower is faster and fine for captioning.
    visual_token_budget = 280

    @staticmethod
    def get_default_prompt() -> str:
        return 'Describe this image in detail.'

    def get_processor(self):
        # `padding_side='left'` is recommended for generation.
        return AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, padding_side='left')

    def get_model_inputs(self, image_prompt: str, image: Image):
        pil_image = self.load_image(image)
        text = self.get_input_text(image_prompt)
        # Image BEFORE text, per Gemma 4 best practices.
        messages = [
            {
                'role': 'system',
                'content': [
                    {'type': 'text',
                     'text': 'You are a helpful image captioner.'}
                ]
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': pil_image},
                    {'type': 'text', 'text': text}
                ]
            }
        ]
        # Optionally control the visual token budget if the processor
        # exposes it (attribute name may vary by transformers version).
        apply_kwargs = {}
        if hasattr(self.processor, 'image_processor') and \
                hasattr(self.processor.image_processor, 'num_soft_tokens'):
            self.processor.image_processor.num_soft_tokens = \
                self.visual_token_budget

        model_inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
            **apply_kwargs
        ).to(self.device, **self.dtype_argument)
        self._input_token_count = model_inputs['input_ids'].shape[-1]
        return model_inputs

    def get_caption_from_generated_tokens(
            self, generated_token_ids: torch.Tensor,
            image_prompt: str) -> str:
        generated_token_ids = generated_token_ids[:, self._input_token_count:]
        caption = self.processor.batch_decode(
            generated_token_ids, skip_special_tokens=True)[0]
        caption = caption.strip()
        if self.caption_start.strip():
            caption = f'{self.caption_start.strip()} {caption}'.strip()
        if self.remove_tag_separators:
            caption = caption.replace(self.thread.tag_separator, ' ')
        return caption
