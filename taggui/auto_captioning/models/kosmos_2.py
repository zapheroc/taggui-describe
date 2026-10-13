from auto_captioning.transformers_captioning_model import TransformersCaptioningModel

class Kosmos2(TransformersCaptioningModel):
    @staticmethod
    def format_prompt(prompt: str) -> str:
        return f'<grounding>{prompt}'

    @staticmethod
    def postprocess_image_prompt(image_prompt: str) -> str:
        return image_prompt.replace('<grounding>', '')

    def postprocess_generated_text(self, generated_text: str) -> str:
        generated_text, _ = self.processor.post_process_generation(
            generated_text)
        return generated_text
