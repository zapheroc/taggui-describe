from auto_captioning.auto_captioning_model import AutoCaptioningModel
from auto_captioning.settings_group import (LLAMA_GROUPS, PIXAI_TAGGER_GROUPS,
                                            SettingGroup, TRANSFORMERS_GROUPS,
                                            WD_TAGGER_GROUPS)

# The model classes are deliberately *not* imported at module level. Importing
# a model class pulls in its backend's heavy libraries (transformers + torch,
# onnxruntime, llama_cpp, bitsandbytes, ...), and `widgets/auto_captioner.py`
# imports this module at app startup to populate the model combo box. Keeping
# the imports out of module scope means a browsing/tagging session that never
# captions loads none of those libraries. `get_model_class` below imports each
# model lazily, and only the one actually selected, so each captioner loads
# only its own libraries (e.g. the WD Tagger loads onnxruntime but not
# transformers).
MODELS = [
    'fancyfeast/llama-joycaption-beta-one-hf-llava',
    'microsoft/Florence-2-large-ft',
    'microsoft/Florence-2-large',
    'microsoft/Florence-2-base-ft',
    'microsoft/Florence-2-base',
    'MiaoshouAI/Florence-2-large-PromptGen-v2.0',
    'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
    'microsoft/Phi-3-vision-128k-instruct',
    'llava-hf/llava-v1.6-mistral-7b-hf',
    'llava-hf/llava-v1.6-vicuna-7b-hf',
    'llava-hf/llava-v1.6-vicuna-13b-hf',
    'llava-hf/llava-v1.6-34b-hf',
    'xtuner/llava-llama-3-8b-v1_1-transformers',
    'vikhyatk/moondream2',
    'vikhyatk/moondream1',
    'SmilingWolf/wd-eva02-large-tagger-v3',
    'SmilingWolf/wd-vit-large-tagger-v3',
    'SmilingWolf/wd-swinv2-tagger-v3',
    'SmilingWolf/wd-convnext-tagger-v3',
    'SmilingWolf/wd-vit-tagger-v3',
    'SmilingWolf/wd-v1-4-moat-tagger-v2',
    'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
    'SmilingWolf/wd-v1-4-convnext-tagger-v2',
    'SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    'SmilingWolf/wd-v1-4-vit-tagger-v2',
    'pixai-labs/pixai-tagger-v1.0',
    'llava-hf/llava-1.5-7b-hf',
    'llava-hf/llava-1.5-13b-hf',
    'llava-hf/bakLlava-v1-hf',
    'Salesforce/instructblip-vicuna-7b',
    'Salesforce/instructblip-vicuna-13b',
    'Salesforce/instructblip-flan-t5-xl',
    'Salesforce/instructblip-flan-t5-xxl',
    'Salesforce/blip2-opt-2.7b',
    'Salesforce/blip2-opt-6.7b',
    'Salesforce/blip2-opt-6.7b-coco',
    'Salesforce/blip2-flan-t5-xl',
    'Salesforce/blip2-flan-t5-xxl',
    'microsoft/kosmos-2-patch14-224',
    'llmfan46/gemma-4-31B-it-uncensored-heretic',
    'HauhauCS/Qwen3.8-27B-Uncensored'
]


def get_model_class(model_id: str) -> type[AutoCaptioningModel]:
    """Return the captioning model class for `model_id`.

    Each model class is imported only when its branch is taken, so importing
    one backend never loads another's libraries (see the module note). The
    dispatch order (substring matching on the lowercased id) is unchanged.
    """
    lowercase_model_id = model_id.lower()
    if 'florence' in lowercase_model_id:
        from auto_captioning.models.florence_2 import Florence2, \
            Florence2Promptgen
        if 'promptgen' in lowercase_model_id:
            return Florence2Promptgen
        return Florence2
    if 'joycaption' in lowercase_model_id:
        from auto_captioning.models.joycaption import Joycaption
        return Joycaption
    if 'kosmos' in lowercase_model_id:
        from auto_captioning.models.kosmos_2 import Kosmos2
        return Kosmos2
    if 'llava-v1.6-34b' in lowercase_model_id:
        from auto_captioning.models.llava_next import LlavaNext34b
        return LlavaNext34b
    if 'llava-v1.6-mistral' in lowercase_model_id:
        from auto_captioning.models.llava_next import LlavaNextMistral
        return LlavaNextMistral
    if 'llava-v1.6-vicuna' in lowercase_model_id:
        from auto_captioning.models.llava_next import LlavaNextVicuna
        return LlavaNextVicuna
    if 'llava-llama-3' in lowercase_model_id:
        from auto_captioning.models.llava_llama_3 import LlavaLlama3
        return LlavaLlama3
    if 'llava' in lowercase_model_id:
        from auto_captioning.models.llava_1_point_5 import Llava1Point5
        return Llava1Point5
    if 'moondream1' in lowercase_model_id:
        from auto_captioning.models.moondream import Moondream1
        return Moondream1
    if 'moondream2' in lowercase_model_id:
        from auto_captioning.models.moondream import Moondream2
        return Moondream2
    if 'phi-3' in lowercase_model_id:
        from auto_captioning.models.phi_3_vision import Phi3Vision
        return Phi3Vision
    if 'pixai' in lowercase_model_id:
        from auto_captioning.models.pixai_tagger import PixAiTagger
        return PixAiTagger
    if 'wd' in lowercase_model_id and 'tagger' in lowercase_model_id:
        from auto_captioning.models.wd_tagger import WdTagger
        return WdTagger
    if 'gemma-4-31b' in lowercase_model_id:
        from auto_captioning.models.gemma_4 import Gemma4
        return Gemma4
    if 'qwen3.8' in lowercase_model_id:
        from auto_captioning.models.qwen_3_8 import Qwen38
        return Qwen38
    # Unknown models default to the generic transformers backend, which can
    # load any Hugging Face repo or local model directory.
    from auto_captioning.transformers_captioning_model import (
        TransformersCaptioningModel)
    return TransformersCaptioningModel


def get_setting_groups(model_id: str) -> set[SettingGroup]:
    """Return the settings rows the model with `model_id` shows, WITHOUT
    importing any model class.

    This mirrors `get_model_class`'s dispatch (same precedence) but returns the
    group-set constants from `settings_group.py` instead of a class, so
    `widgets/auto_captioner.py` can show/hide its settings rows at construction
    time without loading a backend's heavy libraries at startup. It must always
    agree with `get_model_class(model_id).get_setting_groups()` — the model
    classes return these same constants, so the only thing that must stay in
    sync is this dispatch. Keep it in sync with `get_model_class` when adding
    or reordering models.
    """
    lowercase_model_id = model_id.lower()
    # The Hugging Face (transformers) family. In `get_model_class` every
    # florence / joycaption / kosmos / llava / moondream / phi-3 branch is
    # checked *before* the pixai / wd-tagger / llama branches, so any id
    # matching one of these uses the transformers rows even if it also happens
    # to contain a later keyword.
    if ('florence' in lowercase_model_id
            or 'joycaption' in lowercase_model_id
            or 'kosmos' in lowercase_model_id
            or 'llava' in lowercase_model_id
            or 'moondream1' in lowercase_model_id
            or 'moondream2' in lowercase_model_id
            or 'phi-3' in lowercase_model_id):
        return TRANSFORMERS_GROUPS
    if 'pixai' in lowercase_model_id:
        return PIXAI_TAGGER_GROUPS
    if 'wd' in lowercase_model_id and 'tagger' in lowercase_model_id:
        return WD_TAGGER_GROUPS
    if 'gemma-4-31b' in lowercase_model_id:
        return LLAMA_GROUPS
    if 'qwen3.8' in lowercase_model_id:
        return LLAMA_GROUPS
    # Unknown models default to the generic transformers backend.
    return TRANSFORMERS_GROUPS
