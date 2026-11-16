from enum import Enum


class SettingGroup(Enum):
    def __new__(cls, label, is_advanced):
        obj = object.__new__(cls)
        obj._value_ = label
        obj.is_advanced = is_advanced
        return obj

    PROMPT = ("Prompt", False)
    CAPTION_START = ("Start caption with", False)
    DEVICE = ("Device", False)
    LOAD_IN_4_BIT = ("Load in 4-bit (requires bitsandbytes)", False)
    REMOVE_TAG_SEPARATORS = ("Remove tag separators in captions", False)
    DISCOURAGED_WORDS = ("Discourage from caption", True)
    FORCED_WORDS = ("Include in caption", True)
    SEED = ("Seed", True)
    MIN_TOKENS = ("Minimum tokens", True)
    MAX_TOKENS = ("Maximum tokens", True)
    NUM_BEAMS = ("Number of beams", True)
    LENGTH_PENALTY = ("Length penalty", True)
    USE_SAMPLING = ("Use sampling", True)
    TEMPERATURE = ("Temperature", True)
    TOP_K = ("Top-k", True)
    TOP_P = ("Top-p", True)
    REPETITION_PENALTY = ("Repetition penalty", True)
    NO_REPEAT_NGRAM = ("No-repeat n-gram size", True)
    GPU_INDEX = ("GPU index", True)
    # Shared tagger options (show probabilities, maximum tags, tags to
    # exclude), used by both the WD Tagger and the PixAI Tagger.
    TAGGER = ("TAGGER", False)
    # One group to hold the WD Tagger's minimum probability row, since it is
    # unique to the WD Tagger (the PixAI Tagger uses per-category
    # thresholds instead).
    WD_TAGGER = ("WD_TAGGER", False)
    # The PixAI Tagger's per-category thresholds.
    PIXAI_TAGGER = ("PIXAI_TAGGER", True)


# The set of settings rows each captioning family shows. These are defined here
# (rather than inline in the model classes) so that `widgets/auto_captioner.py`
# can show/hide its rows at *widget construction* time via
# `models_list.get_setting_groups(model_id)` — without importing a model class,
# which would load that backend's heavy libraries (torch, transformers,
# onnxruntime, ...) at app startup. The model classes' `get_setting_groups()`
# return these same constants, so the *contents* of each set have a single
# source of truth (only the id→family dispatch is shared with `models_list`).
SHARED_GROUPS: set[SettingGroup] = {
    SettingGroup.PROMPT,
    SettingGroup.REMOVE_TAG_SEPARATORS,
}

WD_TAGGER_GROUPS: set[SettingGroup] = {
    SettingGroup.TAGGER,
    SettingGroup.WD_TAGGER,
    SettingGroup.DEVICE,
    SettingGroup.GPU_INDEX,
}

PIXAI_TAGGER_GROUPS: set[SettingGroup] = {
    SettingGroup.TAGGER,
    SettingGroup.PIXAI_TAGGER,
    SettingGroup.DEVICE,
    SettingGroup.GPU_INDEX,
}

# The Hugging Face (transformers) backend's rows.
TRANSFORMERS_GROUPS: set[SettingGroup] = SHARED_GROUPS | {
    SettingGroup.CAPTION_START,
    SettingGroup.LOAD_IN_4_BIT,
    SettingGroup.DEVICE,
    SettingGroup.DISCOURAGED_WORDS,
    SettingGroup.FORCED_WORDS,
    SettingGroup.MIN_TOKENS,
    SettingGroup.MAX_TOKENS,
    SettingGroup.NUM_BEAMS,
    SettingGroup.LENGTH_PENALTY,
    SettingGroup.USE_SAMPLING,
    SettingGroup.TEMPERATURE,
    SettingGroup.TOP_K,
    SettingGroup.TOP_P,
    SettingGroup.REPETITION_PENALTY,
    SettingGroup.NO_REPEAT_NGRAM,
    SettingGroup.GPU_INDEX,
}

# The llama.cpp (GGUF) backend's rows.
LLAMA_GROUPS: set[SettingGroup] = SHARED_GROUPS | {
    SettingGroup.SEED,
    SettingGroup.MAX_TOKENS,
    SettingGroup.TEMPERATURE,
    SettingGroup.TOP_K,
    SettingGroup.TOP_P,
    SettingGroup.REPETITION_PENALTY,
}

