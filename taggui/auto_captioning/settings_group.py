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

