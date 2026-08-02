from enum import Enum


# TODO Fix up these settings groups. I might do a 1-1 mapping of settings item here
class SettingGroup(Enum):
    def __new__(cls, label, is_advanced):
        obj = object.__new__(cls)
        obj._value_ = label
        obj.is_advanced = is_advanced
        return obj

    PROMPT = ("Prompt", False)
    CAPTION_START = ("Start caption with", False)
    DEVICE = ("False", False)
    LOAD_IN_4_BIT = ("Load in 4-bit (requires bitsandbytes)", False)
    REMOVE_TAG_SEPARATORS = ("Remove tag separators in captions", False)
    DISCOURAGED_WORDS = ("Discourage from caption", True)
    FORCED_WORDS = ("Include in caption", True)
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
    # One group to hold WD-tagger since it has all unique values
    WD_TAGGER = ("WD_TAGGER", False)

