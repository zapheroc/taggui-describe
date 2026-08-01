from enum import StrEnum

# TODO Fix up these settings groups. I might do a 1-1 mapping of settings item here
class SettingGroup(StrEnum):
    # Basic
    PROMPT = 'prompt'
    CAPTION_START = 'caption_start'
    DEVICE = 'device'                       # device combo + load_in_4_bit
    REMOVE_TAG_SEPARATORS = 'remove_tag_separators'
    # Advanced - transformers generation params
    BAD_FORCED_WORDS = 'bad_forced_words'
    MIN_MAX_TOKENS = 'min_max_tokens'
    NUM_BEAMS = 'num_beams'
    LENGTH_PENALTY = 'length_penalty'
    SAMPLING = 'sampling'                    # do_sample, temperature, top_k, top_p
    REPETITION_PENALTY = 'repetition_penalty'
    NO_REPEAT_NGRAM = 'no_repeat_ngram'
    GPU_INDEX = 'gpu_index'
    # Advanced - llama.cpp generation params (subset that overlaps)
    LLAMA_MAX_TOKENS = 'llama_max_tokens'
    LLAMA_SAMPLING = 'llama_sampling'        # temperature, top_k, top_p
    LLAMA_REPEAT_PENALTY = 'llama_repeat_penalty'
    # WD Tagger
    WD_TAGGER = 'wd_tagger'
