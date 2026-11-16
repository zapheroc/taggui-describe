"""A lightweight CLIP ViT-B/32 tokenizer for token counting.

The app only ever *counts* tokens (the Stable Diffusion token counter and the
`tokens:` filter both compute `len(tokenizer(caption).input_ids) - 2`). Loading
`transformers.AutoTokenizer` for that pulls in `transformers` and `torch` at
startup (~0.7 s and a heavy dependency chain) for what is a one-line count.

The `clip-vit-base-patch32/` directory ships a fast-tokenizer `tokenizer.json`,
which the lightweight Rust `tokenizers` library (a pinned dependency of
`transformers`, so it is always installed) loads in milliseconds and produces
byte-identical token IDs to `transformers.AutoTokenizer` for the same input
(including the BOS/EOS tokens that the `- 2` offset accounts for). This wrapper
exposes exactly the interface the app uses: `tokenizer(text).input_ids`.
"""
from pathlib import Path

import tokenizers


class _Encoding:
    """Mimics the `transformers` encoding object's `input_ids` attribute so the
    existing call sites (`len(tokenizer(text).input_ids) - 2`) are unchanged."""
    def __init__(self, input_ids: list[int]):
        self.input_ids = input_ids


class ClipTokenizer:
    """A drop-in replacement for `transformers.AutoTokenizer` that only supports
    the single operation the app needs: encoding text into token IDs."""
    def __init__(self, tokenizer_directory: str | Path):
        tokenizer_path = Path(tokenizer_directory) / 'tokenizer.json'
        self._tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))

    def __call__(self, text: str) -> _Encoding:
        return _Encoding(self._tokenizer.encode(str(text)).ids)
