"""Plain text-only Drama 3 prompt preparation without a model runtime.

Protocol evidence: Fish Speech's fish_omni/generate.py build_drama3_prompt:
https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/fish_omni/generate.py
The checkpoint's generic chat_template.jinja is NOT the TTS prompt builder.
Literals are encoded separately; concatenating them before encoding changes BPE
boundaries. This module implements no references, history, or waveform synthesis.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
from tokenizers import Tokenizer


_WHITE_SPACE = "".join(chr(value) for value in
                      list(range(9, 14)) + [32, 133, 160, 5760]
                      + list(range(8192, 8203)) + [8232, 8233, 8239, 8287, 12288])
_SPEAKER = re.compile(r"<\|speaker:(?:[0-9]|1[0-5])\|>")
_RESERVED = ("<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>",
             "<|interleave|>", "<|semantic:")


@dataclass(frozen=True)
class Drama3Prompt:
    """CPU contiguous int64 rows [N,11], plus a bool semantic mask [N].

    Column zero holds tokenizer IDs; columns 1..10 are zero for text rows, NOT
    audio_pad_token_id. Audio rows would hold ten codec indices and an offset
    semantic ID, but this text-only builder never emits them. semantic_mask is
    therefore entirely false. audio_pad_token_id names a tokenizer special token;
    it is exposed as metadata and is not the filler for codec columns.
    """
    rows: np.ndarray
    semantic_mask: np.ndarray
    normalized_text: str
    segments: tuple
    semantic_start_token_id: int
    semantic_end_token_id: int
    eos_token_id: int
    audio_pad_token_id: int


class Drama3PromptBuilder:
    """Load frozen local tokenizer.json/config.json; perform no downloads."""

    def __init__(self, checkpoint_dir, max_prompt_rows=8192):
        if (not isinstance(max_prompt_rows, int) or isinstance(max_prompt_rows, bool)
                or not 1 <= max_prompt_rows <= 8192):
            raise ValueError("max_prompt_rows must be an integer in 1..8192")
        self.max_prompt_rows = max_prompt_rows
        directory = Path(checkpoint_dir)
        self.config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if isinstance(self.config, dict) and "source_config" in self.config:
            self.config = self.config["source_config"]
        if not isinstance(self.config, dict) or self.config.get("model_type") != "fish_omni":
            raise ValueError("expected a fish_omni checkpoint")
        audio = self.config.get("audio_decoder_config", {})
        if not isinstance(audio, dict) or audio.get("num_codebooks") != 10:
            raise ValueError("plain Drama 3 prompt requires ten codebooks")
        size = audio.get("vocab_size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("invalid audio codebook vocabulary size")
        for field in ("semantic_start_token_id", "semantic_end_token_id", "eos_token_id", "audio_pad_token_id"):
            value = self.config.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"missing or invalid {field}")
        if self.config["semantic_end_token_id"] - self.config["semantic_start_token_id"] + 1 != size:
            raise ValueError("semantic ID range disagrees with codebook size")
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        expected = {"<|im_end|>": self.config["eos_token_id"],
                    "<|audio_pad|>": self.config["audio_pad_token_id"]}
        expected.update({f"<|semantic:{index}|>": self.config["semantic_start_token_id"] + index
                         for index in range(size)})
        for literal, token_id in expected.items():
            if self.tokenizer.token_to_id(literal) != token_id:
                raise ValueError(f"tokenizer/config mismatch: {literal}")
        for literal in ("<|im_start|>", "<|voice|>") + tuple(f"<|speaker:{i}|>" for i in range(16)):
            if self.tokenizer.token_to_id(literal) is None:
                raise ValueError(f"missing tokenizer special token: {literal}")

    def prepare(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        text = unicodedata.normalize("NFC", text)
        if not text.strip(_WHITE_SPACE):
            raise ValueError("text must contain non-whitespace characters")
        if any(marker in text for marker in _RESERVED):
            raise ValueError("text contains reserved prompt structure")
        for marker in re.finditer(re.escape("<|speaker"), text):
            if _SPEAKER.match(text, marker.start()) is None:
                raise ValueError("speaker tags must use canonical IDs 0..15")
        spoken = text if _SPEAKER.match(text.lstrip(_WHITE_SPACE)) else "<|speaker:0|> " + text
        segments = ("<|im_start|>system\n", "convert the provided text to speech",
                    "<|im_end|>\n", "<|im_start|>user\n", spoken,
                    "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>")
        token_ids = []
        for segment in segments:
            token_ids.extend(self.tokenizer.encode(segment).ids)
            if len(token_ids) > self.max_prompt_rows:
                raise ValueError(f"prompt exceeds {self.max_prompt_rows} rows")
        rows = np.zeros((len(token_ids), 11), dtype=np.int64)
        rows[:, 0] = token_ids
        return Drama3Prompt(rows=rows, semantic_mask=np.zeros(len(rows), dtype=np.bool_),
                            normalized_text=text, segments=segments,
                            **{field: self.config[field] for field in
                               ("semantic_start_token_id", "semantic_end_token_id",
                                "eos_token_id", "audio_pad_token_id")})
