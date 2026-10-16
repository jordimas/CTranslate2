"""Independent prompt preparation tests; no CT2 extension or model weights."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tts/python"))
from faster_tts import Drama3PromptBuilder


class Drama3PromptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "Hello": 1, "café": 2}, unk_token="[UNK]"))
        self.tokenizer.normalizer = normalizers.NFC()
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        special = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|audio_pad|>"]
        special += [f"<|speaker:{i}|>" for i in range(16)]
        special += [f"<|semantic:{i}|>" for i in range(4)]
        self.tokenizer.add_special_tokens([AddedToken(token, special=True) for token in special])
        self.tokenizer.save(str(self.root / "tokenizer.json"))
        self.config = {"model_type": "fish_omni", "audio_decoder_config": {"num_codebooks": 10, "vocab_size": 4},
                       "semantic_start_token_id": self.tokenizer.token_to_id("<|semantic:0|>"),
                       "semantic_end_token_id": self.tokenizer.token_to_id("<|semantic:3|>"),
                       "eos_token_id": self.tokenizer.token_to_id("<|im_end|>"),
                       "audio_pad_token_id": self.tokenizer.token_to_id("<|audio_pad|>")}
        self.write_config()

    def write_config(self):
        (self.root / "config.json").write_text(json.dumps(self.config))

    def test_exact_segment_contract_zero_codec_columns_and_mask(self):
        builder = Drama3PromptBuilder(self.root)
        prompt = builder.prepare("Hello")
        expected = ("<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n",
                    "<|im_start|>user\n", "<|speaker:0|> Hello", "<|im_end|>\n",
                    "<|im_start|>assistant\n<|voice|>")
        self.assertEqual(prompt.segments, expected)
        ids = [token for segment in expected for token in self.tokenizer.encode(segment).ids]
        np.testing.assert_array_equal(prompt.rows[:, 0], ids)
        self.assertEqual(prompt.rows.shape, (len(ids), 11))
        self.assertEqual(prompt.rows.dtype, np.int64)
        self.assertTrue(prompt.rows.flags.c_contiguous)
        self.assertFalse(np.any(prompt.rows[:, 1:]))
        self.assertFalse(np.any(prompt.semantic_mask))
        self.assertEqual(prompt.semantic_mask.dtype, np.bool_)
        self.assertEqual(prompt.rows[-1, 0], self.tokenizer.token_to_id("<|voice|>"))

    def test_nfc_and_leading_unicode_whitespace_preserved(self):
        builder = Drama3PromptBuilder(self.root)
        first = builder.prepare("cafe\u0301")
        second = builder.prepare("café")
        np.testing.assert_array_equal(first.rows, second.rows)
        self.assertEqual(first.normalized_text, "café")
        prompt = builder.prepare("\u2003<|speaker:15|> Hello ")
        self.assertEqual(prompt.segments[4], "\u2003<|speaker:15|> Hello ")
        # U+001C is not Unicode White_Space: don't silently strip it.
        self.assertEqual(builder.prepare("\x1c").normalized_text, "\x1c")

    def test_rejects_empty_reserved_and_malformed_speakers(self):
        builder = Drama3PromptBuilder(self.root)
        for text in ("", "\u2003\t", "<|im_start|>bad", "<|im_end|>", "<|voice|>",
                     "<|text|>", "<|interleave|>", "<|semantic:0|>", "<|speaker:16|> Hello",
                     "<|speaker:01|> Hello", "<|speaker:-1|> Hello", "Hello <|speaker:no|>",
                     "<|speaker:١|> Hello", "<|speaker"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                builder.prepare(text)
        with self.assertRaises(TypeError):
            builder.prepare(None)

    def test_row_limit_and_config_tokenizer_mismatches(self):
        with self.assertRaises(ValueError):
            Drama3PromptBuilder(self.root, max_prompt_rows=1).prepare("Hello")
        for count in (0, True, 8193):
            with self.assertRaises(ValueError):
                Drama3PromptBuilder(self.root, max_prompt_rows=count)
        self.config["audio_pad_token_id"] += 1
        self.write_config()
        with self.assertRaises(ValueError):
            Drama3PromptBuilder(self.root)

    def test_frozen_real_tokenizer_assets_if_available(self):
        directories = [ROOT / "tts/models/drama-3-official", Path("/tmp/audiocpp-gguf/dd702bb072bc4d54")]
        directory = next((d for d in directories if (d / "tokenizer.json").is_file()), None)
        if directory is None:
            self.skipTest("frozen Drama 3 tokenizer fixture is unavailable")
        builder = Drama3PromptBuilder(directory)
        for text in ("Hello, this is an English test.", "Hola, esta es una prueba en español."):
            prompt = builder.prepare(text)
            ids = [token for segment in prompt.segments for token in builder.tokenizer.encode(segment).ids]
            np.testing.assert_array_equal(prompt.rows[:, 0], ids)
            self.assertEqual(prompt.rows.shape[1], 11)
            self.assertEqual(prompt.rows[-1, 0], builder.tokenizer.token_to_id("<|voice|>"))
            self.assertFalse(np.any(prompt.semantic_mask))


if __name__ == "__main__":
    unittest.main()
