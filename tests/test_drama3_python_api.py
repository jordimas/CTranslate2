"""Python API boundary tests with a fake native engine; no model inference claims."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tts/python"))
from faster_tts import Model, SynthesisResult
from faster_tts.__main__ import main


class Drama3PythonAPITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "Hello": 1}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        special = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|audio_pad|>"]
        special += [f"<|speaker:{i}|>" for i in range(16)] + [f"<|semantic:{i}|>" for i in range(4)]
        tokenizer.add_special_tokens([AddedToken(token, special=True) for token in special])
        tokenizer.save(str(self.root / "tokenizer.json"))
        config = {"model_type": "fish_omni", "audio_decoder_config": {"num_codebooks": 10, "vocab_size": 4},
                  "semantic_start_token_id": tokenizer.token_to_id("<|semantic:0|>"),
                  "semantic_end_token_id": tokenizer.token_to_id("<|semantic:3|>"),
                  "eos_token_id": tokenizer.token_to_id("<|im_end|>"),
                  "audio_pad_token_id": tokenizer.token_to_id("<|audio_pad|>"),
                  "text_config": {"max_position_embeddings": 8192}}
        (self.root / "config.json").write_text(json.dumps({"model_type": "drama3", "ct2_layout_version": 1,
                                                          "source_config": config, "source_revision": "source-commit",
                                                          "quantization_policy": "all_matrices_int8",
                                                          "matrix_dtype": "int8", "activation_dtype": "float32"}))
        (self.root / "model.bin").write_bytes(b"synthetic assets; native engine is mocked")
        self.result = {"audio": np.zeros(2048, np.float32), "codes": np.zeros((1, 10), np.int32),
                       "sample_rate": 44100, "finish_reason": "eos"}
        self.backend = Mock()
        self.backend.generate.return_value = self.result
        self.constructor = Mock(return_value=self.backend)
        self.native_patch = patch("faster_tts.model._native_class", return_value=self.constructor)
        self.native_patch.start()
        self.addCleanup(self.native_patch.stop)

    def test_reuses_loaded_backend_and_passes_contiguous_int32_prompt(self):
        model = Model(self.root, intra_threads=3, prefill_chunk_size=2)
        self.constructor.assert_called_once_with(str(self.root.resolve()), intra_threads=3, compute_type="int8_float32")
        first = model.synthesize("Hello", max_frames=1)
        second = model.synthesize("Hello", max_frames=2)
        self.assertIsInstance(first, SynthesisResult)
        self.assertEqual(second.finish_reason, "eos")
        self.assertEqual(second.sample_rate, 44100)
        self.assertEqual(second.audio.shape, (2048,))
        self.assertEqual(self.constructor.call_count, 1)
        self.assertEqual(self.backend.generate.call_count, 2)
        rows = self.backend.generate.call_args.args[0]
        self.assertEqual(rows.dtype, np.int32)
        self.assertTrue(rows.flags.c_contiguous)
        self.assertEqual(rows.shape[1], 11)
        self.assertFalse(np.any(rows[:, 1:]))
        self.assertEqual(self.backend.generate.call_args.kwargs, {"max_frames": 2, "prefill_chunk_size": 2,
                                                                  "temperature": .8, "top_k": 30, "top_p": .8,
                                                                  "seed": 0, "greedy": False})

    def test_sampling_parameters_are_validated_and_forwarded_in_greedy_mode_too(self):
        model = Model(self.root)
        invalid = ({"temperature": 0}, {"temperature": float("nan")}, {"temperature": True},
                   {"top_k": 0}, {"top_k": 1.5}, {"top_p": 0}, {"top_p": 1.1},
                   {"top_p": float("inf")}, {"seed": -1}, {"seed": 2 ** 64}, {"seed": True}, {"greedy": 1})
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model.synthesize("Hello", **kwargs)
        self.backend.generate.assert_not_called()
        model.synthesize("Hello", max_frames=1, temperature=.6, top_k=7, top_p=.95,
                          seed=2 ** 64 - 1, greedy=True)
        self.assertEqual(self.backend.generate.call_args.kwargs,
                          {"max_frames": 1, "prefill_chunk_size": 16, "temperature": .6,
                           "top_k": 7, "top_p": .95, "seed": 2 ** 64 - 1, "greedy": True})

    def test_model_explicit_remote_resolves_before_native_load(self):
        snapshot = Mock(return_value=str(self.root))
        hub = Mock(snapshot_download=snapshot)
        cache_dir = self.root / "hub-cache"
        with patch("faster_tts.cache.import_module", return_value=hub):
            model = Model("hf://organization/converted-drama3", revision="pinned-commit",
                          cache_dir=cache_dir, local_files_only=True)
        self.assertEqual(snapshot.call_args.kwargs["repo_id"], "organization/converted-drama3")
        self.assertEqual(snapshot.call_args.kwargs["revision"], "pinned-commit")
        self.assertEqual(snapshot.call_args.kwargs["cache_dir"], str(cache_dir))
        self.assertTrue(snapshot.call_args.kwargs["local_files_only"])
        self.constructor.assert_called_once_with(str(self.root.resolve()), intra_threads=8,
                                                  compute_type="int8_float32")
        self.backend.generate.assert_not_called()
        self.assertIsNotNone(model)

    def test_info_reports_actual_configuration_without_inference(self):
        model = Model(self.root, intra_threads=3, compute_type="float32", prefill_chunk_size=2)
        expected = {"backend": "cpu", "codec_device": "cpu", "mode": "offline", "gpu_supported": False,
                    "compute_type": "float32", "intra_threads": 3, "prefill_chunk_size": 2,
                    "model_path": str(self.root.resolve()), "source_revision": "source-commit",
                    "quantization_policy": "all_matrices_int8", "matrix_dtype": "int8", "activation_dtype": "float32"}
        self.assertEqual(model.info(), expected)
        changed = model.info()
        changed["backend"] = "cuda"
        self.assertEqual(model.info()["backend"], "cpu")
        self.backend.generate.assert_not_called()

    def test_info_does_not_invent_unrecorded_quantization_policy(self):
        path = self.root / "config.json"
        metadata = json.loads(path.read_text())
        metadata.pop("quantization_policy")
        metadata.pop("source_revision")
        path.write_text(json.dumps(metadata))
        info = Model(self.root).info()
        self.assertIsNone(info["quantization_policy"])
        self.assertIsNone(info["source_revision"])
        self.backend.generate.assert_not_called()

    def test_validation_before_native_call(self):
        for args in ({"intra_threads": 0}, {"intra_threads": True}, {"prefill_chunk_size": -1}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Model(self.root, **args)
        model = Model(self.root)
        for frames in (0, -1, True, 1.5, 2 ** 31):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                model.synthesize("Hello", max_frames=frames)
        with self.assertRaises(ValueError):
            model.synthesize("Hello", max_frames=8192)
        with self.assertRaises(ValueError):
            model.synthesize("<|im_end|>")
        self.backend.generate.assert_not_called()

    def test_missing_assets_and_missing_native_binding_are_actionable(self):
        with self.assertRaises(FileNotFoundError):
            Model(self.root / "absent")
        self.native_patch.stop()
        with patch("faster_tts.model.import_module", side_effect=ImportError("no extension")):
            with self.assertRaisesRegex(RuntimeError, "experimental CTranslate2 CPU Drama3 binding"):
                Model(self.root)
        self.native_patch.start()

    def test_native_results_reject_nan_wrong_shapes_and_inconsistent_length(self):
        model = Model(self.root)
        for update in ({"audio": np.full(2048, np.nan, np.float32)},
                       {"audio": np.zeros((1, 2048), np.float32)},
                       {"audio": np.zeros(2048, np.float64)},
                       {"codes": np.zeros((1, 9), np.int32)},
                       {"codes": np.zeros((1, 10), np.int64)},
                       {"audio": np.zeros(2047, np.float32)},
                       {"sample_rate": 22050}, {"finish_reason": "unknown"}):
            self.backend.generate.return_value = dict(self.result, **update)
            with self.subTest(update=list(update)), self.assertRaises(RuntimeError):
                model.synthesize("Hello", max_frames=1)
        self.backend.generate.return_value = dict(self.result, finish_reason="length")
        self.assertEqual(model.synthesize("Hello", max_frames=1).finish_reason, "length")
        self.backend.generate.return_value = {"audio": np.empty(0, np.float32), "codes": np.empty((0, 10), np.int32),
                                               "sample_rate": 44100, "finish_reason": "eos"}
        self.assertEqual(model.synthesize("Hello").audio.size, 0)

    def test_opt_in_retry_preserves_attempts_cost_prompt_and_seed_wrap(self):
        model = Model(self.root)
        first = dict(self.result, finish_reason="length", timings={"backbone": 2., "codec": 1.})
        second = dict(self.result, audio=np.ones(2048, np.float32), timings={"backbone": 3., "codec": .5})
        self.backend.generate.side_effect = [first, second]
        with patch.object(model._prompt, "prepare", wraps=model._prompt.prepare) as prepare:
            result = model.synthesize("Hello", max_frames=1, seed=2 ** 64 - 1, max_retries=1)
        prepare.assert_called_once_with("Hello")
        calls = self.backend.generate.call_args_list
        self.assertIs(calls[0].args[0], calls[1].args[0])
        self.assertEqual([call.kwargs["seed"] for call in calls], [2 ** 64 - 1, 0])
        self.assertEqual(result.finish_reason, "eos")
        np.testing.assert_array_equal(result.audio, second["audio"])
        self.assertEqual(result.timings, {"backbone": 5., "codec": 1.5})
        self.assertIsInstance(result.attempts, tuple)
        self.assertEqual([a["finish_reason"] for a in result.attempts], ["length", "eos"])
        self.assertEqual([a["seed"] for a in result.attempts], [2 ** 64 - 1, 0])
        self.assertTrue(all(a["frames"] == 1 and a["seconds"] >= 0 for a in result.attempts))
        self.assertTrue(all("audio" not in a and "codes" not in a for a in result.attempts))

    def test_retry_is_bounded_returns_last_partial_and_default_or_greedy_skip(self):
        model = Model(self.root)
        self.backend.generate.return_value = dict(self.result, finish_reason="length")
        for kwargs in ({}, {"max_retries": 0}, {"max_retries": 1, "greedy": True}):
            self.backend.generate.reset_mock()
            result = model.synthesize("Hello", max_frames=1, **kwargs)
            self.assertEqual(self.backend.generate.call_count, 1)
            self.assertEqual(len(result.attempts), 1)
        self.backend.generate.reset_mock()
        result = model.synthesize("Hello", max_frames=1, seed=17, max_retries=1)
        self.assertEqual(self.backend.generate.call_count, 2)
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual([a["seed"] for a in result.attempts], [17, 18])
        self.backend.generate.reset_mock()
        self.backend.generate.return_value = self.result
        self.assertEqual(model.synthesize("Hello", max_retries=1).finish_reason, "eos")
        self.backend.generate.assert_called_once()

    def test_retry_validates_each_attempt_and_never_catches_native_errors(self):
        model = Model(self.root)
        for bad in (True, False, -1, 2, 1., None, "1"):
            with self.subTest(max_retries=bad), self.assertRaises(ValueError):
                model.synthesize("Hello", max_retries=bad)
        self.backend.generate.assert_not_called()
        for first in (dict(self.result, finish_reason="length", audio=np.full(2048, np.nan, np.float32)),
                      dict(self.result, finish_reason="length", timings={"codec": -1.}),
                      RuntimeError("native failure")):
            self.backend.generate.reset_mock()
            self.backend.generate.side_effect = [first, self.result]
            with self.subTest(first=type(first)), self.assertRaises(RuntimeError):
                model.synthesize("Hello", max_frames=1, max_retries=1)
            self.backend.generate.assert_called_once()
        self.backend.generate.reset_mock()
        self.backend.generate.side_effect = [dict(self.result, finish_reason="length"),
                                            dict(self.result, sample_rate=1)]
        with self.assertRaises(RuntimeError):
            model.synthesize("Hello", max_frames=1, max_retries=1)
        self.assertEqual(self.backend.generate.call_count, 2)

    def test_cli_writes_mono_pcm16_with_clipping(self):
        output_path = self.root / "sample.wav"
        result = SynthesisResult(np.array([-2, -1, 0, .5, 1, 2], np.float32), 44100,
                                 np.empty((0, 10), np.int32), "length")
        fake = Mock()
        fake.return_value.synthesize.return_value = result
        arguments = ["faster-tts", str(self.root), "--text", "Hello", "--output", str(output_path)]
        with patch("faster_tts.__main__.Model", fake), patch.object(sys, "argv", arguments), patch("builtins.print"):
            main()
        with wave.open(str(output_path), "rb") as stream:
            self.assertEqual((stream.getnchannels(), stream.getsampwidth(), stream.getframerate()), (1, 2, 44100))
            np.testing.assert_array_equal(np.frombuffer(stream.readframes(6), dtype="<i2"),
                                          [-32767, -32767, 0, 16384, 32767, 32767])

    def test_cli_preserves_explicit_remote_identifier_and_cache_options(self):
        fake = Mock()
        fake.return_value.synthesize.return_value = SynthesisResult(
            np.zeros(2048, np.float32), 44100, np.zeros((1, 10), np.int32), "eos")
        arguments = ["faster-tts", "hf://organization/converted-drama3", "--revision", "pinned-commit",
                     "--cache-dir", str(self.root / "cache"), "--local-files-only",
                     "--text", "Hello", "--output", str(self.root / "remote.wav")]
        with patch("faster_tts.__main__.Model", fake), patch.object(sys, "argv", arguments), patch("builtins.print"):
            main()
        fake.assert_called_once_with("hf://organization/converted-drama3", intra_threads=8,
                                     compute_type="int8_float32", revision="pinned-commit",
                                     cache_dir=self.root / "cache", local_files_only=True)

    def test_rejects_zero_stride_native_audio_instead_of_repeating_first_sample(self):
        # Reproduces pybind11 2.11's rank-one allocation metadata under NumPy 2.
        # The underlying buffer can contain correct samples while its public
        # stride makes every element expose the first sample.
        model = Model(self.root)
        samples = np.arange(2048, dtype=np.float32)
        broken = np.lib.stride_tricks.as_strided(samples, shape=(2048,), strides=(0,))
        self.assertEqual(broken.strides, (0,))
        self.backend.generate.return_value = dict(self.result, audio=broken)
        with self.assertRaisesRegex(RuntimeError, "invalid audio/code metadata"):
            model.synthesize("Hello", max_frames=1)


if __name__ == "__main__":
    unittest.main()
