"""Acquisition contract tests; synthetic assets only, no network or native model."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1]/"faster_tts/cache.py"
SPEC = importlib.util.spec_from_file_location("faster_tts_cache_under_test", SOURCE)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)/"converted"
        self.model.mkdir()
        (self.model/"model.bin").write_bytes(b"synthetic-binary-not-loaded")
        (self.model/"tokenizer.json").write_text(json.dumps({"model": {"type": "BPE"}}))
        self.config = {"model_type": "drama3", "ct2_layout_version": 1,
                       "source_config": {"text_config": {"num_hidden_layers": 40},
                                         "audio_decoder_config": {"num_codebooks": 10}}}
        self.save_config()

    def save_config(self):
        (self.model/"config.json").write_text(json.dumps(self.config))

    def test_local_has_no_hub_import(self):
        with patch.object(cache, "import_module", side_effect=AssertionError("Hub imported")):
            self.assertEqual(cache.resolve_model_path(self.model), self.model.resolve())

    def test_missing_local_or_bare_repo_never_downloads(self):
        with patch.object(cache, "import_module", side_effect=AssertionError("Hub imported")):
            for value in (str(self.model/"missing"), "organization/checkpoint"):
                with self.assertRaisesRegex(FileNotFoundError, "hf://"):
                    cache.resolve_model_path(value)

    def test_required_assets(self):
        for name in ("model.bin", "config.json", "tokenizer.json"):
            path = self.model/name
            original = path.read_bytes()
            path.write_bytes(b"")
            with self.assertRaisesRegex(FileNotFoundError, name):
                cache.resolve_model_path(self.model)
            path.write_bytes(original)

    def test_official_or_wrong_layout_rejected(self):
        for key, value in (("model_type", "fish_omni"), ("ct2_layout_version", 2),
                           ("ct2_layout_version", True), ("source_config", {})):
            original = self.config[key]
            self.config[key] = value
            self.save_config()
            with self.assertRaisesRegex(ValueError, "Unsupported converted Drama 3"):
                cache.resolve_model_path(self.model)
            self.config[key] = original
        self.save_config()

    def test_malformed_config_and_tokenizer(self):
        (self.model/"config.json").write_text("{broken")
        with self.assertRaisesRegex(ValueError, "config"):
            cache.resolve_model_path(self.model)
        self.save_config()
        (self.model/"tokenizer.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            cache.resolve_model_path(self.model)

    def test_hub_options_and_allowlist(self):
        download = Mock(return_value=str(self.model))
        with patch.object(cache, "import_module", return_value=types.SimpleNamespace(snapshot_download=download)):
            result = cache.resolve_model_path("hf://organization/checkpoint", revision="deadbeef",
                                              cache_dir=Path(self.temp.name)/"cache", local_files_only=True)
        self.assertEqual(result, self.model.resolve())
        options = download.call_args.kwargs
        self.assertEqual(options["repo_id"], "organization/checkpoint")
        self.assertEqual(options["revision"], "deadbeef")
        self.assertTrue(options["local_files_only"])
        self.assertEqual(options["cache_dir"], str(Path(self.temp.name)/"cache"))
        self.assertIn("model.bin", options["allow_patterns"])
        self.assertNotIn("*.safetensors", options["allow_patterns"])
        self.assertNotIn("*.gguf", options["allow_patterns"])

    def test_missing_optional_dependency(self):
        with patch.object(cache, "import_module", side_effect=ImportError("absent")):
            with self.assertRaisesRegex(ImportError, "pip install huggingface-hub"):
                cache.resolve_model_path("hf://organization/checkpoint")

    def test_hub_failure_preserves_cause_and_has_no_fallback(self):
        failure = OSError("not cached")
        download = Mock(side_effect=failure)
        with patch.object(cache, "import_module", return_value=types.SimpleNamespace(snapshot_download=download)):
            with self.assertRaisesRegex(RuntimeError, "local cache") as result:
                cache.resolve_model_path("hf://organization/checkpoint", local_files_only=True)
        self.assertIs(result.exception.__cause__, failure)
        download.assert_called_once()

    def test_remote_still_requires_converted_assets(self):
        (self.model/"model.bin").unlink()
        with patch.object(cache, "import_module", return_value=types.SimpleNamespace(snapshot_download=Mock(return_value=str(self.model)))):
            with self.assertRaisesRegex(FileNotFoundError, "safetensors and GGUF"):
                cache.resolve_model_path("hf://organization/checkpoint")

    def test_invalid_remote_identifiers_do_not_import(self):
        with patch.object(cache, "import_module", side_effect=AssertionError("Hub imported")):
            for value in ("hf://repo", "hf://org/repo/subdir", "hf://org/repo@main", "hf://../repo"):
                with self.assertRaisesRegex(ValueError, "organization/repository"):
                    cache.resolve_model_path(value)


if __name__ == "__main__":
    unittest.main()
