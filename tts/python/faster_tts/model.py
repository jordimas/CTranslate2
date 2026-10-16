"""Thin CPU inference API for converted Drama 3 checkpoints.

Requires a CTranslate2 build exposing _ext.Drama3. Ordinary upstream wheels may
not expose that experimental binding. Explicit hf:// identifiers optionally resolve
converted Hub snapshots; local paths never trigger downloads. No PyTorch runtime
or external TTS engine fallback is performed. Seeded sampling and diagnostic greedy decoding
do not themselves establish production speech quality.
"""

from dataclasses import dataclass, field
from importlib import import_module
import json

import math
import time
from numbers import Real

import numpy as np

from .prompt import Drama3PromptBuilder
from .cache import resolve_model_path


@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    codes: np.ndarray
    finish_reason: str
    timings: dict = field(default_factory=dict)
    attempts: tuple = ()


def _native_class():
    try:
        extension = import_module("ctranslate2._ext")
        return extension.Drama3
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("This requires the experimental CTranslate2 CPU Drama3 binding; "
                           "install a matching build of this repository.") from exc


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= np.iinfo(np.int32).max:
        raise ValueError(f"{name} must be a positive int32 integer")
    return value


class Model:
    """Load a converted model once, then synthesize plain text on CPU.

    Generation is offline, without cloning or reference audio. Seeded categorical
    sampling follows the model distribution but is not bitwise PyTorch RNG parity.
    Conversion must provide model.bin, a Drama3 layout-version1 config.json with
    source_config, and frozen tokenizer.json.

    Local example: ``Model("/path/to/converted-drama3")``. To opt into Hub
    resolution, install ``faster-tts-ct2[hub]`` and use
    ``Model("hf://YOUR_ORG/YOUR_CONVERTED_REPO", revision="PINNED_COMMIT")``.
    Those uppercase values are placeholders for a compatible converted repository
    and revision you provide; no published model repository is implied.
    ``local_files_only=True`` limits Hub resolution to already cached assets.
    """

    def __init__(self, model_path, intra_threads=8, compute_type="int8_float32", prefill_chunk_size=16,
                 *, revision=None, cache_dir=None, local_files_only=False):
        _positive_int(intra_threads, "intra_threads")
        _positive_int(prefill_chunk_size, "prefill_chunk_size")
        if not isinstance(compute_type, str) or not compute_type:
            raise ValueError("compute_type must be a nonempty string")
        directory = resolve_model_path(model_path, revision=revision, cache_dir=cache_dir,
                                       local_files_only=local_files_only)
        metadata = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self._prompt = Drama3PromptBuilder(directory)
        self._prefill_chunk_size = prefill_chunk_size
        self._backend = _native_class()(str(directory), intra_threads=intra_threads, compute_type=compute_type)
        self._info = {
            "backend": "cpu", "codec_device": "cpu", "mode": "offline", "gpu_supported": False,
            "compute_type": compute_type, "intra_threads": intra_threads,
            "prefill_chunk_size": prefill_chunk_size, "model_path": str(directory),
            "source_revision": metadata.get("source_revision"),
            "quantization_policy": metadata.get("quantization_policy"),
            "matrix_dtype": metadata.get("matrix_dtype"),
            "activation_dtype": metadata.get("activation_dtype"),
        }

    def info(self):
        """Return CPU/offline execution settings and recorded checkpoint metadata.

        Missing revision/quantization fields are returned as None; older converted
        assets may not record them. This does not scan weights or run inference.
        The compute type is the constructor setting accepted by the native loader.
        """
        # Metadata fields are scalars in supported converted configuration files.
        # JSON round-tripping also protects against unexpected nested metadata.
        return json.loads(json.dumps(self._info))

    def synthesize(self, text, max_frames=512, temperature=.8, top_k=30,
                   top_p=.8, seed=0, greedy=False, *, max_retries=0):
        """Return mono float32 audio and generated [frames,10] codec indices.

        finish_reason='length' means the frame budget was exhausted and audio may
        end mid-utterance; 'eos' means the model emitted its stop token.
        Opt-in max_retries=1 retries a valid length-capped sampled result once,
        using seed+1 (wrapping uint64) and fresh native state on the loaded model.
        Greedy decoding never retries. This can double cost and cannot detect
        incomplete speech ending with EOS. attempts records all attempt metadata;
        discarded attempt audio/codes are not retained. timings sums native phase
        timings across attempts; exhausted retries return the last partial result.
        """
        _positive_int(max_frames, "max_frames")
        if (not isinstance(temperature, Real) or isinstance(temperature, bool)
                or not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("temperature must be positive and finite")
        _positive_int(top_k, "top_k")
        if (not isinstance(top_p, Real) or isinstance(top_p, bool)
                or not math.isfinite(top_p) or not 0 < top_p <= 1):
            raise ValueError("top_p must be finite and in (0,1]")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 64:
            raise ValueError("seed must be a nonnegative uint64 integer")
        if not isinstance(greedy, bool):
            raise ValueError("greedy must be a boolean")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries not in (0, 1):
            raise ValueError("max_retries must be 0 or 1")
        prepared = self._prompt.prepare(text)
        maximum = self._prompt.config.get("text_config", {}).get("max_position_embeddings")
        if isinstance(maximum, int) and len(prepared.rows) + max_frames > maximum:
            raise ValueError("prompt plus max_frames exceeds model context capacity")
        if np.any(prepared.rows < 0) or np.any(prepared.rows > np.iinfo(np.int32).max):
            raise ValueError("prompt token IDs do not fit the native int32 interface")
        rows = np.ascontiguousarray(prepared.rows, dtype=np.int32)
        attempts = []
        aggregate_timings = {}
        for attempt in range(1 + (max_retries if not greedy else 0)):
            attempt_seed = (seed + attempt) % (2 ** 64)
            started = time.perf_counter()
            result = self._backend.generate(rows, max_frames=max_frames, prefill_chunk_size=self._prefill_chunk_size,
                                            temperature=float(temperature), top_k=top_k, top_p=float(top_p),
                                            seed=attempt_seed, greedy=greedy)
            if not isinstance(result, dict):
                raise RuntimeError("native Drama3 returned a malformed result")
            audio, codes = result.get("audio"), result.get("codes")
            rate, reason = result.get("sample_rate"), result.get("finish_reason")
            if (not isinstance(audio, np.ndarray) or audio.dtype != np.float32 or audio.ndim != 1
                    or not audio.flags.c_contiguous
                    or not np.all(np.isfinite(audio)) or not isinstance(codes, np.ndarray)
                    or codes.dtype != np.int32 or codes.ndim != 2 or codes.shape[1] != 10
                    or not codes.flags.c_contiguous
                    or len(codes) > max_frames or np.any(codes < 0) or rate != 44100
                    or reason not in ("eos", "length")):
                raise RuntimeError("native Drama3 returned invalid audio/code metadata")
            if audio.size != codes.shape[0] * 2048:
                raise RuntimeError("native Drama3 audio length disagrees with codec frame count")
            timings = result.get("timings", {})
            if (not isinstance(timings, dict)
                    or any(not isinstance(k, str) or not isinstance(v, Real)
                           or not math.isfinite(v) or v < 0 for k, v in timings.items())):
                raise RuntimeError("native Drama3 returned invalid timing metadata")
            attempts.append({"seed": attempt_seed, "finish_reason": reason,
                             "frames": len(codes), "seconds": time.perf_counter() - started,
                             "timings": dict(timings)})
            for phase, seconds in timings.items():
                aggregate_timings[phase] = aggregate_timings.get(phase, 0.) + seconds
            if reason != "length" or attempt == max_retries or greedy:
                break
            # Keep only compact metadata from discarded attempts.
            del result, audio, codes
        return SynthesisResult(audio=np.ascontiguousarray(audio), sample_rate=44100,
                               codes=np.ascontiguousarray(codes), finish_reason=reason,
                               timings=aggregate_timings, attempts=tuple(attempts))
