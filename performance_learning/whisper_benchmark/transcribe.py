import argparse
import time
from faster_whisper import WhisperModel

MODEL_SIZE = "medium"


def transcribe(audio_path: str, device: str = "cpu") -> None:
    compute_type = "int8" if device == "cpu" else "float16"
    print(f"Device: {device} | Compute type: {compute_type} | Model: {MODEL_SIZE}")
    model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)

    start = time.perf_counter()
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"Detected language: {info.language} (probability={info.language_probability:.2f})")
    print(f"Duration: {info.duration:.2f}s\n")

    for segment in segments:
        print(f"[{segment.start:6.2f}s -> {segment.end:6.2f}s] {segment.text}")

    elapsed = time.perf_counter() - start
    print(f"\nTotal transcribe time: {elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    transcribe(args.audio, args.device)
