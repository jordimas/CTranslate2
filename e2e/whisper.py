import time
import ctranslate2
import librosa
import transformers

# --- Load audio ---
audio, sr = librosa.load("dosparlants.mp3", sr=16000, mono=True)
audio_duration = len(audio) / sr

# --- Initialize processor & model ---
processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = ctranslate2.models.Whisper("whisper-medium-ct2", device="cpu", compute_type="auto")

# --- Feature extraction ---
start_gen = time.time()
features = ctranslate2.StorageView.from_array(
    processor(audio, return_tensors="np", sampling_rate=sr).input_features
)
# --- Language detection ---
language = model.detect_language(features)[0][0][0]
print(f"Detected language: {language}")

# --- Prepare prompt & transcription ---
prompt = processor.tokenizer.convert_tokens_to_ids(
    ["<|startoftranscript|>", language, "<|transcribe|>", "<|notimestamps|>"]
)
seq_ids = model.generate(features, [prompt])[0].sequences_ids[0]
gen_duration = time.time() - start_gen

# --- Decode transcription ---
transcription = processor.decode(seq_ids)
print(f"\nTranscription:\n{transcription}\n")

# --- Performance ---
num_tokens = len(seq_ids)
rtf = gen_duration / audio_duration
print(f"Generated {num_tokens} tokens in {gen_duration:.2f} sec | RTF: {rtf:.2f}")

