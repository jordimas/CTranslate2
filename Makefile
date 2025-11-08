
install.convert-transformers:
	pip install torch transformers


convert-whisper:
	ct2-transformers-converter --model openai/whisper-tiny --output_dir whisper-tiny-ct2

