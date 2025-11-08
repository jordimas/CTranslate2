.PHONY: install.convert-transformers convert-whisper run-whisper

install.convert-transformers:
	pip install torch transformers


convert-whisper:
	ct2-transformers-converter --model openai/whisper-medium --output_dir whisper-medium-ct2


run-whisper:
	python e2e/whisper.py 


