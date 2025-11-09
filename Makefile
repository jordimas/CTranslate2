.PHONY: install.convert-transformers convert-whisper run-whisper build-docker

install.convert-transformers:
	pip install torch transformers


convert-whisper:
	ct2-transformers-converter --model openai/whisper-medium --output_dir whisper-medium-ct2


run-whisper:
	python e2e/whisper.py 

docker-build:
	docker build -t ctranslate2 . -f docker/Dockerfile

docker-build-test:
	docker build -t ctranslate2-test . -f e2e/Dockerfile
	
	
docker-run-test:
	docker run ctranslate2-test 
