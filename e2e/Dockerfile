FROM ctranslate2
RUN apt-get update && apt-get install make -y
RUN python3 -m pip install --no-cache-dir librosa transformers torch==2.4.0

COPY * /srv/
WORKDIR /srv/
RUN make convert-whisper


ENTRYPOINT /srv/entrypoint.sh
