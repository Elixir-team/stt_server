FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG WHISPER_MODEL=tiny

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    WHISPER_MODEL=${WHISPER_MODEL} \
    WHISPER_CACHE_DIR=/opt/models/whisper

WORKDIR /app

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install -r requirements.txt

COPY server.py utils.py start.sh ./

RUN chmod +x /app/start.sh

RUN mkdir -p "${WHISPER_CACHE_DIR}" \
    && python -c "import os, whisper; whisper.load_model(os.environ['WHISPER_MODEL'], download_root=os.environ['WHISPER_CACHE_DIR'])"

EXPOSE 8080

CMD ["./start.sh"]
