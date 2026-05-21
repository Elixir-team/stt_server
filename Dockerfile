FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG WHISPER_MODEL=turbo

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    WHISPER_MODEL=${WHISPER_MODEL} \
    WHISPER_CACHE_DIR=/opt/models/whisper

WORKDIR /app

RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i '/jammy-backports/d' /etc/apt/sources.list; \
        sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.edge.kernel.org/ubuntu|g' /etc/apt/sources.list; \
    fi; \
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i '/jammy-backports/d' /etc/apt/sources.list.d/ubuntu.sources; \
        sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.edge.kernel.org/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources; \
    fi; \
    apt-get update -o Acquire::Retries=10 -y; \
    for i in 1 2 3; do \
        apt-get install -o Acquire::Retries=10 --fix-missing -y --no-install-recommends \
            ffmpeg \
            python3 \
            python3-pip && break; \
        sleep 15; \
    done; \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install "setuptools<81" wheel \
    && python -m pip install -r requirements.txt

COPY server.py utils.py start.sh ./

RUN chmod +x /app/start.sh

RUN mkdir -p "${WHISPER_CACHE_DIR}" \
    && python -c "import os; from faster_whisper.utils import download_model; download_model(os.environ['WHISPER_MODEL'], cache_dir=os.environ['WHISPER_CACHE_DIR'])"

EXPOSE 8080

CMD ["./start.sh"]

