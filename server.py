import argparse
import copy
import logging
import os
import re
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import ctranslate2
from faster_whisper import WhisperModel
import uvicorn

from utils import convert_audio_as_numpy_array

SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 60

app = FastAPI()

parser = argparse.ArgumentParser(description="Speech-to-Text Server")
parser.add_argument("--model", type=str, default=os.getenv("WHISPER_MODEL", "turbo"), help="Whisper model type")
parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Server host")
parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")), help="Server port")
parser.add_argument(
    "--max-audio-seconds",
    type=int,
    default=int(os.getenv("MAX_AUDIO_SECONDS", DEFAULT_MAX_AUDIO_SECONDS)),
    help="Maximum accepted audio length in seconds",
)
args = parser.parse_args()

whisper_cache_dir = os.getenv("WHISPER_CACHE_DIR")


class HealthCheckAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return '"GET /ping HTTP/1.1" 200' not in message and '"GET /ping HTTP/1.1" 204' not in message


def build_log_config():
    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    access_logger = log_config["loggers"].get("uvicorn.access")
    if access_logger is None:
        return log_config

    log_config.setdefault("filters", {})
    log_config["filters"]["health_check_access_filter"] = {
        "()": HealthCheckAccessFilter,
    }
    access_logger["filters"] = ["health_check_access_filter"]
    return log_config


def filter_speech(transcription: str) -> str:
    transcription = transcription.strip()
    cleaned_transcription = re.sub(r"\[.*?\]|\(.*?\)", "", transcription)
    cleaned_transcription = re.sub(r"\s+", " ", cleaned_transcription).strip()
    return cleaned_transcription


def resolve_device() -> tuple[str, str]:
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


model_device, compute_type = resolve_device()

print("Initialize faster-whisper:", args.model)
model = WhisperModel(
    args.model,
    device=model_device,
    compute_type=compute_type,
    download_root=whisper_cache_dir,
)
print("Faster-whisper initialized on device:", model_device)


@app.get("/ping")
async def ping():
    return {"status": "healthy", "model": args.model, "device": model_device}


@app.post("/stt/transcribe")
async def rest_endpoint(language: str = Form(...), file: UploadFile = File(...)):
    audio_array = await convert_audio_as_numpy_array(file)

    if len(audio_array) > SAMPLE_RATE * args.max_audio_seconds:
        raise HTTPException(status_code=400, detail="Audio too long")

    start_time = time.time()
    segments, _ = model.transcribe(audio_array, language=language, temperature=0.0)
    text = filter_speech(" ".join(segment.text for segment in segments))

    print("transcribed by", time.time() - start_time, "seconds")
    print(text)

    return {"result": text}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_config=build_log_config())
