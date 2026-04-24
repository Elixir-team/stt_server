import argparse
import os
import re
import time
from datetime import datetime
from threading import Lock, Thread

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
import uvicorn
import whisper

from utils import convert_audio_as_numpy_array

SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 60
DEFAULT_SLOW_REQUEST_SECONDS = 3.0

app = FastAPI()

parser = argparse.ArgumentParser(description="Speech-to-Text Server")
parser.add_argument("--model", type=str, default=os.getenv("WHISPER_MODEL", "tiny"), help="Whisper model type")
parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Server host")
parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")), help="Server port")
parser.add_argument(
    "--max-audio-seconds",
    type=int,
    default=int(os.getenv("MAX_AUDIO_SECONDS", DEFAULT_MAX_AUDIO_SECONDS)),
    help="Maximum accepted audio length in seconds",
)
parser.add_argument(
    "--slow-request-seconds",
    type=float,
    default=float(os.getenv("SLOW_REQUEST_SECONDS", DEFAULT_SLOW_REQUEST_SECONDS)),
    help="Threshold for logging slow requests",
)
args = parser.parse_args()

model = None
model_device = "unknown"
model_load_error = None
model_ready = False
model_lock = Lock()
model_loading_started = False
whisper_cache_dir = os.getenv("WHISPER_CACHE_DIR")


def filter_speech(transcription: str) -> str:
    transcription = transcription.strip()
    cleaned_transcription = re.sub(r"\[.*?\]|\(.*?\)", "", transcription)
    cleaned_transcription = re.sub(r"\s+", " ", cleaned_transcription).strip()
    return cleaned_transcription


def write_logs(request_time, logs):
    print("transcribed by", request_time, "seconds")
    print(logs)
    if request_time > args.slow_request_seconds:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("logs.txt", "a", encoding="utf-8") as file:
            file.write(now + ": transcribed by " + str(request_time) + " seconds\n")


def load_model_once():
    global model
    global model_device
    global model_load_error
    global model_ready
    global model_loading_started

    with model_lock:
        if model_ready or model is not None or model_load_error is not None or model_loading_started:
            return
        model_loading_started = True

    try:
        print("Initialize whisper:", args.model)
        loaded_model = whisper.load_model(args.model, download_root=whisper_cache_dir)
        device = str(loaded_model.device)

        with model_lock:
            model = loaded_model
            model_device = device
            model_ready = True

        print("Whisper initialized on device:", model_device)
    except Exception as exc:
        with model_lock:
            model_load_error = exc
        print("Failed to initialize whisper:", exc)


def ensure_model_loading_started():
    with model_lock:
        already_started = model_loading_started or model_ready or model_load_error is not None

    if already_started:
        return

    loader_thread = Thread(target=load_model_once, daemon=True)
    loader_thread.start()


@app.on_event("startup")
async def startup_event():
    ensure_model_loading_started()


@app.get("/ping")
async def ping():
    if model_ready:
        return {"status": "healthy", "model": args.model, "device": model_device}

    if model_load_error is not None:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {model_load_error}")

    ensure_model_loading_started()
    return Response(status_code=204)


@app.post("/stt/transcribe")
async def rest_endpoint(language: str = Form(...), file: UploadFile = File(...)):
    if model_load_error is not None:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {model_load_error}")

    if not model_ready or model is None:
        raise HTTPException(status_code=503, detail="Model is still initializing")

    audio_array = await convert_audio_as_numpy_array(file)

    if len(audio_array) > SAMPLE_RATE * args.max_audio_seconds:
        raise HTTPException(status_code=400, detail="Audio too long")

    start_time = time.time()
    result = model.transcribe(audio_array, language=language, temperature=0.0)
    text = filter_speech(result["text"])

    write_logs(time.time() - start_time, text)

    return {"result": text}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
