import os
import tempfile
import time
from datetime import datetime
from http.client import HTTPException

from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import whisper
import re
from io import BytesIO
import numpy as np
import wave
import struct
import argparse

SAMPLE_RATE = 16000

app = FastAPI()

parser = argparse.ArgumentParser(description="Speech-to-Text Server")
parser.add_argument("--model", type=str, default="turbo", help="Whisper model type")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
parser.add_argument("--port", type=int, default=8080, help="Server port")
args = parser.parse_args()

print("Initialize whisper:", args.model)
model = whisper.load_model(args.model)
print("Whisper initialized on device:", model.device)

async def get_audio_as_numpy(file: UploadFile):
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    audio = whisper.load_audio(tmp_path)
    os.unlink(tmp_path)

    return audio

def filter_speech(transcription: str) -> str:
    transcription = transcription.strip()
    cleaned_transcription = re.sub(r'\[.*?\]|\(.*?\)', '', transcription)
    cleaned_transcription = re.sub(r'\s+', ' ', cleaned_transcription).strip()
    return cleaned_transcription


def write_logs(time, logs):
    print("trinscribed by", time, "seconds")
    print(logs)
    if time > 3.0:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("logs.txt", "a") as f:
            result = now + ": transcribed by " + str(time) + " seconds\n"
            f.write(result)


@app.post("/stt/transcribe")
async def rest_endpoint(language: str = Form(...), file: UploadFile = File(...)):
    audio_array = await get_audio_as_numpy(file)

    if len(audio_array) > SAMPLE_RATE * 60:
        raise HTTPException(status_code=400, detail="Audio too long")

    start_time = time.time()
    result = model.transcribe(audio_array, language=language, temperature=0.0)

    text = filter_speech(result["text"])

    write_logs(time.time() - start_time, text)

    return {"result": text}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
