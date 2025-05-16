import time
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


def is_wav_file(data: bytes) -> bool:
    if len(data) < 12:
        return False

    riff, size, wave = struct.unpack('<4sI4s', data[:12])

    return riff == b'RIFF' and wave == b'WAVE'


def format_bytes_to_np_array(data: bytes):
    return np.frombuffer(data, dtype=np.int16).flatten().astype(np.float32) / 32768.0


async def get_audio_as_numpy(file: UploadFile):
    audio_bytes = await file.read()
    await file.close()

    if is_wav_file(audio_bytes):
        audio_stream = BytesIO(audio_bytes)
        with wave.open(audio_stream, 'rb') as wf:
            audio_bytes = wf.readframes(wf.getnframes())

    return format_bytes_to_np_array(audio_bytes)


def filter_speech(transcription: str) -> str:
    transcription = transcription.strip()
    cleaned_transcription = re.sub(r'\[.*?\]|\(.*?\)', '', transcription)
    cleaned_transcription = re.sub(r'\s+', ' ', cleaned_transcription).strip()
    return cleaned_transcription


@app.post("/stt/transcribe")
async def rest_endpoint(language: str = Form(...), file: UploadFile = File(...)):
    audio_array = await get_audio_as_numpy(file)

    if len(audio_array) > SAMPLE_RATE * 60:  # 30 секунд
        raise HTTPException(status_code=400, detail="Audio too long")

    start_time = time.time()
    result = model.transcribe(audio_array, language=language, temperature=0.0)
    print("trinscribed from", time.time() - start_time, "seconds")

    text = filter_speech(result["text"])
    print(result)
    return {"result": text}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
