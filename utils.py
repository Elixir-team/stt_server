import os
import tempfile

import numpy as np
from fastapi import UploadFile
from faster_whisper.audio import decode_audio

CONTENT_TYPE_SUFFIXES = {
    "audio/flac": ".flac",
    "audio/m4a": ".m4a",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/x-wav": ".wav",
}


def format_bytes_to_np_array(data: bytes):
    return np.frombuffer(data, dtype=np.int16).flatten().astype(np.float32) / 32768.0


def resolve_suffix(file: UploadFile):
    if file.filename:
        suffix = os.path.splitext(file.filename)[1]
        if suffix:
            return suffix

    if file.content_type:
        return CONTENT_TYPE_SUFFIXES.get(file.content_type.lower(), "")

    return ""


async def convert_audio_as_numpy_array(file: UploadFile):
    audio_bytes = await file.read()
    tmp_path = None

    try:
        suffix = resolve_suffix(file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        return decode_audio(tmp_path, sampling_rate=16000)
    except Exception:
        return format_bytes_to_np_array(audio_bytes)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
