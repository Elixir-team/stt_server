import os
import tempfile

import numpy as np
from fastapi import UploadFile
import whisper


def format_bytes_to_np_array(data: bytes):
    return np.frombuffer(data, dtype=np.int16).flatten().astype(np.float32) / 32768.0


async def convert_audio_as_numpy_array(file: UploadFile):
    audio_bytes = await file.read()
    tmp_path = None

    try:
        suffix = os.path.splitext(file.filename)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        audio = whisper.load_audio(tmp_path)

        return audio
    except Exception:
        return format_bytes_to_np_array(audio_bytes)
    finally:
        if tmp_path:
            os.unlink(tmp_path)
