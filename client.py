import sounddevice as sd
import requests
import io
import wave
from pydub import AudioSegment

# Параметры записи
SAMPLE_RATE = 16000  # Частота дискретизации
DURATION = 3  # Длина чанка в секундах

# SERVER_URL = "https://72mp1d893c4wo6-8080.proxy.runpod.net/stt/transcribe"  # Адрес сервера
SERVER_URL = "http://localhost:8080/stt/transcribe"  # Адрес сервера
LANGUAGE = "ru"  # Язык распознавания


def record_audio_chunk():
    """Записывает аудио-чанк с перекрытием"""
    print(f"Запись {DURATION} секунд аудио...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    return audio


def audio_to_wav_bytes(audio_data):
    """Преобразует массив numpy в WAV-байты"""
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    return wav_io.getvalue()


def wav_bytes_to_mp3_bytes(wav_bytes):
    wav_io = io.BytesIO(wav_bytes)
    audio = AudioSegment.from_file(wav_io, format="wav")
    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3", bitrate="16k")
    return mp3_io.getvalue()


def send_audio_to_server(audio_bytes, audio_name, mime_type):
    """Отправляет аудиофайл в виде байтов на сервер"""
    files = {"file": (audio_name, io.BytesIO(audio_bytes), mime_type)}
    data = {"language": LANGUAGE}
    response = requests.post(SERVER_URL, files=files, data=data)
    return response.json()


def main():
    print("Запуск записи и отправки аудио...")
    while True:
        audio_chunk = record_audio_chunk()
        wav_bytes = audio_to_wav_bytes(audio_chunk)
        mp3_bytes = wav_bytes_to_mp3_bytes(wav_bytes)
        print("Отправляем на сервер")
        import time
        start_time = time.time()
        response0 = send_audio_to_server(audio_chunk.tobytes(), "audio.wav", "audio/wav")
        response1 = send_audio_to_server(wav_bytes, "audio.wav", "audio/wav")
        response2 = send_audio_to_server(mp3_bytes, "audio.mp3", "audio/mpeg")
        print("Время выполнения отправки:", time.time() - start_time, "секунд")
        print("Ответ от сервера wav:", response0.get("result"), response1.get("error", "Ошибка"))
        print("Ответ от сервера wav:", response1.get("result"), response1.get("error", "Ошибка"))
        print("Ответ от сервера mp3:", response2.get("result"), response1.get("error", "Ошибка"))

        time.sleep(1)  # Учитываем перекрытие


if __name__ == "__main__":
    main()
