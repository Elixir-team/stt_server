import sounddevice as sd
import requests
import io
import wave

# Параметры записи
SAMPLE_RATE = 16000  # Частота дискретизации
CHUNK_DURATION = 2  # Длина чанка в секундах
OVERLAP = 1  # Перекрытие в секундах

SERVER_URL = "http://77.104.167.149:55244/transcribe/"  # Адрес сервера
#SERVER_URL = "http://localhost:8000/transcribe/"  # Адрес сервера
LANGUAGE = "ru"  # Язык распознавания

def record_audio_chunk():
    """Записывает аудио-чанк с перекрытием"""
    duration = CHUNK_DURATION + OVERLAP
    print(f"Запись {duration} секунд аудио...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
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


def send_audio_to_server(audio_bytes):
    """Отправляет аудиофайл в виде байтов на сервер"""
    files = {"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
    data = {"language": LANGUAGE}
    response = requests.post(SERVER_URL, files=files, data=data)
    return response.json()


def main():
    print("Запуск записи и отправки аудио...")
    while True:
        audio_chunk = record_audio_chunk()
        wav_bytes = audio_to_wav_bytes(audio_chunk)
        print("Отправляем на сервер")
        import time
        start_time = time.time()
        response = send_audio_to_server(wav_bytes)
        print("Время выполнения отправки:", time.time() - start_time, "секунд")
        print("Ответ от сервера:", response.get("text", response.get("error", "Ошибка")))

        time.sleep(1)  # Учитываем перекрытие


if __name__ == "__main__":
    main()
