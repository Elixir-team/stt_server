# RunPod Dev Notes

## Что делает сервис

Этот сервис переводит речь в текст.

- Маршрут: `POST /stt/transcribe`
- Проверка здоровья: `GET /ping`
- Порт по умолчанию: `8080`

Backend можно оставить как есть: после деплоя нужно будет только заменить старый Pod URL на новый Serverless endpoint URL.

## Почему сервис подходит под RunPod Serverless

Этот репозиторий уже поднимает обычный FastAPI HTTP-сервис.
Поэтому для него подходит именно `Serverless Load Balancing`, а не queue-based Serverless.

Почему:

- RunPod Load Balancing поддерживает кастомные HTTP endpoints
- он умеет работать с любым HTTP framework
- `/ping` используется для проверки готовности
- у сервиса уже есть свои обычные HTTP маршруты

Документация:

- [RunPod Load Balancing Overview](https://docs.runpod.io/serverless/load-balancing/overview)
- [RunPod Endpoints Overview](https://docs.runpod.io/serverless/endpoints/overview)

## Как правильно деплоить

Рекомендуемый порядок такой:

1. Собрать Docker image локально на своем компьютере.
2. Запушить image в container registry.
3. В RunPod Serverless создать новый endpoint через `Import from Docker Registry`.
4. Выбрать тип `Load Balancing`.
5. Подставить новый endpoint URL в backend.

Важно:

- RunPod не видит твой локальный Docker image напрямую.
- Локальная сборка не тратит деньги RunPod.
- Деньги RunPod начнут тратиться, когда будет создан endpoint и пойдут запросы.

Документация:

- [RunPod Quickstart](https://docs.runpod.io/serverless/quickstart)
- [RunPod Workers Overview](https://docs.runpod.io/serverless/workers/overview)

## Что уже готово в этом репозитории

- реализован `GET /ping`
- реализован `POST /stt/transcribe`
- запуск идет через `PORT` и env переменные
- в рантайме больше нет `pip install` и `git pull`
- есть `Dockerfile` для деплоя
- Whisper заранее загружается в image во время `docker build`
- для dev по умолчанию используется модель `tiny`

Это значит, что image должен приезжать в RunPod уже подготовленным, а не скачивать зависимости и модель при каждом старте.

## Оценка готовности

Текущий статус: `хорошо подходит для деплоя`

Почему этот сервис проще:

- архитектура очень хорошо совпадает с RunPod Load Balancing
- здесь одна основная модельная цепочка
- зависимости проще, чем у TTS
- нет дополнительного нестандартного пакета вроде Melo

Что еще не подтверждено на практике:

- первая реальная сборка `docker build`
- первый реальный запуск в RunPod

То есть код уже подходит как база для деплоя, но финальное подтверждение все равно даст первая сборка image и smoke test endpoint.

## Рекомендуемые настройки RunPod для dev

- Тип endpoint: `Load Balancing`
- Тип worker: `GPU`
- Active workers: `0`
- Min workers: `0`
- Max workers: `1`
- Idle timeout: `5s`

Это самый дешевый режим для dev: worker поднимается только когда приходит запрос.

Документация:

- [RunPod Endpoint Settings](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)

## Переменные окружения

- `PORT=8080`
- `WHISPER_MODEL=tiny`
- `MAX_AUDIO_SECONDS=60`
- `SLOW_REQUEST_SECONDS=3.0`

## Сборка image

```bash
docker build --platform linux/amd64 --build-arg WHISPER_MODEL=tiny -t YOUR_REGISTRY/stt-serverless:dev .
docker push YOUR_REGISTRY/stt-serverless:dev
```

## Примечание по аудиоформатам

Сервис принимает не только mp3.
Он старается сохранить входной формат файла и передает его в `whisper.load_audio`.
Поэтому backend обычно может просто переслать тот же формат, который пришел от клиента, если у файла корректное имя или MIME type.

## Примечание по Whisper

Этот image рассчитан на то, что Whisper уже запечен внутрь image во время build.
Для Serverless это правильный подход, потому что не нужно скачивать модель при каждом cold start.

Для dev по умолчанию используется `tiny`.
Для production нужно переключить `WHISPER_MODEL` на `turbo`.

Ссылка:

- [OpenAI Whisper README](https://github.com/openai/whisper)
