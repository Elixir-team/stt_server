#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python server.py \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8080}" \
  --model "${WHISPER_MODEL:-turbo}" \
  --max-audio-seconds "${MAX_AUDIO_SECONDS:-60}"
