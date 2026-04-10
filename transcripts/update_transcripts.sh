#!/usr/bin/env bash
# Run from anywhere — downloads, transcribes, and commits new podcast episodes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Create venv and install dependencies if it doesn't exist yet
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install requests feedparser python-dateutil tqdm unidecode faster-whisper
else
    source "$VENV_DIR/bin/activate"
fi

cd "$SCRIPT_DIR/code"
python3 podcast_pipeline.py "$@"

deactivate
