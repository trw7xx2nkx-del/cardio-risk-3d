#!/usr/bin/env bash
# One-command start: creates a virtual environment, installs dependencies, runs the app.
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
echo "Open http://localhost:8000"
uvicorn api:app --port 8000
