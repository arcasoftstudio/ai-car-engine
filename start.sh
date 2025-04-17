#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive

echo "🧱 Aggiorno pacchetti e installo dipendenze di sistema..."
apt update && apt install -y libgl1 git build-essential

echo "🐍 Installo dipendenze Python..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Avvio FastAPI sulla porta 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
