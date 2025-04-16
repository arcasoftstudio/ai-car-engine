#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive

echo "🧱 Aggiorno pacchetti e installo dipendenze di sistema..."
apt update && apt install -y libgl1 git ninja-build build-essential

echo "🐍 Installo dipendenze Python..."
pip install --upgrade pip
pip install -r requirements.txt

echo "📦 Controllo e compilo GroundingDINO se necessario..."
if [ -d "third_party/GroundingDINO" ]; then
    cd third_party/GroundingDINO
    python setup.py build_ext --inplace
    python setup.py develop
    cd ../../
else
    echo "⚠️  GroundingDINO non trovato nella cartella third_party/"
fi

echo "🚀 Avvio FastAPI sulla porta 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
