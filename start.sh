#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive
echo "✅ START.SH INIZIATO"

echo "🧱 Installo dipendenze di sistema..."
apt update && apt install -y python3-pip git libgl1 libglib2.0-0 build-essential wget unzip curl

echo "🐍 Installo pip e requirements..."
pip3 install --upgrade pip
pip3 install -r /workspace/ai-car-3d-backend/requirements.txt

echo "📦 Scarico e installo Meshroom da Hugging Face..."
cd /workspace
wget https://huggingface.co/ArcaSoftSrudio/ai-car-business/resolve/main/Meshroom-2021.1.0-linux-cuda10.tar.gz
tar -xzf Meshroom-2021.1.0-linux-cuda10.tar.gz
mv Meshroom-2021.1.0 /opt/meshroom
ln -s /opt/meshroom/meshroom_photogrammetry /usr/local/bin/meshroom_photogrammetry

echo "🚀 Avvio FastAPI..."
cd /workspace/ai-car-3d-backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
