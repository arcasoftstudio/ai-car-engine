#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
apt update && apt install -y libgl1 git ninja-build

# Installa Python deps
pip install -r requirements.txt

# Compila GroundingDINO
cd third_party/GroundingDINO
python setup.py build_ext --inplace
python setup.py develop
cd ../../

# Avvia il server FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000
