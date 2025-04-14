# Dockerfile

FROM python:3.10-slim

# 🔧 Installa librerie di sistema richieste da OpenCV, PyTorch ecc.
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 📁 Directory di lavoro
WORKDIR /app

# 📦 Copia requirements e installa torch con supporto CUDA (essenziale)
COPY requirements.txt .

# ✅ Installazione di torch e torchvision con CUDA 11.8 (perfetto per A6000/A100)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 🔌 Installa dipendenze AI e i tuoi requirements
RUN pip install --no-cache-dir opencv-python supervision && \
    pip install --no-cache-dir git+https://github.com/IDEA-Research/GroundingDINO.git && \
    pip install --no-cache-dir -r requirements.txt

# 🧠 Copia tutto il resto del codice
COPY . .

# 🚀 Avvio dell'app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
