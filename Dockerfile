FROM python:3.10-slim

# 🔧 Installa librerie di sistema minime necessarie per rembg + Pillow
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 📁 Directory di lavoro
WORKDIR /app

# 📦 Copia requirements e installa pacchetti
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 📂 Copia il codice
COPY . .

# 🚀 Avvio FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
