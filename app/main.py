# app/main.py

import os
import zipfile
from huggingface_hub import hf_hub_download

# Step 1 – Estrai GroundingDINO se non c'è
extract_path = "third_party/GroundingDINO"
if not os.path.exists(extract_path):
    print("📦 Scarico e estraggo GroundingDINO...")
    zip_path = hf_hub_download(
        repo_id="ArcaSoftSrudio/ai-car-business",
        filename="groundingdinoCode.zip"
    )
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("third_party/")
    print("✅ Estrazione completata.")
else:
    print("✅ GroundingDINO già presente.")

# Aggiungi path ai moduli
import sys
sys.path.append("third_party/GroundingDINO")

# --- resto del codice esistente ---
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from app.remover import remove_background
from app.sam_remover import remove_background_sam
import io
import torch
import glob

app = FastAPI()

@app.post("/remove-background")
async def remove_bg(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result_image = remove_background(image_bytes)
    img_io = io.BytesIO()
    result_image.save(img_io, format='PNG')
    img_io.seek(0)
    return StreamingResponse(img_io, media_type="image/png")

@app.post("/remove-background-sam")
async def remove_bg_sam(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result_image = remove_background_sam(image_bytes)
    img_io = io.BytesIO()
    result_image.save(img_io, format="PNG")
    img_io.seek(0)
    return StreamingResponse(img_io, media_type="image/png")

@app.get("/health")
def health_check():
    try:
        from groundingdino.layers import _C
        dino_ok = True
    except Exception as e:
        print("❌ GroundingDINO _C not loaded:", e)
        dino_ok = False

    return {
        "cuda": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "groundingdino_compiled": dino_ok,
        "models": {
            "sam": os.path.exists("models/sam_vit_h_4b8939.pth"),
            "dino": os.path.exists("models/groundingdino_swint_ogc.pth"),
            "dino_config": os.path.exists("models/GroundingDINO_SwinT_OGC.py")
        }
    }
