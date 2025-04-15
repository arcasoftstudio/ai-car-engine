# app/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from app.remover import remove_background
from app.sam_remover import remove_background_sam
import io
import os
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
    return {
        "cuda": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "groundingdino_compiled": bool(glob.glob("groundingdino/layers/_C*.so")),
        "models": {
            "sam": os.path.exists("models/sam_vit_h_4b8939.pth"),
            "dino": os.path.exists("models/groundingdino_swint_ogc.pth"),
            "dino_config": os.path.exists("models/GroundingDINO_SwinT_OGC.py")
        }
    }
