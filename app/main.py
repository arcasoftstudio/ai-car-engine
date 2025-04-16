from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
import io
import torch
import os

from app.modnet_remover import remove_background_modnet

app = FastAPI()


@app.post("/remove-background-modnet")
async def remove_bg_modnet(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result_image = remove_background_modnet(image_bytes)
        img_io = io.BytesIO()
        result_image.save(img_io, format="PNG")
        img_io.seek(0)
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        print("❌ ERRORE MODNET:", str(e))
        return JSONResponse(
            status_code=500,
            content={"error": f"Errore durante la rimozione dello sfondo: {str(e)}"},
        )


@app.get("/health")
def health_check():
    return {
        "cuda": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "modnet_model": os.path.exists("models/modnet_photographic.pth")
    }
