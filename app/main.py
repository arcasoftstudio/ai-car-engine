from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import io

from app.modnet_remover import remove_background_modnet
from app.sam_remover import remove_background_sam
from app.remover import remove_background  # ← questo è il tuo rembg

app = FastAPI()

# === Endpoint MODNet ===
@app.post("/remove-background-modnet")
async def remove_background_modnet_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result_image = remove_background_modnet(image_bytes)
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print("❌ ERRORE MODNET:", e)
        return {"error": f"Errore durante la rimozione MODNET: {str(e)}"}

# === Endpoint SAM ===
@app.post("/remove-background-sam")
async def remove_background_sam_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result_image = remove_background_sam(image_bytes)
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print("❌ ERRORE SAM:", e)
        return {"error": f"Errore durante la segmentazione con SAM: {str(e)}"}

# === Endpoint Rembg (ISNet) ===
@app.post("/remove-background-rembg")
async def remove_background_rembg_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result_image = remove_background(image_bytes)
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print("❌ ERRORE REMBG:", e)
        return {"error": f"Errore durante la rimozione con rembg: {str(e)}"}
