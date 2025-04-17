from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from app.modnet_remover import remove_background_modnet
from app.sam_remover import remove_background_sam
import io

app = FastAPI()

# === Endpoint MODNET ===
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
        return {"error": f"Errore durante la rimozione dello sfondo: {str(e)}"}

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
