from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from app.modnet_remover import remove_background_modnet
import io

app = FastAPI()

@app.post("/remove-background-modnet")
async def remove_background(file: UploadFile = File(...)):
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
