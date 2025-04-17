from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
from rembg import remove
import io

app = FastAPI()

@app.post("/remove-background-rembg")
async def remove_background_rembg_endpoint(file: UploadFile = File(...)):
    try:
        # 📥 Legge immagine
        image_bytes = await file.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        # ✂️ Applica rembg (ISNet)
        no_bg = remove(input_image)

        # 📤 Restituisce immagine finale con ombra reale mantenuta
        buf = io.BytesIO()
        no_bg.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("❌ ERRORE REMBG:", e)
        return {"error": f"Errore durante la rimozione con rembg: {str(e)}"}
