from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from PIL import Image, ImageFilter
from rembg import remove
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

# === Endpoint Rembg (ISNet) con ombra ===
@app.post("/remove-background-rembg")
async def remove_background_rembg_endpoint(file: UploadFile = File(...)):
    try:
        # Legge l'immagine
        image_bytes = await file.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        # Applica rembg (ISNet)
        no_bg = remove(input_image)

        # Parametri ombra
        width, height = no_bg.size
        shadow_height = int(height * 0.12)  # 12% parte bassa
        shadow_area = no_bg.crop((0, height - shadow_height, width, height))

        # Sfocatura + trasparenza
        blurred_shadow = shadow_area.filter(ImageFilter.GaussianBlur(radius=8))
        alpha = blurred_shadow.split()[-1].point(lambda p: p * 0.3)
        blurred_shadow.putalpha(alpha)

        # Crea nuovo canvas trasparente
        final_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        final_image.paste(blurred_shadow, (0, height - shadow_height), blurred_shadow)
        final_image.paste(no_bg, (0, 0), no_bg)

        # Ritorna il PNG
        buf = io.BytesIO()
        final_image.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("❌ ERRORE REMBG:", e)
        return {"error": f"Errore durante la rimozione con rembg: {str(e)}"}
