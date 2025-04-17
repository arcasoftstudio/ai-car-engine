from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image, ImageFilter
from rembg import remove
import io

app = FastAPI()

# === Endpoint Rembg (ISNet) con ombra sfumata ===
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
        shadow_height = int(height * 0.12)  # Prendiamo 12% dal basso
        shadow_area = no_bg.crop((0, height - shadow_height, width, height))

        # Sfocatura e trasparenza
        blurred_shadow = shadow_area.filter(ImageFilter.GaussianBlur(radius=8))
        alpha = blurred_shadow.split()[-1].point(lambda p: p * 0.3)
        blurred_shadow.putalpha(alpha)

        # Crea canvas trasparente e unisce l'ombra
        final_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        final_image.paste(blurred_shadow, (0, height - shadow_height), blurred_shadow)
        final_image.paste(no_bg, (0, 0), no_bg)

        # Restituisce immagine PNG
        buf = io.BytesIO()
        final_image.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("❌ ERRORE REMBG:", e)
        return {"error": f"Errore durante la rimozione con rembg: {str(e)}"}
