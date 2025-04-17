from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
from rembg import new_session, remove
import io
import onnxruntime

app = FastAPI()

# 🔌 Auto-select CUDA se disponibile
providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else ["CPUExecutionProvider"]
print("⚙️ rembg session using:", providers)
session = new_session("isnet-general-use", providers=providers)

# 🔧 Migliora contrasto dell'immagine in input
def enhance_contrast(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(1.15)
    image = ImageEnhance.Contrast(image).enhance(1.3)
    return image

# 🪟 Rende opachi i vetri anteriori (zona approssimata)
def darken_glass(image: Image.Image) -> Image.Image:
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Modifica questi valori se necessario
    glass_zone = (int(width * 0.2), int(height * 0.15), int(width * 0.8), int(height * 0.45))
    draw.rectangle(glass_zone, fill=(0, 0, 0, 90))  # Nero semitrasparente
    return Image.alpha_composite(image, overlay)

# 🕶️ Aggiunge ombra sfumata sotto l'auto
def add_shadow(image: Image.Image) -> Image.Image:
    width, height = image.size
    shadow_height = int(height * 0.12)
    shadow_area = image.crop((0, height - shadow_height, width, height))
    blurred_shadow = shadow_area.filter(ImageFilter.GaussianBlur(radius=8))
    alpha = blurred_shadow.split()[-1].point(lambda p: p * 0.3)
    blurred_shadow.putalpha(alpha)
    final = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    final.paste(blurred_shadow, (0, height - shadow_height), blurred_shadow)
    final.paste(image, (0, 0), image)
    return final

# === Endpoint unico completo ===
@app.post("/remove-background-rembg")
async def remove_background_rembg_endpoint(file: UploadFile = File(...)):
    try:
        # 📥 Carica immagine
        image_bytes = await file.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        # 🧠 Pre-processing
        input_image = enhance_contrast(input_image)

        # ✂️ Rimozione sfondo con ISNet
        no_bg = remove(input_image, session=session)

        # 🪟 Vetri opachi
        no_bg = darken_glass(no_bg)

        # 🕶️ Aggiunta ombra
        final_image = add_shadow(no_bg)

        # 📤 Output finale
        buf = io.BytesIO()
        final_image.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("❌ ERRORE:", e)
        return {"error": f"Errore: {str(e)}"}
