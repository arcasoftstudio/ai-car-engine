from rembg import new_session, remove
from PIL import Image, ImageEnhance, ImageDraw
import io
import onnxruntime

# ⚡️ Usa CUDA se disponibile, altrimenti CPU
providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else ["CPUExecutionProvider"]
print("💻 rembg session using:", providers)

# Inizializza rembg con il provider selezionato
session = new_session("isnet-general-use", providers=providers)

def enhance_contrast_for_dark_areas(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.15)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)
    return image

def darken_glass_areas(image: Image.Image) -> Image.Image:
    width, height = image.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Zona approssimativa dei vetri anteriori (puoi adattarla)
    glass_zone = (int(width * 0.2), int(height * 0.15), int(width * 0.8), int(height * 0.45))

    # Disegna un rettangolo nero semitrasparente sopra i vetri
    draw.rectangle(glass_zone, fill=(0, 0, 0, 90))  # RGBA, alpha 90 su 255

    # Unisci l’overlay con l’immagine
    image_with_glass = Image.alpha_composite(image, overlay)
    return image_with_glass

def remove_background(image_bytes: bytes):
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    input_image = enhance_contrast_for_dark_areas(input_image)
    output_image = remove(input_image, session=session)
    output_image = darken_glass_areas(output_image)
    return output_image
