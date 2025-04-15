from rembg import new_session, remove
from PIL import Image, ImageEnhance
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

def remove_background(image_bytes: bytes):
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    input_image = enhance_contrast_for_dark_areas(input_image)
    output_image = remove(input_image, session=session)
    return output_image
