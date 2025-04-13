from rembg import new_session, remove
from PIL import Image, ImageEnhance
import io

# Inizializza il modello rembg con isnet-general-use (qualità alta)
session = new_session("isnet-general-use")

def enhance_contrast_for_dark_areas(image: Image.Image) -> Image.Image:
    """
    Aumenta luminosità e contrasto per migliorare la separazione di ruote nere su asfalto.
    """
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.15)  # schiarisce leggermente
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)   # aumenta il contrasto
    return image

def remove_background(image_bytes: bytes):
    """
    Riceve immagine -> boost visivo -> rimozione sfondo con AI -> ritorna PNG pulito.
    """
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    input_image = enhance_contrast_for_dark_areas(input_image)  # 💥 Preprocessing
    output_image = remove(input_image, session=session)
    return output_image
