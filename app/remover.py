# app/remover.py

from rembg import remove
from PIL import Image
import io

def remove_background(image_bytes: bytes) -> Image.Image:
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    output_image = remove(input_image)
    return output_image
