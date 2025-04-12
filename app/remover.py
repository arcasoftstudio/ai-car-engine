from rembg import new_session, remove
from PIL import Image
import io

# Crea una sessione con il modello migliore
session = new_session("isnet-general-use")

def remove_background(image_bytes: bytes):
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    output_image = remove(input_image, session=session)
    return output_image
