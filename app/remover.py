from rembg import new_session, remove
from PIL import Image, ImageFilter
import io

session = new_session("isnet-general-use")

def remove_background(image_bytes: bytes):
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    output_image = remove(input_image, session=session)

    # POST-PROCESSING: miglioramento bordi
    output_image = smooth_edges(output_image)

    return output_image

def smooth_edges(img):
    """Applica blur lieve al canale alpha per bordi più morbidi"""
    r, g, b, a = img.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=1.5))  # sfuma contorno
    return Image.merge("RGBA", (r, g, b, a))
