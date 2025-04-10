import base64
from PIL import Image
from rembg import remove
from io import BytesIO

def handler(event):
    image_data = event['input']['image_base64']

    if image_data.startswith("data:image"):
        image_data = image_data.split(",")[1]

    image_bytes = base64.b64decode(image_data)
    input_image = Image.open(BytesIO(image_bytes)).convert("RGBA")

    output_image = remove(input_image)

    output_io = BytesIO()
    output_image.save(output_io, format="PNG")
    result_base64 = base64.b64encode(output_io.getvalue()).decode("utf-8")

    return {"image_base64": result_base64}
