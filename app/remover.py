from carvekit.api.high import Interface
from carvekit.ml.wrap.modnet import IS_MODNET
from PIL import Image
import io

# Setup CarveKit con MODNet (alta qualità)
interface = Interface(
    seg_pipe=IS_MODNET(),
    pre_pipe=None,
    post_pipe=None,
    device="cuda"  # sfrutta la tua A5000
)

def remove_background(image_bytes: bytes):
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    result = interface([input_image])[0]
    return result
