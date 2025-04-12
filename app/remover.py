from carvekit.api.high import Interface
from carvekit.ml.wrap.modnet import IS_MODNET
from PIL import Image
import io
import numpy as np

# Setup del modello CarveKit con MODNet
interface = Interface(
    seg_pipe=IS_MODNET(),
    pre_pipe=None,
    post_pipe=None,
    device="cuda",  # usa la GPU!
)

def remove_background(image_bytes: bytes):
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    result = interface([input_image])[0]
    return result
