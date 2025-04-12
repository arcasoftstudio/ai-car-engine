from carvekit.api.high import Interface
from carvekit.ml.init import init_interface
from PIL import Image
import io

# Inizializza interfaccia MODNet
interface = init_interface(
    model_type="modnet",
    device="cuda",  # usa la tua GPU
    seg_mask_size=640,
    refine_mode="full",
    trimap_prob_threshold=231,
    trimap_dilation=30,
    trimap_erosion_iters=5
)

def remove_background(image_bytes: bytes):
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    result = interface([input_image])[0]
    return result
