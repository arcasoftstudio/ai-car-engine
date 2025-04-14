import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, predict
import io

# === PERCORSI AI MODELLI ===
SAM_CHECKPOINT = "models/sam_vit_h_4b8939.pth"
DINO_CONFIG_PATH = "models/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS_PATH = "models/groundingdino_swint_ogc.pth"

# === INIZIALIZZA MODELLI ===
def load_sam():
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to("cuda")
    predictor = SamPredictor(sam)
    return predictor

def load_dino():
    model = load_model(DINO_CONFIG_PATH, DINO_WEIGHTS_PATH)
    return model

# === FUNZIONE PRINCIPALE ===
def remove_background_sam(image_bytes: bytes):
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(input_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    dino = load_dino()
    sam = load_sam()

    boxes, logits, phrases = predict(
        model=dino,
        image=image_bgr,
        caption="a car",
        box_threshold=0.3,
        text_threshold=0.25
    )

    if len(boxes) == 0:
        raise Exception("❌ Nessuna auto rilevata nell'immagine")

    sam.set_image(image_np)
    transformed_boxes = sam.transform.apply_boxes_torch(boxes, image_np.shape[:2]).to("cuda")

    masks, scores, _ = sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    mask = masks[0][0].cpu().numpy().astype(np.uint8) * 255
    result_rgba = np.dstack((image_np, mask))
    return Image.fromarray(result_rgba)
