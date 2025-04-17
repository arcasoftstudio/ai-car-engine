import numpy as np
import torch
import cv2
from PIL import Image
import io
from torchvision import transforms
from segment_anything import sam_model_registry, SamPredictor

# === Percorso del modello SAM ===
SAM_CHECKPOINT_PATH = "models/sam_vit_h_4b8939.pth"

# === Carica modello SAM ===
def load_sam():
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    return predictor

# === Segmentazione + rimozione sfondo ===
def remove_background_sam(image_bytes: bytes):
    # Carica immagine
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]

    # Carica SAM
    predictor = load_sam()
    predictor.set_image(image_np)

    # Prompt: centro dell'immagine
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    # Predici maschera
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    # Maschera finale (1 = oggetto da tenere)
    mask = masks[0]
    alpha = (mask * 255).astype(np.uint8)  # Trasparenza sullo sfondo

    # Combina immagine + alpha
    rgba = np.dstack((image_np, alpha))
    return Image.fromarray(rgba)
