import os
import requests
import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import io

# URL Hugging Face per il modello SAM
HF_BASE = "https://huggingface.co/ArcaSoftSrudio/ai-car-business/resolve/main"
SAM_URL = f"{HF_BASE}/sam_vit_h_4b8939.pth"
SAM_PATH = "models/sam_vit_h_4b8939.pth"

# Scarica il modello se non c'è
def download_from_hf_if_missing(url: str, dest_path: str):
    if not os.path.exists(dest_path):
        print(f"📥 Scarico da Hugging Face: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Salvato in: {dest_path}")

download_from_hf_if_missing(SAM_URL, SAM_PATH)

# Carica SAM su CPU
def load_sam():
    sam = sam_model_registry["vit_h"](checkpoint=SAM_PATH).to("cpu")
    predictor = SamPredictor(sam)
    return predictor

# Segmentazione automatica semplice (senza DINO)
def remove_background_sam(image_bytes: bytes):
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(input_image)

    sam = load_sam()
    sam.set_image(image_np)

    # Maschera automatica su tutta l'immagine
    H, W, _ = image_np.shape
    boxes = np.array([[0, 0, W, H]])
    boxes_torch = torch.tensor(boxes, dtype=torch.float32).unsqueeze(0)

    transformed_boxes = sam.transform.apply_boxes_torch(boxes_torch, image_np.shape[:2]).to("cpu")

    masks, _, _ = sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    mask = masks[0][0].cpu().numpy().astype(np.uint8) * 255
    result_rgba = np.dstack((image_np, mask))
    return Image.fromarray(result_rgba)
