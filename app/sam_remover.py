import os
import requests
import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, predict
import torchvision.transforms as T  # IMPORTANTE per la conversione a tensor
import io

# === FUNZIONE: SCARICA FILE DA HUGGING FACE ===
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

# === LINK MODELLI SU HUGGING FACE ===
HF_BASE = "https://huggingface.co/ArcaSoftSrudio/ai-car-business/resolve/main"
SAM_URL = f"{HF_BASE}/sam_vit_h_4b8939.pth"
DINO_URL = f"{HF_BASE}/groundingdino_swint_ogc.pth"
DINO_CONFIG_URL = f"{HF_BASE}/GroundingDINO_SwinT_OGC.py"

# === PERCORSI LOCALI ===
SAM_PATH = "models/sam_vit_h_4b8939.pth"
DINO_PATH = "models/groundingdino_swint_ogc.pth"
DINO_CONFIG_PATH = "models/GroundingDINO_SwinT_OGC.py"

# === SCARICA SE MANCANTI ===
download_from_hf_if_missing(SAM_URL, SAM_PATH)
download_from_hf_if_missing(DINO_URL, DINO_PATH)
download_from_hf_if_missing(DINO_CONFIG_URL, DINO_CONFIG_PATH)

# === INIZIALIZZAZIONE MODELLI ===
def load_sam():
    sam = sam_model_registry["vit_h"](checkpoint=SAM_PATH).to("cuda")
    predictor = SamPredictor(sam)
    return predictor

def load_dino():
    model = load_model(DINO_CONFIG_PATH, DINO_PATH)
    return model

# === FUNZIONE PRINCIPALE: RIMOZIONE SFONDO CON SAM + DINO ===
def remove_background_sam(image_bytes: bytes):
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(input_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Converti immagine in tensor per Grounding DINO
    image_tensor = T.ToTensor()(image_bgr).unsqueeze(0).to("cuda")

    dino = load_dino()
    sam = load_sam()

    boxes, logits, phrases = predict(
        model=dino,
        image=image_tensor,
        caption="a car",
        box_threshold=0.3,
        text_threshold=0.25
    )

    if len(boxes) == 0:
        raise Exception("❌ Nessuna auto rilevata nell'immagine.")

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
