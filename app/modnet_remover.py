import os
import requests
import torch
import numpy as np
import cv2
from PIL import Image
import io
from torchvision import transforms
from torch.nn.functional import interpolate
from app.modnet_arch import MODNet



HF_URL = "https://huggingface.co/ArcaSoftStudio/ai-car-business/resolve/main/modnet_photographic_portrait_matting.ckpt"
MODEL_PATH = "models/modnet_photographic_portrait_matting.ckpt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Scarico modello MODNET da Hugging Face...")
        os.makedirs("models", exist_ok=True)

        response = requests.get(HF_URL, stream=True)
        
        # Verifica codice HTTP
        if response.status_code != 200:
            raise Exception(f"❌ ERRORE durante il download: codice HTTP {response.status_code}")

        # Verifica che non sia HTML (errore silenzioso di Hugging Face)
        first_chunk = next(response.iter_content(1024))
        if first_chunk.strip().startswith(b'<!DOCTYPE html') or b'<html' in first_chunk:
            raise Exception("❌ ERRORE: il file scaricato non è un .ckpt ma una pagina HTML")

        # Scrivi il file (includendo il primo chunk già letto)
        with open(MODEL_PATH, "wb") as f:
            f.write(first_chunk)
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("✅ Modello scaricato correttamente.")


def load_modnet():
    download_model()
    model = MODNet()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model



def remove_background_modnet(image_bytes: bytes):
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    im = np.array(input_image)
    h, w = im.shape[:2]

    ref_size = 512
    im_tensor = transforms.ToTensor()(cv2.resize(im, (ref_size, ref_size))).unsqueeze(0)
    modnet = load_modnet()

    with torch.no_grad():
        matte = modnet(im_tensor)
        matte = interpolate(matte.unsqueeze(0), size=(h, w), mode='bilinear').squeeze().numpy()

    fg = im.astype(np.float32) / 255
    alpha = np.expand_dims(matte, axis=2)
    rgba = np.concatenate((fg, alpha), axis=2)
    rgba = (rgba * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")
