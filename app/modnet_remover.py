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
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024:
        print("✔️ Il modello esiste già.")
        return

    print("📥 Scarico modello MODNET da Hugging Face...")
    os.makedirs("models", exist_ok=True)

    with requests.get(HF_URL, stream=True, allow_redirects=True) as response:
        content_type = response.headers.get("Content-Type", "")
        print("📄 Content-Type:", content_type)

        if response.status_code != 200:
            raise Exception(f"❌ ERRORE HTTP {response.status_code}: {response.text}")

        if "text/html" in content_type or "<html" in response.text[:100].lower():
            raise Exception("❌ Il file scaricato è HTML, probabilmente il link è errato o il repo è privato.")

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"✅ Modello scaricato correttamente! ({round(os.path.getsize(MODEL_PATH)/1024/1024, 2)} MB)")



def load_modnet():
    download_model()
    print(f"📂 Controllo se il file esiste: {os.path.exists(MODEL_PATH)}")
    print(f"📦 Dimensione file: {os.path.getsize(MODEL_PATH)} bytes" if os.path.exists(MODEL_PATH) else "⛔ File non trovato!")
    
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
