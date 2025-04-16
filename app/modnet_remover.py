import os
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import io

from torchvision import transforms
from torch.nn.functional import interpolate

from app.modnet_arch import MODNet  # te lo passo nel prossimo step

HF_URL = "https://huggingface.co/ArcaSoftSrudio/ai-car-business/resolve/main/modnet_photographic.pth"
MODEL_PATH = "models/modnet_photographic.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Scarico modello MODNET da Hugging Face...")
        os.makedirs("models", exist_ok=True)
        r = requests.get(HF_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("✅ Modello scaricato.")

download_model()

def load_modnet():
    model = MODNet(backbone_pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
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
        matte = modnet(im_tensor)[0][0]
        matte = interpolate(matte.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear').squeeze().numpy()

    fg = im.astype(np.float32) / 255
    alpha = np.expand_dims(matte, axis=2)
    rgba = np.concatenate((fg, alpha), axis=2)
    rgba = (rgba * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")
