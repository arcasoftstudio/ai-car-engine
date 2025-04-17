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

HF_URL = "https://github.com/ZHKKKe/MODNet/releases/download/v1/modnet_photographic_portrait_matting.ckpt"
MODEL_PATH = "models/modnet_photographic_portrait_matting.ckpt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Scarico modello MODNET da Hugging Face...")
        os.makedirs("models", exist_ok=True)
        r = requests.get(HF_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Modello scaricato.")

download_model()

def load_modnet():
    model = MODNet(backbone_pretrained=False)
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
        _, _, matte = modnet(im_tensor, inference=True)
        matte = interpolate(matte.unsqueeze(0), size=(h, w), mode='bilinear').squeeze().numpy()

    fg = im.astype(np.float32) / 255
    alpha = np.expand_dims(matte, axis=2)
    rgba = np.concatenate((fg, alpha), axis=2)
    rgba = (rgba * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")
