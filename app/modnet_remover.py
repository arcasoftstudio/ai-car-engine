import os
import gdown
import requests
import torch
import numpy as np
import cv2
from PIL import Image
import io
from torchvision import transforms
from torch.nn.functional import interpolate
from app.modnet_arch import MODNet


GDRIVE_ID = "1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz"
MODEL_PATH = "models/modnet_photographic_portrait_matting.ckpt"

def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024:
        print("✔️ Il modello esiste già.")
        return

    print("📥 Scarico modello da Google Drive...")
    os.makedirs("models", exist_ok=True)

    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

    print(f"✅ Modello scaricato: {round(os.path.getsize(MODEL_PATH)/1024/1024, 2)} MB")




def load_modnet():
    download_model()

    # Carica il file .ckpt
    state_dict = torch.load(MODEL_PATH, map_location='cpu')

    # ✅ Crea modello e carica pesi
    model = MODNet()
    model.load_state_dict(state_dict, strict=False)
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
    matte = interpolate(matte, size=(h, w), mode='bilinear', align_corners=False).squeeze().numpy()

    fg = im.astype(np.float32) / 255
    alpha = np.expand_dims(matte, axis=2)
    rgba = np.concatenate((fg, alpha), axis=2)
    rgba = (rgba * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")

