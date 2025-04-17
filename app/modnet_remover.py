import os
import gdown
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
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model = MODNet()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def resize_with_aspect_ratio(image, ref_size=512):
    h, w = image.shape[:2]
    if max(h, w) < ref_size or min(h, w) > ref_size:
        if w >= h:
            new_w = ref_size
            new_h = int(h * ref_size / w)
        else:
            new_h = ref_size
            new_w = int(w * ref_size / h)
    else:
        new_w, new_h = w, h

    resized = cv2.resize(image, (new_w, new_h))
    pad_h = ref_size - new_h
    pad_w = ref_size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    return padded, (top, bottom, left, right), (h, w)

def remove_background_modnet(image_bytes: bytes):
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    im = np.array(input_image)
    modnet = load_modnet()

    im_resized, padding, original_size = resize_with_aspect_ratio(im)
    im_tensor = transforms.ToTensor()(im_resized).unsqueeze(0)

    with torch.no_grad():
        matte = modnet(im_tensor)[0]
        matte = matte.squeeze().cpu().numpy()

    top, bottom, left, right = padding
    matte_cropped = matte[top:512 - bottom, left:512 - right]
    matte_resized = cv2.resize(matte_cropped, (original_size[1], original_size[0]))

    fg = im.astype(np.float32) / 255
    alpha = np.expand_dims(matte_resized, axis=2)
    rgba = np.concatenate((fg, alpha), axis=2)
    rgba = (rgba * 255).astype(np.uint8)

    return Image.fromarray(rgba, mode="RGBA")
