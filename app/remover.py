# app/remover.py

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import os

from app.models.u2net import U2NET

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'u2net.pth')

def load_model():
    model = U2NET(in_ch=3, out_ch=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

model = load_model()

def remove_background(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        pred = model(input_tensor)[0][0]  # [0][0] since output is [B, 1, H, W]

    pred = pred.numpy()
    mask = (pred > 0.5).astype(np.uint8) * 255
    mask = Image.fromarray(mask).resize(image.size, Image.BILINEAR)

    image = image.convert("RGBA")
    datas = image.getdata()
    newData = [(*item[:3], m) for item, m in zip(datas, mask.getdata())]
    image.putdata(newData)

    return image
