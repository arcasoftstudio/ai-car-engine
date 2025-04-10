### File: app/remover.py
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'u2net.pth')

# Dummy U2NET class placeholder - replace with actual model implementation
class DummyU2Net(torch.nn.Module):
    def forward(self, x):
        return torch.ones((1, 1, x.shape[2], x.shape[3]))  # full mask

def load_model():
    model = DummyU2Net()  # Replace with actual U2NET
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
        pred = model(input_tensor)[0][0].numpy()

    mask = (pred > 0.5).astype(np.uint8) * 255
    mask = Image.fromarray(mask).resize(image.size, Image.BILINEAR)

    image = image.convert("RGBA")
    datas = image.getdata()
    newData = []
    for item, m in zip(datas, mask.getdata()):
        newData.append((*item[:3], m))
    image.putdata(newData)
    return image
