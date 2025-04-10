### File: app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from app.remover import remove_background
import io

app = FastAPI()

@app.post("/remove-background")
async def remove_bg(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result_image = remove_background(image_bytes)
    img_io = io.BytesIO()
    result_image.save(img_io, format='PNG')
    img_io.seek(0)
    return StreamingResponse(img_io, media_type="image/png")
