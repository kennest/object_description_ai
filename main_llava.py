import io
import cv2
import time
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
from PIL import Image
from ultralytics import YOLO
from utils.llava_utils import crop_to_base64, describe_with_llava





# =========================
# App
# =========================
app = FastAPI(title="YOLO + LLaVA Vision API", version="1.0")

# =========================
# YOLOv8 CPU
# =========================
yolo = YOLO("yolov8n.pt")

# =========================
# LLaVA via Ollama
# =========================





# =========================
# Endpoint
# =========================
@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    item_name: Optional[str] = Form(None, description="Nom de l'article pour affiner la description")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image invalide")

    start_total = time.time()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # =========================
    # YOLO detection
    # =========================
    yolo_start = time.time()
    results = yolo(image_rgb, device="cpu", verbose=False)
    yolo_time = time.time() - yolo_start

    objects = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo.names[cls]

            image_b64 = crop_to_base64(image_bgr, (x1, y1, x2, y2))
            if not image_b64:
                continue

            # =========================
            # LLaVA description
            # =========================
            llava_start = time.time()
            description = describe_with_llava(image_b64, item_name)
            llava_time = time.time() - llava_start

            objects.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "llava_description": description,
                "llava_time": round(llava_time, 2)
            })

    total_time = time.time() - start_total

    return {
        "objects": objects,
        "timings": {
            "yolo": round(yolo_time, 3),
            "total": round(total_time, 3)
        }
    }
