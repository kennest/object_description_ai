import io
import cv2
import time
import base64
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from ultralytics import YOLO

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from prompts import LLAVA_PROMPT


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
llava = ChatOllama(
    base_url="http://127.0.0.1:11434",
    model="llava:7b",
    temperature=0.1
)


# =========================
# Utils
# =========================
def crop_to_base64(image_bgr, bbox):
    x1, y1, x2, y2 = bbox
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    _, buffer = cv2.imencode(".png", crop)
    return base64.b64encode(buffer).decode("utf-8")


def describe_with_llava(image_base64):
    message = HumanMessage(
        content=[
            {"type": "text", "text": LLAVA_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
    )
    response = llava.invoke([message])
    return response.content


# =========================
# Endpoint
# =========================
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
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
            description = describe_with_llava(image_b64)
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
