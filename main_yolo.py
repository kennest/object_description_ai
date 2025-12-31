import io
import time
import cv2
import numpy as np
import pytesseract

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from sklearn.cluster import KMeans
from ultralytics import YOLO

app = FastAPI(title="YOLOv8 CPU Image Analysis API", version="1.0")

# =========================
# Charger YOLOv8 (UNE FOIS)
# =========================
yolo_model = YOLO("yolov8n.pt")  # nano = CPU friendly


# =========================
# Utils
# =========================
def extract_dominant_color(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    kmeans = KMeans(n_clusters=3, n_init=5)
    kmeans.fit(pixels)

    return kmeans.cluster_centers_[0].astype(int).tolist()


def run_ocr(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, lang="eng").strip()


# =========================
# Endpoint
# =========================
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fichier non valide")

    start_total = time.time()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # =========================
    # YOLOv8 CPU inference
    # =========================
    start_yolo = time.time()
    results = yolo_model(image_rgb, verbose=False, device="cpu")
    t_yolo = time.time() - start_yolo

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]

            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Couleur dominante
            color = extract_dominant_color(crop)

            # OCR
            text = run_ocr(crop)

            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "dominant_color_hsv": color,
                "ocr_text": text,
            })

    total_time = time.time() - start_total

    return {
        "detections": detections,
        "timings": {
            "yolo": round(t_yolo, 3),
            "total": round(total_time, 3),
        },
    }
