import io
import time
import cv2
import numpy as np
import pytesseract
import torch
import torchvision.transforms as T

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from sklearn.cluster import KMeans
from torchvision.models import efficientnet_b0

app = FastAPI(title="Image Analysis API", version="1.0")

# =========================
# Chargement du modèle (UNE FOIS)
# =========================
model = efficientnet_b0(weights="IMAGENET1K_V1")
model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================
# Utils
# =========================
def extract_dominant_color(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    kmeans = KMeans(n_clusters=3, n_init=5)
    kmeans.fit(pixels)

    color = kmeans.cluster_centers_[0]
    return color.astype(int).tolist()


def run_ocr(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang="eng")
    return text.strip()


def classify_image(image_rgb):
    tensor = transform(Image.fromarray(image_rgb))
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probs, 0)

    return {
        "imagenet_class_index": int(predicted.item()),
        "confidence": round(float(confidence.item()), 3),
    }


# =========================
# Endpoint principal
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

    # Couleur
    start_color = time.time()
    dominant_color = extract_dominant_color(image_bgr)
    t_color = time.time() - start_color

    # OCR
    start_ocr = time.time()
    text = run_ocr(image_bgr)
    t_ocr = time.time() - start_ocr

    # Classification
    start_cls = time.time()
    classification = classify_image(image_rgb)
    t_cls = time.time() - start_cls

    total_time = time.time() - start_total

    return {
        "classification": classification,
        "dominant_color_hsv": dominant_color,
        "ocr_text": text,
        "timings": {
            "color": round(t_color, 3),
            "ocr": round(t_ocr, 3),
            "classification": round(t_cls, 3),
            "total": round(total_time, 3),
        },
    }
