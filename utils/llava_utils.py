import os
import cv2
import base64
from langchain_core.messages import HumanMessage
from utils.prompts import LLAVA_PROMPT

from langchain_ollama import ChatOllama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

llava = ChatOllama(
    base_url=OLLAMA_HOST,
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
