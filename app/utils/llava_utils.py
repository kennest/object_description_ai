import os
import cv2
import base64
from langchain_core.messages import HumanMessage
from utils.prompts import LLAVA_PROMPT, LLAVA_PROMPT_WITH_ITEM

from langchain_ollama import ChatOllama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

llava = ChatOllama(base_url=OLLAMA_HOST, model="llava:7b", temperature=0.1)


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


def describe_with_llava(image_base64, item_name: str = None):
    if item_name:
        prompt = LLAVA_PROMPT_WITH_ITEM.format(item_name=item_name)
    else:
        prompt = LLAVA_PROMPT

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ]
    )
    # print("LLaVA message:", message.content[0]['text'][:100], "...")  # Print only the first 100 characters of the text part
    response = llava.invoke([message])
    return response.content
