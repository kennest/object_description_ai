import os
import logging
import httpx
from typing import Optional
from utils.prompts import LLAVA_PROMPT, LLAVA_PROMPT_WITH_ITEM
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
REQUEST_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "60"))

# =========================
# Configuration du logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.info(f"📂  Modèle (OpenRouter): {MODEL}")


# =========================
# Utils
# =========================
async def describe_with_openrouter(
    image_base64: str, 
    item_name: Optional[str] = None
) -> str:
    """Appelle l'API OpenRouter avec une image pour obtenir une description"""
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY non configurée. Ajoutez-la dans le fichier .env")
    
    # Sélection du prompt
    if item_name:
        prompt = LLAVA_PROMPT_WITH_ITEM.format(item_name=item_name)
    else:
        prompt = LLAVA_PROMPT
    
    # Construction du payload
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.1
    }
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Object Description API"
    }
    
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            OPENROUTER_BASE_URL,
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
