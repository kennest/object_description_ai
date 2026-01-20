import os
import io
import base64
import json
import logging
import httpx
from typing import Optional
from PIL import Image, ImageEnhance
from utils.prompts import LLAVA_PROMPT, LLAVA_PROMPT_WITH_ITEM
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
REQUEST_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "60"))

# =========================
# Configuration
# =========================
class ImageConfig:
    """Configuration pour le pré-traitement d'image"""
    # Pré-traitement
    ENHANCE_CONTRAST = True
    CONTRAST_FACTOR = 1.2
    ENHANCE_SHARPNESS = True  
    SHARPNESS_FACTOR = 1.3
    
    # Taille max pour optimiser l'envoi
    MAX_IMAGE_SIZE = 1024

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


def preprocess_image(image: Image.Image, config: ImageConfig) -> Image.Image:
    """Pré-traitement pour améliorer la qualité de l'image"""
    logger.debug(f"Pré-traitement image - Taille originale: {image.size}")
    enhanced = image.copy()
    
    # Redimensionner si trop grande
    if max(enhanced.size) > config.MAX_IMAGE_SIZE:
        old_size = enhanced.size
        enhanced.thumbnail((config.MAX_IMAGE_SIZE, config.MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
        logger.info(f"📐 Image redimensionnée: {old_size} → {enhanced.size}")
    
    # Amélioration du contraste
    if config.ENHANCE_CONTRAST:
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(config.CONTRAST_FACTOR)
        logger.debug(f"Contraste amélioré (facteur: {config.CONTRAST_FACTOR})")
    
    # Amélioration de la netteté
    if config.ENHANCE_SHARPNESS:
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(config.SHARPNESS_FACTOR)
        logger.debug(f"Netteté améliorée (facteur: {config.SHARPNESS_FACTOR})")
    
    logger.info("✅ Pré-traitement terminé")
    return enhanced


def image_to_base64(image: Image.Image) -> str:
    """Convertit une image PIL en base64"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    logger.debug(f"Image encodée en base64 ({len(b64_str)} caractères)")
    return b64_str


def parse_json_response(response: str) -> dict:
    """Extrait et parse le JSON d'une réponse qui peut contenir du markdown"""
    # Nettoyer la réponse
    cleaned = response.strip()
    
    # Supprimer les blocs markdown ```json ... ```
    if "```json" in cleaned:
        start = cleaned.find("```json") + 7
        end = cleaned.rfind("```")
        if end > start:
            cleaned = cleaned[start:end].strip()
    elif "```" in cleaned:
        start = cleaned.find("```") + 3
        end = cleaned.rfind("```")
        if end > start:
            cleaned = cleaned[start:end].strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ Impossible de parser le JSON: {e}")
        return {"raw_response": response}
