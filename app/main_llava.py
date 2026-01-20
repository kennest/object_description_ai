import io
import base64
import time
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
from PIL import Image, ImageEnhance
from utils.llava_utils import describe_with_llava


# =========================
# Configuration du logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


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
    
    # Taille max pour optimiser l'envoi à LLaVA
    MAX_IMAGE_SIZE = 1024


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





# =========================
# App
# =========================
app = FastAPI(title="LLaVA Vision API - Gestion de Stock", version="2.0")

# =========================
# Configuration globale
# =========================
config = ImageConfig()
logger.info("🚀 LLaVA Vision API démarrée")
logger.info(f"📋 Config: max_size={config.MAX_IMAGE_SIZE}, contrast={config.CONTRAST_FACTOR}, sharpness={config.SHARPNESS_FACTOR}")


# =========================
# Endpoint
# =========================
@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    item_name: Optional[str] = Form(None, description="Nom de l'article pour affiner la description")
):
    """Analyse une image d'article pour la gestion de stock via LLaVA"""
    logger.info("=" * 50)
    logger.info(f"📥 Nouvelle requête: {file.filename} ({file.content_type})")
    if item_name:
        logger.info(f"🏷️  Article recherché: {item_name}")
    
    if not file.content_type.startswith("image/"):
        logger.warning(f"❌ Type de fichier invalide: {file.content_type}")
        raise HTTPException(status_code=400, detail="Image invalide")

    start_total = time.time()

    # Charger l'image
    logger.info("📂 Chargement de l'image...")
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size
    logger.info(f"📷 Image chargée: {original_size[0]}x{original_size[1]} pixels")

    # =========================
    # Pré-traitement de l'image
    # =========================
    logger.info("🔧 Pré-traitement en cours...")
    preprocess_start = time.time()
    image_enhanced = preprocess_image(image, config)
    preprocess_time = time.time() - preprocess_start
    logger.info(f"⏱️  Pré-traitement: {preprocess_time:.3f}s")
    
    # Convertir en base64 pour LLaVA
    logger.debug("Conversion en base64...")
    image_b64 = image_to_base64(image_enhanced)

    # =========================
    # Description avec LLaVA
    # =========================
    logger.info("🤖 Envoi à LLaVA pour analyse...")
    llava_start = time.time()
    description = describe_with_llava(image_b64, item_name)
    llava_time = time.time() - llava_start
    logger.info(f"✅ Réponse LLaVA reçue en {llava_time:.3f}s")
    logger.debug(f"Description: {description[:200]}..." if len(description) > 200 else f"Description: {description}")

    total_time = time.time() - start_total
    logger.info(f"🏁 Requête terminée en {total_time:.3f}s")
    logger.info("=" * 50)

    return {
        "filename": file.filename,
        "item_searched": item_name,
        "original_size": original_size,
        "description": description,
        "timings": {
            "preprocessing": round(preprocess_time, 3),
            "llava_inference": round(llava_time, 3),
            "total": round(total_time, 3)
        }
    }


@app.get("/config")
async def get_config():
    """Retourne la configuration actuelle"""
    logger.debug("Requête config reçue")
    return {
        "max_image_size": config.MAX_IMAGE_SIZE,
        "preprocessing": {
            "contrast_enhancement": config.ENHANCE_CONTRAST,
            "contrast_factor": config.CONTRAST_FACTOR,
            "sharpness_enhancement": config.ENHANCE_SHARPNESS,
            "sharpness_factor": config.SHARPNESS_FACTOR
        }
    }


@app.get("/health")
async def health_check():
    """Vérification de l'état du service"""
    logger.debug("Health check")
    return {
        "status": "healthy",
        "service": "LLaVA Vision API"
    }
