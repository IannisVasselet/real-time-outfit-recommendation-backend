"""
API endpoints pour l'analyse d'images.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Query
from app.services.image_analysis import image_analysis_service
from app.database.image_database import ImageDatabase, ClothingCategory, ClothingStyle

router = APIRouter()
db = ImageDatabase()

@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    category: Optional[ClothingCategory] = None,
    style: Optional[ClothingStyle] = None,
    color: Optional[str] = None,
    top_k: int = Query(5, ge=1, le=20)
) -> Dict[str, Any]:
    """
    Analyse une image et trouve des vêtements similaires.
    
    Args:
        file: L'image à analyser
        category: Catégorie de vêtement à filtrer
        style: Style de vêtement à filtrer
        color: Couleur à filtrer
        top_k: Nombre de résultats à retourner
        
    Returns:
        Dict[str, Any]: Les résultats de l'analyse et les vêtements similaires
    """
    # Analyse le vêtement pour détecter ses caractéristiques
    analysis_result = await image_analysis_service.analyze_clothing(file)
    
    # Analyse l'image pour les embeddings (fonctionnalité existante)
    embeddings = analysis_result["embeddings"]
    
    # Construire les filtres
    filters = {}
    if category:
        filters["category"] = category.value
    if style:
        filters["style"] = style.value
    if color:
        filters["color"] = color
    
    # Trouver des vêtements similaires dans la base de données
    similar_images = db.get_similar_images(embeddings, top_k, filters)
    
    # Retourner les résultats de l'analyse + vêtements similaires
    return {
        "analysis": {
            "color": analysis_result["color"],
            "predicted_type": analysis_result["predicted_type"],
            "predicted_season": analysis_result["predicted_season"],
            "predicted_style": analysis_result["predicted_style"]
        },
        "similar_items": similar_images
    }

@router.put("/metadata/{image_name}")
async def update_image_metadata(
    image_name: str,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Met à jour les métadonnées d'une image.
    
    Args:
        image_name: Nom de l'image
        metadata: Nouvelles métadonnées
        
    Returns:
        Dict[str, Any]: Résultat de la mise à jour
    """
    success = db.update_metadata(image_name, metadata)
    return {
        "success": success,
        "message": "Métadonnées mises à jour avec succès" if success else "Image non trouvée"
    } 