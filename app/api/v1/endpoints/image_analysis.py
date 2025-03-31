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
    # Analyse l'image
    embeddings = await image_analysis_service.analyze_image(file)
    
    # Construire les filtres
    filters = {}
    if category:
        filters["category"] = category.value
    if style:
        filters["style"] = style.value
    if color:
        filters["color"] = color
    
    # Trouve des vêtements similaires
    similar_items = db.get_similar_images(embeddings, top_k=top_k, filters=filters)
    
    return {
        "similar_items": similar_items,
        "filters_applied": filters
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