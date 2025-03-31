"""
Configuration des routes API.
"""
from fastapi import APIRouter
from app.api.v1.endpoints import categories, clothes, outfits, image_analysis, recommendations

api_router = APIRouter()

# Routes pour les catégories
api_router.include_router(
    categories.router,
    prefix="/categories",
    tags=["categories"]
)

# Routes pour les vêtements
api_router.include_router(
    clothes.router,
    prefix="/clothes",
    tags=["clothes"]
)

# Routes pour les tenues
api_router.include_router(
    outfits.router,
    prefix="/outfits",
    tags=["outfits"]
)

# Routes pour l'analyse d'images
api_router.include_router(
    image_analysis.router,
    prefix="/image-analysis",
    tags=["image-analysis"]
) 

# Routes pour les recommandations
api_router.include_router(
    recommendations.router,
    prefix="/recommendations",
    tags=["recommendations"]
)