"""
Schémas Pydantic pour les tenues.
"""
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.schemas.clothing import Clothing


class OutfitItemBase(BaseModel):
    """Schéma de base pour un élément de tenue."""
    clothing_id: int = Field(..., description="ID du vêtement")
    position: int = Field(..., description="Position du vêtement dans la tenue")


class OutfitItemCreate(OutfitItemBase):
    """Schéma pour la création d'un élément de tenue."""
    pass


class OutfitItem(OutfitItemBase):
    """Schéma complet d'un élément de tenue."""
    id: int
    outfit_id: int
    clothing_id: Optional[int] = None
    clothing: Optional[Clothing] = None
    
    class Config:
        """Configuration du modèle Pydantic."""
        from_attributes = True


class OutfitBase(BaseModel):
    """Schéma de base pour une tenue."""
    name: str = Field(..., description="Nom de la tenue")
    description: Optional[str] = Field(None, description="Description de la tenue")
    style: str = Field(..., description="Style de la tenue")
    season: str = Field(..., description="Saison appropriée")
    occasion: str = Field(..., description="Occasion pour laquelle la tenue est appropriée")


class OutfitCreate(OutfitBase):
    """Schéma pour la création d'une tenue."""
    items: List[OutfitItemCreate] = Field(..., description="Liste des éléments de la tenue")


class Outfit(OutfitBase):
    """Schéma complet d'une tenue."""
    id: int
    created_at: datetime
    updated_at: datetime
    popularity_score: float = Field(default=0.0, description="Score de popularité de la tenue")
    items: List[OutfitItem]

    class Config:
        """Configuration du modèle Pydantic."""
        from_attributes = True 