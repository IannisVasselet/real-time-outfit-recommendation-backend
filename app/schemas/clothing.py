"""
Schémas Pydantic pour les vêtements et les catégories.
"""
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class CategoryBase(BaseModel):
    """Schéma de base pour une catégorie."""
    name: str = Field(..., description="Nom de la catégorie")
    description: Optional[str] = Field(None, description="Description de la catégorie")


class CategoryCreate(CategoryBase):
    """Schéma pour la création d'une catégorie."""
    pass


class Category(CategoryBase):
    """Schéma complet d'une catégorie."""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        """Configuration du modèle Pydantic."""
        from_attributes = True


class ClothingBase(BaseModel):
    """Schéma de base pour un vêtement."""
    name: str = Field(..., description="Nom du vêtement")
    description: Optional[str] = Field(None, description="Description du vêtement")
    image_url: str = Field(..., description="URL de l'image du vêtement")
    category_id: int = Field(..., description="ID de la catégorie")
    color: str = Field(..., description="Couleur principale du vêtement")
    pattern: Optional[str] = Field(None, description="Motif du vêtement")
    style: str = Field(..., description="Style du vêtement")
    season: str = Field(..., description="Saison appropriée")
    embeddings: Optional[Dict] = Field(None, description="Embeddings du vêtement")


class ClothingCreate(ClothingBase):
    """Schéma pour la création d'un vêtement."""
    pass


class Clothing(ClothingBase):
    """Schéma complet d'un vêtement."""
    id: int
    created_at: datetime
    updated_at: datetime
    category: Category

    class Config:
        """Configuration du modèle Pydantic."""
        from_attributes = True 