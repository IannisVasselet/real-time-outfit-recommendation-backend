"""
Module contenant les modèles pour les vêtements.
"""
from typing import List
from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship

from .base import Base


class Category(Base):
    """
    Modèle pour les catégories de vêtements.
    """
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)

    # Relations
    clothes = relationship("Clothing", back_populates="category")


class Clothing(Base):
    """
    Modèle pour les vêtements.
    """
    __tablename__ = "clothes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    image_url = Column(String)
    category_id = Column(Integer, ForeignKey("categories.id"))
    
    # Caractéristiques visuelles
    color = Column(String)
    pattern = Column(String, nullable=True)
    style = Column(String, nullable=True)
    season = Column(String, nullable=True)
    
    # Vecteur d'embeddings pour la recherche de similarité
    embeddings = Column(JSON)
    
    # Relations
    category = relationship("Category", back_populates="clothes")
    outfits = relationship("OutfitItem", back_populates="clothing")


class Outfit(Base):
    """
    Modèle pour les tenues complètes.
    """
    __tablename__ = "outfits"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    style = Column(String)
    season = Column(String)
    occasion = Column(String, nullable=True)
    
    # Score de popularité
    popularity_score = Column(Float, default=0.0)
    
    # Relations
    items = relationship("OutfitItem", back_populates="outfit")


class OutfitItem(Base):
    """
    Table de liaison entre les tenues et les vêtements.
    """
    __tablename__ = "outfit_items"

    id = Column(Integer, primary_key=True, index=True)
    outfit_id = Column(Integer, ForeignKey("outfits.id"))
    clothing_id = Column(Integer, ForeignKey("clothes.id"))
    position = Column(Integer)  # Position dans la tenue (haut, bas, etc.)
    
    # Relations
    outfit = relationship("Outfit", back_populates="items")
    clothing = relationship("Clothing", back_populates="outfits") 