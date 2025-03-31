"""
Endpoints pour la gestion des tenues.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload

from app import deps
from app.models.clothing import Outfit, OutfitItem, Clothing
from app.schemas.outfit import OutfitCreate, Outfit as OutfitSchema

router = APIRouter()


@router.post("/", response_model=OutfitSchema, status_code=status.HTTP_201_CREATED)
def create_outfit(
    *,
    db: Session = Depends(deps.get_db),
    outfit_in: OutfitCreate
) -> Outfit:
    """
    Crée une nouvelle tenue.
    
    Args:
        db: Session de base de données.
        outfit_in: Données de la tenue à créer.
        
    Returns:
        Outfit: La tenue créée.
        
    Raises:
        HTTPException: Si un vêtement n'existe pas.
    """
    # Vérification de l'existence des vêtements
    for item in outfit_in.items:
        clothing = db.query(Clothing).filter(Clothing.id == item.clothing_id).first()
        if not clothing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vêtement avec l'ID {item.clothing_id} non trouvé"
            )
    
    # Création de la tenue
    outfit = Outfit(
        name=outfit_in.name,
        description=outfit_in.description,
        style=outfit_in.style,
        season=outfit_in.season,
        occasion=outfit_in.occasion
    )
    db.add(outfit)
    db.flush()  # Pour obtenir l'ID de la tenue
    
    # Création des éléments de la tenue
    for item in outfit_in.items:
        outfit_item = OutfitItem(
            outfit_id=outfit.id,
            clothing_id=item.clothing_id,
            position=item.position
        )
        db.add(outfit_item)
    
    db.commit()
    db.refresh(outfit)
    return outfit


@router.get("/", response_model=List[OutfitSchema])
def read_outfits(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    style: str = None,
    season: str = None,
    occasion: str = None
) -> List[Outfit]:
    """
    Récupère la liste des tenues.
    
    Args:
        db: Session de base de données.
        skip: Nombre d'éléments à sauter.
        limit: Nombre maximum d'éléments à retourner.
        style: Style pour filtrer les tenues.
        season: Saison pour filtrer les tenues.
        occasion: Occasion pour filtrer les tenues.
        
    Returns:
        List[Outfit]: Liste des tenues.
    """
    query = db.query(Outfit).options(
        joinedload(Outfit.items).joinedload(OutfitItem.clothing).joinedload(Clothing.category)
    )    
    if style:
        query = query.filter(Outfit.style == style)
    if season:
        query = query.filter(Outfit.season == season)
    if occasion:
        query = query.filter(Outfit.occasion == occasion)
    
    outfits = query.offset(skip).limit(limit).all()
    return outfits


@router.get("/{outfit_id}", response_model=OutfitSchema)
def read_outfit(
    *,
    db: Session = Depends(deps.get_db),
    outfit_id: int
) -> Outfit:
    """
    Récupère une tenue par son ID.
    
    Args:
        db: Session de base de données.
        outfit_id: ID de la tenue.
        
    Returns:
        Outfit: La tenue demandée.
        
    Raises:
        HTTPException: Si la tenue n'existe pas.
    """
    outfit = db.query(Outfit).filter(Outfit.id == outfit_id).first()
    if not outfit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenue non trouvée"
        )
    return outfit


@router.put("/{outfit_id}", response_model=OutfitSchema)
def update_outfit(
    *,
    db: Session = Depends(deps.get_db),
    outfit_id: int,
    outfit_in: OutfitCreate
) -> Outfit:
    """
    Met à jour une tenue.
    
    Args:
        db: Session de base de données.
        outfit_id: ID de la tenue à mettre à jour.
        outfit_in: Nouvelles données de la tenue.
        
    Returns:
        Outfit: La tenue mise à jour.
        
    Raises:
        HTTPException: Si la tenue ou un vêtement n'existe pas.
    """
    outfit = db.query(Outfit).filter(Outfit.id == outfit_id).first()
    if not outfit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenue non trouvée"
        )
    
    # Vérification de l'existence des vêtements
    for item in outfit_in.items:
        clothing = db.query(Clothing).filter(Clothing.id == item.clothing_id).first()
        if not clothing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vêtement avec l'ID {item.clothing_id} non trouvé"
            )
    
    # Mise à jour des champs de base
    outfit.name = outfit_in.name
    outfit.description = outfit_in.description
    outfit.style = outfit_in.style
    outfit.season = outfit_in.season
    outfit.occasion = outfit_in.occasion
    
    # Suppression des anciens éléments
    db.query(OutfitItem).filter(OutfitItem.outfit_id == outfit_id).delete()
    
    # Création des nouveaux éléments
    for item in outfit_in.items:
        outfit_item = OutfitItem(
            outfit_id=outfit_id,
            clothing_id=item.clothing_id,
            position=item.position
        )
        db.add(outfit_item)
    
    db.commit()
    db.refresh(outfit)
    return outfit


@router.delete("/{outfit_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_outfit(
    *,
    db: Session = Depends(deps.get_db),
    outfit_id: int
) -> None:
    """
    Supprime une tenue.
    
    Args:
        db: Session de base de données.
        outfit_id: ID de la tenue à supprimer.
        
    Raises:
        HTTPException: Si la tenue n'existe pas.
    """
    outfit = db.query(Outfit).filter(Outfit.id == outfit_id).first()
    if not outfit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenue non trouvée"
        )
    
    db.delete(outfit)
    db.commit() 