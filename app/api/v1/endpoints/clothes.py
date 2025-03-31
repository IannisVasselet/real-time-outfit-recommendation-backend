"""
Endpoints pour la gestion des vêtements.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import deps
from app.models.clothing import Clothing, Category
from app.schemas.clothing import ClothingCreate, Clothing as ClothingSchema

router = APIRouter()


@router.post("/", response_model=ClothingSchema, status_code=status.HTTP_201_CREATED)
def create_clothing(
    *,
    db: Session = Depends(deps.get_db),
    clothing_in: ClothingCreate
) -> Clothing:
    """
    Crée un nouveau vêtement.
    
    Args:
        db: Session de base de données.
        clothing_in: Données du vêtement à créer.
        
    Returns:
        Clothing: Le vêtement créé.
        
    Raises:
        HTTPException: Si la catégorie n'existe pas.
    """
    # Vérification de l'existence de la catégorie
    category = db.query(Category).filter(Category.id == clothing_in.category_id).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Catégorie non trouvée"
        )
    
    clothing = Clothing(
        name=clothing_in.name,
        description=clothing_in.description,
        image_url=clothing_in.image_url,
        category_id=clothing_in.category_id,
        color=clothing_in.color,
        pattern=clothing_in.pattern,
        style=clothing_in.style,
        season=clothing_in.season,
        embeddings=clothing_in.embeddings
    )
    db.add(clothing)
    db.commit()
    db.refresh(clothing)
    return clothing


@router.get("/", response_model=List[ClothingSchema])
def read_clothes(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    category_id: int = None
) -> List[Clothing]:
    """
    Récupère la liste des vêtements.
    
    Args:
        db: Session de base de données.
        skip: Nombre d'éléments à sauter.
        limit: Nombre maximum d'éléments à retourner.
        category_id: ID de la catégorie pour filtrer les vêtements.
        
    Returns:
        List[Clothing]: Liste des vêtements.
    """
    query = db.query(Clothing)
    if category_id:
        query = query.filter(Clothing.category_id == category_id)
    clothes = query.offset(skip).limit(limit).all()
    return clothes


@router.get("/{clothing_id}", response_model=ClothingSchema)
def read_clothing(
    *,
    db: Session = Depends(deps.get_db),
    clothing_id: int
) -> Clothing:
    """
    Récupère un vêtement par son ID.
    
    Args:
        db: Session de base de données.
        clothing_id: ID du vêtement.
        
    Returns:
        Clothing: Le vêtement demandé.
        
    Raises:
        HTTPException: Si le vêtement n'existe pas.
    """
    clothing = db.query(Clothing).filter(Clothing.id == clothing_id).first()
    if not clothing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vêtement non trouvé"
        )
    return clothing


@router.put("/{clothing_id}", response_model=ClothingSchema)
def update_clothing(
    *,
    db: Session = Depends(deps.get_db),
    clothing_id: int,
    clothing_in: ClothingCreate
) -> Clothing:
    """
    Met à jour un vêtement.
    
    Args:
        db: Session de base de données.
        clothing_id: ID du vêtement à mettre à jour.
        clothing_in: Nouvelles données du vêtement.
        
    Returns:
        Clothing: Le vêtement mis à jour.
        
    Raises:
        HTTPException: Si le vêtement ou la catégorie n'existe pas.
    """
    clothing = db.query(Clothing).filter(Clothing.id == clothing_id).first()
    if not clothing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vêtement non trouvé"
        )
    
    # Vérification de l'existence de la nouvelle catégorie
    if clothing_in.category_id != clothing.category_id:
        category = db.query(Category).filter(Category.id == clothing_in.category_id).first()
        if not category:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Catégorie non trouvée"
            )
    
    # Mise à jour des champs
    for field, value in clothing_in.dict(exclude_unset=True).items():
        setattr(clothing, field, value)
    
    db.add(clothing)
    db.commit()
    db.refresh(clothing)
    return clothing


@router.delete("/{clothing_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_clothing(
    *,
    db: Session = Depends(deps.get_db),
    clothing_id: int
) -> None:
    """
    Supprime un vêtement.
    
    Args:
        db: Session de base de données.
        clothing_id: ID du vêtement à supprimer.
        
    Raises:
        HTTPException: Si le vêtement n'existe pas.
    """
    clothing = db.query(Clothing).filter(Clothing.id == clothing_id).first()
    if not clothing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vêtement non trouvé"
        )
    
    db.delete(clothing)
    db.commit() 