"""
Endpoints pour la gestion des catégories.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import deps
from app.models.clothing import Category
from app.schemas.clothing import CategoryCreate, Category as CategorySchema

router = APIRouter()


@router.post("/", response_model=CategorySchema, status_code=status.HTTP_201_CREATED)
def create_category(
    *,
    db: Session = Depends(deps.get_db),
    category_in: CategoryCreate
) -> Category:
    """
    Crée une nouvelle catégorie.
    
    Args:
        db: Session de base de données.
        category_in: Données de la catégorie à créer.
        
    Returns:
        Category: La catégorie créée.
    """
    category = Category(
        name=category_in.name,
        description=category_in.description
    )
    db.add(category)
    db.commit()
    db.refresh(category)
    return category


@router.get("/", response_model=List[CategorySchema])
def read_categories(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100
) -> List[Category]:
    """
    Récupère la liste des catégories.
    
    Args:
        db: Session de base de données.
        skip: Nombre d'éléments à sauter.
        limit: Nombre maximum d'éléments à retourner.
        
    Returns:
        List[Category]: Liste des catégories.
    """
    categories = db.query(Category).offset(skip).limit(limit).all()
    return categories


@router.get("/{category_id}", response_model=CategorySchema)
def read_category(
    *,
    db: Session = Depends(deps.get_db),
    category_id: int
) -> Category:
    """
    Récupère une catégorie par son ID.
    
    Args:
        db: Session de base de données.
        category_id: ID de la catégorie.
        
    Returns:
        Category: La catégorie demandée.
        
    Raises:
        HTTPException: Si la catégorie n'existe pas.
    """
    category = db.query(Category).filter(Category.id == category_id).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Catégorie non trouvée"
        )
    return category


@router.put("/{category_id}", response_model=CategorySchema)
def update_category(
    *,
    db: Session = Depends(deps.get_db),
    category_id: int,
    category_in: CategoryCreate
) -> Category:
    """
    Met à jour une catégorie.
    
    Args:
        db: Session de base de données.
        category_id: ID de la catégorie à mettre à jour.
        category_in: Nouvelles données de la catégorie.
        
    Returns:
        Category: La catégorie mise à jour.
        
    Raises:
        HTTPException: Si la catégorie n'existe pas.
    """
    category = db.query(Category).filter(Category.id == category_id).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Catégorie non trouvée"
        )
    
    category.name = category_in.name
    category.description = category_in.description
    
    db.add(category)
    db.commit()
    db.refresh(category)
    return category


@router.delete("/{category_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_category(
    *,
    db: Session = Depends(deps.get_db),
    category_id: int
) -> None:
    """
    Supprime une catégorie.
    
    Args:
        db: Session de base de données.
        category_id: ID de la catégorie à supprimer.
        
    Raises:
        HTTPException: Si la catégorie n'existe pas.
    """
    category = db.query(Category).filter(Category.id == category_id).first()
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Catégorie non trouvée"
        )
    
    db.delete(category)
    db.commit() 