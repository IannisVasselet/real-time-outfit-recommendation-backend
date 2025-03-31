"""
API endpoints pour les recommandations de tenues.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, joinedload

from app import deps
from app.models.clothing import Clothing, Category, Outfit, OutfitItem
from app.schemas.outfit import Outfit as OutfitSchema

router = APIRouter()

@router.get("/", response_model=List[OutfitSchema])
def get_outfit_recommendations(
    db: Session = Depends(deps.get_db),
    weather: Optional[str] = None,
    temperature: Optional[float] = None,
    occasion: Optional[str] = None,
    preferred_styles: Optional[str] = Query(None),
    preferred_colors: Optional[str] = Query(None),
    excluded_clothing_ids: Optional[str] = Query(None)
):
    """
    Générer des recommandations de tenues basées sur divers critères.
    
    Args:
        db: Session de base de données
        weather: Condition météo actuelle (ensoleillé, nuageux, pluvieux, etc.)
        temperature: Température actuelle en °C
        occasion: Type d'occasion (quotidien, travail, soirée, etc.)
        preferred_styles: Styles préférés séparés par virgules
        preferred_colors: Couleurs préférées séparées par virgules
        excluded_clothing_ids: IDs de vêtements à exclure séparés par virgules
        
    Returns:
        List[OutfitSchema]: Liste des tenues recommandées
    """
    # Initialiser la requête de base
    query = db.query(Outfit).options(
        joinedload(Outfit.items).joinedload(OutfitItem.clothing).joinedload(Clothing.category)
    )
    
    # Appliquer les filtres basés sur les paramètres fournis
    
    # Filtre par occasion
    if occasion:
        query = query.filter(Outfit.occasion == occasion)
    
    # Filtre par styles préférés
    if preferred_styles:
        styles_list = preferred_styles.split(",")
        if styles_list:
            query = query.filter(Outfit.style.in_(styles_list))
    
    # Logique de recommandation basée sur la météo et la température
    if weather or temperature is not None:
        # En cas de pluie, recommander des tenues d'automne/hiver
        if weather == "pluvieux":
            query = query.filter(Outfit.season.in_(["automne", "hiver", "toutes saisons"]))
        
        # En cas de neige, recommander des tenues d'hiver
        elif weather == "neigeux":
            query = query.filter(Outfit.season.in_(["hiver", "toutes saisons"]))
        
        # Basé sur la température
        if temperature is not None:
            if temperature < 5:
                query = query.filter(Outfit.season.in_(["hiver", "toutes saisons"]))
            elif temperature < 15:
                query = query.filter(Outfit.season.in_(["automne", "printemps", "toutes saisons"]))
            elif temperature < 25:
                query = query.filter(Outfit.season.in_(["printemps", "été", "toutes saisons"]))
            else:
                query = query.filter(Outfit.season.in_(["été", "toutes saisons"]))
    
    # Exclure les tenues contenant des vêtements spécifiques
    if excluded_clothing_ids:
        excluded_ids = [int(id) for id in excluded_clothing_ids.split(",") if id.isdigit()]
        if excluded_ids:
            query = query.filter(
                ~Outfit.items.any(OutfitItem.clothing_id.in_(excluded_ids))
            )
    
    # Trier par popularité (les plus populaires d'abord)
    query = query.order_by(Outfit.popularity_score.desc())
    
    # Limiter les résultats
    outfits = query.limit(10).all()
    
    # Si pas suffisamment de résultats, relâcher certains filtres
    if len(outfits) < 3:
        # Initialiser une nouvelle requête sans filtres de style ni d'occasion
        backup_query = db.query(Outfit).options(
            joinedload(Outfit.items).joinedload(OutfitItem.clothing).joinedload(Clothing.category)
        )
        
        # Garder seulement les filtres de météo/température s'ils existaient
        if weather or temperature is not None:
            if temperature is not None:
                if temperature < 5:
                    backup_query = backup_query.filter(Outfit.season.in_(["hiver", "toutes saisons"]))
                elif temperature < 15:
                    backup_query = backup_query.filter(Outfit.season.in_(["automne", "printemps", "toutes saisons"]))
                elif temperature < 25:
                    backup_query = backup_query.filter(Outfit.season.in_(["printemps", "été", "toutes saisons"]))
                else:
                    backup_query = backup_query.filter(Outfit.season.in_(["été", "toutes saisons"]))
        
        # Trier et limiter les résultats de secours
        backup_query = backup_query.order_by(Outfit.popularity_score.desc())
        backup_outfits = backup_query.limit(10 - len(outfits)).all()
        
        # Ajouter les résultats de secours
        outfits.extend(backup_outfits)
    
    return outfits