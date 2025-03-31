"""
Script d'initialisation de la base de données d'images.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.database.image_database import ImageDatabase, ClothingCategory, ClothingStyle
from app.services.image_analysis import ImageAnalysisService

def get_default_metadata(image_name: str) -> Dict[str, Any]:
    """
    Génère des métadonnées par défaut basées sur le nom de l'image.
    
    Args:
        image_name (str): Nom de l'image
        
    Returns:
        Dict[str, Any]: Métadonnées par défaut
    """
    # Exemple de métadonnées basées sur le nom de l'image
    name_lower = image_name.lower()
    
    # Détecter la catégorie
    category = None
    for cat in ClothingCategory:
        if cat.value in name_lower:
            category = cat.value
            break
    
    # Détecter le style
    style = None
    for st in ClothingStyle:
        if st.value in name_lower:
            style = st.value
            break
    
    return {
        "category": category or ClothingCategory.HAUT.value,
        "style": style or ClothingStyle.CASUAL.value,
        "color": "noir",  # Par défaut
        "brand": "inconnu",
        "season": "toutes",
        "price_range": "moyen"
    }

async def process_image(image_path: Path, db: ImageDatabase, image_service: ImageAnalysisService) -> None:
    """
    Traite une image et l'ajoute à la base de données.
    
    Args:
        image_path (Path): Chemin de l'image
        db (ImageDatabase): Instance de la base de données
        image_service (ImageAnalysisService): Service d'analyse d'image
    """
    print(f"Analyse de l'image : {image_path.name}")
    
    try:
        # Obtenir les embeddings de l'image
        embeddings = await image_service.analyze_image(str(image_path))
        
        # Générer les métadonnées par défaut
        metadata = get_default_metadata(image_path.name)
        
        # Ajouter l'image à la base de données
        db.add_image(str(image_path), embeddings, metadata)
        print(f"✓ Image ajoutée avec succès : {image_path.name}")
        print(f"  Métadonnées : {metadata}")
        
    except Exception as e:
        print(f"✗ Erreur lors de l'analyse de {image_path.name}: {str(e)}")

async def main():
    """
    Fonction principale pour initialiser la base de données.
    """
    # Initialiser les services
    db = ImageDatabase()
    image_service = ImageAnalysisService()
    
    # Chemin vers le dossier des images
    images_dir = Path("app/static/images")
    
    # Créer une liste de tâches pour chaque image
    tasks = []
    for image_path in images_dir.glob("*"):
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            tasks.append(process_image(image_path, db, image_service))
    
    # Exécuter toutes les tâches en parallèle
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main()) 