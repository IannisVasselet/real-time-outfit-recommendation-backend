"""
Script de préparation des données d'entraînement.
"""
import os
import sys
import shutil
from pathlib import Path
import random
from typing import List, Dict, Any
import json

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.models.clothing_classifier import ClothingClassifier, SeasonClassifier

def create_directory_structure(base_dir: str, classifiers: List[Dict[str, Any]]):
    """
    Crée la structure de répertoires pour les données d'entraînement.
    
    Args:
        base_dir: Répertoire de base pour les données
        classifiers: Liste des classificateurs et leurs classes
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for classifier in classifiers:
        classifier_dir = base_path / classifier["name"]
        classifier_dir.mkdir(exist_ok=True)
        
        for class_name in classifier["classes"]:
            class_dir = classifier_dir / class_name
            class_dir.mkdir(exist_ok=True)

def organize_images(
    source_dir: str,
    target_dir: str,
    classifiers: List[Dict[str, Any]],
    train_ratio: float = 0.8
):
    """
    Organise les images dans la structure de répertoires appropriée.
    
    Args:
        source_dir: Répertoire source contenant les images
        target_dir: Répertoire cible pour les données organisées
        classifiers: Liste des classificateurs et leurs classes
        train_ratio: Ratio des données pour l'entraînement
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Créer la structure de répertoires
    create_directory_structure(target_dir, classifiers)
    
    # Parcourir les images source
    for img_path in source_path.glob("**/*.jpg"):
        # Lire les métadonnées
        metadata_path = img_path.with_suffix(".json")
        if not metadata_path.exists():
            print(f"Pas de métadonnées pour {img_path}")
            continue
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Copier l'image dans les répertoires appropriés
        for classifier in classifiers:
            class_name = metadata.get(classifier["metadata_key"])
            if class_name and class_name in classifier["classes"]:
                # Déterminer si l'image va dans train ou val
                is_train = random.random() < train_ratio
                subdir = "train" if is_train else "val"
                
                # Créer le chemin cible
                target_img_path = (
                    target_path /
                    classifier["name"] /
                    class_name /
                    subdir /
                    img_path.name
                )
                
                # Copier l'image
                shutil.copy2(img_path, target_img_path)
                
                # Copier les métadonnées
                target_metadata_path = target_img_path.with_suffix(".json")
                shutil.copy2(metadata_path, target_metadata_path)

def main():
    """
    Fonction principale pour la préparation des données.
    """
    # Configuration
    source_dir = "data/raw"
    target_dir = "data/processed"
    
    # Définir les classificateurs et leurs classes
    classifiers = [
        {
            "name": "clothing",
            "metadata_key": "type",
            "classes": [
                "t-shirt",
                "pantalon",
                "pull",
                "robe",
                "manteau",
                "sandale",
                "chemise",
                "baskets",
                "sac",
                "bottines"
            ]
        },
        {
            "name": "seasons",
            "metadata_key": "season",
            "classes": [
                "printemps",
                "été",
                "automne",
                "hiver"
            ]
        }
    ]
    
    print("Préparation des données d'entraînement...")
    print(f"Source: {source_dir}")
    print(f"Cible: {target_dir}")
    
    # Organiser les images
    organize_images(source_dir, target_dir, classifiers)
    
    print("\nPréparation terminée !")
    print(f"Les données ont été organisées dans {target_dir}")

if __name__ == "__main__":
    main() 