"""
Module de gestion de la base de données d'images et leurs embeddings.
"""

import os
import json
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
from enum import Enum

class ClothingCategory(str, Enum):
    """
    Catégories de vêtements disponibles.
    """
    HAUT = "haut"
    BAS = "bas"
    ROBE = "robe"
    VESTE = "veste"
    CHAUSSURES = "chaussures"
    ACCESSOIRE = "accessoire"

class ClothingStyle(str, Enum):
    """
    Styles de vêtements disponibles.
    """
    CASUAL = "casual"
    ELEGANT = "elegant"
    SPORT = "sport"
    BUSINESS = "business"
    VINTAGE = "vintage"
    MODERNE = "moderne"

class ImageDatabase:
    """
    Classe pour gérer la base de données d'images et leurs embeddings.
    """
    
    def __init__(self, db_path: str = "image_database.json"):
        """
        Initialise la base de données d'images.
        
        Args:
            db_path (str): Chemin vers le fichier de base de données
        """
        self.db_path = db_path
        self.images_dir = Path("app/static/images")
        self.database: Dict[str, Dict] = {}
        
        # Créer le dossier parent si nécessaire
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        
        # Charger ou créer la base de données
        self._load_database()
    
    def _load_database(self) -> None:
        """
        Charge la base de données depuis le fichier JSON.
        Si le fichier n'existe pas, crée une base de données vide.
        """
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    self.database = json.load(f)
            else:
                # Créer une base de données vide et la sauvegarder
                self.database = {}
                self._save_database()
        except json.JSONDecodeError:
            print(f"Erreur de décodage JSON dans {self.db_path}. Création d'une nouvelle base de données.")
            self.database = {}
            self._save_database()
    
    def _save_database(self) -> None:
        """
        Sauvegarde la base de données dans le fichier JSON.
        """
        with open(self.db_path, 'w') as f:
            json.dump(self.database, f, indent=2)
    
    def add_image(
        self,
        image_path: str,
        embeddings: List[float],
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Ajoute une nouvelle image à la base de données.
        
        Args:
            image_path (str): Chemin relatif de l'image
            embeddings (List[float]): Embeddings de l'image
            metadata (Optional[Dict]): Métadonnées du vêtement
        """
        image_name = os.path.basename(image_path)
        self.database[image_name] = {
            "path": image_path,
            "embeddings": embeddings,
            "metadata": metadata or {}
        }
        self._save_database()
        print(f"Image {image_name} ajoutée à la base de données")
    
    def update_metadata(self, image_name: str, metadata: Dict) -> bool:
        """
        Met à jour les métadonnées d'une image.
        
        Args:
            image_name (str): Nom de l'image
            metadata (Dict): Nouvelles métadonnées
            
        Returns:
            bool: True si la mise à jour a réussi, False sinon
        """
        if image_name in self.database:
            self.database[image_name]["metadata"].update(metadata)
            self._save_database()
            print(f"Métadonnées mises à jour pour {image_name}")
            return True
        return False
    
    def get_similar_images(
        self,
        target_embeddings: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Trouve les images les plus similaires à partir des embeddings cibles.
        
        Args:
            target_embeddings (List[float]): Embeddings de l'image cible
            top_k (int): Nombre d'images similaires à retourner
            filters (Optional[Dict]): Filtres à appliquer sur les métadonnées
            
        Returns:
            List[Dict]: Liste des images similaires avec leurs scores
        """
        if not self.database:
            print("La base de données est vide")
            return []
            
        target_embeddings = np.array(target_embeddings)
        similarities = []
        
        for image_name, data in self.database.items():
            # Vérifier les filtres si présents
            if filters:
                metadata = data.get("metadata", {})
                if not all(metadata.get(k) == v for k, v in filters.items()):
                    continue
            
            db_embeddings = np.array(data["embeddings"])
            
            # Vérifier la compatibilité des dimensions
            if len(target_embeddings) != len(db_embeddings):
                print(f"Incompatibilité de dimensions: {len(target_embeddings)} vs {len(db_embeddings)} pour {image_name}")
                # Ignorer cet élément plutôt que de planter
                continue
            
            # Calculer la similarité seulement si les dimensions correspondent
            similarity = np.dot(target_embeddings, db_embeddings) / (
                np.linalg.norm(target_embeddings) * np.linalg.norm(db_embeddings)
            )
            similarities.append({
                "image_name": image_name,
                "path": data["path"],
                "metadata": data.get("metadata", {}),
                "similarity": float(similarity)
            })
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def get_image_embeddings(self, image_name: str) -> Optional[List[float]]:
        """
        Récupère les embeddings d'une image spécifique.
        
        Args:
            image_name (str): Nom de l'image
            
        Returns:
            Optional[List[float]]: Embeddings de l'image ou None si non trouvée
        """
        return self.database.get(image_name, {}).get("embeddings")
    
    def get_image_metadata(self, image_name: str) -> Optional[Dict]:
        """
        Récupère les métadonnées d'une image spécifique.
        
        Args:
            image_name (str): Nom de l'image
            
        Returns:
            Optional[Dict]: Métadonnées de l'image ou None si non trouvée
        """
        return self.database.get(image_name, {}).get("metadata")
    
    def remove_image(self, image_name: str) -> bool:
        """
        Supprime une image de la base de données.
        
        Args:
            image_name (str): Nom de l'image à supprimer
            
        Returns:
            bool: True si l'image a été supprimée, False sinon
        """
        if image_name in self.database:
            del self.database[image_name]
            self._save_database()
            print(f"Image {image_name} supprimée de la base de données")
            return True
        return False 