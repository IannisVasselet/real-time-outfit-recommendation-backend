"""
Service d'analyse d'images utilisant ResNet50 et des modèles spécialisés pour la détection de caractéristiques.
"""
import os
import colorsys
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from fastapi import UploadFile
from sklearn.cluster import KMeans
from app.core.config import settings
from app.services.file_storage import file_storage_service

class ImageAnalysisService:
    def __init__(self):
        """
        Initialise le service d'analyse d'images.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Charger les modèles spécialisés ou définir des classifications
        self.clothing_types = {
            0: "t-shirt",
            1: "pantalon",
            2: "pull",
            3: "robe",
            4: "manteau",
            5: "sandale",
            6: "chemise",
            7: "baskets",
            8: "sac",
            9: "bottines"
        }
        
        self.seasons_map = {
            "t-shirt": ["printemps", "été"],
            "pantalon": ["toutes saisons"],
            "pull": ["automne", "hiver"],
            "robe": ["printemps", "été"],
            "manteau": ["automne", "hiver"],
            "sandale": ["été"],
            "chemise": ["printemps", "automne"],
            "baskets": ["toutes saisons"],
            "sac": ["toutes saisons"],
            "bottines": ["automne", "hiver"]
        }
        
        self.styles_map = {
            "t-shirt": "casual",
            "pantalon": "casual",
            "pull": "casual",
            "robe": ["casual", "formal"],
            "manteau": ["casual", "formal"],
            "sandale": "casual",
            "chemise": ["casual", "formal", "business"],
            "baskets": "sport",
            "sac": ["casual", "formal"],
            "bottines": ["casual", "vintage"]
        }
        
        # Préparation du prétraitement
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Prétraite une image pour l'analyse.

        Args:
            image: L'image à prétraiter

        Returns:
            torch.Tensor: L'image prétraitée
        """
        return self.transform(image).unsqueeze(0).to(self.device)

    def extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Extrait les caractéristiques d'une image.

        Args:
            image_tensor: Le tenseur de l'image

        Returns:
            np.ndarray: Les caractéristiques extraites
        """
        with torch.no_grad():
            features = self.model(image_tensor)
            features = features.squeeze().cpu().numpy()
        return features

    async def analyze_image(self, file: Union[str, UploadFile]) -> List[float]:
        """
        Analyse une image et retourne ses caractéristiques.

        Args:
            file: Le fichier image à analyser (chemin ou UploadFile)

        Returns:
            List[float]: Les caractéristiques de l'image
        """
        if isinstance(file, str):
            # Si c'est un chemin de fichier
            image_path = Path(file)
            image = Image.open(image_path).convert('RGB')
        else:
            # Si c'est un UploadFile
            # Sauvegarde l'image
            image_url = await file_storage_service.save_image(file)
            image_path = Path("app/static/images") / image_url.split("/")[-1]
            image = Image.open(image_path).convert('RGB')
        
        # Prétraite et extrait les caractéristiques
        image_tensor = self.preprocess_image(image)
        features = self.extract_features(image_tensor)
        
        return features.tolist()

    def find_similar_items(
        self,
        target_features: np.ndarray,
        items: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Trouve les items les plus similaires à une image cible.

        Args:
            target_features: Les caractéristiques de l'image cible
            items: La liste des items à comparer
            top_k: Le nombre d'items similaires à retourner

        Returns:
            List[Dict[str, Any]]: Les items les plus similaires
        """
        similarities = []
        
        for item in items:
            if item.get("embeddings"):
                item_features = np.array(item["embeddings"])
                similarity = np.dot(target_features, item_features) / (
                    np.linalg.norm(target_features) * np.linalg.norm(item_features)
                )
                similarities.append((similarity, item))
        
        # Trie par similarité décroissante
        similarities.sort(reverse=True)
        
        return [item for _, item in similarities[:top_k]]
    
    def detect_dominant_color(self, image: Image.Image) -> Tuple[str, List[float]]:
        """
        Détecte la couleur dominante dans une image.
        
        Args:
            image: L'image à analyser
            
        Returns:
            Tuple contenant le nom de la couleur et ses valeurs RGB (0-1)
        """
        # Redimensionner l'image pour accélérer le traitement
        small_image = image.resize((100, 100))
        # Convertir en tableau numpy
        np_image = np.array(small_image)
        # Remodeler pour KMeans
        pixels = np_image.reshape(-1, 3)
        
        # Appliquer KMeans pour trouver les couleurs dominantes
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(pixels)
        
        # Identifier la couleur la plus fréquente
        counts = np.bincount(kmeans.labels_)
        dominant_cluster = np.argmax(counts)
        dominant_color = kmeans.cluster_centers_[dominant_cluster]
        
        # Normaliser les valeurs RGB (0-255 -> 0-1)
        r, g, b = dominant_color / 255.0
        
        # Convertir RGB en HSV pour faciliter la classification des couleurs
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        # Mapper HSV à un nom de couleur
        color_name = self.get_color_name(h, s, v)
        
        return color_name, [r, g, b]
    
    def get_color_name(self, h: float, s: float, v: float) -> str:
        """
        Convertit les valeurs HSV en nom de couleur.
        
        Args:
            h: Teinte (0-1)
            s: Saturation (0-1)
            v: Valeur (0-1)
            
        Returns:
            Le nom de la couleur
        """
        # Si saturation est très basse, c'est une nuance de gris
        if s < 0.1:
            if v < 0.3: return "noir"
            if v < 0.7: return "gris"
            return "blanc"
        
        # Classification basée sur la teinte
        if h < 0.04: return "rouge"
        if h < 0.08: return "orange"
        if h < 0.13: return "jaune"
        if h < 0.33: return "vert"
        if h < 0.5: return "bleu"
        if h < 0.7: return "violet"
        if h < 0.83: return "rose"
        return "rouge"  # Boucle de la teinte (retour au rouge)
    
    def predict_clothing_type(self, features: np.ndarray) -> str:
        """
        Prédit le type de vêtement basé sur les caractéristiques extraites.
        
        Args:
            features: Caractéristiques de l'image
            
        Returns:
            Le type de vêtement prédit
        """
        # Prédiction simulée basée sur un sous-ensemble de caractéristiques
        # Dans une implémentation réelle, vous utiliseriez un modèle entraîné
        sub_features = features[:10]
        value = abs(hash(tuple(sub_features.round(2)))) % 10
        return self.clothing_types[value]
    
    def predict_season(self, clothing_type: str) -> str:
        """
        Prédit la saison appropriée pour un type de vêtement.
        
        Args:
            clothing_type: Le type de vêtement
            
        Returns:
            La saison prédite
        """
        seasons = self.seasons_map.get(clothing_type, ["toutes saisons"])
        if isinstance(seasons, list):
            return seasons[0]  # Retourne la première saison suggérée
        return seasons
    
    def predict_style(self, clothing_type: str) -> str:
        """
        Prédit le style d'un type de vêtement.
        
        Args:
            clothing_type: Le type de vêtement
            
        Returns:
            Le style prédit
        """
        styles = self.styles_map.get(clothing_type, "casual")
        if isinstance(styles, list):
            return styles[0]  # Retourne le premier style suggéré
        return styles
    
    async def analyze_clothing(self, file: Union[str, UploadFile]) -> Dict[str, Any]:
        """
        Analyse un vêtement et détecte ses caractéristiques.
        
        Args:
            file: Le fichier image à analyser
            
        Returns:
            Un dictionnaire contenant les caractéristiques détectées
        """
        if isinstance(file, str):
            # Si c'est un chemin de fichier
            image_path = Path(file)
            image = Image.open(image_path).convert('RGB')
        else:
            # Si c'est un UploadFile
            # Sauvegarde l'image
            image_url = await file_storage_service.save_image(file)
            image_path = Path("app/static/images") / image_url.split("/")[-1]
            image = Image.open(image_path).convert('RGB')
        
        # Prétraite et extrait les caractéristiques
        image_tensor = self.preprocess_image(image)
        features = self.extract_features(image_tensor)
        
        # Détecte la couleur dominante
        color_name, rgb_values = self.detect_dominant_color(image)
        
        # Prédit le type de vêtement
        clothing_type = self.predict_clothing_type(features)
        
        # Prédit la saison et le style
        season = self.predict_season(clothing_type)
        style = self.predict_style(clothing_type)
        
        # Retourne toutes les caractéristiques détectées
        return {
            "embeddings": features.tolist(),
            "color": color_name,
            "rgb_values": rgb_values,
            "predicted_type": clothing_type,
            "predicted_season": season,
            "predicted_style": style
        }

# Instance du service
image_analysis_service = ImageAnalysisService() 