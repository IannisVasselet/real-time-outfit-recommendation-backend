"""
Modèles de classification pour l'analyse des vêtements.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from colorsys import rgb_to_hsv

class ClothingClassifier(nn.Module):
    """
    Modèle de classification fine-tuné pour les vêtements.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Charger ResNet50 pré-entraîné
        self.model = models.resnet50(pretrained=True)
        
        # Modifier la première couche pour accepter des images en niveaux de gris
        self.model.conv1 = nn.Conv2d(
            1,  # 1 canal pour les images en niveaux de gris
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Modifier la dernière couche pour notre nombre de classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Définir les transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Mapping des classes
        self.class_mapping = {
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def predict(self, image: Image.Image) -> Dict[str, float]:
        """
        Prédit le type de vêtement avec des scores de confiance.
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Dict contenant les prédictions et leurs scores
        """
        self.eval()
        with torch.no_grad():
            # Prétraiter l'image
            input_tensor = self.transform(image).unsqueeze(0)
            
            # Obtenir les prédictions
            outputs = self(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Convertir en dictionnaire
            predictions = {}
            for idx, prob in enumerate(probabilities[0]):
                predictions[self.class_mapping[idx]] = float(prob)
            
            return predictions

class ColorAnalyzer:
    """
    Analyseur de couleurs amélioré utilisant K-means et analyse HSV.
    """
    def __init__(self, n_clusters: int = 8):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Définition des plages de couleurs en HSV
        self.color_ranges = {
            "rouge": [(0, 0.04), (0.95, 1.0)],
            "orange": [(0.04, 0.08)],
            "jaune": [(0.08, 0.13)],
            "vert": [(0.13, 0.33)],
            "bleu": [(0.33, 0.5)],
            "violet": [(0.5, 0.7)],
            "rose": [(0.7, 0.83)],
            "noir": [(0, 1), (0, 0.1)],  # Basé sur la valeur
            "blanc": [(0, 1), (0.9, 1)],  # Basé sur la valeur
            "gris": [(0, 1), (0.1, 0.9)]  # Basé sur la valeur
        }
    
    def analyze_colors(self, image: Image.Image) -> Dict[str, List[Dict[str, any]]]:
        """
        Analyse les couleurs dominantes et secondaires dans une image.
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Dict contenant les couleurs dominantes et secondaires
        """
        # Redimensionner l'image pour accélérer le traitement
        small_image = image.resize((100, 100))
        np_image = np.array(small_image)
        
        # Remodeler pour KMeans
        pixels = np_image.reshape(-1, 3)
        
        # Appliquer KMeans
        self.kmeans.fit(pixels)
        
        # Obtenir les centres des clusters et leurs fréquences
        centers = self.kmeans.cluster_centers_
        counts = np.bincount(self.kmeans.labels_)
        
        # Trier par fréquence
        sorted_indices = np.argsort(counts)[::-1]
        
        # Analyser chaque couleur
        colors = []
        for idx in sorted_indices:
            rgb = centers[idx] / 255.0
            h, s, v = rgb_to_hsv(*rgb)
            
            # Déterminer le nom de la couleur
            color_name = self.get_color_name(h, s, v)
            
            colors.append({
                "name": color_name,
                "rgb": rgb.tolist(),
                "percentage": float(counts[idx] / len(pixels))
            })
        
        return {
            "dominant": colors[0],
            "secondary": colors[1:],
            "all_colors": colors
        }
    
    def get_color_name(self, h: float, s: float, v: float) -> str:
        """
        Convertit les valeurs HSV en nom de couleur.
        """
        # Vérifier d'abord les couleurs basées sur la valeur
        if v < 0.1:
            return "noir"
        if v > 0.9 and s < 0.1:
            return "blanc"
        if 0.1 <= v <= 0.9 and s < 0.1:
            return "gris"
        
        # Vérifier les couleurs basées sur la teinte
        for color_name, ranges in self.color_ranges.items():
            for h_range in ranges:
                if h_range[0] <= h <= h_range[1]:
                    return color_name
        
        return "inconnu"

class SeasonClassifier(nn.Module):
    """
    Modèle de classification pour les saisons.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # Charger ResNet50 pré-entraîné
        self.model = models.resnet50(pretrained=True)
        
        # Modifier la première couche pour accepter des images en niveaux de gris
        self.model.conv1 = nn.Conv2d(
            1,  # 1 canal pour les images en niveaux de gris
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Modifier la dernière couche
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Définir les transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Mapping des saisons
        self.season_mapping = {
            0: "printemps",
            1: "été",
            2: "automne",
            3: "hiver"
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def predict(self, image: Image.Image) -> Dict[str, float]:
        """
        Prédit les saisons appropriées avec des scores de confiance.
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Dict contenant les prédictions de saisons et leurs scores
        """
        self.eval()
        with torch.no_grad():
            # Prétraiter l'image
            input_tensor = self.transform(image).unsqueeze(0)
            
            # Obtenir les prédictions
            outputs = self(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Convertir en dictionnaire
            predictions = {}
            for idx, prob in enumerate(probabilities[0]):
                predictions[self.season_mapping[idx]] = float(prob)
            
            return predictions 