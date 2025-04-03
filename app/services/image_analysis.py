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
from app.core.config import settings
from app.services.file_storage import file_storage_service
from app.models.clothing_classifier import ClothingClassifier, ColorAnalyzer, SeasonClassifier
from colorthief import ColorThief
import io

class ImageAnalysisService:
    def __init__(self):
        """
        Initialise le service d'analyse d'images.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilisation du device: {self.device}")
        
        # Initialiser les modèles
        self.clothing_classifier = ClothingClassifier().to(self.device)
        self.color_analyzer = ColorAnalyzer()
        self.season_classifier = SeasonClassifier().to(self.device)
        
        # Charger les poids des modèles
        self._load_model_weights()
        
        # Définir les transformations d'image
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            )
        ])
    
    def _load_model_weights(self):
        """
        Charge les poids des modèles depuis les fichiers de sauvegarde.
        """
        # Chemins des fichiers de poids
        clothing_weights = Path("app/models/weights/clothing_classifier.pth")
        season_weights = Path("app/models/weights/season_classifier.pth")
        
        # Charger les poids si disponibles
        if clothing_weights.exists():
            print(f"Chargement des poids du classificateur de vêtements depuis {clothing_weights}")
            self.clothing_classifier.load_state_dict(torch.load(clothing_weights, map_location=self.device))
            self.clothing_classifier.eval()  # Mode évaluation
            print("Poids du classificateur de vêtements chargés avec succès")
        else:
            print(f"ATTENTION: Fichier de poids non trouvé: {clothing_weights}")
        
        if season_weights.exists():
            print(f"Chargement des poids du classificateur de saisons depuis {season_weights}")
            self.season_classifier.load_state_dict(torch.load(season_weights, map_location=self.device))
            self.season_classifier.eval()  # Mode évaluation
            print("Poids du classificateur de saisons chargés avec succès")
        else:
            print(f"ATTENTION: Fichier de poids non trouvé: {season_weights}")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Prétraite une image pour l'analyse.
        
        Args:
            image: Image PIL à prétraiter (déjà en niveaux de gris)
            
        Returns:
            Tensor prétraité
        """
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Extrait les caractéristiques d'une image.
        
        Args:
            image_tensor: Tensor d'image prétraité
            
        Returns:
            Caractéristiques extraites
        """
        with torch.no_grad():
            features = self.clothing_classifier.model.avgpool(
                self.clothing_classifier.model.layer4(
                    self.clothing_classifier.model.layer3(
                        self.clothing_classifier.model.layer2(
                            self.clothing_classifier.model.layer1(
                                self.clothing_classifier.model.maxpool(
                                    self.clothing_classifier.model.relu(
                                        self.clothing_classifier.model.bn1(
                                            self.clothing_classifier.model.conv1(image_tensor)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            return features.squeeze().cpu().numpy()
    
    def _get_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[Dict[str, Any]]:
        """
        Extrait les couleurs dominantes d'une image.
        
        Args:
            image: Image PIL
            num_colors: Nombre de couleurs à extraire
            
        Returns:
            List[Dict]: Liste des couleurs avec leur nom et pourcentage
        """
        # Convertir l'image PIL en bytes pour ColorThief
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Utiliser ColorThief pour extraire les couleurs
        color_thief = ColorThief(img_byte_arr)
        palette = color_thief.get_palette(color_count=num_colors, quality=1)
        
        # Calculer les pourcentages (approximatif)
        total_pixels = image.width * image.height
        colors = []
        
        for i, (r, g, b) in enumerate(palette):
            # Convertir RGB en HSV pour une meilleure identification des couleurs
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            
            # Déterminer le nom de la couleur
            color_name = self._get_color_name(h, s, v)
            
            # Calculer un pourcentage approximatif
            percentage = (num_colors - i) / num_colors * 100
            
            colors.append({
                "name": color_name,
                "rgb": [r, g, b],
                "percentage": percentage
            })
            
        return colors
    
    def _get_color_name(self, h: float, s: float, v: float) -> str:
        """
        Détermine le nom d'une couleur à partir de ses valeurs HSV.
        
        Args:
            h: Teinte (0-1)
            s: Saturation (0-1)
            v: Valeur/Luminosité (0-1)
            
        Returns:
            str: Nom de la couleur
        """
        if v < 0.2:
            return "noir"
        elif v > 0.8 and s < 0.2:
            return "blanc"
        elif s < 0.2:
            return "gris"
            
        # Convertir h en degrés (0-360)
        h_deg = h * 360
        
        if s < 0.3:
            return "gris"
        elif h_deg < 30 or h_deg >= 330:
            return "rouge"
        elif h_deg < 60:
            return "orange"
        elif h_deg < 90:
            return "jaune"
        elif h_deg < 150:
            return "vert"
        elif h_deg < 210:
            return "cyan"
        elif h_deg < 270:
            return "bleu"
        elif h_deg < 330:
            return "violet"
        else:
            return "inconnu"
    
    async def analyze_clothing(self, file: Union[str, UploadFile]) -> Dict[str, Any]:
        """
        Analyse un vêtement et détecte ses caractéristiques.
        
        Args:
            file: Le fichier image à analyser
            
        Returns:
            Un dictionnaire contenant les caractéristiques détectées
        """
        # Charger l'image
        if isinstance(file, str):
            image_path = Path(file)
            image = Image.open(image_path)
        else:
            image_url = await file_storage_service.save_image(file)
            image_path = Path("app/static/images") / image_url.split("/")[-1]
            image = Image.open(image_path)
        
        # Convertir en RGB pour l'analyse des couleurs
        rgb_image = image.convert('RGB')
        
        # Convertir en niveaux de gris pour le modèle
        gray_image = image.convert('L')
        
        # Prétraiter l'image pour le modèle
        image_tensor = self.preprocess_image(gray_image)
        
        # Extraire les caractéristiques
        features = self.extract_features(image_tensor)
        
        # Obtenir les prédictions des différents modèles
        clothing_predictions = self.clothing_classifier.predict(gray_image)
        color_analysis = self.color_analyzer.analyze_colors(rgb_image)
        season_predictions = self.season_classifier.predict(gray_image)
        
        # Trouver le type de vêtement le plus probable
        predicted_type = max(clothing_predictions.items(), key=lambda x: x[1])[0]
        
        # Trouver les saisons les plus probables (score > 0.3)
        predicted_seasons = [
            season for season, score in season_predictions.items()
            if score > 0.3
        ]
        
        # Si aucune saison n'a un score suffisant, utiliser "toutes saisons"
        if not predicted_seasons:
            predicted_seasons = ["toutes saisons"]
        
        # Détecter les couleurs
        colors = self._get_dominant_colors(image)
        
        # Retourner toutes les caractéristiques détectées
        return {
            "embeddings": features.tolist(),
            "color_analysis": {
                "dominant": colors[0],
                "palette": colors
            },
            "predicted_type": predicted_type,
            "type_confidence": clothing_predictions[predicted_type],
            "predicted_seasons": predicted_seasons,
            "season_confidences": season_predictions,
            "all_predictions": {
                "clothing_types": clothing_predictions,
                "seasons": season_predictions
            }
        }

# Instance du service
image_analysis_service = ImageAnalysisService() 