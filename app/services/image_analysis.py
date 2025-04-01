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

class ImageAnalysisService:
    def __init__(self):
        """
        Initialise le service d'analyse d'images.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialiser les modèles
        self.clothing_classifier = ClothingClassifier().to(self.device)
        self.color_analyzer = ColorAnalyzer()
        self.season_classifier = SeasonClassifier().to(self.device)
        
        # Charger les poids des modèles si disponibles
        self._load_model_weights()
        
        # Définir les transformations d'image
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
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
            self.clothing_classifier.load_state_dict(torch.load(clothing_weights))
            print("Poids du classificateur de vêtements chargés")
        
        if season_weights.exists():
            self.season_classifier.load_state_dict(torch.load(season_weights))
            print("Poids du classificateur de saisons chargés")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Prétraite une image pour l'analyse.
        
        Args:
            image: Image PIL à prétraiter
            
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
            image = Image.open(image_path).convert('RGB')
        else:
            image_url = await file_storage_service.save_image(file)
            image_path = Path("app/static/images") / image_url.split("/")[-1]
            image = Image.open(image_path).convert('RGB')
        
        # Prétraiter l'image
        image_tensor = self.preprocess_image(image)
        
        # Extraire les caractéristiques
        features = self.extract_features(image_tensor)
        
        # Obtenir les prédictions des différents modèles
        clothing_predictions = self.clothing_classifier.predict(image)
        color_analysis = self.color_analyzer.analyze_colors(image)
        season_predictions = self.season_classifier.predict(image)
        
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
        
        # Retourner toutes les caractéristiques détectées
        return {
            "embeddings": features.tolist(),
            "color_analysis": color_analysis,
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