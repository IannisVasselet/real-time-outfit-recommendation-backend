"""
Service d'analyse d'images utilisant ResNet50.
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from fastapi import UploadFile
from app.core.config import settings
from app.services.file_storage import file_storage_service

class ImageAnalysisService:
    """
    Service d'analyse d'images utilisant ResNet50.
    """
    def __init__(self):
        """
        Initialise le service d'analyse d'images.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        self.model.to(self.device)
        
        # Supprime la dernière couche de classification
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
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

# Instance du service
image_analysis_service = ImageAnalysisService() 