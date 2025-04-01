"""
Script d'entraînement des modèles de classification.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Tuple
import pandas as pd

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.models.clothing_classifier import ClothingClassifier, SeasonClassifier

class FashionMNISTDataset(Dataset):
    """
    Dataset personnalisé pour Fashion-MNIST.
    """
    def __init__(
        self,
        root_dir: str,
        transform: transforms.Compose,
        split: str = "train"
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        
        # Charger les annotations
        self.annotations = pd.read_csv(
            self.root_dir / f"{split}_annotations.csv"
        )
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.annotations.iloc[idx]
        img_path = self.root_dir / self.split / row["image_name"]
        
        # Charger et prétraiter l'image
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        
        return image, row["label"]

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device
) -> List[Dict[str, float]]:
    """
    Entraîne un modèle de classification.
    """
    history = []
    
    for epoch in range(num_epochs):
        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Phase de validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Enregistrer les métriques
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_acc
        })
        
        print(f"Époque {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 60)
    
    return history

def main():
    """
    Fonction principale pour l'entraînement des modèles.
    """
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    
    # Paramètres d'entraînement
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # Créer les dossiers pour les poids
    weights_dir = Path("app/models/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Définir les transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])
    
    # Chemin vers le dataset
    dataset_dir = "data/fashion_mnist"
    
    # Entraîner le classificateur de vêtements
    print("\nEntraînement du classificateur de vêtements...")
    clothing_model = ClothingClassifier().to(device)
    
    # Créer les datasets
    train_dataset = FashionMNISTDataset(
        dataset_dir,
        transform,
        split="train"
    )
    val_dataset = FashionMNISTDataset(
        dataset_dir,
        transform,
        split="val"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clothing_model.parameters(), lr=learning_rate)
    
    clothing_history = train_model(
        clothing_model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs,
        device
    )
    
    # Sauvegarder le modèle de vêtements
    torch.save(
        clothing_model.state_dict(),
        weights_dir / "clothing_classifier.pth"
    )
    
    print("\nEntraînement terminé !")
    print(f"Les modèles ont été sauvegardés dans {weights_dir}")

if __name__ == "__main__":
    main() 