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
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))
print(f"Chemin du projet : {current_dir}")

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
        print(f"Initialisation du dataset {split} avec le répertoire : {self.root_dir}")
        print(f"Contenu du répertoire : {list(self.root_dir.iterdir())}")
        
        self.transform = transform
        self.split = split
        
        # Vérifier l'existence du fichier d'annotations
        annotations_path = self.root_dir / f"{split}_annotations.csv"
        print(f"Recherche du fichier d'annotations : {annotations_path}")
        
        if not annotations_path.exists():
            raise FileNotFoundError(
                f"Le fichier d'annotations {annotations_path} n'existe pas. "
                f"Vérifiez que le chemin est correct et que le fichier a été copié dans Colab."
            )
        
        # Charger les annotations
        print(f"Chargement des annotations depuis {annotations_path}")
        self.annotations = pd.read_csv(annotations_path)
        print(f"Nombre d'annotations chargées : {len(self.annotations)}")
    
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
    total_batches = len(train_loader)
    
    for epoch in range(num_epochs):
        print(f"\n{'='*20} Époque {epoch + 1}/{num_epochs} {'='*20}")
        print(f"Phase d'entraînement...")
        
        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
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
            
            # Afficher la progression
            if (batch_idx + 1) % 50 == 0:  # Afficher tous les 50 batches
                print(f"Batch [{batch_idx + 1}/{total_batches}] "
                      f"Loss: {loss.item():.4f} "
                      f"Acc: {100.*train_correct/train_total:.2f}%")
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        print(f"\nPhase de validation...")
        # Phase de validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Afficher la progression
                if (batch_idx + 1) % 20 == 0:  # Afficher tous les 20 batches
                    print(f"Batch [{batch_idx + 1}/{len(val_loader)}] "
                          f"Loss: {loss.item():.4f} "
                          f"Acc: {100.*val_correct/val_total:.2f}%")
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Enregistrer les métriques
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        })
        
        print(f"\nRésumé de l'époque {epoch + 1}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 60)
    
    return history

def main():
    """
    Fonction principale pour l'entraînement des modèles.
    """
    print("\nInitialisation de l'entraînement...")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Paramètres d'entraînement
    batch_size = 64  # Augmenté pour GPU
    num_epochs = 2  # Réduit à 2 époques
    learning_rate = 0.001
    
    print(f"\nParamètres d'entraînement:")
    print(f"- Batch size: {batch_size}")
    print(f"- Nombre d'époques: {num_epochs}")
    print(f"- Learning rate: {learning_rate}")
    
    # Créer les dossiers pour les poids
    weights_dir = Path("app/models/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDossier de sauvegarde: {weights_dir}")
    
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
    print(f"\nChargement du dataset depuis: {dataset_dir}")
    
    # Entraîner le classificateur de vêtements
    print("\nInitialisation du modèle...")
    clothing_model = ClothingClassifier().to(device)
    
    # Créer les datasets
    print("\nPréparation des datasets...")
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
    
    print(f"Nombre d'images d'entraînement: {len(train_dataset)}")
    print(f"Nombre d'images de validation: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Réduit pour éviter les problèmes
        pin_memory=True  # Optimisation pour GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # Réduit pour éviter les problèmes
        pin_memory=True  # Optimisation pour GPU
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clothing_model.parameters(), lr=learning_rate)
    
    print("\nDémarrage de l'entraînement...")
    print("=" * 60)
    
    # Sauvegarde intermédiaire
    best_val_acc = 0.0
    
    try:
        clothing_history = train_model(
            clothing_model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs,
            device
        )
        
        # Sauvegarder le modèle final
        print("\nSauvegarde du modèle final...")
        torch.save(
            clothing_model.state_dict(),
            weights_dir / "clothing_classifier.pth"
        )
        
        print("\nEntraînement terminé avec succès !")
        print(f"Les modèles ont été sauvegardés dans {weights_dir}")
        
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur.")
        print("Sauvegarde du modèle...")
        torch.save(
            clothing_model.state_dict(),
            weights_dir / "clothing_classifier_interrupted.pth"
        )
        print(f"Modèle sauvegardé dans {weights_dir}/clothing_classifier_interrupted.pth")
    except Exception as e:
        print(f"\nErreur pendant l'entraînement : {str(e)}")
        print("Sauvegarde du modèle...")
        torch.save(
            clothing_model.state_dict(),
            weights_dir / "clothing_classifier_error.pth"
        )
        print(f"Modèle sauvegardé dans {weights_dir}/clothing_classifier_error.pth")

if __name__ == "__main__":
    main() 