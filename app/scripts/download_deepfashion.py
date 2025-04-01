"""
Script de téléchargement et préparation du dataset Fashion-MNIST.
"""
import os
import sys
import requests
import zipfile
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import shutil
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def download_file(url: str, destination: Path) -> None:
    """
    Télécharge un fichier avec une barre de progression.
    
    Args:
        url: URL du fichier à télécharger
        destination: Chemin de destination
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convertit un tenseur PyTorch en image PIL.
    
    Args:
        tensor: Tenseur PyTorch de forme (C, H, W) ou (H, W)
        
    Returns:
        Image PIL
    """
    # Convertir en numpy et dénormaliser
    img = tensor.numpy()
    img = (img * 0.5 + 0.5) * 255  # Dénormaliser de [-1,1] à [0,255]
    
    # Gérer les différents formats d'entrée
    if img.ndim == 2:
        # Image en niveaux de gris
        img = img.astype(np.uint8)
        return Image.fromarray(img, mode='L')
    elif img.ndim == 3:
        # Image RGB
        img = img.transpose(1, 2, 0)  # Convertir de (C,H,W) à (H,W,C)
        img = img.astype(np.uint8)
        return Image.fromarray(img)
    else:
        raise ValueError(f"Format de tenseur non supporté: {img.ndim} dimensions")

def prepare_fashion_mnist_dataset():
    """
    Télécharge et prépare le dataset Fashion-MNIST.
    """
    # Configuration
    base_dir = Path("data/fashion_mnist")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer les dossiers pour les images
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Définir les transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Télécharger le dataset
    print("Téléchargement du dataset Fashion-MNIST...")
    trainset = torchvision.datasets.FashionMNIST(
        root=str(base_dir),
        train=True,
        download=True,
        transform=transform
    )
    
    # Diviser en train/val (80/20)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        trainset, [train_size, val_size]
    )
    
    # Créer les annotations
    print("\nCréation des annotations...")
    
    # Mapping des classes
    class_mapping = {
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
    
    # Créer les annotations pour le train
    train_annotations = []
    print("Sauvegarde des images d'entraînement...")
    for i in tqdm(range(len(train_dataset))):
        img, label = train_dataset[i]
        # Convertir le tenseur en numpy et dénormaliser
        img_np = img.squeeze().numpy()  # Supprimer la dimension inutile
        img_np = (img_np * 0.5 + 0.5) * 255  # Dénormaliser de [-1,1] à [0,255]
        img_np = img_np.astype(np.uint8)
        
        # Créer l'image PIL
        img_pil = Image.fromarray(img_np, mode='L')
        img_path = train_dir / f"train_{i}.png"
        img_pil.save(img_path)
        
        train_annotations.append({
            "image_name": f"train_{i}.png",
            "category": class_mapping[label],
            "label": label
        })
    
    # Créer les annotations pour la validation
    val_annotations = []
    print("Sauvegarde des images de validation...")
    for i in tqdm(range(len(val_dataset))):
        img, label = val_dataset[i]
        # Convertir le tenseur en numpy et dénormaliser
        img_np = img.squeeze().numpy()  # Supprimer la dimension inutile
        img_np = (img_np * 0.5 + 0.5) * 255  # Dénormaliser de [-1,1] à [0,255]
        img_np = img_np.astype(np.uint8)
        
        # Créer l'image PIL
        img_pil = Image.fromarray(img_np, mode='L')
        img_path = val_dir / f"val_{i}.png"
        img_pil.save(img_path)
        
        val_annotations.append({
            "image_name": f"val_{i}.png",
            "category": class_mapping[label],
            "label": label
        })
    
    # Sauvegarder les annotations
    pd.DataFrame(train_annotations).to_csv(
        base_dir / "train_annotations.csv",
        index=False
    )
    pd.DataFrame(val_annotations).to_csv(
        base_dir / "val_annotations.csv",
        index=False
    )
    
    print("\nPréparation du dataset terminée !")
    print(f"Dataset disponible dans {base_dir}")
    print(f"Nombre d'images d'entraînement : {len(train_dataset)}")
    print(f"Nombre d'images de validation : {len(val_dataset)}")

if __name__ == "__main__":
    prepare_fashion_mnist_dataset() 