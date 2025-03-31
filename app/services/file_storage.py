"""
Service de gestion du stockage des fichiers.
"""
import os
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
from app.core.config import settings

class FileStorageService:
    """
    Service de gestion du stockage des fichiers.
    """
    def __init__(self):
        """
        Initialise le service de stockage des fichiers.
        """
        self.upload_dir = Path("app/static/images")
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def save_image(self, file: UploadFile, filename: Optional[str] = None) -> str:
        """
        Sauvegarde une image et retourne son URL.

        Args:
            file: Le fichier à sauvegarder
            filename: Le nom du fichier (optionnel)

        Returns:
            str: L'URL de l'image
        """
        if not filename:
            filename = file.filename

        # Assurez-vous que le nom du fichier est sécurisé
        safe_filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_', '.'))
        
        file_path = self.upload_dir / safe_filename
        
        # Sauvegardez le fichier
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Retournez l'URL relative
        return f"/static/images/{safe_filename}"

    def delete_image(self, image_url: str) -> bool:
        """
        Supprime une image.

        Args:
            image_url: L'URL de l'image à supprimer

        Returns:
            bool: True si l'image a été supprimée, False sinon
        """
        try:
            # Extrait le nom du fichier de l'URL
            filename = image_url.split("/")[-1]
            file_path = self.upload_dir / filename
            
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception:
            return False

# Instance du service
file_storage_service = FileStorageService() 