"""
Configuration de l'application.
"""
from typing import Any, Dict, Optional

from pydantic import PostgresDsn, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuration de l'application.
    """
    # Configuration de l'API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = True

    # Configuration de la base de données
    DATABASE_URL: str = "postgresql://postgres:#Timepowa533@localhost:5432/outfit_recommender"
    
    # Configuration de sécurité
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Configuration du modèle
    MODEL_PATH: str = "models/resnet50.pth"
    EMBEDDINGS_DIMENSION: int = 2048
    BATCH_SIZE: int = 32

    class Config:
        """Configuration de l'environnement."""
        env_file = ".env"
        case_sensitive = True


settings = Settings() 