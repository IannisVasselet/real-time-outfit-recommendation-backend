"""
Module contenant les modèles de base pour SQLAlchemy.
"""
from datetime import datetime
from typing import Any

from sqlalchemy.ext.declarative import as_declarative, declared_attr


@as_declarative()
class Base:
    """
    Classe de base pour tous les modèles SQLAlchemy.
    """
    id: Any
    __name__: str

    # Génère automatiquement le nom de la table
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    # Champs communs à tous les modèles
    created_at = datetime.utcnow()
    updated_at = datetime.utcnow() 