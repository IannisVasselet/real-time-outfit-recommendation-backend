"""
Dépendances de l'application.
"""
from typing import Generator
from sqlalchemy.orm import Session

from app.core.database import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Dépendance pour obtenir une session de base de données.
    
    Yields:
        Session: Session de base de données SQLAlchemy.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 