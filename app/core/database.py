"""
Configuration de la base de données.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import settings

# Création du moteur SQLAlchemy
engine = create_engine(str(settings.DATABASE_URL))

# Création de la session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    Générateur de session de base de données.
    
    Yields:
        Session: Session de base de données.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 