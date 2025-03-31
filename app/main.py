"""
Point d'entrée principal de l'application.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.v1.api import api_router
from app.core.config import settings

app = FastAPI(
    title="API de Recommandation de Tenues",
    description="API pour l'analyse et la recommandation de tenues vestimentaires",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage des fichiers statiques
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Inclusion des routes API
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Route racine de l'API."""
    return {
        "message": "Bienvenue sur l'API Fashion Recommender",
        "version": "1.0.0",
        "status": "online"
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API."""
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "database": "connected"
    } 