"""
Point d'entrée principal de l'application FastAPI.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Recommandateur de Collections de Mode",
    description="API pour le système de recommandation de tenues de mode en temps réel",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Ajout de Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """
    Route racine de l'API.
    
    Returns:
        dict: Message de bienvenue
    """
    return {
        "message": "Bienvenue sur l'API du Recommandateur de Collections de Mode",
        "status": "online"
    }

@app.get("/health")
async def health_check():
    """
    Route de vérification de la santé de l'API.
    
    Returns:
        dict: État de l'API
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)