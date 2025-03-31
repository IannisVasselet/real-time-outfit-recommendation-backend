# Guide de Démarrage

## 🚀 Introduction

Ce guide vous aidera à démarrer rapidement avec l'API de recommandation d'outfits. Nous allons couvrir l'installation, la configuration et les premiers pas avec l'API.

## 📋 Table des matières

- [Installation](#installation)
- [Configuration](#configuration)
- [Premiers pas](#premiers-pas)
- [Exemples d'utilisation](#exemples-dutilisation)
- [Dépannage](#dépannage)

## 🛠️ Installation

### 1. Prérequis

Assurez-vous d'avoir installé :
- Python 3.8 ou supérieur
- PostgreSQL
- pip (gestionnaire de paquets Python)

### 2. Cloner le Repository

```bash
git clone https://github.com/IannisVasselet/real-time-outfit-recommendation-backend.git
cd real-time-outfit-recommendation-backend
```

### 3. Environnement Virtuel

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Sur Windows :
.\venv\Scripts\activate
# Sur macOS/Linux :
source venv/bin/activate
```

### 4. Dépendances

```bash
pip install -r requirements.txt
```

### 5. Configuration de la Base de Données

```bash
# Créer la base de données PostgreSQL
createdb outfit_recommendation

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos configurations
```

### 6. Migrations

```bash
# Initialiser Alembic
alembic init alembic

# Créer la première migration
alembic revision --autogenerate -m "initial migration"

# Appliquer les migrations
alembic upgrade head
```

## ⚙️ Configuration

### Variables d'Environnement

Créez un fichier `.env` avec les variables suivantes :

```env
DATABASE_URL=postgresql://user:password@localhost:5432/outfit_recommendation
SECRET_KEY=votre_clé_secrète
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

### Configuration de l'API

L'API est configurée par défaut pour :
- Port : 8000
- Host : 0.0.0.0
- Mode debug : activé en développement

## 🚀 Premiers pas

### 1. Démarrer le Serveur

```bash
uvicorn main:app --reload
```

### 2. Vérifier l'Installation

Visitez `http://localhost:8000/docs` pour accéder à la documentation Swagger UI.

### 3. Créer un Compte

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "password123"}'
```

### 4. Obtenir un Token

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "password123"}'
```

## 💡 Exemples d'Utilisation

### 1. Analyser une Image

```python
import requests

url = "http://localhost:8000/api/v1/analyze"
files = {"image": open("chemin/vers/image.jpg", "rb")}
headers = {"Authorization": "Bearer votre_token"}
response = requests.post(url, files=files, headers=headers)
print(response.json())
```

### 2. Obtenir des Recommandations

```python
import requests

url = "http://localhost:8000/api/v1/outfits"
headers = {"Authorization": "Bearer votre_token"}
response = requests.get(url, headers=headers)
print(response.json())
```

## 🔍 Dépannage

### Problèmes Courants

1. **Erreur de connexion à la base de données**
   - Vérifiez les variables d'environnement
   - Assurez-vous que PostgreSQL est en cours d'exécution

2. **Erreur d'authentification**
   - Vérifiez que le token est valide
   - Assurez-vous que le token est inclus dans l'en-tête

3. **Erreur de migration**
   - Supprimez le dossier `alembic/versions`
   - Réinitialisez les migrations

### Logs

Les logs sont disponibles dans :
- Console en mode développement
- Fichier `logs/app.log` en production

## 📚 Ressources Supplémentaires

- [Documentation API complète](../api/overview.md)
- [Guide avancé](advanced-usage.md)
- [FAQ](../faq.md) 