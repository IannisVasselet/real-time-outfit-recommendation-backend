# Real-Time Outfit Recommendation - Backend

## 🚀 Vue d'ensemble

Backend de l'application de recommandation d'outfits en temps réel. Cette API permet d'analyser des images de vêtements et de générer des recommandations d'outfits personnalisées.

## 🛠️ Technologies

- Python 3.8+
- FastAPI
- PyTorch
- PostgreSQL
- Alembic (migrations)

## 📋 Prérequis

- Python 3.8 ou supérieur
- PostgreSQL
- pip (gestionnaire de paquets Python)

## 🚀 Installation

1. Cloner le repository :
```bash
git clone https://github.com/IannisVasselet/real-time-outfit-recommendation-backend.git
cd real-time-outfit-recommendation-backend
```

2. Créer et activer l'environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Configurer les variables d'environnement :
```bash
cp .env.example .env
# Éditer .env avec vos configurations
```

5. Initialiser la base de données :
```bash
alembic upgrade head
```

## 🚀 Démarrage

```bash
uvicorn main:app --reload
```

L'API sera disponible à l'adresse : `http://localhost:8000`

## 📚 Documentation

- [Documentation complète](docs/README.md)
- [Guide de démarrage](docs/guides/getting-started.md)
- [Documentation API](docs/api/overview.md)
- [Guide de contribution](docs/CONTRIBUTING.md)

## 🔍 Tests

```bash
pytest
```

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🤝 Contribution

Les contributions sont les bienvenues ! Consultez notre [guide de contribution](docs/CONTRIBUTING.md) pour commencer. 