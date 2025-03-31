# Documentation du Backend

## 📚 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Installation](#installation)
- [Configuration](#configuration)
- [API](#api)
- [Architecture](#architecture)
- [Développement](#développement)
- [Tests](#tests)
- [Déploiement](#déploiement)

## 🚀 Vue d'ensemble

Le backend de l'application de recommandation d'outfits est une API RESTful construite avec FastAPI. Il fournit des fonctionnalités pour :
- L'analyse d'images de vêtements
- La génération de recommandations d'outfits
- La gestion des vêtements et des outfits

## 📋 Installation

Consultez notre [guide de démarrage](guides/getting-started.md) pour une installation détaillée.

## ⚙️ Configuration

### Variables d'Environnement

Les variables d'environnement requises sont :
- `DATABASE_URL` : URL de connexion à la base de données
- `SECRET_KEY` : Clé secrète pour JWT
- `ALGORITHM` : Algorithme de hachage (HS256)
- `ACCESS_TOKEN_EXPIRE_MINUTES` : Durée de validité des tokens

### Base de Données

La base de données utilise PostgreSQL avec SQLAlchemy comme ORM. Les migrations sont gérées avec Alembic.

## 🔌 API

Consultez la [documentation API](api/overview.md) pour plus de détails sur les endpoints disponibles.

## 🏗️ Architecture

### Structure du Projet

```
backend/
├── app/
│   ├── api/           # Routes et endpoints
│   ├── core/          # Configuration et utilitaires
│   ├── models/        # Modèles de données
│   ├── schemas/       # Schémas Pydantic
│   ├── services/      # Logique métier
│   └── static/        # Fichiers statiques
├── tests/             # Tests unitaires et d'intégration
├── alembic/           # Migrations de base de données
└── docs/              # Documentation
```

### Technologies Utilisées

- **Framework** : FastAPI
- **Base de données** : PostgreSQL
- **ORM** : SQLAlchemy
- **Migrations** : Alembic
- **Traitement d'images** : PyTorch
- **Tests** : pytest
- **Documentation** : Swagger UI

## 💻 Développement

### Standards de Code

- Suivez PEP 8
- Utilisez des docstrings
- Écrivez des tests unitaires
- Documentez les changements

### Workflow Git

1. Créez une branche feature/fix
2. Développez et testez
3. Créez une Pull Request
4. Attendez la revue de code
5. Merge dans develop

## 🧪 Tests

### Types de Tests

- Tests unitaires
- Tests d'intégration
- Tests de performance

### Exécution des Tests

```bash
# Tous les tests
pytest

# Tests avec couverture
pytest --cov=app tests/

# Tests de style
flake8
black --check .
```

## 🚀 Déploiement

### Prérequis

- Python 3.8+
- PostgreSQL
- Redis (optionnel, pour le cache)

### Étapes de Déploiement

1. Configurer l'environnement
2. Installer les dépendances
3. Appliquer les migrations
4. Démarrer le serveur

### Production

- Utiliser Gunicorn comme serveur WSGI
- Configurer un reverse proxy (Nginx)
- Activer HTTPS
- Configurer les logs

## 📚 Ressources

- [Guide de Contribution](CONTRIBUTING.md)
- [Code de Conduite](CODE_OF_CONDUCT.md)
- [FAQ](faq.md)
- [Changelog](changelog.md) 