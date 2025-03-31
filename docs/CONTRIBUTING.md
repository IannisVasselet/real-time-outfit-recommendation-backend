# Guide de Contribution

## 🎯 Introduction

Nous sommes ravis que vous souhaitiez contribuer à notre projet ! Ce guide vous aidera à comprendre comment contribuer de manière efficace.

## 📋 Table des matières

- [Code de Conduite](#code-de-conduite)
- [Comment Contribuer](#comment-contribuer)
- [Processus de Développement](#processus-de-développement)
- [Standards de Code](#standards-de-code)
- [Tests](#tests)
- [Documentation](#documentation)

## 🤝 Code de Conduite

Ce projet et tous ceux qui y participent sont régis par notre [Code de Conduite](CODE_OF_CONDUCT.md). En participant, vous êtes tenu de respecter ce code.

## 💡 Comment Contribuer

### 1. Fork du Projet

1. Fork le repository sur GitHub
2. Clone votre fork localement
3. Configurez le remote upstream :
```bash
git remote add upstream https://github.com/IannisVasselet/real-time-outfit-recommendation-backend.git
```

### 2. Création d'une Branche

```bash
git checkout develop
git pull upstream develop
git checkout -b feature/nom-de-la-feature
# ou
git checkout -b fix/nom-du-bug
```

### 3. Développement

- Suivez les [standards de code](#standards-de-code)
- Écrivez des tests pour les nouvelles fonctionnalités
- Mettez à jour la documentation si nécessaire

### 4. Commit

```bash
git add .
git commit -m "type: description concise"
```

Types de commit :
- `feat`: Nouvelle fonctionnalité
- `fix`: Correction de bug
- `docs`: Documentation
- `style`: Formatage, point-virgules, etc.
- `refactor`: Refactorisation du code
- `test`: Ajout ou modification de tests
- `chore`: Maintenance

### 5. Push

```bash
git push origin feature/nom-de-la-feature
```

### 6. Pull Request

1. Créez une Pull Request sur GitHub
2. Remplissez le template de PR
3. Attendez la revue de code

## 🔄 Processus de Développement

1. **Planification**
   - Créez une issue pour discuter de la fonctionnalité
   - Obtenez l'approbation de l'équipe

2. **Développement**
   - Suivez les standards de code
   - Écrivez des tests
   - Documentez vos changements

3. **Revue**
   - Répondez aux commentaires
   - Faites les modifications demandées
   - Assurez-vous que tous les tests passent

4. **Merge**
   - Une fois approuvé, la PR sera mergée dans `develop`
   - Les releases seront mergées dans `main`

## 📝 Standards de Code

### Python

- Suivez PEP 8
- Utilisez des docstrings pour les fonctions et classes
- Nommez les variables de manière descriptive
- Limitez la longueur des lignes à 88 caractères (black)

### Tests

- Écrivez des tests unitaires pour chaque fonctionnalité
- Maintenez une couverture de tests > 80%
- Utilisez pytest pour les tests

### Documentation

- Documentez toutes les nouvelles fonctionnalités
- Mettez à jour la documentation existante si nécessaire
- Utilisez un langage clair et concis

## 🔍 Tests

```bash
# Lancer tous les tests
pytest

# Lancer les tests avec couverture
pytest --cov=app tests/

# Lancer les tests de style
flake8
black --check .
```

## 📚 Documentation

- Documentez les nouvelles fonctionnalités dans `/docs`
- Mettez à jour l'API documentation si nécessaire
- Ajoutez des exemples d'utilisation

## ❓ Questions ?

Si vous avez des questions, n'hésitez pas à :
1. Ouvrir une issue
2. Rejoindre notre canal de discussion
3. Contacter les mainteneurs 