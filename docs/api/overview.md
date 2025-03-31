# Documentation API

## 🚀 Vue d'ensemble

L'API de recommandation d'outfits est construite avec FastAPI et fournit des endpoints RESTful pour :
- L'analyse d'images de vêtements
- La génération de recommandations d'outfits
- La gestion des vêtements et des outfits

## 🔑 Authentification

L'API utilise JWT (JSON Web Tokens) pour l'authentification. Les tokens doivent être inclus dans l'en-tête `Authorization` :

```
Authorization: Bearer <votre_token>
```

## 📝 Endpoints

### Vêtements

#### GET /api/v1/clothes
Récupère la liste des vêtements.

**Paramètres de requête :**
- `category` (optionnel) : Filtrer par catégorie
- `limit` (optionnel) : Nombre maximum d'éléments (défaut: 10)
- `offset` (optionnel) : Nombre d'éléments à sauter (défaut: 0)

**Réponse :**
```json
{
  "items": [
    {
      "id": "string",
      "name": "string",
      "category": "string",
      "image_url": "string",
      "created_at": "datetime",
      "updated_at": "datetime"
    }
  ],
  "total": "integer",
  "limit": "integer",
  "offset": "integer"
}
```

#### POST /api/v1/clothes
Ajoute un nouveau vêtement.

**Corps de la requête :**
```json
{
  "name": "string",
  "category": "string",
  "image": "file"
}
```

### Outfits

#### GET /api/v1/outfits
Récupère la liste des outfits recommandés.

**Paramètres de requête :**
- `limit` (optionnel) : Nombre maximum d'éléments (défaut: 10)
- `offset` (optionnel) : Nombre d'éléments à sauter (défaut: 0)

**Réponse :**
```json
{
  "items": [
    {
      "id": "string",
      "name": "string",
      "clothes": [
        {
          "id": "string",
          "name": "string",
          "category": "string",
          "image_url": "string"
        }
      ],
      "created_at": "datetime"
    }
  ],
  "total": "integer",
  "limit": "integer",
  "offset": "integer"
}
```

### Analyse d'Images

#### POST /api/v1/analyze
Analyse une image de vêtement.

**Corps de la requête :**
```json
{
  "image": "file"
}
```

**Réponse :**
```json
{
  "category": "string",
  "attributes": {
    "color": "string",
    "style": "string",
    "pattern": "string"
  },
  "confidence": "float"
}
```

## 🔍 Codes d'Erreur

- `400 Bad Request` : Requête invalide
- `401 Unauthorized` : Non authentifié
- `403 Forbidden` : Non autorisé
- `404 Not Found` : Ressource non trouvée
- `422 Validation Error` : Données invalides
- `500 Internal Server Error` : Erreur serveur

## 📚 Documentation Interactive

Une documentation interactive Swagger UI est disponible à :
```
http://localhost:8000/docs
```

## 🔒 Sécurité

- Tous les endpoints nécessitent une authentification
- Les tokens JWT expirent après 24 heures
- Les mots de passe sont hachés avec bcrypt
- Les requêtes sont limitées à 100 par minute par IP

## 🚀 Versions

- Version actuelle : v1
- Base URL : `/api/v1` 