# Neural Chat Engine API

Une API FastAPI complète pour un moteur de chat neuronal avec capacités multimodales, apprentissage continu et recommandations contextuelles.

## 🚀 Fonctionnalités

- **Chat en temps réel** : Communication bidirectionnelle via WebSocket
- **Multimodal** : Support texte, images, audio et documents
- **Apprentissage continu** : Adaptation basée sur les interactions utilisateur
- **Recommandations** : Suggestions contextuelles intelligentes
- **Authentification** : Sessions utilisateur sécurisées
- **Limites de taux** : Protection contre les abus
- **Métriques** : Monitoring complet des performances
- **Export** : Conversation exportable (JSON, TXT)

## 📋 Prérequis

- Python 3.8+
- pip pour la gestion des dépendances

## 🛠️ Installation

1. **Cloner le repository** :
```bash
git clone <repository-url>
cd neural-chat-engine
```

2. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

3. **Créer la configuration** :
```bash
python start.py create-env
```

4. **Vérifier les dépendances** :
```bash
python start.py check-deps
```

## 🚀 Démarrage

### Démarrage rapide
```bash
python start.py start
```

L'API sera disponible sur `http://localhost:8000`

### Options de démarrage
```bash
python start.py start --host 0.0.0.0 --port 8080 --workers 4
```

### Documentation API
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc
- **OpenAPI** : http://localhost:8000/openapi.json

## 📚 API Endpoints

### Chat

#### Envoyer un message
```http
POST /chat/message
Content-Type: application/json

{
  "message": "Bonjour, comment allez-vous ?",
  "session_id": "optional-session-id",
  "message_type": "text",
  "stream": false
}
```

#### Récupérer l'historique
```http
GET /chat/history/{session_id}?limit=50&cursor=optional-cursor
```

#### Upload de fichier
```http
POST /chat/upload
Content-Type: multipart/form-data

file: <fichier>
session_id: <session-id>
description: <description-optionnelle>
```

#### Exporter une conversation
```http
GET /chat/export/{session_id}?format=json
```

#### Soumettre un feedback
```http
POST /chat/feedback
Content-Type: application/json

{
  "session_id": "session-id",
  "message_id": "message-id",
  "rating": 5,
  "feedback_type": "quality",
  "comment": "Excellent réponse !"
}
```

### Authentification

#### Connexion
```http
POST /auth/login
```

#### Déconnexion
```http
POST /auth/logout
Content-Type: application/json

{
  "session_id": "session-id"
}
```

### Monitoring

#### État de santé
```http
GET /health
```

#### Métriques
```http
GET /metrics
```

## 🌐 WebSocket

### Connexion au chat
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat/{session_id}');

// Écouter les messages
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Message reçu:', data);
};

// Envoyer un message
ws.send(JSON.stringify({
    type: 'message',
    content: 'Bonjour !',
    message_type: 'text'
}));
```

### Types de messages WebSocket

#### Message utilisateur
```json
{
  "type": "message",
  "content": "Votre message",
  "message_type": "text",
  "attachments": []
}
```

#### Message assistant
```json
{
  "type": "response",
  "content": "Réponse de l'assistant",
  "message_type": "text",
  "metadata": {
    "model": "neural-chat-engine-v1",
    "confidence": 0.95
  }
}
```

#### Indicateur de frappe
```json
{
  "type": "typing",
  "is_typing": true
}
```

#### Erreur
```json
{
  "type": "error",
  "error": "Description de l'erreur",
  "code": 500
}
```

## 🔧 Configuration

### Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `DEBUG` | `false` | Mode debug |
| `HOST` | `0.0.0.0` | Hôte du serveur |
| `PORT` | `8000` | Port du serveur |
| `WORKERS` | `1` | Nombre de workers |
| `SECRET_KEY` | Généré | Clé secrète pour les sessions |
| `JWT_SECRET` | Généré | Clé secrète JWT |
| `DATABASE_URL` | `sqlite:///./chat_engine.db` | URL de la base de données |
| `ALLOWED_ORIGINS` | `*` | Origines CORS autorisées |
| `RATE_LIMIT_REQUESTS` | `100` | Requêtes max par fenêtre |
| `RATE_LIMIT_WINDOW` | `60` | Fenêtre de limite de taux (secondes) |
| `WS_MAX_CONNECTIONS` | `1000` | Connexions WebSocket max |
| `UPLOAD_DIR` | `/tmp/uploads` | Répertoire d'upload |

## 🧪 Tests

### Exécuter tous les tests
```bash
python start.py test
```

### Tests avec couverture
```bash
pytest tests/ --cov=src --cov-report=html
```

## 🎨 Formatage du code

### Formatter automatiquement
```bash
python start.py format
```

### Vérifier le style
```bash
python start.py lint
```

## 📊 Architecture

```
src/
├── api/
│   ├── main.py              # Application FastAPI principale
│   ├── api_types.py         # Modèles Pydantic pour l'API
│   ├── websocket_manager.py # Gestionnaire WebSocket
│   └── chat_endpoints.py    # Endpoints de chat
├── core/
│   ├── chat_engine.py       # Moteur de chat principal
│   ├── multimodal.py        # Traitement multimodal
│   └── memory.py            # Système de mémoire
├── models/
│   └── recommendation.py    # Modèle de recommandations
└── utils/
    ├── config.py            # Configuration
    └── logging.py           # Configuration du logging
```

## 🔒 Sécurité

- **Authentification JWT** : Tokens sécurisés pour les sessions
- **Limites de taux** : Protection contre les abus
- **CORS** : Contrôle des origines autorisées
- **Validation** : Validation stricte des données avec Pydantic
- **Logging** : Audit complet des actions

## 📈 Monitoring

### Métriques disponibles
- Nombre total de tokens utilisés
- Nombre total de requêtes
- Sessions actives
- Temps de réponse moyen
- Taux d'erreur

### Health checks
- État de santé général
- Connexion à la base de données
- Connexions WebSocket actives

## 🚀 Déploiement

### Production
```bash
export DEBUG=false
export WORKERS=4
python start.py start --no-reload
```

### Avec Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "start.py", "start", "--host", "0.0.0.0"]
```

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📞 Support

Pour le support, ouvrez une issue sur GitHub ou contactez l'équipe de développement.

---

**Développé avec ❤️ par l'équipe Neural Chat Engine**