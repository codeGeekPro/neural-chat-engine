# API Neural Chat Engine

Ce dossier contient l'implémentation de l'API REST et WebSocket pour le Neural Chat Engine.

## Structure

- `main.py` : Point d'entrée principal de l'API FastAPI
- `chat_endpoints.py` : Endpoints pour les fonctionnalités de chat
- `api_types.py` : Définitions des types et modèles Pydantic
- `websocket_manager.py` : Gestionnaire des connexions WebSocket

## Utilisation

L'API expose plusieurs endpoints :

- `POST /chat` : Envoyer un message et recevoir une réponse
- `WS /ws/chat` : Connexion WebSocket pour le chat en temps réel
- `GET /health` : Vérifier l'état de l'API

## Configuration

La configuration de l'API se fait via des variables d'environnement ou le fichier `.env` :

- `API_HOST` : Hôte de l'API (défaut: "0.0.0.0")
- `API_PORT` : Port de l'API (défaut: 8000)
- `API_WORKERS` : Nombre de workers (défaut: 1)

## Développement

Pour lancer l'API en mode développement :

```bash
uvicorn src.api.main:app --reload
```