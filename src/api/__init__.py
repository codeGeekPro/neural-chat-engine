"""
API Backend - Neural Chat Engine

API REST haute performance avec FastAPI :
- Endpoints de conversation en temps réel
- Gestion des sessions utilisateur
- Intégration avec Redis et PostgreSQL
- WebSocket pour communication temps réel
"""

from .main import app
from .api_types import *
from .websocket_manager import WebSocketManager
from .chat_endpoints import router as chat_router

__all__ = [
    "app",
    "WebSocketManager",
    "chat_router"
]