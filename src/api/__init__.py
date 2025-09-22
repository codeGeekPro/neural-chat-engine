"""
API Backend - Neural Chat Engine

API REST haute performance avec FastAPI :
- Endpoints de conversation en temps réel
- Gestion des sessions utilisateur
- Intégration avec Redis et PostgreSQL
- WebSocket pour communication temps réel
"""

from .main import app
from .chat_endpoint import ChatEndpoint
from .session_manager import SessionManager
from .websocket_handler import WebSocketHandler

__all__ = [
    "app",
    "ChatEndpoint", 
    "SessionManager",
    "WebSocketHandler"
]