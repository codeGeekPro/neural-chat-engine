"""Gestionnaire WebSocket pour chat temps réel."""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from collections import defaultdict
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from .api_types import WebSocketMessage, StreamingChunk, ChatMessage, MessageRole


class WebSocketManager:
    """Gestionnaire de connexions WebSocket pour chat temps réel."""

    def __init__(self, max_connections_per_session: int = 5, heartbeat_interval: int = 30):
        """
        Initialise le gestionnaire WebSocket.

        Args:
            max_connections_per_session: Nombre maximum de connexions par session
            heartbeat_interval: Intervalle de heartbeat en secondes
        """
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.session_managers: Dict[str, 'SessionManager'] = {}
        self.max_connections_per_session = max_connections_per_session
        self.heartbeat_interval = heartbeat_interval

        # Statistiques
        self.stats = {
            'total_connections': 0,
            'active_sessions': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0
        }

        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Démarrer le nettoyage périodique
        asyncio.create_task(self._periodic_cleanup())

    def _setup_logging(self):
        """Configure le système de logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """
        Connecte un nouveau WebSocket à une session.

        Args:
            websocket: Instance WebSocket
            session_id: ID de la session

        Returns:
            bool: True si la connexion a réussi
        """
        try:
            # Vérifier les limites de connexion
            if len(self.active_connections[session_id]) >= self.max_connections_per_session:
                await websocket.close(code=1008, reason="Trop de connexions pour cette session")
                return False

            # Accepter la connexion
            await websocket.accept()

            # Ajouter à la liste des connexions actives
            self.active_connections[session_id].add(websocket)
            self.stats['total_connections'] += 1

            # Créer le gestionnaire de session si nécessaire
            if session_id not in self.session_managers:
                self.session_managers[session_id] = SessionManager(session_id)
                self.stats['active_sessions'] += 1

            # Démarrer le heartbeat pour cette connexion
            asyncio.create_task(self._heartbeat_worker(websocket, session_id))

            self.logger.info(f"Nouvelle connexion WebSocket pour session {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"Erreur lors de la connexion WebSocket: {e}")
            self.stats['errors'] += 1
            return False

    async def disconnect(self, websocket: WebSocket, session_id: str):
        """
        Déconnecte un WebSocket d'une session.

        Args:
            websocket: Instance WebSocket à déconnecter
            session_id: ID de la session
        """
        try:
            # Retirer de la liste des connexions actives
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)

            # Nettoyer la session si plus de connexions
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
                if session_id in self.session_managers:
                    del self.session_managers[session_id]
                    self.stats['active_sessions'] -= 1

            self.logger.info(f"Déconnexion WebSocket pour session {session_id}")

        except Exception as e:
            self.logger.error(f"Erreur lors de la déconnexion WebSocket: {e}")
            self.stats['errors'] += 1

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """
        Envoie un message personnel à un WebSocket spécifique.

        Args:
            message: Message à envoyer
            websocket: WebSocket destinataire
        """
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
                self.stats['messages_sent'] += 1
            else:
                self.logger.warning("Tentative d'envoi à un WebSocket déconnecté")

        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi du message personnel: {e}")
            self.stats['errors'] += 1

    async def broadcast_to_session(self, message: Dict[str, Any], session_id: str, exclude_websocket: Optional[WebSocket] = None):
        """
        Diffuse un message à tous les WebSockets d'une session.

        Args:
            message: Message à diffuser
            session_id: ID de la session
            exclude_websocket: WebSocket à exclure (optionnel)
        """
        if session_id not in self.active_connections:
            return

        disconnected = []
        for websocket in self.active_connections[session_id]:
            if websocket == exclude_websocket:
                continue

            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(message)
                    self.stats['messages_sent'] += 1
                else:
                    disconnected.append(websocket)
            except Exception as e:
                self.logger.error(f"Erreur lors de la diffusion: {e}")
                disconnected.append(websocket)
                self.stats['errors'] += 1

        # Nettoyer les connexions déconnectées
        for websocket in disconnected:
            await self.disconnect(websocket, session_id)

    async def send_typing_indicator(self, session_id: str, user_id: str, is_typing: bool = True):
        """
        Envoie un indicateur de frappe.

        Args:
            session_id: ID de la session
            user_id: ID de l'utilisateur
            is_typing: Si l'utilisateur est en train de taper
        """
        message = {
            "type": "typing_indicator",
            "data": {
                "user_id": user_id,
                "is_typing": is_typing,
                "timestamp": datetime.now().isoformat()
            }
        }

        await self.broadcast_to_session(message, session_id)

    async def handle_message_stream(self, websocket: WebSocket, session_id: str, message_data: Dict[str, Any]):
        """
        Gère le streaming d'un message.

        Args:
            websocket: WebSocket source
            session_id: ID de la session
            message_data: Données du message
        """
        try:
            # Créer un ID de message unique
            message_id = str(uuid.uuid4())

            # Notifier le début du streaming
            start_message = {
                "type": "stream_start",
                "data": {
                    "message_id": message_id,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
            await self.send_personal_message(start_message, websocket)

            # Simuler le streaming (en production, ceci viendrait du modèle)
            full_response = message_data.get("content", "")
            chunks = self._chunk_text(full_response)

            for i, chunk in enumerate(chunks):
                is_final = (i == len(chunks) - 1)

                stream_chunk = {
                    "type": "stream_chunk",
                    "data": {
                        "message_id": message_id,
                        "session_id": session_id,
                        "chunk": chunk,
                        "is_final": is_final,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "timestamp": datetime.now().isoformat()
                    }
                }

                await self.send_personal_message(stream_chunk, websocket)

                # Petit délai pour simuler le temps de génération
                if not is_final:
                    await asyncio.sleep(0.05)

            # Notifier la fin du streaming
            end_message = {
                "type": "stream_end",
                "data": {
                    "message_id": message_id,
                    "session_id": session_id,
                    "total_chunks": len(chunks),
                    "timestamp": datetime.now().isoformat()
                }
            }
            await self.send_personal_message(end_message, websocket)

        except Exception as e:
            self.logger.error(f"Erreur lors du streaming du message: {e}")
            self.stats['errors'] += 1

    def _chunk_text(self, text: str, chunk_size: int = 10) -> List[str]:
        """Divise le texte en chunks pour le streaming."""
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks if chunks else [text]

    async def _heartbeat_worker(self, websocket: WebSocket, session_id: str):
        """Worker pour maintenir la connexion active."""
        try:
            while websocket.client_state == WebSocketState.CONNECTED:
                await asyncio.sleep(self.heartbeat_interval)

                # Envoyer un ping
                ping_message = {
                    "type": "ping",
                    "data": {
                        "timestamp": datetime.now().isoformat()
                    }
                }

                try:
                    await websocket.send_json(ping_message)
                except Exception:
                    # La connexion est probablement fermée
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Erreur dans le heartbeat worker: {e}")

        # Nettoyer la connexion
        await self.disconnect(websocket, session_id)

    async def _periodic_cleanup(self):
        """Nettoie périodiquement les connexions mortes."""
        while True:
            try:
                await asyncio.sleep(60)  # Nettoyer chaque minute

                disconnected_sessions = []

                for session_id, websockets in self.active_connections.items():
                    disconnected_websockets = []

                    for websocket in websockets:
                        if websocket.client_state != WebSocketState.CONNECTED:
                            disconnected_websockets.append(websocket)

                    # Nettoyer les WebSockets déconnectés
                    for websocket in disconnected_websockets:
                        await self.disconnect(websocket, session_id)

                    # Marquer les sessions vides
                    if not self.active_connections[session_id]:
                        disconnected_sessions.append(session_id)

                # Nettoyer les sessions vides
                for session_id in disconnected_sessions:
                    if session_id in self.session_managers:
                        del self.session_managers[session_id]
                        self.stats['active_sessions'] -= 1

            except Exception as e:
                self.logger.error(f"Erreur dans le nettoyage périodique: {e}")

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Retourne les statistiques d'une session."""
        return {
            'session_id': session_id,
            'active_connections': len(self.active_connections.get(session_id, set())),
            'has_manager': session_id in self.session_managers
        }

    def get_global_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques globales."""
        return {
            **self.stats,
            'total_sessions': len(self.active_connections),
            'sessions_with_managers': len(self.session_managers)
        }


class SessionManager:
    """Gestionnaire de session pour le chat."""

    def __init__(self, session_id: str):
        """
        Initialise le gestionnaire de session.

        Args:
            session_id: ID de la session
        """
        self.session_id = session_id
        self.message_history: List[ChatMessage] = []
        self.typing_users: Set[str] = set()
        self.last_activity = datetime.now()
        self.metadata: Dict[str, Any] = {}

    def add_message(self, message: ChatMessage):
        """Ajoute un message à l'historique."""
        self.message_history.append(message)
        self.last_activity = datetime.now()

    def get_recent_messages(self, limit: int = 50) -> List[ChatMessage]:
        """Retourne les messages récents."""
        return self.message_history[-limit:]

    def add_typing_user(self, user_id: str):
        """Ajoute un utilisateur en train de taper."""
        self.typing_users.add(user_id)

    def remove_typing_user(self, user_id: str):
        """Retire un utilisateur qui ne tape plus."""
        self.typing_users.discard(user_id)

    def get_typing_users(self) -> List[str]:
        """Retourne la liste des utilisateurs en train de taper."""
        return list(self.typing_users)

    def update_metadata(self, key: str, value: Any):
        """Met à jour les métadonnées de session."""
        self.metadata[key] = value
        self.last_activity = datetime.now()

    def get_metadata(self, key: Optional[str] = None) -> Any:
        """Retourne les métadonnées de session."""
        if key:
            return self.metadata.get(key)
        return self.metadata.copy()

    def is_active(self, timeout_minutes: int = 30) -> bool:
        """Vérifie si la session est active."""
        from datetime import timedelta
        return (datetime.now() - self.last_activity) < timedelta(minutes=timeout_minutes)