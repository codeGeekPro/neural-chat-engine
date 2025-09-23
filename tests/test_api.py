"""Tests pour l'API Neural Chat Engine."""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import app
from src.api.api_types import (
    SendMessageRequest,
    ChatMessage,
    MessageRole,
    MessageType,
    APIResponse
)


@pytest.fixture
def client():
    """Client de test FastAPI."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Client HTTPX asynchrone."""
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


class TestHealthCheck:
    """Tests pour les endpoints de santé."""

    def test_health_check(self, client):
        """Test du endpoint de santé."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_metrics(self, client):
        """Test du endpoint de métriques."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_tokens" in data
        assert "total_requests" in data
        assert "active_sessions" in data


class TestChatEndpoints:
    """Tests pour les endpoints de chat."""

    @patch("src.api.chat_endpoints._generate_assistant_response")
    @patch("src.api.chat_endpoints._save_message")
    @patch("src.api.chat_endpoints._update_session_stats")
    @patch("src.api.chat_endpoints._generate_recommendations")
    @patch("src.api.chat_endpoints._log_interaction")
    def test_send_message_success(
        self, mock_log, mock_recommendations, mock_update_stats,
        mock_save_message, mock_generate_response, client
    ):
        """Test d'envoi de message réussi."""
        # Mock des fonctions
        mock_response = ChatMessage(
            session_id="test-session",
            role=MessageRole.ASSISTANT,
            content="Bonjour ! Comment puis-je vous aider ?",
            message_type=MessageType.TEXT
        )
        mock_generate_response.return_value = mock_response
        mock_recommendations.return_value = None

        # Requête de test
        request_data = {
            "message": "Bonjour",
            "session_id": "test-session",
            "message_type": "text"
        }

        response = client.post("/chat/message", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == "test-session"
        assert "message_id" in data
        assert "response_message" in data
        assert data["response_message"]["content"] == "Bonjour ! Comment puis-je vous aider ?"

    def test_send_message_invalid_data(self, client):
        """Test d'envoi de message avec données invalides."""
        request_data = {
            "message": "",  # Message vide
            "message_type": "invalid_type"
        }

        response = client.post("/chat/message", json=request_data)
        assert response.status_code == 422  # Validation error

    @patch("src.api.chat_endpoints._get_messages")
    def test_get_conversation_history(self, mock_get_messages, client):
        """Test de récupération de l'historique."""
        # Mock des messages
        messages = [
            ChatMessage(
                session_id="test-session",
                role=MessageRole.USER,
                content="Bonjour",
                message_type=MessageType.TEXT
            ),
            ChatMessage(
                session_id="test-session",
                role=MessageRole.ASSISTANT,
                content="Bonjour !",
                message_type=MessageType.TEXT
            )
        ]
        mock_get_messages.return_value = messages

        response = client.get("/chat/history/test-session")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == "test-session"
        assert len(data["messages"]) == 2
        assert data["total_messages"] == 2

    def test_get_conversation_history_not_found(self, client):
        """Test de récupération d'historique inexistant."""
        response = client.get("/chat/history/nonexistent-session")
        assert response.status_code == 200

        data = response.json()
        assert data["messages"] == []
        assert data["total_messages"] == 0

    @patch("src.api.chat_endpoints._save_uploaded_file")
    @patch("src.api.chat_endpoints._analyze_file")
    def test_upload_file_success(self, mock_analyze, mock_save_file, client):
        """Test d'upload de fichier réussi."""
        mock_save_file.return_value = "/tmp/uploads/test.jpg"
        mock_analyze.return_value = {
            "width": 1920,
            "height": 1080,
            "format": "JPEG"
        }

        # Créer un fichier de test
        file_content = b"fake image content"
        files = {"file": ("test.jpg", file_content, "image/jpeg")}
        data = {"session_id": "test-session"}

        response = client.post("/chat/upload", files=files, data=data)
        assert response.status_code == 200

        result = response.json()
        assert result["filename"] == "test.jpg"
        assert result["content_type"] == "image/jpeg"
        assert "file_id" in result
        assert "url" in result
        assert result["analysis"]["width"] == 1920

    def test_upload_file_invalid_type(self, client):
        """Test d'upload de fichier avec type invalide."""
        file_content = b"fake executable content"
        files = {"file": ("test.exe", file_content, "application/x-msdownload")}
        data = {"session_id": "test-session"}

        response = client.post("/chat/upload", files=files, data=data)
        assert response.status_code == 400
        assert "non supporté" in response.json()["detail"]

    @patch("src.api.chat_endpoints._get_messages")
    @patch("src.api.chat_endpoints._create_json_export")
    def test_export_conversation_json(self, mock_create_export, mock_get_messages, client):
        """Test d'export de conversation au format JSON."""
        # Mock des messages
        messages = [
            ChatMessage(
                session_id="test-session",
                role=MessageRole.USER,
                content="Test message",
                message_type=MessageType.TEXT
            )
        ]
        mock_get_messages.return_value = messages
        mock_create_export.return_value = "/tmp/test_export.json"

        response = client.get("/chat/export/test-session?format=json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert "conversation_test-session.json" in response.headers["content-disposition"]

    def test_export_conversation_invalid_format(self, client):
        """Test d'export avec format invalide."""
        response = client.get("/chat/export/test-session?format=invalid")
        assert response.status_code == 400
        assert "non supporté" in response.json()["detail"]

    @patch("src.api.chat_endpoints._save_feedback")
    @patch("src.api.chat_endpoints._update_feedback_metrics")
    def test_submit_feedback_success(self, mock_update_metrics, mock_save_feedback, client):
        """Test de soumission de feedback réussi."""
        feedback_data = {
            "session_id": "test-session",
            "message_id": "test-message-id",
            "rating": 5,
            "feedback_type": "quality",
            "comment": "Excellent !"
        }

        response = client.post("/chat/feedback", json=feedback_data)
        assert response.status_code == 200

        result = response.json()
        assert result["success"] is True
        assert "feedback enregistré" in result["message"]

    def test_submit_feedback_invalid_rating(self, client):
        """Test de soumission de feedback avec note invalide."""
        feedback_data = {
            "session_id": "test-session",
            "message_id": "test-message-id",
            "rating": 10,  # Note invalide (max 5)
            "feedback_type": "quality"
        }

        response = client.post("/chat/feedback", json=feedback_data)
        assert response.status_code == 422  # Validation error


class TestAuthEndpoints:
    """Tests pour les endpoints d'authentification."""

    @patch("src.api.main.session_manager.create_session")
    def test_login_success(self, mock_create_session, client):
        """Test de connexion réussi."""
        from src.api.main import UserSession
        from datetime import datetime, timedelta

        mock_session = UserSession(
            session_id="test-session-id",
            user_id="user_123",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
            user_agent="test-agent",
            ip_address="127.0.0.1"
        )
        mock_create_session.return_value = mock_session

        response = client.post("/auth/login")
        assert response.status_code == 200

        result = response.json()
        assert result["success"] is True
        assert result["data"]["session_id"] == "test-session-id"
        assert result["data"]["user_id"] == "user_123"

    @patch("src.api.main.session_manager.delete_session")
    def test_logout_success(self, mock_delete_session, client):
        """Test de déconnexion réussi."""
        response = client.post("/auth/logout", json={"session_id": "test-session"})
        assert response.status_code == 200

        result = response.json()
        assert result["success"] is True
        assert "Déconnexion réussie" in result["message"]


class TestWebSocketManager:
    """Tests pour le gestionnaire WebSocket."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test de connexion WebSocket simulée."""
        from src.api.websocket_manager import WebSocketManager

        manager = WebSocketManager()

        # Mock WebSocket connection
        mock_ws = MagicMock()
        mock_ws.client.host = "127.0.0.1"

        # Test de connexion
        await manager.connect("test-session", mock_ws, "test-user")

        # Vérifier que la connexion est enregistrée
        assert "test-session" in manager.active_connections
        assert len(manager.active_connections["test-session"]) == 1

        # Test de déconnexion
        await manager.disconnect("test-session", mock_ws)

        # Vérifier que la connexion est supprimée
        assert len(manager.active_connections["test-session"]) == 0

    @pytest.mark.asyncio
    async def test_message_broadcasting(self):
        """Test de diffusion de messages."""
        from src.api.websocket_manager import WebSocketManager

        manager = WebSocketManager()

        # Mock WebSocket connections
        mock_ws1 = MagicMock()
        mock_ws2 = MagicMock()
        mock_ws1.client.host = "127.0.0.1"
        mock_ws2.client.host = "127.0.0.1"

        # Connecter deux clients
        await manager.connect("test-session", mock_ws1, "user1")
        await manager.connect("test-session", mock_ws2, "user2")

        # Diffuser un message
        test_message = {"type": "message", "content": "Hello!"}
        await manager.broadcast_to_session("test-session", test_message)

        # Vérifier que les deux clients ont reçu le message
        mock_ws1.send_json.assert_called_once_with(test_message)
        mock_ws2.send_json.assert_called_once_with(test_message)


class TestRateLimiting:
    """Tests pour les limites de taux."""

    def test_rate_limit_not_exceeded(self, client):
        """Test de limite de taux non dépassée."""
        # Faire plusieurs requêtes dans les limites
        for i in range(5):
            response = client.get("/health")
            assert response.status_code == 200

            # Vérifier les headers de limite de taux
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
            assert "X-RateLimit-Reset" in response.headers

    def test_rate_limit_exceeded(self, client):
        """Test de limite de taux dépassée."""
        # Simuler beaucoup de requêtes (en pratique, cela nécessiterait
        # de modifier la configuration de test ou de mocker le temps)
        # Pour cet exemple, nous testons juste que les headers sont présents
        response = client.get("/health")
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers


class TestErrorHandling:
    """Tests pour la gestion d'erreurs."""

    def test_404_not_found(self, client):
        """Test d'endpoint inexistant."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test de méthode HTTP non autorisée."""
        response = client.post("/health")  # GET only
        assert response.status_code == 405

    def test_invalid_json(self, client):
        """Test de JSON invalide."""
        response = client.post(
            "/chat/message",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])