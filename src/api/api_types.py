"""Types de données pour l'API du Neural Chat Engine."""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import uuid


class MessageRole(str, Enum):
    """Rôles possibles pour les messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, Enum):
    """Types de messages."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    FILE = "file"
    MULTIMODAL = "multimodal"


class SessionStatus(str, Enum):
    """Statuts de session."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CLOSED = "closed"
    ERROR = "error"


class UserSession(BaseModel):
    """Session utilisateur."""
    session_id: str = Field(..., description="ID unique de la session")
    user_id: str = Field(..., description="ID de l'utilisateur")
    created_at: datetime = Field(default_factory=datetime.now, description="Date de création")
    expires_at: datetime = Field(..., description="Date d'expiration")
    user_agent: Optional[str] = Field(default=None, description="User-Agent du client")
    ip_address: Optional[str] = Field(default=None, description="Adresse IP du client")
    is_active: bool = Field(default=True, description="Session active")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Métadonnées supplémentaires")


class APIResponse(BaseModel):
    """Réponse API standardisée."""
    success: bool = Field(default=True, description="Statut de succès")
    message: str = Field(default="", description="Message descriptif")
    data: Optional[Any] = Field(default=None, description="Données de réponse")
    errors: Optional[List[str]] = Field(default=None, description="Erreurs éventuelles")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp de la réponse")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="ID unique de la requête")


class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée."""
    error: str = Field(..., description="Type d'erreur")
    message: str = Field(..., description="Message d'erreur détaillé")
    error_code: str = Field(default="UNKNOWN_ERROR", description="Code d'erreur")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Détails supplémentaires")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp de l'erreur")


class HealthCheck(BaseModel):
    """Vérification de l'état de santé du service."""
    status: str = Field(..., description="Statut du service")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    version: str = Field(..., description="Version du service")
    uptime: Optional[float] = Field(default=None, description="Temps de fonctionnement en secondes")
    database: Optional[str] = Field(default=None, description="Statut de la base de données")
    websocket_connections: Optional[int] = Field(default=None, description="Nombre de connexions WebSocket actives")


class ChatMessage(BaseModel):
    """Message de chat."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="ID unique du message")
    session_id: str = Field(..., description="ID de la session")
    role: MessageRole = Field(..., description="Rôle de l'expéditeur")
    content: str = Field(..., description="Contenu du message")
    message_type: MessageType = Field(default=MessageType.TEXT, description="Type de message")
    attachments: Optional[List[Dict[str, Any]]] = Field(default=None, description="Pièces jointes")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Métadonnées")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp du message")

    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Valider le contenu du message."""
        if not v or not v.strip():
            raise ValueError('Le contenu du message ne peut pas être vide')
        if len(v) > 10000:  # Limite de 10k caractères
            raise ValueError('Le contenu du message est trop long (max 10000 caractères)')
        return v.strip()


class SendMessageRequest(BaseModel):
    """Requête d'envoi de message."""
    session_id: Optional[str] = Field(default=None, description="ID de session (optionnel pour nouvelle session)")
    message: str = Field(..., description="Contenu du message")
    message_type: MessageType = Field(default=MessageType.TEXT, description="Type de message")
    attachments: Optional[List[Dict[str, Any]]] = Field(default=None, description="Pièces jointes")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Contexte supplémentaire")
    stream: bool = Field(default=False, description="Streaming de la réponse")

    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        """Valider le message de la requête."""
        if not v or not v.strip():
            raise ValueError('Le message ne peut pas être vide')
        if len(v) > 5000:  # Limite de 5k caractères pour les requêtes
            raise ValueError('Le message est trop long (max 5000 caractères)')
        return v.strip()


class SendMessageResponse(BaseModel):
    """Réponse d'envoi de message."""
    session_id: str = Field(..., description="ID de la session")
    message_id: str = Field(..., description="ID du message envoyé")
    response_message: ChatMessage = Field(..., description="Message de réponse")
    recommendations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Recommandations")


class ConversationHistory(BaseModel):
    """Historique de conversation."""
    session_id: str = Field(..., description="ID de la session")
    messages: List[ChatMessage] = Field(..., description="Liste des messages")
    total_messages: int = Field(..., description="Nombre total de messages")
    has_more: bool = Field(default=False, description="S'il y a plus de messages")
    next_cursor: Optional[str] = Field(default=None, description="Curseur pour la pagination")


class ChatSession(BaseModel):
    """Session de chat."""
    id: str = Field(..., description="ID de la session")
    user_id: str = Field(..., description="ID de l'utilisateur")
    title: Optional[str] = Field(default=None, description="Titre de la session")
    message_count: int = Field(default=0, description="Nombre de messages")
    created_at: datetime = Field(default_factory=datetime.now, description="Date de création")
    updated_at: datetime = Field(default_factory=datetime.now, description="Date de mise à jour")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Statut de la session")


class UploadResponse(BaseModel):
    """Réponse d'upload de fichier."""
    file_id: str = Field(..., description="ID unique du fichier")
    filename: str = Field(..., description="Nom du fichier")
    content_type: str = Field(..., description="Type de contenu")
    size: int = Field(..., description="Taille du fichier en octets")
    url: str = Field(..., description="URL d'accès au fichier")
    thumbnail_url: Optional[str] = Field(default=None, description="URL de la miniature")
    analysis: Optional[Dict[str, Any]] = Field(default=None, description="Résultats d'analyse")


class RateLimitInfo(BaseModel):
    """Informations sur les limites de taux."""
    allowed: bool = Field(default=True, description="Accès autorisé")
    limit: int = Field(..., description="Limite de requêtes")
    remaining: int = Field(..., description="Requêtes restantes")
    reset_time: datetime = Field(..., description="Temps de réinitialisation")
    retry_after: Optional[int] = Field(default=None, description="Secondes avant nouvelle tentative")


class TokenUsage(BaseModel):
    """Utilisation de tokens pour les modèles."""
    session_id: str = Field(..., description="ID de la session")
    model_name: str = Field(..., description="Nom du modèle")
    prompt_tokens: int = Field(..., description="Tokens utilisés pour le prompt")
    completion_tokens: int = Field(..., description="Tokens utilisés pour la complétion")
    total_tokens: int = Field(..., description="Total des tokens")
    cost: Optional[float] = Field(default=None, description="Coût estimé")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")


class FeedbackRequest(BaseModel):
    """Requête de feedback utilisateur."""
    session_id: str = Field(..., description="ID de la session")
    message_id: str = Field(..., description="ID du message")
    rating: int = Field(..., ge=1, le=5, description="Note de 1 à 5")
    feedback_type: str = Field(..., description="Type de feedback")
    comment: Optional[str] = Field(default=None, description="Commentaire optionnel")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Métadonnées supplémentaires")


class WebSocketMessage(BaseModel):
    """Message WebSocket."""
    type: str = Field(..., description="Type de message")
    data: Dict[str, Any] = Field(..., description="Données du message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")


class StreamingChunk(BaseModel):
    """Chunk de réponse en streaming."""
    session_id: str = Field(..., description="ID de la session")
    chunk: str = Field(..., description="Chunk de texte")
    is_final: bool = Field(default=False, description="Si c'est le dernier chunk")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Métadonnées")
