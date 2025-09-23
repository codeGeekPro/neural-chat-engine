"""Endpoints API pour les fonctionnalités de chat."""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json
import uuid

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from starlette.requests import Request
from starlette.responses import JSONResponse

from .api_types import (
    SendMessageRequest,
    SendMessageResponse,
    ConversationHistory,
    ChatMessage,
    ChatSession,
    MessageRole,
    MessageType,
    UploadResponse,
    APIResponse,
    ErrorResponse,
    TokenUsage,
    FeedbackRequest
)
from .websocket_manager import WebSocketManager


# Configuration du router
router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={
        404: {"description": "Session ou message non trouvé"},
        422: {"description": "Données invalides"},
        500: {"description": "Erreur interne du serveur"}
    }
)

# Instance globale du gestionnaire WebSocket (à injecter depuis main.py)
websocket_manager: Optional[WebSocketManager] = None

# Logger
logger = logging.getLogger(__name__)


def set_websocket_manager(manager: WebSocketManager):
    """Configure le gestionnaire WebSocket."""
    global websocket_manager
    websocket_manager = manager


@router.post("/message", response_model=SendMessageResponse)
async def send_message(
    request: SendMessageRequest,
    background_tasks: BackgroundTasks,
    req: Request
) -> SendMessageResponse:
    """
    Envoie un message et reçoit une réponse.

    - **session_id**: ID de la session (optionnel pour nouvelle session)
    - **message**: Contenu du message
    - **message_type**: Type de message (text, image, audio, etc.)
    - **attachments**: Pièces jointes optionnelles
    - **context**: Contexte supplémentaire
    - **stream**: Si la réponse doit être streamée
    """
    try:
        # Générer un nouvel ID de session si non fourni
        session_id = request.session_id or str(uuid.uuid4())

        # Créer le message utilisateur
        user_message = ChatMessage(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.message,
            message_type=request.message_type,
            attachments=request.attachments,
            metadata={
                "user_agent": req.headers.get("user-agent"),
                "ip_address": req.client.host if req.client else None,
                "context": request.context
            }
        )

        # Sauvegarder le message utilisateur (simulation)
        await _save_message(user_message)

        # Générer la réponse de l'assistant
        assistant_response = await _generate_assistant_response(
            user_message,
            stream=request.stream
        )

        # Sauvegarder la réponse
        await _save_message(assistant_response)

        # Mettre à jour la session
        await _update_session_stats(session_id)

        # Générer des recommandations si pertinent
        recommendations = await _generate_recommendations(user_message)

        # Tâches en arrière-plan
        background_tasks.add_task(_log_interaction, user_message, assistant_response)

        response = SendMessageResponse(
            session_id=session_id,
            message_id=user_message.id,
            response_message=assistant_response,
            recommendations=recommendations
        )

        logger.info(f"Message envoyé pour session {session_id}")
        return response

    except Exception as e:
        logger.error(f"Erreur lors de l'envoi du message: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")


@router.get("/history/{session_id}", response_model=ConversationHistory)
async def get_conversation_history(
    session_id: str,
    limit: int = 50,
    cursor: Optional[str] = None
) -> ConversationHistory:
    """
    Récupère l'historique de conversation d'une session.

    - **session_id**: ID de la session
    - **limit**: Nombre maximum de messages (défaut: 50)
    - **cursor**: Curseur pour la pagination
    """
    try:
        # Récupérer les messages depuis la base de données (simulation)
        messages = await _get_messages(session_id, limit, cursor)

        # Calculer s'il y a plus de messages
        has_more = len(messages) == limit
        next_cursor = messages[-1].id if has_more and messages else None

        history = ConversationHistory(
            session_id=session_id,
            messages=messages,
            total_messages=len(messages),
            has_more=has_more,
            next_cursor=next_cursor
        )

        logger.info(f"Historique récupéré pour session {session_id}: {len(messages)} messages")
        return history

    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'historique: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération de l'historique")


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    description: Optional[str] = Form(None)
) -> UploadResponse:
    """
    Upload un fichier pour une session de chat.

    - **file**: Fichier à uploader
    - **session_id**: ID de la session
    - **description**: Description optionnelle du fichier
    """
    try:
        # Valider le type de fichier
        allowed_types = {
            'image': ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
            'audio': ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp4'],
            'document': ['application/pdf', 'text/plain', 'application/msword']
        }

        content_type = file.content_type
        file_category = None

        for category, types in allowed_types.items():
            if content_type in types:
                file_category = category
                break

        if not file_category:
            raise HTTPException(
                status_code=400,
                detail=f"Type de fichier non supporté: {content_type}"
            )

        # Générer un ID unique pour le fichier
        file_id = str(uuid.uuid4())

        # Sauvegarder le fichier (simulation)
        file_path = await _save_uploaded_file(file, file_id, session_id)

        # Analyser le fichier si c'est une image ou audio
        analysis = None
        if file_category in ['image', 'audio']:
            analysis = await _analyze_file(file_path, file_category)

        # Générer l'URL d'accès
        file_url = f"/api/files/{file_id}"

        # Générer la miniature pour les images
        thumbnail_url = None
        if file_category == 'image':
            thumbnail_url = f"/api/files/{file_id}/thumbnail"

        response = UploadResponse(
            file_id=file_id,
            filename=file.filename,
            content_type=content_type,
            size=file.size or 0,
            url=file_url,
            thumbnail_url=thumbnail_url,
            analysis=analysis
        )

        logger.info(f"Fichier uploadé: {file.filename} ({file.size} bytes) pour session {session_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'upload du fichier: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'upload du fichier")


@router.get("/export/{session_id}")
async def export_conversation(session_id: str, format: str = "json") -> FileResponse:
    """
    Exporte une conversation au format spécifié.

    - **session_id**: ID de la session à exporter
    - **format**: Format d'export (json, txt, pdf)
    """
    try:
        # Récupérer tous les messages de la session
        messages = await _get_messages(session_id, limit=1000)

        if not messages:
            raise HTTPException(status_code=404, detail="Conversation non trouvée")

        # Générer le fichier d'export
        if format == "json":
            export_data = {
                "session_id": session_id,
                "exported_at": datetime.now().isoformat(),
                "messages": [msg.dict() for msg in messages]
            }

            export_path = await _create_json_export(export_data, session_id)
            media_type = "application/json"
            filename = f"conversation_{session_id}.json"

        elif format == "txt":
            export_content = await _create_text_export(messages, session_id)
            export_path = await _save_text_export(export_content, session_id)
            media_type = "text/plain"
            filename = f"conversation_{session_id}.txt"

        else:
            raise HTTPException(status_code=400, detail="Format d'export non supporté")

        logger.info(f"Conversation exportée: {session_id} au format {format}")
        return FileResponse(
            path=export_path,
            media_type=media_type,
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'export de la conversation: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'export")


@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest) -> APIResponse:
    """
    Soumet un feedback pour un message.

    - **session_id**: ID de la session
    - **message_id**: ID du message
    - **rating**: Note de 1 à 5
    - **feedback_type**: Type de feedback
    - **comment**: Commentaire optionnel
    """
    try:
        # Sauvegarder le feedback (simulation)
        await _save_feedback(feedback)

        # Mettre à jour les métriques
        await _update_feedback_metrics(feedback)

        logger.info(f"Feedback soumis pour message {feedback.message_id}: {feedback.rating}/5")

        return APIResponse(
            success=True,
            message="Feedback enregistré avec succès",
            data={"feedback_id": str(uuid.uuid4())}
        )

    except Exception as e:
        logger.error(f"Erreur lors de la soumission du feedback: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'enregistrement du feedback")


@router.get("/sessions", response_model=List[ChatSession])
async def list_user_sessions(user_id: str) -> List[ChatSession]:
    """
    Liste les sessions de chat d'un utilisateur.

    - **user_id**: ID de l'utilisateur
    """
    try:
        sessions = await _get_user_sessions(user_id)

        logger.info(f"Sessions récupérées pour utilisateur {user_id}: {len(sessions)} sessions")
        return sessions

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des sessions: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des sessions")


@router.delete("/session/{session_id}")
async def delete_session(session_id: str) -> APIResponse:
    """
    Supprime une session de chat.

    - **session_id**: ID de la session à supprimer
    """
    try:
        # Supprimer la session et tous ses messages (simulation)
        await _delete_session(session_id)

        # Fermer les connexions WebSocket associées
        if websocket_manager:
            # Note: Dans une vraie implémentation, il faudrait identifier et fermer
            # toutes les connexions WebSocket pour cette session
            pass

        logger.info(f"Session supprimée: {session_id}")

        return APIResponse(
            success=True,
            message="Session supprimée avec succès"
        )

    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la session: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la suppression de la session")


# Fonctions utilitaires (à remplacer par de vraies implémentations de base de données)

async def _save_message(message: ChatMessage) -> None:
    """Sauvegarde un message (simulation)."""
    # Simulation - en production, sauvegarder dans une base de données
    logger.debug(f"Message sauvegardé: {message.id} pour session {message.session_id}")


async def _generate_assistant_response(message: ChatMessage, stream: bool = False) -> ChatMessage:
    """Génère une réponse d'assistant (simulation)."""
    # Simulation d'une réponse intelligente
    response_content = await _generate_smart_response(message.content)

    response = ChatMessage(
        session_id=message.session_id,
        role=MessageRole.ASSISTANT,
        content=response_content,
        message_type=MessageType.TEXT,
        metadata={
            "model": "neural-chat-engine-v1",
            "processing_time": 0.5,
            "confidence": 0.95
        }
    )

    return response


async def _generate_smart_response(user_message: str) -> str:
    """Génère une réponse intelligente (simulation)."""
    # Simulation simple - en production, utiliser un vrai modèle
    responses = [
        "Je comprends votre question. Laissez-moi vous aider avec cela.",
        "C'est une excellente question ! Voici ce que je peux vous dire :",
        "Intéressant ! Permettez-moi d'analyser cela pour vous.",
        "Je vois que vous vous intéressez à ce sujet. Voici quelques informations :",
        "Merci pour votre message. Je vais vous fournir une réponse détaillée."
    ]

    # Réponse basée sur le contenu
    if "bonjour" in user_message.lower() or "salut" in user_message.lower():
        return "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
    elif "?" in user_message:
        return "Excellente question ! Laissez-moi réfléchir à cela..."
    else:
        return responses[len(user_message) % len(responses)]


async def _generate_recommendations(message: ChatMessage) -> Optional[List[Dict[str, Any]]]:
    """Génère des recommandations (simulation)."""
    # Simulation - en production, utiliser le vrai système de recommandations
    if "recommandation" in message.content.lower() or "suggérer" in message.content.lower():
        return [
            {
                "type": "product",
                "title": "Produit Suggéré",
                "description": "Basé sur votre intérêt",
                "score": 0.85
            }
        ]
    return None


async def _get_messages(session_id: str, limit: int = 50, cursor: Optional[str] = None) -> List[ChatMessage]:
    """Récupère les messages d'une session (simulation)."""
    # Simulation - en production, récupérer depuis la base de données
    messages = []

    # Créer quelques messages d'exemple
    for i in range(min(limit, 10)):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        content = f"Message d'exemple {i+1}" if role == MessageRole.USER else f"Réponse d'exemple {i+1}"

        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            message_type=MessageType.TEXT,
            timestamp=datetime.now()
        )
        messages.append(message)

    return messages


async def _update_session_stats(session_id: str) -> None:
    """Met à jour les statistiques de session (simulation)."""
    logger.debug(f"Statistiques mises à jour pour session {session_id}")


async def _log_interaction(user_message: ChatMessage, assistant_response: ChatMessage) -> None:
    """Log l'interaction pour l'analyse (simulation)."""
    logger.debug(f"Interaction loggée: {user_message.id} -> {assistant_response.id}")


async def _save_uploaded_file(file: UploadFile, file_id: str, session_id: str) -> str:
    """Sauvegarde un fichier uploadé (simulation)."""
    # Simulation - en production, sauvegarder sur le système de fichiers ou cloud
    file_path = f"/tmp/uploads/{file_id}_{file.filename}"
    logger.debug(f"Fichier sauvegardé: {file_path}")
    return file_path


async def _analyze_file(file_path: str, file_category: str) -> Optional[Dict[str, Any]]:
    """Analyse un fichier (simulation)."""
    if file_category == "image":
        return {
            "width": 1920,
            "height": 1080,
            "format": "JPEG",
            "objects_detected": ["person", "computer"],
            "description": "Une personne utilisant un ordinateur"
        }
    elif file_category == "audio":
        return {
            "duration": 45.2,
            "language": "fr",
            "transcription": "Contenu audio transcrit..."
        }
    return None


async def _create_json_export(data: Dict[str, Any], session_id: str) -> str:
    """Crée un export JSON (simulation)."""
    export_path = f"/tmp/exports/conversation_{session_id}.json"
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return export_path


async def _create_text_export(messages: List[ChatMessage], session_id: str) -> str:
    """Crée un export texte (simulation)."""
    lines = [f"Conversation {session_id}\n{'='*50}\n"]

    for msg in messages:
        timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        role = "Utilisateur" if msg.role == MessageRole.USER else "Assistant"
        lines.append(f"[{timestamp}] {role}: {msg.content}\n")

    return "\n".join(lines)


async def _save_text_export(content: str, session_id: str) -> str:
    """Sauvegarde un export texte (simulation)."""
    export_path = f"/tmp/exports/conversation_{session_id}.txt"
    with open(export_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return export_path


async def _save_feedback(feedback: FeedbackRequest) -> None:
    """Sauvegarde un feedback (simulation)."""
    logger.debug(f"Feedback sauvegardé: {feedback.message_id} - {feedback.rating}/5")


async def _update_feedback_metrics(feedback: FeedbackRequest) -> None:
    """Met à jour les métriques de feedback (simulation)."""
    logger.debug(f"Métriques mises à jour pour feedback: {feedback.feedback_type}")


async def _get_user_sessions(user_id: str) -> List[ChatSession]:
    """Récupère les sessions d'un utilisateur (simulation)."""
    sessions = []

    # Créer quelques sessions d'exemple
    for i in range(3):
        session = ChatSession(
            id=f"session_{i+1}",
            user_id=user_id,
            title=f"Conversation {i+1}",
            message_count=10 + i * 5,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        sessions.append(session)

    return sessions


async def _delete_session(session_id: str) -> None:
    """Supprime une session (simulation)."""
    logger.debug(f"Session supprimée: {session_id}")