"""Application FastAPI principale pour le Neural Chat Engine."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import secrets

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from .api_types import (
    UserSession,
    APIResponse,
    ErrorResponse,
    HealthCheck,
    TokenUsage,
    RateLimitInfo
)
from .websocket_manager import WebSocketManager
from .chat_endpoints import router as chat_router, set_websocket_manager as set_chat_ws_manager


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration de l'application
class Config:
    """Configuration de l'application."""
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    WORKERS = int(os.getenv("WORKERS", "1"))

    # CORS
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Rate limiting
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # secondes

    # Base de données (simulation)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_engine.db")

    # Sécurité
    JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION = int(os.getenv("JWT_EXPIRATION", "3600"))  # secondes

    # WebSocket
    WS_MAX_CONNECTIONS = int(os.getenv("WS_MAX_CONNECTIONS", "1000"))
    WS_MESSAGE_TIMEOUT = int(os.getenv("WS_MESSAGE_TIMEOUT", "30"))

    # Fichiers
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB


config = Config()


# Gestionnaire de sessions utilisateur (simulation)
class SessionManager:
    """Gestionnaire de sessions utilisateur."""

    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}

    async def create_session(self, user_id: str, user_agent: str = None) -> UserSession:
        """Crée une nouvelle session utilisateur."""
        session_id = secrets.token_urlsafe(32)
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),
            user_agent=user_agent,
            ip_address=None  # Sera défini par le middleware
        )
        self.sessions[session_id] = session
        logger.info(f"Session créée: {session_id} pour utilisateur {user_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Récupère une session par son ID."""
        session = self.sessions.get(session_id)
        if session and session.expires_at > datetime.now():
            return session
        elif session:
            # Session expirée
            await self.delete_session(session_id)
        return None

    async def delete_session(self, session_id: str) -> None:
        """Supprime une session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session supprimée: {session_id}")

    async def check_rate_limit(self, user_id: str, endpoint: str) -> RateLimitInfo:
        """Vérifie les limites de taux pour un utilisateur."""
        now = datetime.now()
        user_limits = self.rate_limits.get(user_id, {})

        # Nettoyer les anciennes entrées
        window_start = now - timedelta(seconds=config.RATE_LIMIT_WINDOW)
        user_limits = {
            k: v for k, v in user_limits.items()
            if isinstance(v, list) and v and v[0] > window_start
        }

        # Compter les requêtes dans la fenêtre
        requests = user_limits.get(endpoint, [])
        requests = [t for t in requests if t > window_start]

        if len(requests) >= config.RATE_LIMIT_REQUESTS:
            reset_time = requests[0] + timedelta(seconds=config.RATE_LIMIT_WINDOW)
            return RateLimitInfo(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                limit=config.RATE_LIMIT_REQUESTS
            )

        # Ajouter la nouvelle requête
        requests.append(now)
        user_limits[endpoint] = requests
        self.rate_limits[user_id] = user_limits

        remaining = config.RATE_LIMIT_REQUESTS - len(requests)
        reset_time = now + timedelta(seconds=config.RATE_LIMIT_WINDOW)

        return RateLimitInfo(
            allowed=True,
            remaining=remaining,
            reset_time=reset_time,
            limit=config.RATE_LIMIT_REQUESTS
        )


# Instance globale du gestionnaire de sessions
session_manager = SessionManager()


# Middleware de sécurité et logging
class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware pour la sécurité et le logging des requêtes."""

    async def dispatch(self, request: Request, call_next):
        start_time = datetime.now()

        # Log de la requête
        logger.info(f"Requête: {request.method} {request.url.path} - IP: {request.client.host}")

        # Vérifier les limites de taux
        user_id = request.headers.get("X-User-ID", "anonymous")
        rate_limit = await session_manager.check_rate_limit(user_id, request.url.path)

        if not rate_limit.allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Trop de requêtes",
                    "retry_after": int((rate_limit.reset_time - datetime.now()).total_seconds())
                }
            )

        # Ajouter les headers de limite de taux
        response = await call_next(request)

        response.headers["X-RateLimit-Limit"] = str(rate_limit.limit)
        response.headers["X-RateLimit-Remaining"] = str(rate_limit.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(rate_limit.reset_time.timestamp()))

        # Log de la réponse
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Réponse: {response.status_code} - Durée: {duration:.2f}s")

        return response


# Gestionnaire d'authentification
security = HTTPBearer(auto_error=False)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Récupère l'utilisateur actuel depuis le token JWT."""
    if not credentials:
        return None

    try:
        # Simulation de validation JWT - en production, utiliser une vraie bibliothèque JWT
        token = credentials.credentials
        if token.startswith("Bearer "):
            token = token[7:]

        # Validation simple (simulation)
        if len(token) > 10:  # Token valide simulé
            return "user_123"  # ID utilisateur simulé

    except Exception as e:
        logger.warning(f"Erreur de validation du token: {e}")

    return None


# Gestionnaire d'erreurs global
async def global_exception_handler(request: Request, exc: Exception):
    """Gestionnaire d'erreurs global."""
    logger.error(f"Erreur non gérée: {exc}", exc_info=True)

    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="Erreur HTTP",
                message=str(exc.detail),
                code=exc.status_code
            ).dict()
        )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Erreur interne",
            message="Une erreur inattendue s'est produite",
            code=500
        ).dict()
    )


# Gestionnaire de lifespan pour l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de lifespan pour l'initialisation et le nettoyage."""
    # Initialisation
    logger.info("Démarrage du Neural Chat Engine...")

    # Créer les répertoires nécessaires
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs("/tmp/exports", exist_ok=True)
    os.makedirs("/tmp/logs", exist_ok=True)

    # Initialiser le gestionnaire WebSocket
    websocket_manager = WebSocketManager(
        max_connections=config.WS_MAX_CONNECTIONS,
        message_timeout=config.WS_MESSAGE_TIMEOUT
    )

    # Configurer le gestionnaire WebSocket dans les endpoints
    set_chat_ws_manager(websocket_manager)

    # Démarrer les tâches en arrière-plan
    background_tasks = [
        asyncio.create_task(websocket_manager.cleanup_expired_connections()),
        asyncio.create_task(_periodic_cleanup())
    ]

    logger.info("Neural Chat Engine démarré avec succès")

    yield

    # Nettoyage
    logger.info("Arrêt du Neural Chat Engine...")

    # Annuler les tâches en arrière-plan
    for task in background_tasks:
        task.cancel()

    try:
        await asyncio.gather(*background_tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass

    logger.info("Neural Chat Engine arrêté")


# Création de l'application FastAPI
app = FastAPI(
    title="Neural Chat Engine API",
    description="API FastAPI pour le moteur de chat neuronal avec capacités multimodales",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Ajout des middlewares
app.add_middleware(SecurityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gestionnaire d'erreurs global
app.add_exception_handler(Exception, global_exception_handler)

# Inclusion des routers
app.include_router(chat_router)

# Monter les fichiers statiques (pour les uploads)
app.mount("/files", StaticFiles(directory=config.UPLOAD_DIR), name="files")


# Endpoints de base
@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil avec documentation."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural Chat Engine</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 40px; }
            .endpoints { margin-top: 30px; }
            .endpoint { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Neural Chat Engine</h1>
                <p>API FastAPI pour le chat neuronal avec capacités multimodales</p>
                <a href="/docs">📚 Documentation API</a> |
                <a href="/redoc">📖 Documentation ReDoc</a>
            </div>

            <h2>🚀 Démarrage rapide</h2>
            <div class="endpoints">
                <div class="endpoint">
                    <strong>POST /chat/message</strong><br>
                    Envoie un message et reçoit une réponse
                </div>
                <div class="endpoint">
                    <strong>GET /chat/history/{session_id}</strong><br>
                    Récupère l'historique de conversation
                </div>
                <div class="endpoint">
                    <strong>WebSocket /ws/chat/{session_id}</strong><br>
                    Connexion WebSocket pour le chat en temps réel
                </div>
            </div>

            <h2>📊 Monitoring</h2>
            <div class="endpoints">
                <div class="endpoint">
                    <strong>GET /health</strong><br>
                    Vérification de l'état de santé
                </div>
                <div class="endpoint">
                    <strong>GET /metrics</strong><br>
                    Métriques d'utilisation
                </div>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Vérification de l'état de santé du service."""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime=0,  # À calculer depuis le démarrage
        database="connected",  # Simulation
        websocket_connections=0  # À récupérer depuis le gestionnaire WS
    )


@app.get("/metrics", response_model=TokenUsage)
async def get_metrics():
    """Récupère les métriques d'utilisation."""
    # Simulation des métriques
    return TokenUsage(
        total_tokens=10000,
        total_requests=500,
        active_sessions=25,
        average_response_time=0.8,
        error_rate=0.02
    )


@app.post("/auth/login")
async def login(request: Request):
    """Authentification utilisateur (simulation)."""
    # Simulation d'authentification
    user_id = "user_123"

    # Créer une session
    session = await session_manager.create_session(
        user_id=user_id,
        user_agent=request.headers.get("user-agent")
    )

    return APIResponse(
        success=True,
        message="Authentification réussie",
        data={
            "session_id": session.session_id,
            "user_id": user_id,
            "expires_at": session.expires_at.isoformat()
        }
    )


@app.post("/auth/logout")
async def logout(session_id: str):
    """Déconnexion utilisateur."""
    await session_manager.delete_session(session_id)

    return APIResponse(
        success=True,
        message="Déconnexion réussie"
    )


# Tâches périodiques
async def _periodic_cleanup():
    """Nettoyage périodique des données expirées."""
    while True:
        try:
            # Nettoyer les sessions expirées
            expired_sessions = [
                sid for sid, session in session_manager.sessions.items()
                if session.expires_at < datetime.now()
            ]

            for sid in expired_sessions:
                await session_manager.delete_session(sid)

            logger.debug(f"Sessions expirées nettoyées: {len(expired_sessions)}")

            # Attendre 5 minutes avant le prochain nettoyage
            await asyncio.sleep(300)

        except Exception as e:
            logger.error(f"Erreur lors du nettoyage périodique: {e}")
            await asyncio.sleep(60)  # Attendre 1 minute en cas d'erreur


# Point d'entrée pour le développement
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        workers=config.WORKERS,
        log_level="info"
    )