"""
Configuration Management System - Neural Chat Engine

Système complet de gestion de configuration avec validation Pydantic,
variables d'environnement, paramètres de base de données, configuration des modèles,
gestion des clés API et setup de logging.
"""

import os
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from enum import Enum

from pydantic import validator, Field
from pydantic import PostgresDsn, RedisDsn, HttpUrl
from pydantic_settings import BaseSettings


class EnvironmentType(str, Enum):
    """Types d'environnement supportés."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Niveaux de logging disponibles."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelProvider(str, Enum):
    """Fournisseurs de modèles IA supportés."""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    LOCAL = "local"


class VectorDatabaseType(str, Enum):
    """Types de bases de données vectorielles."""
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CHROMA = "chroma"
    FAISS = "faiss"


class DatabaseConfig(BaseSettings):
    """Configuration de la base de données PostgreSQL."""
    
    # PostgreSQL
    postgres_server: str = Field(default="localhost", env="POSTGRES_SERVER")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_user: str = Field(default="neural_user", env="POSTGRES_USER")
    postgres_password: str = Field(default="neural_pass", env="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="neural_chat_db", env="POSTGRES_DB")
    postgres_schema: str = Field(default="public", env="POSTGRES_SCHEMA")
    
    # Pool de connexions
    postgres_pool_size: int = Field(default=10, env="POSTGRES_POOL_SIZE")
    postgres_max_overflow: int = Field(default=20, env="POSTGRES_MAX_OVERFLOW")
    postgres_pool_timeout: int = Field(default=30, env="POSTGRES_POOL_TIMEOUT")
    
    @property
    def postgres_url(self) -> str:
        """URL de connexion PostgreSQL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_server}:{self.postgres_port}/{self.postgres_db}"
    
    @property  
    def async_postgres_url(self) -> str:
        """URL de connexion PostgreSQL asynchrone."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_server}:{self.postgres_port}/{self.postgres_db}"


class RedisConfig(BaseSettings):
    """Configuration Redis pour cache et sessions."""
    
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_ssl: bool = Field(default=False, env="REDIS_SSL")
    
    # Configuration du cache
    redis_cache_ttl: int = Field(default=3600, env="REDIS_CACHE_TTL")  # 1 heure
    redis_session_ttl: int = Field(default=86400, env="REDIS_SESSION_TTL")  # 24 heures
    
    # Pool de connexions
    redis_max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    redis_retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")
    
    @property
    def redis_url(self) -> str:
        """URL de connexion Redis."""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        protocol = "rediss" if self.redis_ssl else "redis"
        return f"{protocol}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


class VectorDatabaseConfig(BaseSettings):
    """Configuration des bases de données vectorielles."""
    
    # Type de base vectorielle
    vector_db_type: VectorDatabaseType = Field(default=VectorDatabaseType.WEAVIATE, env="VECTOR_DB_TYPE")
    
    # Pinecone
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="neural-chat-vectors", env="PINECONE_INDEX_NAME")
    
    # Weaviate
    weaviate_url: str = Field(default="http://localhost:8080", env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    weaviate_class_name: str = Field(default="ConversationVector", env="WEAVIATE_CLASS_NAME")
    
    # Configuration générale
    vector_dimension: int = Field(default=384, env="VECTOR_DIMENSION")  # all-MiniLM-L6-v2
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    max_results: int = Field(default=10, env="MAX_VECTOR_RESULTS")


class ModelConfig(BaseSettings):
    """Configuration des modèles IA."""
    
    # Modèle de classification d'intentions
    intent_model_name: str = Field(default="distilbert-base-multilingual-cased", env="INTENT_MODEL_NAME")
    intent_model_path: Optional[str] = Field(default=None, env="INTENT_MODEL_PATH")
    intent_max_length: int = Field(default=128, env="INTENT_MAX_LENGTH")
    intent_batch_size: int = Field(default=32, env="INTENT_BATCH_SIZE")
    
    # Modèle de génération de réponses
    response_model_provider: ModelProvider = Field(default=ModelProvider.HUGGINGFACE, env="RESPONSE_MODEL_PROVIDER")
    response_model_name: str = Field(default="microsoft/DialoGPT-medium", env="RESPONSE_MODEL_NAME")
    response_max_length: int = Field(default=512, env="RESPONSE_MAX_LENGTH")
    response_temperature: float = Field(default=0.7, env="RESPONSE_TEMPERATURE")
    response_top_p: float = Field(default=0.9, env="RESPONSE_TOP_P")
    response_top_k: int = Field(default=50, env="RESPONSE_TOP_K")
    
    # Modèle d'embeddings
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL_NAME")
    embedding_model_device: str = Field(default="cpu", env="EMBEDDING_MODEL_DEVICE")
    
    # Configuration GPU/CPU
    use_gpu: bool = Field(default=False, env="USE_GPU")
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    
    # Cache des modèles
    model_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    enable_model_caching: bool = Field(default=True, env="ENABLE_MODEL_CACHING")


class APIKeysConfig(BaseSettings):
    """Gestion sécurisée des clés API."""
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_organization: Optional[str] = Field(default=None, env="OPENAI_ORGANIZATION")
    
    # Anthropic Claude
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # Cohere
    cohere_api_key: Optional[str] = Field(default=None, env="COHERE_API_KEY")
    
    # Hugging Face
    huggingface_token: Optional[str] = Field(default=None, env="HUGGINGFACE_TOKEN")
    
    # Monitoring & Analytics
    wandb_api_key: Optional[str] = Field(default=None, env="WANDB_API_KEY")
    wandb_project: str = Field(default="neural-chat-engine", env="WANDB_PROJECT")
    
    # Security
    jwt_secret_key: str = Field(default="your-secret-key-change-in-production", env="JWT_SECRET_KEY")
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    
    @validator("jwt_secret_key")
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v


class ServerConfig(BaseSettings):
    """Configuration du serveur API."""
    
    # Serveur
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    workers: int = Field(default=1, env="API_WORKERS")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # secondes
    
    # Security
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    trusted_proxies: List[str] = Field(default=[], env="TRUSTED_PROXIES")


class LoggingConfig(BaseSettings):
    """Configuration du système de logging."""
    
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    log_max_size: int = Field(default=10485760, env="LOG_MAX_SIZE")  # 10MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # Logging spécialisé
    enable_request_logging: bool = Field(default=True, env="ENABLE_REQUEST_LOGGING")
    enable_sql_logging: bool = Field(default=False, env="ENABLE_SQL_LOGGING")
    enable_model_logging: bool = Field(default=True, env="ENABLE_MODEL_LOGGING")
    
    # Monitoring externe
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    enable_sentry: bool = Field(default=False, env="ENABLE_SENTRY")


class CeleryConfig(BaseSettings):
    """Configuration Celery pour les tâches asynchrones."""
    
    broker_url: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    result_backend: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    
    # Configuration des tâches
    task_serializer: str = Field(default="json", env="CELERY_TASK_SERIALIZER")
    result_serializer: str = Field(default="json", env="CELERY_RESULT_SERIALIZER")
    accept_content: List[str] = Field(default=["json"], env="CELERY_ACCEPT_CONTENT")
    
    # Timeout et retry
    task_soft_time_limit: int = Field(default=300, env="CELERY_TASK_SOFT_TIME_LIMIT")  # 5 minutes
    task_time_limit: int = Field(default=600, env="CELERY_TASK_TIME_LIMIT")  # 10 minutes
    task_max_retries: int = Field(default=3, env="CELERY_TASK_MAX_RETRIES")
    task_retry_delay: int = Field(default=60, env="CELERY_TASK_RETRY_DELAY")  # 1 minute


class Settings(BaseSettings):
    """Configuration principale de l'application."""
    
    # Métadonnées du projet
    project_name: str = "Neural Chat Engine"
    project_version: str = "0.1.0"
    project_description: str = "Chatbot IA Avancé avec Deep Learning et NLP"
    
    # Environnement
    environment: EnvironmentType = Field(default=EnvironmentType.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Configurations des sous-systèmes
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    vector_db: VectorDatabaseConfig = VectorDatabaseConfig()
    models: ModelConfig = ModelConfig()
    api_keys: APIKeysConfig = APIKeysConfig()
    server: ServerConfig = ServerConfig()
    logging: LoggingConfig = LoggingConfig()
    celery: CeleryConfig = CeleryConfig()
    
    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("./data"))
    models_dir: Path = Field(default_factory=lambda: Path("./models"))
    logs_dir: Path = Field(default_factory=lambda: Path("./logs"))
    
    # Features flags
    prefer_local_embeddings: bool = Field(default=False, env="PREFER_LOCAL_EMBEDDINGS")
    enable_conversation_memory: bool = Field(default=True, env="ENABLE_CONVERSATION_MEMORY")
    enable_personality_adaptation: bool = Field(default=True, env="ENABLE_PERSONALITY_ADAPTATION")
    enable_emotion_analysis: bool = Field(default=True, env="ENABLE_EMOTION_ANALYSIS")
    enable_multilingual: bool = Field(default=True, env="ENABLE_MULTILINGUAL")
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("data_dir", "models_dir", "logs_dir", pre=True)
    def create_directories(cls, v):
        """Crée les répertoires s'ils n'existent pas."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_database_url(self, async_mode: bool = False) -> str:
        """Retourne l'URL appropriée de la base de données."""
        return self.database.async_postgres_url if async_mode else self.database.postgres_url
    
    def is_production(self) -> bool:
        """Vérifie si on est en environnement de production."""
        return self.environment == EnvironmentType.PRODUCTION
    
    def is_development(self) -> bool:
        """Vérifie si on est en environnement de développement."""
        return self.environment == EnvironmentType.DEVELOPMENT


# Instance globale des settings
settings = Settings()


def get_settings() -> Settings:
    """Factory function pour obtenir les settings (utile pour l'injection de dépendances)."""
    return settings


def setup_logging() -> None:
    """Configure le système de logging basé sur les settings."""
    logging_config = settings.logging
    
    # Configuration de base
    log_level = getattr(logging, logging_config.log_level.value)
    
    # Format du logging
    formatter = logging.Formatter(logging_config.log_format)
    
    # Logger principal
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler si spécifié
    if logging_config.log_file:
        from logging.handlers import RotatingFileHandler
        
        file_handler = RotatingFileHandler(
            logging_config.log_file,
            maxBytes=logging_config.log_max_size,
            backupCount=logging_config.log_backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Configuration Sentry si activé
    if logging_config.enable_sentry and logging_config.sentry_dsn:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration
            
            sentry_logging = LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR
            )
            
            sentry_sdk.init(
                dsn=logging_config.sentry_dsn,
                integrations=[sentry_logging],
                environment=settings.environment.value,
                release=settings.project_version
            )
        except ImportError:
            logger.warning("Sentry SDK not installed, skipping Sentry configuration")


def validate_configuration() -> bool:
    """Valide la configuration et affiche les warnings appropriés."""
    logger = logging.getLogger(__name__)
    is_valid = True
    
    # Vérifications de sécurité en production
    if settings.is_production():
        if settings.api_keys.jwt_secret_key == "your-secret-key-change-in-production":
            logger.error("JWT secret key must be changed in production!")
            is_valid = False
            
        if not settings.database.postgres_password or len(settings.database.postgres_password) < 8:
            logger.error("Database password must be strong in production!")
            is_valid = False
            
        if not settings.redis.redis_password:
            logger.warning("Redis password not set in production")
    
    # Vérifications des clés API selon les modèles configurés
    model_config = settings.models
    api_keys = settings.api_keys
    
    if model_config.response_model_provider == ModelProvider.OPENAI and not api_keys.openai_api_key:
        logger.error("OpenAI API key required for OpenAI models")
        is_valid = False
    
    if model_config.response_model_provider == ModelProvider.ANTHROPIC and not api_keys.anthropic_api_key:
        logger.error("Anthropic API key required for Claude models")
        is_valid = False
    
    # Vérifications des bases de données vectorielles
    vector_config = settings.vector_db
    if vector_config.vector_db_type == VectorDatabaseType.PINECONE:
        if not vector_config.pinecone_api_key or not vector_config.pinecone_environment:
            logger.error("Pinecone API key and environment required")
            is_valid = False
    
    logger.info(f"Configuration validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    return is_valid


if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Neural Chat Engine - Configuration Test")
    print(f"Environment: {settings.environment.value}")
    print(f"Debug: {settings.debug}")
    print(f"Database URL: {settings.get_database_url()}")
    print(f"Redis URL: {settings.redis.redis_url}")
    print(f"Models Dir: {settings.models_dir}")
    
    setup_logging()
    validate_configuration()