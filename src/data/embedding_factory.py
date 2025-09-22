from __future__ import annotations

from ..config import Settings, get_settings
from .embedding_generator import EmbeddingGenerator


def create_embedding_generator(settings: Settings | None = None) -> EmbeddingGenerator:
    """Factory pour créer EmbeddingGenerator depuis la configuration.
    
    Args:
        settings: Settings optionnelles, sinon utilise get_settings()
    
    Returns:
        EmbeddingGenerator configuré (sentence-transformers si possible, sinon fallback)
    """
    if settings is None:
        settings = get_settings()

    return EmbeddingGenerator(
        model_name=settings.models.embedding_model_name,
        device=settings.models.embedding_model_device if settings.models.use_gpu else "cpu",
        prefer_local_fallback=settings.prefer_local_embeddings,
    )