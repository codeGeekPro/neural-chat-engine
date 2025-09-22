"""
Pipeline de Données - Neural Chat Engine

Gestion complète des données conversationnelles :
- Extraction depuis multiples sources (GitHub, Stack Overflow, Reddit)
- Nettoyage et normalisation
- Augmentation des données
- Création d'embeddings vectoriels
"""

from .data_pipeline import DataPipeline
from .conversation_processor import ConversationProcessor
from .embedding_generator import EmbeddingGenerator
from .data_augmentation import DataAugmentation

__all__ = [
    "DataPipeline",
    "ConversationProcessor",
    "EmbeddingGenerator", 
    "DataAugmentation"
]