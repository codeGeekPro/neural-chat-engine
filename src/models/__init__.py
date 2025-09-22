"""
Modèles IA - Neural Chat Engine

Ce module contient tous les modèles d'intelligence artificielle utilisés :
- Classification d'intentions avec DistilBERT
- Génération de réponses avec T5/GPT
- Embeddings avec Sentence-BERT
- Système de mémoire contextuelle avec Graph Neural Networks
"""

from .intent_classifier import IntentClassifier
from .response_generator import ResponseGenerator
from .personality_engine import PersonalityEngine
from .emotion_analyzer import EmotionAnalyzer
from .memory_system import ContextualMemory

__all__ = [
    "IntentClassifier",
    "ResponseGenerator", 
    "PersonalityEngine",
    "EmotionAnalyzer",
    "ContextualMemory"
]