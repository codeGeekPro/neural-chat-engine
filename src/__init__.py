"""
Neural Chat Engine - Chatbot IA Avancé

Un système de chatbot intelligent multi-domaines avec capacités avancées de 
compréhension contextuelle, mémoire conversationnelle, et apprentissage continu.

Modules principaux:
- models: Modèles IA (Transformers, Classification, RAG)
- data: Pipeline de données et preprocessing  
- training: Scripts d'entraînement et fine-tuning
- api: Backend API FastAPI
- frontend: Interface utilisateur
"""

__version__ = "0.1.0"
__author__ = "CodeGeekPro"
__email__ = "contact@codegeekpro.com"

from .models import *
from .data import *
from .api import *

__all__ = [
    "models",
    "data", 
    "training",
    "api",
    "frontend"
]