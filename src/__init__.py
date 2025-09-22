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
- config: Système de gestion de configuration avec Pydantic
- config_loader: Chargeur de configurations YAML avancé
"""

__version__ = "0.1.0"
__author__ = "CodeGeekPro"
__email__ = "contact@codegeekpro.com"

# Import des modules principaux
try:
    from .config import Settings, get_settings, setup_logging
    from .config_loader import get_config_loader, ConfigurationLoader
except ImportError:
    # Imports optionnels en cas de dépendances manquantes
    pass

try:
    from .models import *
except ImportError:
    pass

try:
    from .data import *
except ImportError:
    pass

try:
    from .api import *
except ImportError:
    pass

__all__ = [
    "models",
    "data", 
    "training",
    "api",
    "frontend",
    "config",
    "config_loader",
    "Settings",
    "get_settings",
    "get_config_loader"
]