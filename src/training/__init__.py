"""
Scripts d'Entraînement - Neural Chat Engine

Scripts pour l'entraînement et le fine-tuning des modèles :
- Fine-tuning DistilBERT pour classification d'intentions
- Entraînement générateur de réponses T5/GPT
- Apprentissage continu et prévention de l'oubli catastrophique
"""

from .intent_trainer import IntentTrainer
from .response_trainer import ResponseTrainer
from .continual_learning import ContinualLearning
from .model_evaluator import ModelEvaluator

__all__ = [
    "IntentTrainer",
    "ResponseTrainer",
    "ContinualLearning",
    "ModelEvaluator"
]