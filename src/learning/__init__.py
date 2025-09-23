"""Module d'apprentissage continu."""

from .continual_learning import ContinualLearningSystem
from .learning_types import LearningTask, ModelVersion, PerformanceMetrics

__all__ = [
    "ContinualLearningSystem",
    "LearningTask",
    "ModelVersion",
    "PerformanceMetrics"
]