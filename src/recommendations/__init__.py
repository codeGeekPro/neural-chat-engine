"""Module de recommandations contextuelles."""

from .recommendation_engine import RecommendationEngine
from .recommendation_types import (
    UserProfile,
    Item,
    ItemCatalog,
    UserInteraction,
    ContextSnapshot,
    Recommendation,
    RecommendationResult,
    RecommendationModel,
    RecommendationType,
    InteractionType,
    ContextType,
    FeedbackData
)

__all__ = [
    "RecommendationEngine",
    "UserProfile",
    "Item",
    "ItemCatalog",
    "UserInteraction",
    "ContextSnapshot",
    "Recommendation",
    "RecommendationResult",
    "RecommendationModel",
    "RecommendationType",
    "InteractionType",
    "ContextType",
    "FeedbackData"
]