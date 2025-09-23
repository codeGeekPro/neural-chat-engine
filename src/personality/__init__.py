"""Module de personnalité adaptative pour le chatbot."""

from .personality_engine import PersonalityEngine
from .personality_types import PersonalityProfile, CommunicationStyle

__all__ = [
    "PersonalityEngine",
    "PersonalityProfile",
    "CommunicationStyle"
]