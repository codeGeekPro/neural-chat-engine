"""Types et structures pour le système de personnalité."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class CommunicationStyle(Enum):
    """Styles de communication possibles."""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    SIMPLE = "simple"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CREATIVE = "creative"
    DIRECT = "direct"


class PersonalityDimension(Enum):
    """Dimensions de personnalité."""
    FORMALITY = "formality"  # Formal vs Casual
    COMPLEXITY = "complexity"  # Technical vs Simple
    FRIENDLINESS = "friendliness"  # Friendly vs Professional
    CREATIVITY = "creativity"  # Creative vs Direct


@dataclass
class PersonalityProfile:
    """Profil de personnalité d'un utilisateur."""

    user_id: str
    communication_style: CommunicationStyle = CommunicationStyle.CASUAL
    personality_dimensions: Dict[PersonalityDimension, float] = field(
        default_factory=lambda: {
            PersonalityDimension.FORMALITY: 0.5,
            PersonalityDimension.COMPLEXITY: 0.5,
            PersonalityDimension.FRIENDLINESS: 0.5,
            PersonalityDimension.CREATIVITY: 0.5
        }
    )
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    sample_size: int = 0

    def update_from_interaction(
        self,
        interaction_data: Dict[str, Any],
        confidence: float = 1.0
    ) -> None:
        """Met à jour le profil depuis une interaction."""
        self.interaction_history.append(interaction_data)
        self.last_updated = datetime.now()
        self.sample_size += 1

        # Met à jour les scores de confiance
        for key, value in interaction_data.items():
            if key in self.confidence_scores:
                # Moyenne pondérée
                old_conf = self.confidence_scores[key]
                self.confidence_scores[key] = (
                    old_conf * (self.sample_size - 1) + confidence
                ) / self.sample_size
            else:
                self.confidence_scores[key] = confidence

    def get_personality_vector(self) -> np.ndarray:
        """Retourne un vecteur représentant la personnalité."""
        return np.array([
            self.personality_dimensions[PersonalityDimension.FORMALITY],
            self.personality_dimensions[PersonalityDimension.COMPLEXITY],
            self.personality_dimensions[PersonalityDimension.FRIENDLINESS],
            self.personality_dimensions[PersonalityDimension.CREATIVITY]
        ])

    def is_confident(self, threshold: float = 0.7) -> bool:
        """Vérifie si le profil est suffisamment confiant."""
        if self.sample_size < 5:
            return False

        avg_confidence = np.mean(list(self.confidence_scores.values()))
        return avg_confidence >= threshold


@dataclass
class PersonalityAdaptation:
    """Adaptation de personnalité pour une réponse."""

    original_tone: str
    adapted_tone: str
    adaptation_factors: Dict[str, float]
    confidence_score: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "original_tone": self.original_tone,
            "adapted_tone": self.adapted_tone,
            "adaptation_factors": self.adaptation_factors,
            "confidence_score": self.confidence_score,
            "reasoning": self.reasoning
        }


@dataclass
class StyleAnalysis:
    """Analyse du style de communication."""

    detected_style: CommunicationStyle
    style_confidence: float
    linguistic_features: Dict[str, float]
    emotional_indicators: Dict[str, float]
    complexity_metrics: Dict[str, float]
    formality_score: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "detected_style": self.detected_style.value,
            "style_confidence": self.style_confidence,
            "linguistic_features": self.linguistic_features,
            "emotional_indicators": self.emotional_indicators,
            "complexity_metrics": self.complexity_metrics,
            "formality_score": self.formality_score,
            "timestamp": self.timestamp.isoformat()
        }