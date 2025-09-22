"""Types et constantes pour l'analyse émotionnelle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class EmotionCategory(str, Enum):
    """Catégories d'émotions de base et complexes."""
    
    # Émotions primaires (Plutchik)
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    
    # Émotions secondaires
    LOVE = "love"  # joy + trust
    GUILT = "guilt"  # sadness + fear
    PRIDE = "pride"  # joy + anticipation
    SHAME = "shame"  # sadness + disgust
    ANXIETY = "anxiety"  # fear + anticipation
    
    # États neutres
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class EmotionIntensity(float, Enum):
    """Niveaux d'intensité des émotions."""
    NONE = 0.0
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 1.0


@dataclass
class EmotionState:
    """État émotionnel complet à un instant donné."""
    
    # Émotion dominante
    primary_emotion: EmotionCategory
    intensity: float  # [0.0, 1.0]
    
    # Distribution complète
    emotion_scores: Dict[EmotionCategory, float] = field(default_factory=dict)
    
    # Méta-informations
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0  # [0.0, 1.0]
    context_window: Optional[List[str]] = None
    
    @property
    def secondary_emotions(self) -> List[EmotionCategory]:
        """Retourne les émotions secondaires significatives (score > 0.3)."""
        return [
            emotion for emotion, score in self.emotion_scores.items()
            if emotion != self.primary_emotion and score > 0.3
        ]


@dataclass
class EmotionProfile:
    """Profil émotionnel avec historique et tendances."""
    
    # Historique des états
    emotion_history: List[EmotionState] = field(default_factory=list)
    
    # Statistiques
    avg_intensities: Dict[EmotionCategory, float] = field(default_factory=dict)
    emotion_frequencies: Dict[EmotionCategory, int] = field(default_factory=dict)
    
    # Patterns identifiés
    common_transitions: Dict[EmotionCategory, Dict[EmotionCategory, int]] = field(
        default_factory=lambda: {}
    )
    
    def add_emotion_state(self, state: EmotionState) -> None:
        """Ajoute un nouvel état émotionnel et met à jour les stats."""
        self.emotion_history.append(state)
        
        # Met à jour les fréquences
        self.emotion_frequencies[state.primary_emotion] = (
            self.emotion_frequencies.get(state.primary_emotion, 0) + 1
        )
        
        # Met à jour les intensités moyennes
        for emotion, score in state.emotion_scores.items():
            current_avg = self.avg_intensities.get(emotion, 0.0)
            n = self.emotion_frequencies.get(emotion, 0)
            if n > 0:
                self.avg_intensities[emotion] = (
                    (current_avg * n + score) / (n + 1)
                )
        
        # Met à jour les transitions si on a un historique
        if len(self.emotion_history) > 1:
            prev_state = self.emotion_history[-2]
            prev_emotion = prev_state.primary_emotion
            curr_emotion = state.primary_emotion
            
            if prev_emotion not in self.common_transitions:
                self.common_transitions[prev_emotion] = {}
            
            self.common_transitions[prev_emotion][curr_emotion] = (
                self.common_transitions[prev_emotion].get(curr_emotion, 0) + 1
            )

    def get_dominant_emotions(self, top_k: int = 3) -> List[EmotionCategory]:
        """Retourne les k émotions les plus fréquentes."""
        return sorted(
            self.emotion_frequencies.keys(),
            key=lambda e: self.emotion_frequencies[e],
            reverse=True
        )[:top_k]

    def get_emotion_trend(self, window: int = 5) -> Optional[str]:
        """Analyse la tendance émotionnelle récente."""
        if len(self.emotion_history) < window:
            return None
            
        recent = self.emotion_history[-window:]
        avg_intensity = sum(s.intensity for s in recent) / window
        
        if avg_intensity > 0.7:
            return "heightened"
        elif avg_intensity < 0.3:
            return "subdued"
        return "stable"