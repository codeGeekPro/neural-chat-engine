from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
from loguru import logger


class ConversationFormat(str, Enum):
    """Formats de sérialisation des conversations supportés."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    XML = "xml"
    MARKDOWN = "md"
    TEXT = "txt"


@dataclass
class ConversationStats:
    """Statistiques sur une conversation."""
    
    # Métriques basiques
    num_turns: int = 0
    num_tokens: int = 0
    duration_seconds: float = 0.0
    avg_response_time: float = 0.0
    
    # Qualité et engagement
    user_sentiment: float = 0.0  # [-1, 1]
    assistant_sentiment: float = 0.0
    coherence_score: float = 0.0  # [0, 1]
    engagement_score: float = 0.0  # [0, 1]
    
    # Annotations
    detected_intents: List[str] = field(default_factory=list)
    detected_emotions: List[str] = field(default_factory=list)
    detected_languages: List[str] = field(default_factory=list)
    
    # Timestamps
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_minutes(self) -> float:
        """Durée en minutes."""
        return self.duration_seconds / 60.0

    @property
    def avg_tokens_per_turn(self) -> float:
        """Moyenne de tokens par tour."""
        return self.num_tokens / self.num_turns if self.num_turns > 0 else 0.0


class ConversationMetrics:
    """Calcule et suit les métriques de qualité des conversations."""

    def __init__(self) -> None:
        self._metrics_history: List[ConversationStats] = []

    def analyze_conversation(self, convo: List[Tuple[str, str]]) -> ConversationStats:
        """Analyse une conversation et calcule ses métriques."""
        stats = ConversationStats()
        
        # Métriques basiques
        stats.num_turns = len(convo)
        stats.num_tokens = sum(len(text.split()) for _, text in convo)  # simple word count for now
        
        # Engagement & cohérence simplifiée
        stats.coherence_score = self._estimate_coherence(convo)
        stats.engagement_score = self._estimate_engagement(convo)
        
        # Placeholder pour sentiment/émotions/langues
        # TODO: Intégrer des analyseurs dédiés
        stats.user_sentiment = np.random.uniform(-1, 1)
        stats.assistant_sentiment = np.random.uniform(-1, 1)
        stats.detected_emotions = ["neutral"]
        stats.detected_languages = ["fr", "en"]
        
        self._metrics_history.append(stats)
        return stats

    def _estimate_coherence(self, convo: List[Tuple[str, str]]) -> float:
        """Estime la cohérence des réponses (heuristique simple)."""
        if len(convo) < 2:
            return 1.0
        
        # Ratio questions/réponses pertinentes (simpliste)
        question_pattern = re.compile(r"\?|quoi|comment|pourquoi|qui|où|quand", re.IGNORECASE)
        answer_pattern = re.compile(r"\.|!|voici|voilà|effectivement|en effet", re.IGNORECASE)
        
        score = 0
        pairs = zip(convo[:-1], convo[1:])
        for (r1, c1), (r2, c2) in pairs:
            if question_pattern.search(c1) and answer_pattern.search(c2):
                score += 1
        return min(1.0, score / (len(convo) - 1)) if len(convo) > 1 else 1.0

    def _estimate_engagement(self, convo: List[Tuple[str, str]]) -> float:
        """Estime l'engagement (heuristique simple)."""
        if not convo:
            return 0.0
        
        # Moyenne de longueur, présence de questions et variété de réponses
        avg_length = np.mean([len(text.split()) for _, text in convo])
        engagement = min(1.0, avg_length / 50.0)  # normalise vers [0, 1]
        return engagement

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques sur toutes les conversations."""
        if not self._metrics_history:
            return {}
        
        return {
            "total_conversations": len(self._metrics_history),
            "avg_turns": np.mean([s.num_turns for s in self._metrics_history]),
            "avg_coherence": np.mean([s.coherence_score for s in self._metrics_history]),
            "avg_engagement": np.mean([s.engagement_score for s in self._metrics_history]),
            "common_intents": self._most_common([i for s in self._metrics_history for i in s.detected_intents]),
            "common_emotions": self._most_common([e for s in self._metrics_history for e in s.detected_emotions]),
        }

    def _most_common(self, items: List[str], top_k: int = 3) -> List[str]:
        """Retourne les k items les plus fréquents."""
        if not items:
            return []
        counter = {}
        for item in items:
            counter[item] = counter.get(item, 0) + 1
        return sorted(counter, key=counter.get, reverse=True)[:top_k]  # type: ignore