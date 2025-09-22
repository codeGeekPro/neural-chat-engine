"""Types et constantes pour la génération de réponses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel


class PersonalityTrait(str, Enum):
    """Traits de personnalité principaux."""
    
    # Big Five (OCEAN)
    OPENNESS = "openness"  # Curiosité et créativité
    CONSCIENTIOUSNESS = "conscientiousness"  # Organisation et fiabilité
    EXTRAVERSION = "extraversion"  # Sociabilité et énergie
    AGREEABLENESS = "agreeableness"  # Empathie et coopération
    NEUROTICISM = "neuroticism"  # Stabilité émotionnelle
    
    # Traits supplémentaires
    FORMALITY = "formality"  # Niveau de formalité
    HUMOR = "humor"  # Sens de l'humour
    EMPATHY = "empathy"  # Capacité empathique
    EXPERTISE = "expertise"  # Niveau d'expertise
    PROACTIVITY = "proactivity"  # Initiative


class ResponseFormat(str, Enum):
    """Types de réponses supportés."""
    TEXT = "text"  # Texte simple
    MARKDOWN = "markdown"  # Texte formaté
    JSON = "json"  # Structure de données
    ACTION = "action"  # Commande ou action
    SUGGESTION = "suggestion"  # Liste de suggestions


class PersonalityProfile(BaseModel):
    """Profil de personnalité configurable."""
    
    # Traits OCEAN (échelle -1 à 1)
    openness: float = 0.0
    conscientiousness: float = 0.0
    extraversion: float = 0.0
    agreeableness: float = 0.0
    neuroticism: float = 0.0
    
    # Traits supplémentaires (échelle 0 à 1)
    formality: float = 0.5
    humor: float = 0.3
    empathy: float = 0.5
    expertise: float = 0.7
    proactivity: float = 0.5
    
    # Templates et phrases types
    greeting_templates: List[str] = field(default_factory=list)
    closing_templates: List[str] = field(default_factory=list)
    filler_phrases: List[str] = field(default_factory=list)
    
    class Config:
        validate_assignment = True

    def adjust_trait(
        self,
        trait: PersonalityTrait,
        value: float,
        weight: float = 0.5
    ) -> None:
        """Ajuste un trait de personnalité avec lissage."""
        current = getattr(self, trait.value)
        if trait.value in {"openness", "conscientiousness", "extraversion", 
                          "agreeableness", "neuroticism"}:
            # OCEAN: échelle -1 à 1
            new_value = np.clip(
                current * (1 - weight) + value * weight,
                -1.0,
                1.0
            )
        else:
            # Autres traits: échelle 0 à 1
            new_value = np.clip(
                current * (1 - weight) + value * weight,
                0.0,
                1.0
            )
        setattr(self, trait.value, new_value)


@dataclass
class GeneratedResponse:
    """Réponse générée avec métadonnées."""
    
    # Contenu principal
    content: str
    format: ResponseFormat = ResponseFormat.TEXT
    
    # Métadonnées
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    # Contenu additionnel
    suggestions: Optional[List[str]] = None
    actions: Optional[List[Dict[str, Any]]] = None
    
    # Contrôle qualité
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_high_quality(self) -> bool:
        """Vérifie si la réponse est de haute qualité."""
        if not self.quality_metrics:
            return True
        return all(score >= 0.7 for score in self.quality_metrics.values())


@dataclass
class RAGResult:
    """Résultat d'une recherche RAG."""
    
    # Documents retrouvés
    documents: List[Dict[str, Any]]
    scores: List[float]
    
    # Métadonnées
    query_vector: Optional[List[float]] = None
    search_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_best_documents(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Retourne les meilleurs documents au-dessus du seuil."""
        return [
            doc for doc, score in zip(self.documents, self.scores)
            if score >= threshold
        ]

    def get_formatted_context(self) -> str:
        """Formate les documents en contexte pour le LLM."""
        context_parts = []
        for i, (doc, score) in enumerate(zip(self.documents, self.scores)):
            context_parts.append(
                f"[Document {i+1}] (pertinence: {score:.2f})\n"
                f"{doc.get('content', '')}\n"
            )
        return "\n".join(context_parts)