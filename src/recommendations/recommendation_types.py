"""Types de données pour le système de recommandations."""

from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np


class RecommendationType(Enum):
    """Types de recommandations disponibles."""
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"
    POPULARITY = "popularity"


class InteractionType(Enum):
    """Types d'interactions utilisateur."""
    VIEW = "view"
    LIKE = "like"
    DISLIKE = "dislike"
    PURCHASE = "purchase"
    SHARE = "share"
    COMMENT = "comment"
    SAVE = "save"
    SKIP = "skip"
    RATING = "rating"


class ContextType(Enum):
    """Types de contexte pour les recommandations."""
    CONVERSATION = "conversation"
    TIME_BASED = "time_based"
    LOCATION_BASED = "location_based"
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    SITUATIONAL = "situational"


@dataclass
class UserProfile:
    """Profil utilisateur pour les recommandations."""
    user_id: str
    demographics: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, float] = field(default_factory=dict)
    interaction_history: List['UserInteraction'] = field(default_factory=list)
    behavioral_patterns: Dict[str, Any] = field(default_factory=dict)
    context_history: List['ContextSnapshot'] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def add_interaction(self, interaction: 'UserInteraction'):
        """Ajoute une interaction à l'historique."""
        self.interaction_history.append(interaction)
        self.last_updated = datetime.now()

    def update_preferences(self, item_id: str, preference_score: float):
        """Met à jour les préférences pour un élément."""
        self.preferences[item_id] = preference_score
        self.last_updated = datetime.now()

    def get_recent_interactions(self, limit: int = 10) -> List['UserInteraction']:
        """Retourne les interactions récentes."""
        return sorted(self.interaction_history, key=lambda x: x.timestamp, reverse=True)[:limit]


@dataclass
class Item:
    """Élément du catalogue (produit, contenu, etc.)."""
    item_id: str
    title: str
    description: str
    category: str
    tags: Set[str] = field(default_factory=set)
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    popularity_score: float = 0.0
    average_rating: float = 0.0
    interaction_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def add_tag(self, tag: str):
        """Ajoute un tag à l'élément."""
        self.tags.add(tag)

    def update_popularity(self, new_score: float):
        """Met à jour le score de popularité."""
        self.popularity_score = new_score

    def get_similarity_score(self, other_item: 'Item') -> float:
        """Calcule le score de similarité avec un autre élément."""
        # Similarité basée sur les tags communs
        if not self.tags or not other_item.tags:
            return 0.0

        common_tags = self.tags.intersection(other_item.tags)
        union_tags = self.tags.union(other_item.tags)

        if not union_tags:
            return 0.0

        return len(common_tags) / len(union_tags)


@dataclass
class UserInteraction:
    """Interaction utilisateur avec un élément."""
    user_id: str
    item_id: str
    interaction_type: InteractionType
    timestamp: datetime = field(default_factory=datetime.now)
    context: Optional['ContextSnapshot'] = None
    rating: Optional[float] = None
    duration: Optional[float] = None  # en secondes
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'interaction en dictionnaire."""
        return {
            'user_id': self.user_id,
            'item_id': self.item_id,
            'interaction_type': self.interaction_type.value,
            'timestamp': self.timestamp.isoformat(),
            'rating': self.rating,
            'duration': self.duration,
            'metadata': self.metadata
        }


@dataclass
class ContextSnapshot:
    """Snapshot du contexte utilisateur."""
    context_type: ContextType
    features: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    location: Optional[Dict[str, float]] = None  # lat, lon
    time_of_day: Optional[str] = None
    day_of_week: Optional[str] = None
    season: Optional[str] = None
    weather: Optional[str] = None
    device_type: Optional[str] = None
    conversation_context: Optional[str] = None

    def get_context_vector(self) -> np.ndarray:
        """Retourne un vecteur numérique représentant le contexte."""
        # Implémentation simplifiée - à étendre selon les besoins
        vector = []

        # Encodage du type de contexte
        context_types = list(ContextType)
        type_vector = [1 if ct == self.context_type else 0 for ct in context_types]
        vector.extend(type_vector)

        # Encodage temporel
        if self.time_of_day:
            hour = int(self.time_of_day.split(':')[0])
            time_vector = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
            vector.extend(time_vector)

        # Normalisation et retour
        return np.array(vector) if vector else np.array([])


@dataclass
class Recommendation:
    """Recommandation générée par le système."""
    item_id: str
    score: float
    recommendation_type: RecommendationType
    confidence: float = 1.0
    explanation: str = ""
    context_used: Optional[ContextSnapshot] = None
    similar_items: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit la recommandation en dictionnaire."""
        return {
            'item_id': self.item_id,
            'score': self.score,
            'recommendation_type': self.recommendation_type.value,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'similar_items': self.similar_items,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RecommendationResult:
    """Résultat d'une requête de recommandations."""
    user_id: str
    recommendations: List[Recommendation]
    context_used: Optional[ContextSnapshot] = None
    processing_time: float = 0.0
    algorithm_used: str = ""
    total_candidates: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def get_top_recommendations(self, limit: int = 10) -> List[Recommendation]:
        """Retourne les meilleures recommandations."""
        return sorted(self.recommendations, key=lambda x: x.score, reverse=True)[:limit]

    def get_recommendations_by_type(self, rec_type: RecommendationType) -> List[Recommendation]:
        """Retourne les recommandations d'un type spécifique."""
        return [r for r in self.recommendations if r.recommendation_type == rec_type]


@dataclass
class RecommendationModel:
    """Modèle de recommandation entraîné."""
    model_type: RecommendationType
    parameters: Dict[str, Any] = field(default_factory=dict)
    trained_at: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_data: Optional[Any] = None  # Pour stocker le modèle entraîné

    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Vérifie si le modèle est expiré."""
        age = datetime.now() - self.trained_at
        return age.total_seconds() > (max_age_hours * 3600)

    def update_performance(self, metrics: Dict[str, float]):
        """Met à jour les métriques de performance."""
        self.performance_metrics.update(metrics)


@dataclass
class ItemCatalog:
    """Catalogue d'éléments pour les recommandations."""
    items: Dict[str, Item] = field(default_factory=dict)
    categories: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    last_updated: datetime = field(default_factory=datetime.now)

    def add_item(self, item: Item):
        """Ajoute un élément au catalogue."""
        self.items[item.item_id] = item
        self.categories.add(item.category)
        self.tags.update(item.tags)
        self.last_updated = datetime.now()

    def get_item(self, item_id: str) -> Optional[Item]:
        """Récupère un élément par son ID."""
        return self.items.get(item_id)

    def get_items_by_category(self, category: str) -> List[Item]:
        """Retourne les éléments d'une catégorie."""
        return [item for item in self.items.values() if item.category == category]

    def get_items_by_tag(self, tag: str) -> List[Item]:
        """Retourne les éléments ayant un tag spécifique."""
        return [item for item in self.items.values() if tag in item.tags]

    def search_similar_items(self, query_item: Item, limit: int = 10) -> List[tuple]:
        """Recherche les éléments similaires."""
        similarities = []
        for item in self.items.values():
            if item.item_id != query_item.item_id:
                score = query_item.get_similarity_score(item)
                similarities.append((item, score))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:limit]


@dataclass
class FeedbackData:
    """Données de feedback pour l'amélioration des recommandations."""
    user_id: str
    item_id: str
    recommended_at: datetime
    feedback_type: str  # "accepted", "rejected", "viewed", "purchased"
    feedback_score: float  # -1.0 à 1.0
    context: Optional[ContextSnapshot] = None
    timestamp: datetime = field(default_factory=datetime.now)
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit le feedback en dictionnaire."""
        return {
            'user_id': self.user_id,
            'item_id': self.item_id,
            'recommended_at': self.recommended_at.isoformat(),
            'feedback_type': self.feedback_type,
            'feedback_score': self.feedback_score,
            'timestamp': self.timestamp.isoformat(),
            'additional_data': self.additional_data
        }