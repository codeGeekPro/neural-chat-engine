"""Moteur de recommandations contextuelles pour le Neural Chat Engine."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import os

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


class RecommendationEngine:
    """Moteur de recommandations contextuelles avec approche hybride."""

    def __init__(self,
                 user_profiles: Dict[str, UserProfile],
                 item_catalog: ItemCatalog,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialise le moteur de recommandations.

        Args:
            user_profiles: Dictionnaire des profils utilisateurs
            item_catalog: Catalogue des éléments
            config: Configuration optionnelle
        """
        self.user_profiles = user_profiles
        self.item_catalog = item_catalog
        self.config = config or self._get_default_config()

        # Modèles de recommandation
        self.models: Dict[RecommendationType, RecommendationModel] = {}

        # Cache pour les calculs coûteux
        self._similarity_cache: Dict[str, Dict[str, float]] = {}
        self._user_vectors_cache: Dict[str, np.ndarray] = {}
        self._item_vectors_cache: Dict[str, np.ndarray] = {}

        # Métriques de performance
        self.performance_metrics = {
            'total_recommendations': 0,
            'cache_hit_rate': 0.0,
            'average_processing_time': 0.0
        }

        # Configuration du logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Initialisation des modèles
        self._initialize_models()

    def _get_default_config(self) -> Dict[str, Any]:
        """Retourne la configuration par défaut."""
        return {
            'max_recommendations': 20,
            'min_similarity_threshold': 0.1,
            'context_weight': 0.3,
            'collaborative_weight': 0.4,
            'content_weight': 0.3,
            'temporal_decay_factor': 0.95,
            'min_interactions_for_similarity': 5,
            'cache_max_size': 10000,
            'model_update_interval_hours': 24,
            'feedback_learning_rate': 0.1,
            'diversity_factor': 0.2,
            'novelty_boost': 0.1
        }

    def _setup_logging(self):
        """Configure le système de logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _initialize_models(self):
        """Initialise les modèles de recommandation."""
        self.logger.info("Initialisation des modèles de recommandation...")

        # Modèle de filtrage collaboratif
        self.models[RecommendationType.COLLABORATIVE] = RecommendationModel(
            model_type=RecommendationType.COLLABORATIVE,
            parameters={'k_neighbors': 10, 'similarity_metric': 'cosine'}
        )

        # Modèle basé sur le contenu
        self.models[RecommendationType.CONTENT_BASED] = RecommendationModel(
            model_type=RecommendationType.CONTENT_BASED,
            parameters={'feature_weights': {'tags': 0.4, 'category': 0.3, 'description': 0.3}}
        )

        # Modèle hybride
        self.models[RecommendationType.HYBRID] = RecommendationModel(
            model_type=RecommendationType.HYBRID,
            parameters={
                'collaborative_weight': self.config['collaborative_weight'],
                'content_weight': self.config['content_weight'],
                'context_weight': self.config['context_weight']
            }
        )

        # Modèle contextuel
        self.models[RecommendationType.CONTEXTUAL] = RecommendationModel(
            model_type=RecommendationType.CONTEXTUAL,
            parameters={'context_features': ['time', 'location', 'conversation']}
        )

        self.logger.info(f"Modèles initialisés: {list(self.models.keys())}")

    def analyze_conversation_context(self, current_conversation: str) -> ContextSnapshot:
        """
        Analyse le contexte de la conversation actuelle.

        Args:
            current_conversation: Texte de la conversation actuelle

        Returns:
            ContextSnapshot: Snapshot du contexte analysé
        """
        self.logger.debug("Analyse du contexte de conversation...")

        # Analyse temporelle
        now = datetime.now()
        time_of_day = f"{now.hour:02d}:{now.minute:02d}"
        day_of_week = now.strftime('%A')
        season = self._get_season(now)

        # Analyse du contenu de la conversation
        conversation_features = self._extract_conversation_features(current_conversation)

        context = ContextSnapshot(
            context_type=ContextType.CONVERSATION,
            features=conversation_features,
            timestamp=now,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            season=season,
            conversation_context=current_conversation
        )

        self.logger.debug(f"Contexte analysé: {len(conversation_features)} features extraites")
        return context

    def _extract_conversation_features(self, conversation: str) -> Dict[str, Any]:
        """Extrait les features du texte de conversation."""
        # Analyse simplifiée - peut être étendue avec NLP
        features = {
            'length': len(conversation),
            'word_count': len(conversation.split()),
            'has_questions': '?' in conversation,
            'has_exclamation': '!' in conversation,
            'sentiment_indicators': self._detect_sentiment_indicators(conversation),
            'topics': self._extract_topics(conversation),
            'urgency_level': self._detect_urgency(conversation)
        }
        return features

    def _detect_sentiment_indicators(self, text: str) -> Dict[str, float]:
        """Détecte les indicateurs de sentiment."""
        positive_words = ['bon', 'bien', 'excellent', 'super', 'génial', 'parfait']
        negative_words = ['mauvais', 'mal', 'terrible', 'horrible', 'nul', 'déçu']

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

        return {
            'positive': positive_count / total_sentiment_words,
            'negative': negative_count / total_sentiment_words,
            'neutral': (len(words) - total_sentiment_words) / len(words)
        }

    def _extract_topics(self, text: str) -> List[str]:
        """Extrait les topics principaux du texte."""
        # Implémentation simplifiée - peut utiliser des modèles NLP avancés
        topic_keywords = {
            'technology': ['ordinateur', 'logiciel', 'internet', 'app', 'application'],
            'food': ['manger', 'restaurant', 'cuisine', 'recette', 'plat'],
            'travel': ['voyage', 'destination', 'hôtel', 'avion', 'train'],
            'entertainment': ['film', 'musique', 'livre', 'jeu', 'divertissement'],
            'shopping': ['acheter', 'prix', 'magasin', 'produit', 'marque']
        }

        text_lower = text.lower()
        detected_topics = []

        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)

        return detected_topics[:3]  # Maximum 3 topics

    def _detect_urgency(self, text: str) -> float:
        """Détecte le niveau d'urgence dans le texte."""
        urgency_indicators = ['urgent', 'vite', 'rapidement', 'immédiatement', 'asap']
        text_lower = text.lower()

        urgency_score = sum(1 for indicator in urgency_indicators if indicator in text_lower)
        return min(urgency_score / 3.0, 1.0)  # Normalisation

    def _get_season(self, date: datetime) -> str:
        """Détermine la saison basée sur la date."""
        month = date.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

    def generate_proactive_suggestions(self,
                                     user_profile: UserProfile,
                                     context: Optional[ContextSnapshot] = None,
                                     max_suggestions: int = 10) -> RecommendationResult:
        """
        Génère des suggestions proactives basées sur le profil et le contexte.

        Args:
            user_profile: Profil de l'utilisateur
            context: Contexte actuel (optionnel)
            max_suggestions: Nombre maximum de suggestions

        Returns:
            RecommendationResult: Résultats des recommandations
        """
        start_time = datetime.now()

        self.logger.info(f"Génération de suggestions pour l'utilisateur {user_profile.user_id}")

        # Analyse du contexte si non fourni
        if context is None:
            context = ContextSnapshot(context_type=ContextType.BEHAVIORAL)

        # Récupération des éléments candidats
        candidate_items = self._get_candidate_items(user_profile, context)

        # Calcul des scores de recommandation
        recommendations = []
        for item in candidate_items:
            score = self._calculate_hybrid_score(user_profile, item, context)
            if score > self.config['min_similarity_threshold']:
                recommendation = Recommendation(
                    item_id=item.item_id,
                    score=score,
                    recommendation_type=RecommendationType.HYBRID,
                    confidence=self._calculate_confidence(score, user_profile),
                    explanation=self._generate_explanation(user_profile, item, context),
                    context_used=context,
                    similar_items=self._find_similar_items(item.item_id, limit=3)
                )
                recommendations.append(recommendation)

        # Tri et limitation des recommandations
        recommendations.sort(key=lambda x: x.score, reverse=True)
        recommendations = recommendations[:max_suggestions]

        # Application de la diversité
        recommendations = self._apply_diversity(recommendations)

        processing_time = (datetime.now() - start_time).total_seconds()

        result = RecommendationResult(
            user_id=user_profile.user_id,
            recommendations=recommendations,
            context_used=context,
            processing_time=processing_time,
            algorithm_used="hybrid_contextual",
            total_candidates=len(candidate_items)
        )

        self._update_performance_metrics(processing_time)
        self.logger.info(f"Généré {len(recommendations)} suggestions en {processing_time:.2f}s")

        return result

    def _get_candidate_items(self,
                           user_profile: UserProfile,
                           context: ContextSnapshot) -> List[Item]:
        """Récupère les éléments candidats pour les recommandations."""
        all_items = list(self.item_catalog.items.values())

        # Filtrage basé sur l'historique utilisateur
        interacted_items = {interaction.item_id for interaction in user_profile.interaction_history}
        candidate_items = [item for item in all_items if item.item_id not in interacted_items]

        # Filtrage contextuel
        if context.context_type == ContextType.CONVERSATION and context.conversation_context:
            topics = context.features.get('topics', [])
            if topics:
                # Prioriser les éléments liés aux topics de conversation
                topic_filtered = []
                for item in candidate_items:
                    item_topics = set(item.tags) & set(topics)
                    if item_topics:
                        topic_filtered.append(item)
                if topic_filtered:
                    candidate_items = topic_filtered

        # Limitation du nombre de candidats pour la performance
        max_candidates = min(len(candidate_items), self.config['max_recommendations'] * 5)
        return candidate_items[:max_candidates]

    def _calculate_hybrid_score(self,
                               user_profile: UserProfile,
                               item: Item,
                               context: ContextSnapshot) -> float:
        """Calcule le score hybride pour un élément."""
        # Score collaboratif
        collaborative_score = self._calculate_collaborative_score(user_profile, item)

        # Score basé sur le contenu
        content_score = self._calculate_content_score(user_profile, item)

        # Score contextuel
        context_score = self._calculate_context_score(user_profile, item, context)

        # Combinaison hybride
        weights = self.models[RecommendationType.HYBRID].parameters
        hybrid_score = (
            collaborative_score * weights['collaborative_weight'] +
            content_score * weights['content_weight'] +
            context_score * weights['context_weight']
        )

        # Application de la décroissance temporelle
        temporal_factor = self._calculate_temporal_factor(user_profile, item)
        final_score = hybrid_score * temporal_factor

        return max(0.0, min(1.0, final_score))

    def _calculate_collaborative_score(self, user_profile: UserProfile, item: Item) -> float:
        """Calcule le score collaboratif."""
        if user_profile.user_id not in self._user_vectors_cache:
            self._build_user_similarity_matrix()

        if user_profile.user_id not in self._user_vectors_cache:
            return 0.0

        user_vector = self._user_vectors_cache[user_profile.user_id]

        # Trouver les utilisateurs similaires
        similar_users = self._find_similar_users(user_profile.user_id, k=10)

        if not similar_users:
            return 0.0

        # Calculer le score basé sur les interactions des utilisateurs similaires
        collaborative_score = 0.0
        total_weight = 0.0

        for similar_user_id, similarity in similar_users:
            if similar_user_id in self.user_profiles:
                similar_profile = self.user_profiles[similar_user_id]
                interaction_score = self._get_user_item_interaction_score(similar_profile, item.item_id)
                if interaction_score > 0:
                    collaborative_score += interaction_score * similarity
                    total_weight += similarity

        return collaborative_score / total_weight if total_weight > 0 else 0.0

    def _calculate_content_score(self, user_profile: UserProfile, item: Item) -> float:
        """Calcule le score basé sur le contenu."""
        if not user_profile.preferences:
            return item.popularity_score

        # Calcul de similarité basé sur les préférences utilisateur
        content_score = 0.0
        total_weight = 0.0

        for preferred_item_id, preference_score in user_profile.preferences.items():
            preferred_item = self.item_catalog.get_item(preferred_item_id)
            if preferred_item:
                similarity = item.get_similarity_score(preferred_item)
                content_score += similarity * preference_score
                total_weight += preference_score

        if total_weight > 0:
            content_score /= total_weight

        # Boost basé sur la popularité
        popularity_boost = item.popularity_score * 0.1
        content_score += popularity_boost

        return min(1.0, content_score)

    def _calculate_context_score(self,
                               user_profile: UserProfile,
                               item: Item,
                               context: ContextSnapshot) -> float:
        """Calcule le score contextuel."""
        context_score = 0.0

        if context.context_type == ContextType.CONVERSATION:
            # Score basé sur les topics de conversation
            conversation_topics = context.features.get('topics', [])
            item_topics = set(item.tags)

            if conversation_topics and item_topics:
                topic_overlap = len(set(conversation_topics) & item_topics)
                context_score += topic_overlap / len(conversation_topics)

            # Score basé sur le sentiment
            sentiment = context.features.get('sentiment_indicators', {})
            if sentiment.get('positive', 0) > 0.5:
                # Recommander des éléments positifs
                context_score += 0.2

        elif context.context_type == ContextType.TIME_BASED:
            # Score basé sur l'heure de la journée
            if context.time_of_day:
                hour = int(context.time_of_day.split(':')[0])
                if 6 <= hour <= 12:  # Matinée
                    if 'breakfast' in item.tags or 'morning' in item.tags:
                        context_score += 0.3
                elif 12 <= hour <= 17:  # Après-midi
                    if 'lunch' in item.tags or 'afternoon' in item.tags:
                        context_score += 0.3
                elif 18 <= hour <= 22:  # Soirée
                    if 'dinner' in item.tags or 'evening' in item.tags:
                        context_score += 0.3

        return min(1.0, context_score)

    def _calculate_temporal_factor(self, user_profile: UserProfile, item: Item) -> float:
        """Calcule le facteur de décroissance temporelle."""
        # Vérifier si l'utilisateur a récemment interagi avec des éléments similaires
        recent_interactions = user_profile.get_recent_interactions(limit=20)

        similar_recent_items = []
        for interaction in recent_interactions:
            interacted_item = self.item_catalog.get_item(interaction.item_id)
            if interacted_item and interacted_item.get_similarity_score(item) > 0.5:
                similar_recent_items.append(interaction)

        if not similar_recent_items:
            return 1.0

        # Calculer la décroissance basée sur le temps écoulé
        most_recent = max(similar_recent_items, key=lambda x: x.timestamp)
        hours_since = (datetime.now() - most_recent.timestamp).total_seconds() / 3600

        # Décroissance exponentielle
        decay = self.config['temporal_decay_factor'] ** (hours_since / 24)
        return decay

    def _find_similar_users(self, user_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """Trouve les utilisateurs similaires."""
        if user_id not in self._user_vectors_cache:
            return []

        user_vector = self._user_vectors_cache[user_id]
        similarities = []

        for other_user_id, other_vector in self._user_vectors_cache.items():
            if other_user_id != user_id:
                similarity = cosine_similarity([user_vector], [other_vector])[0][0]
                similarities.append((other_user_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def _get_user_item_interaction_score(self, user_profile: UserProfile, item_id: str) -> float:
        """Calcule le score d'interaction utilisateur-élément."""
        interactions = [i for i in user_profile.interaction_history if i.item_id == item_id]

        if not interactions:
            return 0.0

        # Calcul du score basé sur le type et la fréquence des interactions
        score = 0.0
        for interaction in interactions:
            if interaction.interaction_type == InteractionType.LIKE:
                score += 1.0
            elif interaction.interaction_type == InteractionType.PURCHASE:
                score += 2.0
            elif interaction.interaction_type == InteractionType.SHARE:
                score += 1.5
            elif interaction.interaction_type == InteractionType.RATING and interaction.rating:
                score += interaction.rating / 5.0  # Normalisation
            elif interaction.interaction_type == InteractionType.VIEW:
                score += 0.5
            elif interaction.interaction_type == InteractionType.DISLIKE:
                score -= 0.5

        return max(0.0, min(2.0, score / len(interactions)))

    def _build_user_similarity_matrix(self):
        """Construit la matrice de similarité utilisateurs."""
        self.logger.debug("Construction de la matrice de similarité utilisateurs...")

        # Création des vecteurs utilisateur basés sur les interactions
        user_item_matrix = defaultdict(lambda: defaultdict(float))

        for user_profile in self.user_profiles.values():
            for interaction in user_profile.interaction_history:
                score = self._get_interaction_weight(interaction)
                user_item_matrix[user_profile.user_id][interaction.item_id] = score

        # Conversion en matrice numpy
        all_users = list(user_item_matrix.keys())
        all_items = list(set(item for user_items in user_item_matrix.values() for item in user_items))

        if not all_users or not all_items:
            return

        matrix = np.zeros((len(all_users), len(all_items)))

        for i, user_id in enumerate(all_users):
            for j, item_id in enumerate(all_items):
                matrix[i, j] = user_item_matrix[user_id].get(item_id, 0.0)

        # Normalisation
        if matrix.shape[0] > 1:
            scaler = StandardScaler()
            matrix_normalized = scaler.fit_transform(matrix)

            # Réduction de dimensionnalité si nécessaire
            if matrix.shape[1] > 50:
                pca = PCA(n_components=min(50, matrix.shape[0], matrix.shape[1]))
                matrix_normalized = pca.fit_transform(matrix_normalized)

            # Mise en cache des vecteurs
            for i, user_id in enumerate(all_users):
                self._user_vectors_cache[user_id] = matrix_normalized[i]

        self.logger.debug(f"Matrice construite pour {len(all_users)} utilisateurs")

    def _get_interaction_weight(self, interaction: UserInteraction) -> float:
        """Calcule le poids d'une interaction."""
        base_weights = {
            InteractionType.VIEW: 0.1,
            InteractionType.LIKE: 0.5,
            InteractionType.COMMENT: 0.7,
            InteractionType.SHARE: 0.8,
            InteractionType.PURCHASE: 1.0,
            InteractionType.RATING: 0.6,
            InteractionType.SAVE: 0.4,
            InteractionType.DISLIKE: -0.3,
            InteractionType.SKIP: -0.1
        }

        weight = base_weights.get(interaction.interaction_type, 0.0)

        # Ajustement basé sur la durée si disponible
        if interaction.duration and interaction.interaction_type == InteractionType.VIEW:
            weight *= min(interaction.duration / 60, 2.0)  # Max 2x pour 2 minutes

        # Ajustement basé sur la note si disponible
        if interaction.rating and interaction.interaction_type == InteractionType.RATING:
            weight *= interaction.rating / 5.0

        return weight

    def learn_user_preferences(self,
                             user_interactions: List[UserInteraction],
                             feedback: Optional[List[FeedbackData]] = None):
        """
        Apprentissage des préférences utilisateur.

        Args:
            user_interactions: Nouvelles interactions utilisateur
            feedback: Données de feedback (optionnel)
        """
        self.logger.info(f"Apprentissage des préférences pour {len(user_interactions)} interactions")

        # Mise à jour des profils utilisateur
        user_updates = defaultdict(list)
        for interaction in user_interactions:
            user_updates[interaction.user_id].append(interaction)

        for user_id, interactions in user_updates.items():
            if user_id in self.user_profiles:
                user_profile = self.user_profiles[user_id]
                for interaction in interactions:
                    user_profile.add_interaction(interaction)

                # Mise à jour des préférences
                self._update_user_preferences(user_profile)

        # Apprentissage basé sur le feedback
        if feedback:
            self._learn_from_feedback(feedback)

        # Reconstruction des matrices de similarité si nécessaire
        if len(user_interactions) > self.config['min_interactions_for_similarity']:
            self._user_vectors_cache.clear()
            self._build_user_similarity_matrix()

        self.logger.info("Apprentissage des préférences terminé")

    def _update_user_preferences(self, user_profile: UserProfile):
        """Met à jour les préférences d'un utilisateur."""
        preferences = defaultdict(float)
        total_weight = 0.0

        for interaction in user_profile.interaction_history[-100:]:  # Dernières 100 interactions
            weight = self._get_interaction_weight(interaction)
            if weight > 0:
                preferences[interaction.item_id] += weight
                total_weight += weight

        # Normalisation
        if total_weight > 0:
            for item_id in preferences:
                preferences[item_id] /= total_weight

        # Mise à jour du profil
        user_profile.preferences = dict(preferences)
        user_profile.last_updated = datetime.now()

    def _learn_from_feedback(self, feedback_data: List[FeedbackData]):
        """Apprentissage basé sur les données de feedback."""
        for feedback in feedback_data:
            if feedback.user_id in self.user_profiles:
                user_profile = self.user_profiles[feedback.user_id]

                # Ajustement des préférences basé sur le feedback
                learning_rate = self.config['feedback_learning_rate']

                if feedback.feedback_type == "accepted":
                    adjustment = learning_rate
                elif feedback.feedback_type == "rejected":
                    adjustment = -learning_rate
                elif feedback.feedback_type == "purchased":
                    adjustment = learning_rate * 2
                else:
                    adjustment = feedback.feedback_score * learning_rate

                current_pref = user_profile.preferences.get(feedback.item_id, 0.5)
                new_pref = max(0.0, min(1.0, current_pref + adjustment))

                user_profile.update_preferences(feedback.item_id, new_pref)

    def update_recommendation_models(self, new_interaction_data: List[UserInteraction]):
        """
        Met à jour les modèles de recommandation avec de nouvelles données.

        Args:
            new_interaction_data: Nouvelles données d'interaction
        """
        self.logger.info(f"Mise à jour des modèles avec {len(new_interaction_data)} nouvelles interactions")

        # Apprentissage des préférences
        self.learn_user_preferences(new_interaction_data)

        # Mise à jour des métriques des éléments
        self._update_item_metrics(new_interaction_data)

        # Reconstruction des caches si nécessaire
        cache_size = len(self._user_vectors_cache) + len(self._item_vectors_cache)
        if cache_size > self.config['cache_max_size']:
            self._clear_old_cache_entries()

        # Mise à jour des timestamps des modèles
        for model in self.models.values():
            model.trained_at = datetime.now()

        self.logger.info("Modèles de recommandation mis à jour")

    def _update_item_metrics(self, interactions: List[UserInteraction]):
        """Met à jour les métriques des éléments."""
        item_stats = defaultdict(lambda: {'views': 0, 'likes': 0, 'purchases': 0, 'ratings': [], 'total': 0})

        for interaction in interactions:
            stats = item_stats[interaction.item_id]
            stats['total'] += 1

            if interaction.interaction_type == InteractionType.VIEW:
                stats['views'] += 1
            elif interaction.interaction_type == InteractionType.LIKE:
                stats['likes'] += 1
            elif interaction.interaction_type == InteractionType.PURCHASE:
                stats['purchases'] += 1
            elif interaction.interaction_type == InteractionType.RATING and interaction.rating:
                stats['ratings'].append(interaction.rating)

        # Mise à jour des éléments du catalogue
        for item_id, stats in item_stats.items():
            item = self.item_catalog.get_item(item_id)
            if item:
                # Calcul du score de popularité
                popularity = (
                    stats['views'] * 0.1 +
                    stats['likes'] * 0.3 +
                    stats['purchases'] * 0.6
                )

                if stats['ratings']:
                    avg_rating = sum(stats['ratings']) / len(stats['ratings'])
                    item.average_rating = avg_rating
                    popularity += avg_rating * 0.2

                item.popularity_score = min(1.0, popularity / 10.0)  # Normalisation
                item.interaction_count += stats['total']

    def _clear_old_cache_entries(self):
        """Nettoie les anciennes entrées du cache."""
        # Implémentation simplifiée - garder seulement les plus récents
        max_entries = self.config['cache_max_size'] // 2

        # Nettoyer le cache utilisateurs (LRU simplifié)
        if len(self._user_vectors_cache) > max_entries:
            # Garder les utilisateurs les plus actifs
            active_users = set()
            for user_profile in self.user_profiles.values():
                if len(user_profile.interaction_history) > 10:
                    active_users.add(user_profile.user_id)

            self._user_vectors_cache = {
                uid: vector for uid, vector in self._user_vectors_cache.items()
                if uid in active_users
            }

    def explain_recommendations(self,
                              recommended_items: List[Recommendation],
                              user_context: Optional[ContextSnapshot] = None) -> Dict[str, str]:
        """
        Génère des explications pour les recommandations.

        Args:
            recommended_items: Liste des recommandations
            user_context: Contexte utilisateur

        Returns:
            Dict[str, str]: Explications par élément
        """
        explanations = {}

        for recommendation in recommended_items:
            item = self.item_catalog.get_item(recommendation.item_id)
            if not item:
                continue

            explanation_parts = []

            # Explication basée sur le type de recommandation
            if recommendation.recommendation_type == RecommendationType.COLLABORATIVE:
                explanation_parts.append("Recommandé car des utilisateurs similaires l'ont apprécié")
            elif recommendation.recommendation_type == RecommendationType.CONTENT_BASED:
                explanation_parts.append("Similaire à ce que vous avez aimé auparavant")
            elif recommendation.recommendation_type == RecommendationType.CONTEXTUAL:
                if user_context and user_context.conversation_context:
                    explanation_parts.append("Pertinent pour votre conversation actuelle")
            else:  # HYBRID
                explanation_parts.append("Combinaison de vos préférences et de celles d'autres utilisateurs")

            # Explication basée sur les features
            if recommendation.similar_items:
                similar_names = []
                for similar_id in recommendation.similar_items[:2]:
                    similar_item = self.item_catalog.get_item(similar_id)
                    if similar_item:
                        similar_names.append(similar_item.title)

                if similar_names:
                    explanation_parts.append(f"Similaire à: {', '.join(similar_names)}")

            # Explication contextuelle
            if user_context:
                if user_context.time_of_day:
                    hour = int(user_context.time_of_day.split(':')[0])
                    if 6 <= hour <= 12 and 'breakfast' in item.tags:
                        explanation_parts.append("Idéal pour le matin")
                    elif 18 <= hour <= 22 and 'dinner' in item.tags:
                        explanation_parts.append("Parfait pour le dîner")

            explanations[recommendation.item_id] = ". ".join(explanation_parts)

        return explanations

    def _calculate_confidence(self, score: float, user_profile: UserProfile) -> float:
        """Calcule le niveau de confiance d'une recommandation."""
        # Facteurs influençant la confiance
        interaction_count = len(user_profile.interaction_history)
        preference_strength = sum(user_profile.preferences.values()) / max(len(user_profile.preferences), 1)

        # Confiance basée sur la quantité de données
        data_confidence = min(interaction_count / 50, 1.0)  # Max à 50 interactions

        # Confiance basée sur la force des préférences
        preference_confidence = min(preference_strength, 1.0)

        # Combinaison
        confidence = (score * 0.6 + data_confidence * 0.2 + preference_confidence * 0.2)

        return min(1.0, confidence)

    def _generate_explanation(self,
                           user_profile: UserProfile,
                           item: Item,
                           context: ContextSnapshot) -> str:
        """Génère une explication pour une recommandation."""
        explanations = self.explain_recommendations([Recommendation(
            item_id=item.item_id,
            score=0.0,
            recommendation_type=RecommendationType.HYBRID,
            context_used=context
        )], context)

        return explanations.get(item.item_id, "Recommandation personnalisée")

    def _find_similar_items(self, item_id: str, limit: int = 5) -> List[str]:
        """Trouve les éléments similaires."""
        if item_id not in self._similarity_cache:
            self._build_item_similarity_cache()

        if item_id in self._similarity_cache:
            similarities = self._similarity_cache[item_id]
            sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            return [item_id for item_id, score in sorted_items[:limit] if score > 0.1]

        return []

    def _build_item_similarity_cache(self):
        """Construit le cache de similarité des éléments."""
        self.logger.debug("Construction du cache de similarité des éléments...")

        items = list(self.item_catalog.items.values())
        self._similarity_cache = {}

        for i, item1 in enumerate(items):
            similarities = {}
            for j, item2 in enumerate(items):
                if i != j:
                    similarity = item1.get_similarity_score(item2)
                    if similarity > 0.1:  # Seuil minimum
                        similarities[item2.item_id] = similarity

            self._similarity_cache[item1.item_id] = similarities

        self.logger.debug(f"Cache construit pour {len(items)} éléments")

    def _apply_diversity(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Applique la diversité aux recommandations."""
        if len(recommendations) <= 1:
            return recommendations

        diversity_factor = self.config['diversity_factor']

        # Calcul des scores de diversité
        diverse_scores = []
        for i, rec1 in enumerate(recommendations):
            diversity_score = 0.0
            for j, rec2 in enumerate(recommendations):
                if i != j:
                    item1 = self.item_catalog.get_item(rec1.item_id)
                    item2 = self.item_catalog.get_item(rec2.item_id)
                    if item1 and item2:
                        similarity = item1.get_similarity_score(item2)
                        diversity_score += (1 - similarity)

            diversity_score /= max(len(recommendations) - 1, 1)
            diverse_scores.append(diversity_score)

        # Combinaison des scores original et diversité
        for i, rec in enumerate(recommendations):
            original_score = rec.score
            diversity_bonus = diverse_scores[i] * diversity_factor
            rec.score = original_score * (1 - diversity_factor) + diversity_bonus

        # Re-tri après application de la diversité
        recommendations.sort(key=lambda x: x.score, reverse=True)

        return recommendations

    def _update_performance_metrics(self, processing_time: float):
        """Met à jour les métriques de performance."""
        self.performance_metrics['total_recommendations'] += 1
        current_avg = self.performance_metrics['average_processing_time']
        total_count = self.performance_metrics['total_recommendations']

        # Moyenne glissante
        self.performance_metrics['average_processing_time'] = (
            (current_avg * (total_count - 1)) + processing_time
        ) / total_count

    def get_performance_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de performance."""
        return {
            **self.performance_metrics,
            'cache_size': len(self._user_vectors_cache) + len(self._item_vectors_cache),
            'total_users': len(self.user_profiles),
            'total_items': len(self.item_catalog.items),
            'models_status': {
                model_type.value: {
                    'trained_at': model.trained_at.isoformat(),
                    'is_expired': model.is_expired()
                }
                for model_type, model in self.models.items()
            }
        }

    def save_models(self, directory: str = "models/recommendations"):
        """Sauvegarde les modèles."""
        os.makedirs(directory, exist_ok=True)

        # Sauvegarde des modèles
        for model_type, model in self.models.items():
            model_path = os.path.join(directory, f"{model_type.value}_model.json")
            model_data = {
                'model_type': model_type.value,
                'parameters': model.parameters,
                'trained_at': model.trained_at.isoformat(),
                'performance_metrics': model.performance_metrics,
                'feature_importance': model.feature_importance
            }

            with open(model_path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Modèles sauvegardés dans {directory}")

    def load_models(self, directory: str = "models/recommendations"):
        """Charge les modèles."""
        if not os.path.exists(directory):
            self.logger.warning(f"Répertoire de modèles non trouvé: {directory}")
            return

        for model_file in os.listdir(directory):
            if model_file.endswith('_model.json'):
                model_path = os.path.join(directory, model_file)
                try:
                    with open(model_path, 'r', encoding='utf-8') as f:
                        model_data = json.load(f)

                    model_type = RecommendationType(model_data['model_type'])
                    self.models[model_type] = RecommendationModel(
                        model_type=model_type,
                        parameters=model_data['parameters'],
                        trained_at=datetime.fromisoformat(model_data['trained_at']),
                        performance_metrics=model_data.get('performance_metrics', {}),
                        feature_importance=model_data.get('feature_importance', {})
                    )

                except Exception as e:
                    self.logger.error(f"Erreur lors du chargement du modèle {model_file}: {e}")

        self.logger.info(f"Modèles chargés depuis {directory}")