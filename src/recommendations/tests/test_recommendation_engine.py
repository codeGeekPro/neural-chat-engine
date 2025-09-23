"""Tests pour le système de recommandations."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.recommendations.recommendation_engine import RecommendationEngine
from src.recommendations.recommendation_types import (
    UserProfile,
    Item,
    ItemCatalog,
    UserInteraction,
    ContextSnapshot,
    Recommendation,
    RecommendationResult,
    RecommendationType,
    InteractionType,
    ContextType,
    FeedbackData
)


@pytest.fixture
def sample_user_profiles():
    """Fixture pour créer des profils utilisateur d'exemple."""
    profiles = {}

    # Utilisateur 1 - Préférences tech
    user1 = UserProfile(
        user_id="user1",
        demographics={"age": 25, "gender": "M", "location": "Paris"},
        preferences={"item1": 0.8, "item2": 0.6, "item3": 0.4}
    )
    user1.add_interaction(UserInteraction(
        user_id="user1",
        item_id="item1",
        interaction_type=InteractionType.LIKE,
        timestamp=datetime.now() - timedelta(hours=1)
    ))
    profiles["user1"] = user1

    # Utilisateur 2 - Préférences food
    user2 = UserProfile(
        user_id="user2",
        demographics={"age": 30, "gender": "F", "location": "Lyon"},
        preferences={"item4": 0.9, "item5": 0.7}
    )
    user2.add_interaction(UserInteraction(
        user_id="user2",
        item_id="item4",
        interaction_type=InteractionType.PURCHASE,
        timestamp=datetime.now() - timedelta(hours=2)
    ))
    profiles["user2"] = user2

    return profiles


@pytest.fixture
def sample_item_catalog():
    """Fixture pour créer un catalogue d'éléments d'exemple."""
    catalog = ItemCatalog()

    # Éléments tech
    item1 = Item(
        item_id="item1",
        title="Ordinateur Portable",
        description="Ordinateur portable haute performance",
        category="technology",
        tags={"ordinateur", "tech", "portable"},
        popularity_score=0.8
    )
    catalog.add_item(item1)

    item2 = Item(
        item_id="item2",
        title="Smartphone",
        description="Smartphone dernière génération",
        category="technology",
        tags={"smartphone", "tech", "mobile"},
        popularity_score=0.7
    )
    catalog.add_item(item2)

    # Éléments food
    item3 = Item(
        item_id="item3",
        title="Recette Pasta",
        description="Recette de pâtes italiennes",
        category="food",
        tags={"pasta", "recette", "italien"},
        popularity_score=0.6
    )
    catalog.add_item(item3)

    item4 = Item(
        item_id="item4",
        title="Restaurant Italien",
        description="Restaurant italien traditionnel",
        category="food",
        tags={"restaurant", "italien", "diner"},
        popularity_score=0.9
    )
    catalog.add_item(item4)

    return catalog


@pytest.fixture
def recommendation_engine(sample_user_profiles, sample_item_catalog):
    """Fixture pour créer un moteur de recommandations."""
    return RecommendationEngine(sample_user_profiles, sample_item_catalog)


class TestRecommendationEngine:
    """Tests pour le moteur de recommandations."""

    def test_initialization(self, recommendation_engine):
        """Test de l'initialisation du moteur."""
        assert recommendation_engine.user_profiles is not None
        assert recommendation_engine.item_catalog is not None
        assert len(recommendation_engine.models) == 4  # 4 types de modèles
        assert RecommendationType.COLLABORATIVE in recommendation_engine.models
        assert RecommendationType.CONTENT_BASED in recommendation_engine.models
        assert RecommendationType.HYBRID in recommendation_engine.models
        assert RecommendationType.CONTEXTUAL in recommendation_engine.models

    def test_analyze_conversation_context(self, recommendation_engine):
        """Test de l'analyse du contexte de conversation."""
        conversation = "Je cherche un bon restaurant italien pour ce soir ?"
        context = recommendation_engine.analyze_conversation_context(conversation)

        assert context.context_type == ContextType.CONVERSATION
        assert context.conversation_context == conversation
        assert 'topics' in context.features
        assert 'sentiment_indicators' in context.features
        assert context.time_of_day is not None

    def test_generate_proactive_suggestions(self, recommendation_engine, sample_user_profiles):
        """Test de la génération de suggestions proactives."""
        user_profile = sample_user_profiles["user1"]
        result = recommendation_engine.generate_proactive_suggestions(user_profile)

        assert isinstance(result, RecommendationResult)
        assert result.user_id == user_profile.user_id
        assert isinstance(result.recommendations, list)
        assert result.processing_time >= 0
        assert result.total_candidates > 0

        if result.recommendations:
            rec = result.recommendations[0]
            assert isinstance(rec, Recommendation)
            assert 0 <= rec.score <= 1
            assert rec.recommendation_type == RecommendationType.HYBRID

    def test_calculate_hybrid_score(self, recommendation_engine, sample_user_profiles, sample_item_catalog):
        """Test du calcul du score hybride."""
        user_profile = sample_user_profiles["user1"]
        item = sample_item_catalog.get_item("item3")  # Recette pasta
        context = ContextSnapshot(context_type=ContextType.BEHAVIORAL)

        score = recommendation_engine._calculate_hybrid_score(user_profile, item, context)

        assert 0 <= score <= 1
        assert isinstance(score, float)

    def test_learn_user_preferences(self, recommendation_engine, sample_user_profiles):
        """Test de l'apprentissage des préférences utilisateur."""
        user_profile = sample_user_profiles["user1"]

        # Ajouter de nouvelles interactions
        new_interactions = [
            UserInteraction(
                user_id="user1",
                item_id="item2",
                interaction_type=InteractionType.LIKE,
                timestamp=datetime.now()
            )
        ]

        initial_prefs = user_profile.preferences.copy()
        initial_interaction_count = len(user_profile.interaction_history)

        recommendation_engine.learn_user_preferences(new_interactions)

        # Vérifier que les interactions ont été ajoutées
        assert len(user_profile.interaction_history) == initial_interaction_count + len(new_interactions)

        # Vérifier que les préférences ont été mises à jour
        assert user_profile.preferences != initial_prefs  # Les préférences ont changé

    def test_explain_recommendations(self, recommendation_engine, sample_item_catalog):
        """Test de l'explication des recommandations."""
        recommendations = [
            Recommendation(
                item_id="item1",
                score=0.8,
                recommendation_type=RecommendationType.HYBRID
            ),
            Recommendation(
                item_id="item4",
                score=0.7,
                recommendation_type=RecommendationType.COLLABORATIVE
            )
        ]

        explanations = recommendation_engine.explain_recommendations(recommendations)

        assert isinstance(explanations, dict)
        assert len(explanations) == 2
        assert "item1" in explanations
        assert "item4" in explanations

        for explanation in explanations.values():
            assert isinstance(explanation, str)
            assert len(explanation) > 0

    def test_update_recommendation_models(self, recommendation_engine):
        """Test de la mise à jour des modèles."""
        new_interactions = [
            UserInteraction(
                user_id="user1",
                item_id="item3",
                interaction_type=InteractionType.VIEW,
                timestamp=datetime.now()
            )
        ]

        # Capture l'état initial
        initial_cache_size = len(recommendation_engine._user_vectors_cache)

        recommendation_engine.update_recommendation_models(new_interactions)

        # Vérifier que les modèles ont été mis à jour
        for model in recommendation_engine.models.values():
            assert model.trained_at is not None

    def test_get_performance_stats(self, recommendation_engine):
        """Test de récupération des statistiques de performance."""
        stats = recommendation_engine.get_performance_stats()

        assert isinstance(stats, dict)
        assert 'total_recommendations' in stats
        assert 'average_processing_time' in stats
        assert 'cache_size' in stats
        assert 'total_users' in stats
        assert 'total_items' in stats
        assert 'models_status' in stats

    def test_context_types(self):
        """Test des différents types de contexte."""
        # Test conversation context
        conv_context = ContextSnapshot(
            context_type=ContextType.CONVERSATION,
            conversation_context="Je veux manger italien"
        )
        assert conv_context.context_type == ContextType.CONVERSATION

        # Test time-based context
        time_context = ContextSnapshot(
            context_type=ContextType.TIME_BASED,
            time_of_day="19:00",
            day_of_week="Friday"
        )
        assert time_context.time_of_day == "19:00"

        # Test vector generation
        vector = conv_context.get_context_vector()
        assert isinstance(vector, np.ndarray)

    def test_interaction_types(self):
        """Test des types d'interactions."""
        interaction = UserInteraction(
            user_id="user1",
            item_id="item1",
            interaction_type=InteractionType.LIKE,
            rating=4.5,
            duration=120.0
        )

        assert interaction.interaction_type == InteractionType.LIKE
        assert interaction.rating == 4.5
        assert interaction.duration == 120.0

        # Test serialization
        data = interaction.to_dict()
        assert data['user_id'] == "user1"
        assert data['interaction_type'] == "like"

    def test_item_similarity(self, sample_item_catalog):
        """Test de la similarité entre éléments."""
        item1 = sample_item_catalog.get_item("item1")  # Ordinateur
        item2 = sample_item_catalog.get_item("item2")  # Smartphone
        item3 = sample_item_catalog.get_item("item3")  # Recette pasta

        # Items tech devraient être similaires
        tech_similarity = item1.get_similarity_score(item2)
        assert tech_similarity > 0

        # Item tech et food devraient être moins similaires
        cross_similarity = item1.get_similarity_score(item3)
        assert cross_similarity < tech_similarity

    def test_feedback_learning(self, recommendation_engine, sample_user_profiles):
        """Test de l'apprentissage basé sur le feedback."""
        user_profile = sample_user_profiles["user1"]

        feedback_data = [
            FeedbackData(
                user_id="user1",
                item_id="item1",
                feedback_type="accepted",
                feedback_score=1.0,
                recommended_at=datetime.now() - timedelta(hours=1)
            ),
            FeedbackData(
                user_id="user1",
                item_id="item2",
                feedback_type="rejected",
                feedback_score=-0.5,
                recommended_at=datetime.now() - timedelta(hours=1)
            )
        ]

        initial_pref_item1 = user_profile.preferences.get("item1", 0.5)
        initial_pref_item2 = user_profile.preferences.get("item2", 0.5)

        recommendation_engine._learn_from_feedback(feedback_data)

        # Vérifier que les préférences ont été ajustées
        new_pref_item1 = user_profile.preferences.get("item1", 0.5)
        new_pref_item2 = user_profile.preferences.get("item2", 0.5)

        assert new_pref_item1 >= initial_pref_item1  # Devrait augmenter
        assert new_pref_item2 <= initial_pref_item2  # Devrait diminuer

    @patch('src.recommendations.recommendation_engine.cosine_similarity')
    def test_collaborative_filtering(self, mock_cosine, recommendation_engine, sample_user_profiles):
        """Test du filtrage collaboratif."""
        # Mock de la similarité cosinus
        mock_cosine.return_value = np.array([[0.8]])

        user_profile = sample_user_profiles["user1"]
        item = recommendation_engine.item_catalog.get_item("item4")

        score = recommendation_engine._calculate_collaborative_score(user_profile, item)

        # Vérifier que cosine_similarity a été appelée
        assert mock_cosine.called

    def test_diversity_application(self, recommendation_engine):
        """Test de l'application de la diversité."""
        recommendations = [
            Recommendation(item_id="item1", score=0.8, recommendation_type=RecommendationType.HYBRID),
            Recommendation(item_id="item2", score=0.7, recommendation_type=RecommendationType.HYBRID),
            Recommendation(item_id="item3", score=0.6, recommendation_type=RecommendationType.HYBRID)
        ]

        diverse_recs = recommendation_engine._apply_diversity(recommendations)

        assert len(diverse_recs) == len(recommendations)
        # Les scores devraient être ajustés pour la diversité
        assert all(rec.score >= 0 for rec in diverse_recs)

    def test_cache_management(self, recommendation_engine):
        """Test de la gestion du cache."""
        # Ajouter des données au cache
        recommendation_engine._user_vectors_cache["test_user"] = np.array([1, 2, 3])
        recommendation_engine._item_vectors_cache["test_item"] = np.array([4, 5, 6])

        initial_cache_size = len(recommendation_engine._user_vectors_cache) + len(recommendation_engine._item_vectors_cache)

        # Simuler un cache plein
        recommendation_engine.config['cache_max_size'] = initial_cache_size - 1

        recommendation_engine._clear_old_cache_entries()

        # Le cache devrait avoir été nettoyé
        final_cache_size = len(recommendation_engine._user_vectors_cache) + len(recommendation_engine._item_vectors_cache)
        assert final_cache_size <= initial_cache_size

    def test_season_detection(self, recommendation_engine):
        """Test de la détection de saison."""
        # Test hiver
        winter_date = datetime(2024, 1, 15)
        season = recommendation_engine._get_season(winter_date)
        assert season == "winter"

        # Test été
        summer_date = datetime(2024, 7, 15)
        season = recommendation_engine._get_season(summer_date)
        assert season == "summer"

        # Test automne
        fall_date = datetime(2024, 10, 15)
        season = recommendation_engine._get_season(fall_date)
        assert season == "fall"

        # Test printemps
        spring_date = datetime(2024, 4, 15)
        season = recommendation_engine._get_season(spring_date)
        assert season == "spring"

    def test_urgency_detection(self, recommendation_engine):
        """Test de la détection d'urgence."""
        urgent_text = "J'ai besoin de ça URGENT ! Vite !"
        urgency = recommendation_engine._extract_conversation_features(urgent_text)['urgency_level']
        assert urgency > 0

        normal_text = "Je cherche quelque chose d'intéressant."
        urgency_normal = recommendation_engine._extract_conversation_features(normal_text)['urgency_level']
        assert urgency_normal < urgency

    def test_sentiment_analysis(self, recommendation_engine):
        """Test de l'analyse de sentiment."""
        positive_text = "J'adore ça ! C'est excellent et super !"
        sentiment = recommendation_engine._extract_conversation_features(positive_text)['sentiment_indicators']

        assert sentiment['positive'] > sentiment['negative']
        assert sentiment['positive'] > 0

        negative_text = "C'est nul et horrible, je déteste ça."
        sentiment_neg = recommendation_engine._extract_conversation_features(negative_text)['sentiment_indicators']

        assert sentiment_neg['negative'] > sentiment_neg['positive']
        assert sentiment_neg['negative'] > 0

    def test_topic_extraction(self, recommendation_engine):
        """Test de l'extraction de topics."""
        food_text = "Je veux manger au restaurant italien ce soir."
        features = recommendation_engine._extract_conversation_features(food_text)

        assert 'food' in features['topics']

        tech_text = "J'ai besoin d'un nouvel ordinateur portable."
        tech_features = recommendation_engine._extract_conversation_features(tech_text)

        assert 'technology' in tech_features['topics']

    def test_model_save_load(self, recommendation_engine, tmp_path):
        """Test de la sauvegarde et chargement des modèles."""
        model_dir = tmp_path / "test_models"
        model_dir.mkdir()

        # Sauvegarder les modèles
        recommendation_engine.save_models(str(model_dir))

        # Créer un nouveau moteur
        new_engine = RecommendationEngine({}, ItemCatalog())

        # Charger les modèles
        new_engine.load_models(str(model_dir))

        # Vérifier que les modèles ont été chargés
        assert len(new_engine.models) == len(recommendation_engine.models)

        for model_type in recommendation_engine.models:
            assert model_type in new_engine.models
            original_model = recommendation_engine.models[model_type]
            loaded_model = new_engine.models[model_type]

            assert loaded_model.model_type == original_model.model_type
            assert loaded_model.parameters == original_model.parameters


class TestRecommendationTypes:
    """Tests pour les types de données de recommandation."""

    def test_user_profile_creation(self):
        """Test de création de profil utilisateur."""
        profile = UserProfile(
            user_id="test_user",
            demographics={"age": 25, "city": "Paris"}
        )

        assert profile.user_id == "test_user"
        assert profile.demographics["age"] == 25
        assert len(profile.interaction_history) == 0

    def test_item_creation(self):
        """Test de création d'élément."""
        item = Item(
            item_id="test_item",
            title="Test Item",
            description="A test item",
            category="test",
            tags={"tag1", "tag2"}
        )

        assert item.item_id == "test_item"
        assert item.title == "Test Item"
        assert "tag1" in item.tags
        assert item.category == "test"

    def test_interaction_creation(self):
        """Test de création d'interaction."""
        interaction = UserInteraction(
            user_id="user1",
            item_id="item1",
            interaction_type=InteractionType.LIKE,
            rating=4.0,
            duration=60.0
        )

        assert interaction.user_id == "user1"
        assert interaction.interaction_type == InteractionType.LIKE
        assert interaction.rating == 4.0
        assert interaction.duration == 60.0

    def test_recommendation_creation(self):
        """Test de création de recommandation."""
        rec = Recommendation(
            item_id="item1",
            score=0.85,
            recommendation_type=RecommendationType.HYBRID,
            confidence=0.9,
            explanation="Recommandation personnalisée"
        )

        assert rec.item_id == "item1"
        assert rec.score == 0.85
        assert rec.recommendation_type == RecommendationType.HYBRID
        assert rec.confidence == 0.9
        assert rec.explanation == "Recommandation personnalisée"

    def test_catalog_operations(self):
        """Test des opérations du catalogue."""
        catalog = ItemCatalog()

        item1 = Item("item1", "Item 1", "Desc 1", "cat1", {"tag1"})
        item2 = Item("item2", "Item 2", "Desc 2", "cat1", {"tag1", "tag2"})
        item3 = Item("item3", "Item 3", "Desc 3", "cat2", {"tag2"})

        catalog.add_item(item1)
        catalog.add_item(item2)
        catalog.add_item(item3)

        assert len(catalog.items) == 3
        assert len(catalog.categories) == 2
        assert len(catalog.tags) == 2

        # Test de recherche par catégorie
        cat1_items = catalog.get_items_by_category("cat1")
        assert len(cat1_items) == 2

        # Test de recherche par tag
        tag1_items = catalog.get_items_by_tag("tag1")
        assert len(tag1_items) == 2

    def test_feedback_data(self):
        """Test des données de feedback."""
        feedback = FeedbackData(
            user_id="user1",
            item_id="item1",
            feedback_type="accepted",
            feedback_score=1.0,
            recommended_at=datetime.now()
        )

        assert feedback.user_id == "user1"
        assert feedback.feedback_type == "accepted"
        assert feedback.feedback_score == 1.0

        # Test de sérialisation
        data = feedback.to_dict()
        assert data['user_id'] == "user1"
        assert data['feedback_type'] == "accepted"