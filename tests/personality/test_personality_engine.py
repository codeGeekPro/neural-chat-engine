"""Tests pour le moteur de personnalité."""

import pytest
import numpy as np
from datetime import datetime

from src.personality.personality_engine import PersonalityEngine
from src.personality.personality_types import (
    CommunicationStyle,
    PersonalityDimension,
    PersonalityProfile,
    StyleAnalysis
)


@pytest.fixture
def personality_engine():
    """Fixture pour le moteur de personnalité."""
    return PersonalityEngine()


@pytest.fixture
def sample_conversation():
    """Fixture pour un exemple de conversation."""
    return [
        {
            "role": "user",
            "content": "Salut ! Peux-tu m'aider avec un problème ?"
        },
        {
            "role": "assistant",
            "content": "Bien sûr ! Quel est le problème ?"
        },
        {
            "role": "user",
            "content": "J'ai du mal à comprendre les algorithmes de graphe."
        },
        {
            "role": "assistant",
            "content": "Les algorithmes de graphe peuvent être complexes. Commençons par les bases..."
        }
    ]


def test_personality_engine_initialization(personality_engine):
    """Teste l'initialisation du moteur de personnalité."""
    assert isinstance(personality_engine.embedding_model, object)
    assert isinstance(personality_engine.user_profiles, dict)
    assert personality_engine.style_analyzer is not None


def test_analyze_user_communication_style(personality_engine, sample_conversation):
    """Teste l'analyse du style de communication."""
    analysis = personality_engine.analyze_user_communication_style(sample_conversation)

    assert isinstance(analysis, StyleAnalysis)
    assert isinstance(analysis.detected_style, CommunicationStyle)
    assert 0 <= analysis.style_confidence <= 1
    assert isinstance(analysis.linguistic_features, dict)
    assert isinstance(analysis.emotional_indicators, dict)
    assert isinstance(analysis.complexity_metrics, dict)
    assert 0 <= analysis.formality_score <= 1


def test_create_personality_profile(personality_engine, sample_conversation):
    """Teste la création d'un profil de personnalité."""
    user_id = "test_user"

    profile = personality_engine.create_personality_profile(
        sample_conversation, user_id
    )

    assert isinstance(profile, PersonalityProfile)
    assert profile.user_id == user_id
    assert profile in personality_engine.user_profiles.values()
    assert len(profile.interaction_history) > 0
    assert profile.sample_size > 0


def test_adapt_response_tone(personality_engine):
    """Teste l'adaptation du ton des réponses."""
    base_response = "Bonjour, comment puis-je vous aider ?"

    # Crée un profil formel
    formal_profile = PersonalityProfile(
        user_id="formal_user",
        personality_dimensions={
            PersonalityDimension.FORMALITY: 0.9,
            PersonalityDimension.COMPLEXITY: 0.5,
            PersonalityDimension.FRIENDLINESS: 0.3,
            PersonalityDimension.CREATIVITY: 0.5
        }
    )

    adaptation = personality_engine.adapt_response_tone(
        base_response, formal_profile
    )

    assert isinstance(adaptation, object)
    assert adaptation.original_tone is not None
    assert adaptation.adapted_tone is not None
    assert isinstance(adaptation.adaptation_factors, dict)
    assert 0 <= adaptation.confidence_score <= 1


def test_maintain_personality_consistency(personality_engine):
    """Teste la maintenance de la cohérence de personnalité."""
    conversation_context = [
        {"role": "user", "content": "Salut"},
        {"role": "assistant", "content": "Salut ! Comment ça va ?"},
        {"role": "user", "content": "Bien merci"},
        {"role": "assistant", "content": "Super ! Que puis-je faire pour toi ?"}
    ]

    current_personality = PersonalityProfile(user_id="test")

    consistency = personality_engine.maintain_personality_consistency(
        conversation_context, current_personality
    )

    assert isinstance(consistency, dict)
    assert "consistency_score" in consistency
    assert "recommendations" in consistency
    assert 0 <= consistency["consistency_score"] <= 1


def test_learn_user_preferences(personality_engine):
    """Teste l'apprentissage des préférences utilisateur."""
    user_feedback = [
        {"sentiment": "positive", "response_id": "resp1"},
        {"sentiment": "negative", "response_id": "resp2"},
        {"sentiment": "positive", "response_id": "resp3"}
    ]

    response_history = [
        {"id": "resp1", "content": "Salut ! Je t'aide avec plaisir."},
        {"id": "resp2", "content": "Voici une explication technique détaillée."},
        {"id": "resp3", "content": "Pas de problème, on va résoudre ça ensemble !"}
    ]

    preferences = personality_engine.learn_user_preferences(
        user_feedback, response_history
    )

    assert isinstance(preferences, dict)
    assert "preferred_tones" in preferences
    assert "avoided_patterns" in preferences
    assert "learning_confidence" in preferences
    assert 0 <= preferences["learning_confidence"] <= 1


def test_personality_profile_methods():
    """Teste les méthodes du profil de personnalité."""
    profile = PersonalityProfile(user_id="test_user")

    # Test du vecteur de personnalité
    vector = profile.get_personality_vector()
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (4,)  # 4 dimensions

    # Test de la confiance
    assert not profile.is_confident()  # Pas assez d'échantillons

    # Ajoute des interactions
    for i in range(10):
        profile.update_from_interaction({"test": f"value_{i}"}, confidence=0.8)

    assert profile.sample_size == 10
    assert profile.is_confident()


def test_style_analysis():
    """Teste l'analyse de style."""
    analysis = StyleAnalysis(
        detected_style=CommunicationStyle.CASUAL,
        style_confidence=0.8,
        linguistic_features={"test": 1.0},
        emotional_indicators={"test": 0.5},
        complexity_metrics={"test": 0.3},
        formality_score=0.4
    )

    # Test de la conversion en dict
    data = analysis.to_dict()
    assert isinstance(data, dict)
    assert data["detected_style"] == "casual"
    assert data["style_confidence"] == 0.8


def test_empty_conversation_handling(personality_engine):
    """Teste la gestion des conversations vides."""
    analysis = personality_engine.analyze_user_communication_style([])

    assert analysis.detected_style == CommunicationStyle.CASUAL
    assert analysis.style_confidence == 0.0

    profile = personality_engine.create_personality_profile([], "empty_user")
    assert profile.user_id == "empty_user"
    assert profile.sample_size == 0