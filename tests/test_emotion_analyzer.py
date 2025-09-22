"""Tests pour le système d'analyse des émotions."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.models.emotion_analyzer import EmotionAnalyzer
from src.models.emotion_types import EmotionCategory, EmotionProfile, EmotionState


@pytest.fixture
def analyzer():
    """Instance d'analyseur pour les tests."""
    return EmotionAnalyzer(
        model_name="SamLowe/roberta-base-go_emotions"
    )


@pytest.fixture
def sample_conversation():
    """Conversation d'exemple pour les tests."""
    return [
        ("user", "Je suis très content de cette nouvelle !"),
        ("assistant", "C'est une excellente nouvelle en effet !"),
        ("user", "Par contre, j'ai un peu peur des changements à venir..."),
        ("assistant", "C'est normal d'avoir des appréhensions.")
    ]


def test_emotion_detection(analyzer):
    """Teste la détection d'émotions basique."""
    text = "Je suis très content de cette nouvelle !"
    state = analyzer.analyze_message(text)
    
    assert state.primary_emotion == EmotionCategory.JOY
    assert 0.0 <= state.intensity <= 1.0
    assert state.emotion_scores
    assert state.timestamp


def test_conversation_analysis(analyzer, sample_conversation):
    """Teste l'analyse d'une conversation complète."""
    states = analyzer.analyze_conversation(sample_conversation)
    
    assert len(states) == 2  # Seulement les messages utilisateur
    
    # Premier message: joie
    assert states[0].primary_emotion == EmotionCategory.JOY
    
    # Second message: peur
    assert states[1].primary_emotion == EmotionCategory.FEAR
    

def test_user_profile_tracking(analyzer, sample_conversation):
    """Teste le suivi du profil émotionnel."""
    user_id = "test_user"
    
    # Analyse avec ID utilisateur
    analyzer.analyze_conversation(sample_conversation, user_id=user_id)
    
    # Vérifie le profil
    profile = analyzer.get_user_profile(user_id)
    assert isinstance(profile, EmotionProfile)
    assert len(profile.emotion_history) == 2
    assert profile.emotion_frequencies
    assert profile.avg_intensities
    
    # Vérifie les statistiques
    dominant = profile.get_dominant_emotions(top_k=2)
    assert EmotionCategory.JOY in dominant
    assert EmotionCategory.FEAR in dominant


def test_emotion_state_properties():
    """Teste les propriétés des états émotionnels."""
    state = EmotionState(
        primary_emotion=EmotionCategory.JOY,
        intensity=0.8,
        emotion_scores={
            EmotionCategory.JOY: 0.8,
            EmotionCategory.TRUST: 0.4,
            EmotionCategory.ANTICIPATION: 0.2
        }
    )
    
    # Vérifie les émotions secondaires
    secondary = state.secondary_emotions
    assert len(secondary) == 1
    assert EmotionCategory.TRUST in secondary


def test_emotion_profile_trends(analyzer):
    """Teste l'analyse des tendances émotionnelles."""
    profile = EmotionProfile()
    
    # Simule une séquence d'états
    base_time = datetime.now()
    emotions = [
        (EmotionCategory.JOY, 0.8),
        (EmotionCategory.JOY, 0.9),
        (EmotionCategory.ANTICIPATION, 0.7),
        (EmotionCategory.FEAR, 0.6),
        (EmotionCategory.FEAR, 0.8)
    ]
    
    for i, (emotion, intensity) in enumerate(emotions):
        state = EmotionState(
            primary_emotion=emotion,
            intensity=intensity,
            timestamp=base_time + timedelta(minutes=i)
        )
        profile.add_emotion_state(state)
    
    # Vérifie les tendances
    assert profile.get_emotion_trend(window=3) == "stable"
    assert len(profile.get_dominant_emotions(top_k=2)) == 2


def test_profile_persistence(analyzer, sample_conversation, tmp_path):
    """Teste la persistance des profils émotionnels."""
    user_id = "test_user"
    save_path = tmp_path / "profiles.json"
    
    # Crée et sauvegarde un profil
    analyzer.analyze_conversation(sample_conversation, user_id=user_id)
    analyzer.save_profiles(str(save_path))
    
    # Crée un nouvel analyseur et charge les profils
    new_analyzer = EmotionAnalyzer()
    new_analyzer.load_profiles(str(save_path))
    
    # Vérifie que le profil est identique
    original_profile = analyzer.get_user_profile(user_id)
    loaded_profile = new_analyzer.get_user_profile(user_id)
    
    assert original_profile.emotion_frequencies == loaded_profile.emotion_frequencies
    assert original_profile.avg_intensities == loaded_profile.avg_intensities


@pytest.mark.parametrize("text,expected_emotion", [
    ("Je suis très heureux !", EmotionCategory.JOY),
    ("C'est vraiment triste...", EmotionCategory.SADNESS),
    ("Ça me met en colère !", EmotionCategory.ANGER),
    ("J'ai très peur...", EmotionCategory.FEAR),
    ("Wow, quelle surprise !", EmotionCategory.SURPRISE)
])
def test_basic_emotions(analyzer, text, expected_emotion):
    """Teste la détection des émotions de base."""
    state = analyzer.analyze_message(text)
    assert state.primary_emotion == expected_emotion