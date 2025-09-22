"""Test du processeur de conversations avancé avec mesures."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.data.conversation_processor import ConversationProcessor
from src.data.conversation_types import ConversationFormat


@pytest.fixture
def sample_convo():
    return [
        ("user", "Bonjour, comment ça va ?"),
        ("assistant", "Très bien merci ! Je peux vous aider ?"),
        ("user", "Oui, j'ai une question sur les embeddings."),
        ("assistant", "Je vous écoute, que voulez-vous savoir sur les embeddings ?")
    ]


def test_load_json_conversation(sample_convo):
    proc = ConversationProcessor()
    json_str = proc.save_conversation(sample_convo, ConversationFormat.JSON)
    loaded = proc.load_conversation(json_str, ConversationFormat.JSON)
    assert len(loaded) == len(sample_convo)
    assert all(a == b for a, b in zip(loaded, sample_convo))


def test_conversation_stats(sample_convo):
    proc = ConversationProcessor()
    proc.start_session()
    # Simuler un délai
    proc.start_time = datetime.now() - timedelta(seconds=30)
    
    stats = proc.analyze_conversation(sample_convo)
    assert stats.num_turns == 4
    assert stats.num_tokens > 0
    assert stats.duration_seconds >= 30.0
    assert 0 <= stats.coherence_score <= 1.0
    assert 0 <= stats.engagement_score <= 1.0


@pytest.mark.parametrize("fmt", list(ConversationFormat))
def test_conversation_formats(sample_convo, fmt):
    proc = ConversationProcessor()
    serialized = proc.save_conversation(sample_convo, fmt)
    loaded = proc.load_conversation(serialized, fmt)
    # Les rôles et contenus essentiels sont préservés
    assert len(loaded) == len(sample_convo)
    assert all(a[0] == b[0] for a, b in zip(loaded, sample_convo))  # rôles préservés
    assert all(len(a[1]) > 0 for a in loaded)  # contenus non vides