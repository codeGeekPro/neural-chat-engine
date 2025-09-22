"""Tests pour le système de mémoire conversationnelle."""

import pytest
import torch
import numpy as np
from datetime import datetime, timedelta

from src.models.memory_system import ConversationMemory, MemoryGNN
from src.models.memory_types import (
    ConversationMemoryItem,
    EntityType,
    KnowledgeGraphEntity,
    KnowledgeGraphRelation,
    MemoryEmbedding,
    MemoryType,
    RelationType,
    UserProfile
)


@pytest.fixture
def memory_system():
    """Fixture pour le système de mémoire."""
    return ConversationMemory(
        max_context_length=100,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )


@pytest.fixture
def sample_conversation():
    """Fixture pour un exemple de conversation."""
    return [
        {
            "role": "user",
            "content": "Bonjour, comment allez-vous ?"
        },
        {
            "role": "assistant",
            "content": "Je vais très bien, merci ! Comment puis-je vous aider ?"
        },
        {
            "role": "user",
            "content": "J'aimerais en savoir plus sur l'intelligence artificielle."
        }
    ]


def test_memory_gnn_initialization():
    """Teste l'initialisation du GNN."""
    model = MemoryGNN(input_dim=384)  # dimension de MiniLM
    
    assert isinstance(model.gat1, torch.nn.Module)
    assert isinstance(model.gat2, torch.nn.Module)
    assert isinstance(model.mlp, torch.nn.Sequential)


def test_memory_system_initialization(memory_system):
    """Teste l'initialisation du système de mémoire."""
    assert memory_system.max_context_length == 100
    assert memory_system.device == "cpu"
    assert len(memory_system.short_term) == 0
    assert len(memory_system.long_term) == 0
    assert len(memory_system.compressed) == 0


def test_store_conversation_turn(memory_system):
    """Teste le stockage d'un tour de conversation."""
    user_input = "Bonjour !"
    bot_response = "Bonjour, comment puis-je vous aider ?"
    
    memory_system.store_conversation_turn(
        user_input=user_input,
        bot_response=bot_response,
        metadata={"user_id": "test_user"}
    )
    
    assert len(memory_system.short_term) == 2  # un pour chaque message
    assert all(isinstance(item, ConversationMemoryItem) 
              for item in memory_system.short_term)
    
    # Vérifie les embeddings
    for item in memory_system.short_term:
        assert item.embedding.vector.shape == (384,)  # dimension de MiniLM
        assert isinstance(item.embedding.timestamp, datetime)


def test_retrieve_relevant_context(memory_system, sample_conversation):
    """Teste la récupération du contexte pertinent."""
    # Stocke quelques conversations
    for turn in sample_conversation[:-1]:
        if turn["role"] == "user":
            memory_system.store_conversation_turn(
                user_input=turn["content"],
                bot_response="Test response"
            )
    
    # Teste la récupération
    context = memory_system.retrieve_relevant_context(
        current_input="Parlez-moi de l'IA",
        max_turns=2
    )
    
    assert isinstance(context, list)
    assert len(context) <= 2
    assert all(isinstance(msg, str) for msg in context)


def test_compress_old_conversations(memory_system):
    """Teste la compression des anciennes conversations."""
    # Ajoute plusieurs conversations
    for i in range(5):
        memory_system.store_conversation_turn(
            user_input=f"Message utilisateur {i}",
            bot_response=f"Réponse bot {i}"
        )
    
    # Simule des conversations anciennes
    old_timestamp = datetime.now() - timedelta(days=7)
    for item in memory_system.short_term[:4]:
        item.embedding.timestamp = old_timestamp
    
    # Compresse
    memory_system.compress_old_conversations(compression_threshold=2)
    
    assert len(memory_system.compressed) > 0
    assert all(item.memory_type == MemoryType.COMPRESSED 
              for item in memory_system.compressed)


def test_build_user_profile(memory_system, sample_conversation):
    """Teste la construction du profil utilisateur."""
    user_id = "test_user"
    
    profile = memory_system.build_user_profile(
        conversation_history=sample_conversation,
        user_id=user_id
    )
    
    assert isinstance(profile, UserProfile)
    assert profile.user_id == user_id
    assert len(memory_system.user_profiles) == 1
    assert user_id in memory_system.user_profiles


def test_update_relationship_graph(memory_system):
    """Teste la mise à jour du graphe de relations."""
    entities = [
        {
            "id": "user1",
            "type": "USER",
            "name": "Alice",
            "attributes": {"age": 30}
        },
        {
            "id": "topic1",
            "type": "TOPIC",
            "name": "Intelligence Artificielle",
        }
    ]
    
    relationships = [
        {
            "source": "user1",
            "target": "topic1",
            "type": "INTERESTED_IN",
            "weight": 0.8
        }
    ]
    
    memory_system.update_relationship_graph(entities, relationships)
    
    assert len(memory_system.graph.entities) == 2
    assert len(memory_system.graph.relations) == 1
    
    # Vérifie l'entité
    user = memory_system.graph.entities.get("user1")
    assert isinstance(user, KnowledgeGraphEntity)
    assert user.type == EntityType.USER
    assert user.name == "Alice"
    
    # Vérifie la relation
    relation = next(iter(memory_system.graph.relations))
    assert isinstance(relation, KnowledgeGraphRelation)
    assert relation.type == RelationType.INTERESTED_IN
    assert relation.weight == 0.8


def test_memory_compression_threshold(memory_system):
    """Teste le respect de la limite de contexte."""
    # Ajoute des messages jusqu'à dépasser la limite
    long_message = "Test " * 30  # 150 caractères
    
    for _ in range(3):  # Dépassera la limite de 100
        memory_system.store_conversation_turn(
            user_input=long_message,
            bot_response=long_message
        )
    
    total_length = sum(len(item.content) 
                      for item in memory_system.short_term)
    
    assert total_length <= memory_system.max_context_length
    assert len(memory_system.long_term) > 0  # Certains messages ont été déplacés