"""Tests pour le générateur de réponses avec RAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from sentence_transformers import SentenceTransformer

from src.models.response_generator import ResponseGenerator
from src.models.response_types import (
    GeneratedResponse,
    PersonalityProfile,
    PersonalityTrait,
    RAGResult,
    ResponseFormat
)


class MockVectorStore:
    """Base de données vectorielle simulée pour les tests."""
    
    def __init__(self):
        self.docs = [
            {
                "content": "Les transformers sont des modèles d'attention.",
                "metadata": {"source": "doc1.txt"}
            },
            {
                "content": "PyTorch est un framework de deep learning.",
                "metadata": {"source": "doc2.txt"}
            }
        ]
        
    def search(self, query_vector: np.ndarray, k: int = 3, min_score: float = 0.6):
        """Simule une recherche vectorielle."""
        # Retourne toujours les mêmes documents avec des scores simulés
        return {
            "documents": self.docs[:k],
            "scores": [0.8, 0.7][:k]
        }


@pytest.fixture
def generator():
    """Instance de générateur pour les tests."""
    return ResponseGenerator(
        base_model="google/flan-t5-small",  # Petit modèle pour les tests
        device="cpu"
    )


@pytest.fixture
def rag_generator(generator):
    """Générateur configuré avec RAG."""
    vector_store = MockVectorStore()
    generator.setup_rag_system(
        vector_store,
        retriever_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return generator


@pytest.fixture
def sample_conversation():
    """Conversation d'exemple pour les tests."""
    return [
        {
            "role": "user",
            "content": "Qu'est-ce qu'un transformer ?"
        },
        {
            "role": "assistant",
            "content": "Un transformer est un modèle d'attention qui..."
        },
        {
            "role": "user",
            "content": "Comment utiliser PyTorch ?"
        }
    ]


def test_basic_response_generation(generator):
    """Teste la génération de réponse basique."""
    response = generator.generate_response(
        user_input="Bonjour, comment ça va ?",
        max_length=50
    )
    
    assert isinstance(response, GeneratedResponse)
    assert response.content
    assert response.format == ResponseFormat.TEXT
    assert response.metadata
    assert response.quality_metrics


def test_rag_response_generation(rag_generator):
    """Teste la génération avec RAG."""
    response = rag_generator.generate_response(
        user_input="Expliquez-moi les transformers.",
        max_length=50
    )
    
    assert response.metadata["rag_result"]
    assert len(response.metadata["rag_result"]["documents"]) > 0


def test_personality_adaptation(generator):
    """Teste l'adaptation de personnalité."""
    # Test avec personnalité formelle
    formal_override = {
        PersonalityTrait.FORMALITY: 1.0,
        PersonalityTrait.EXPERTISE: 0.9
    }
    
    response = generator.generate_response(
        user_input="Qu'est-ce que PyTorch ?",
        max_length=50
    )
    
    adapted = generator.adapt_personality(
        response.content,
        personality_override=formal_override
    )
    
    assert adapted != response.content
    assert len(adapted) > 0


def test_response_quality_validation(generator):
    """Teste la validation de qualité des réponses."""
    metrics = generator.validate_response_quality(
        "Cette réponse est cohérente et informative à propos de PyTorch.",
        context="Question sur PyTorch et son utilisation."
    )
    
    assert "coherence_score" in metrics
    assert "length_score" in metrics
    assert all(0.0 <= score <= 1.0 for score in metrics.values())


def test_conversation_context_handling(generator, sample_conversation):
    """Teste la gestion du contexte conversationnel."""
    response = generator.generate_response(
        user_input="Et pour TensorFlow ?",
        conversation_context=sample_conversation,
        max_length=50
    )
    
    assert response.content
    assert response.metadata["prompt_length"] > 0


def test_multi_format_responses(generator):
    """Teste les différents formats de réponse."""
    formats = [
        ResponseFormat.TEXT,
        ResponseFormat.MARKDOWN,
        ResponseFormat.JSON
    ]
    
    for fmt in formats:
        response = generator.generate_response(
            user_input="Donnez-moi des informations sur PyTorch.",
            response_format=fmt,
            max_length=50
        )
        assert response.format == fmt


def test_personality_profile_adjustments():
    """Teste les ajustements de profil de personnalité."""
    profile = PersonalityProfile()
    
    # Test d'ajustement de traits OCEAN
    profile.adjust_trait(PersonalityTrait.OPENNESS, 0.5)
    assert -1.0 <= profile.openness <= 1.0
    
    # Test d'ajustement de traits supplémentaires
    profile.adjust_trait(PersonalityTrait.FORMALITY, 0.8)
    assert 0.0 <= profile.formality <= 1.0


def test_rag_result_processing():
    """Teste le traitement des résultats RAG."""
    result = RAGResult(
        documents=[
            {"content": "Doc 1", "score": 0.9},
            {"content": "Doc 2", "score": 0.7},
            {"content": "Doc 3", "score": 0.5}
        ],
        scores=[0.9, 0.7, 0.5]
    )
    
    # Test du filtrage par seuil
    best_docs = result.get_best_documents(threshold=0.6)
    assert len(best_docs) == 2
    
    # Test du formatage
    context = result.get_formatted_context()
    assert "[Document 1]" in context
    assert "[Document 2]" in context


@pytest.mark.parametrize("trait,value,expected", [
    (PersonalityTrait.FORMALITY, 0.8, "formel et professionnel"),
    (PersonalityTrait.FORMALITY, 0.2, "décontracté et familier"),
    (PersonalityTrait.FORMALITY, 0.5, "naturel et équilibré")
])
def test_personality_styles(generator, trait, value, expected):
    """Teste les styles basés sur la personnalité."""
    generator.personality.adjust_trait(trait, value)
    style = generator._get_personality_style()
    assert style == expected