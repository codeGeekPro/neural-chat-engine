"""Tests pour le système d'apprentissage continu."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from src.learning.continual_learning import ContinualLearningSystem
from src.learning.learning_types import (
    LearningTask,
    ModelVersion,
    ModelVersionStatus,
    PerformanceMetrics,
    ElasticWeightConsolidation,
    LearningSchedule,
    TrainingBatch
)


@pytest.fixture
def simple_model():
    """Fixture pour un modèle simple."""
    return nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )


@pytest.fixture
def continual_learning_system(simple_model):
    """Fixture pour le système d'apprentissage continu."""
    with tempfile.TemporaryDirectory() as temp_dir:
        models = {
            "conversation_processing": simple_model,
            "emotion_analysis": simple_model
        }

        system = ContinualLearningSystem(
            base_models=models,
            model_dir=temp_dir,
            backup_dir=temp_dir
        )
        yield system


@pytest.fixture
def sample_conversations():
    """Fixture pour des conversations d'exemple."""
    return [
        {
            "user_input": "Bonjour, comment allez-vous ?",
            "bot_response": "Je vais très bien, merci !",
            "emotion": "positive",
            "timestamp": datetime.now().isoformat()
        },
        {
            "user_input": "J'ai un problème avec mon code.",
            "bot_response": "Pouvez-vous me donner plus de détails ?",
            "emotion": "neutral",
            "timestamp": datetime.now().isoformat()
        }
    ]


def test_continual_learning_initialization(continual_learning_system):
    """Teste l'initialisation du système."""
    assert len(continual_learning_system.base_models) == 2
    assert "conversation_processing" in continual_learning_system.base_models
    assert "emotion_analysis" in continual_learning_system.base_models

    # Vérifie les versions initiales
    assert len(continual_learning_system.model_versions) == 2
    for model_name in continual_learning_system.base_models:
        assert model_name in continual_learning_system.model_versions
        assert len(continual_learning_system.model_versions[model_name]) == 1
        assert continual_learning_system.current_versions[model_name] == "1.0.0"


def test_update_model_incrementally(continual_learning_system, sample_conversations):
    """Teste la mise à jour incrémentale d'un modèle."""
    model_name = "conversation_processing"

    # Mise à jour incrémentale
    result = continual_learning_system.update_model_incrementally(
        sample_conversations, model_name
    )

    assert result["success"] is True
    assert "version" in result
    assert "performance" in result
    assert result["version"] != "1.0.0"  # Version mise à jour

    # Vérifie que la version a été créée
    versions = continual_learning_system.model_versions[model_name]
    assert len(versions) == 2  # Version initiale + nouvelle


def test_prevent_catastrophic_forgetting(continual_learning_system, sample_conversations):
    """Teste la prévention de l'oubli catastrophique."""
    model_name = "conversation_processing"

    # Données des anciennes tâches
    old_tasks_data = {
        model_name: sample_conversations
    }

    # Applique EWC
    success = continual_learning_system.prevent_catastrophic_forgetting(
        old_tasks_data, model_name
    )

    assert success is True
    assert model_name in continual_learning_system.ewc_systems

    ewc_system = continual_learning_system.ewc_systems[model_name]
    assert isinstance(ewc_system, ElasticWeightConsolidation)


def test_evaluate_model_drift(continual_learning_system, sample_conversations):
    """Teste l'évaluation de la dérive du modèle."""
    model_name = "conversation_processing"

    test_data = [{"conversations": sample_conversations}]

    metrics = continual_learning_system.evaluate_model_drift(test_data, model_name)

    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.accuracy >= 0
    assert metrics.f1_score >= 0
    assert metrics.loss >= 0

    # Vérifie les métriques de surveillance
    assert model_name in continual_learning_system.monitoring_metrics
    monitoring = continual_learning_system.monitoring_metrics[model_name]
    assert "last_evaluation" in monitoring
    assert "current_performance" in monitoring


def test_rollback_to_previous_version(continual_learning_system, sample_conversations):
    """Teste le rollback vers une version précédente."""
    model_name = "conversation_processing"

    # Crée une nouvelle version
    continual_learning_system.update_model_incrementally(
        sample_conversations, model_name
    )

    # Rollback vers la version initiale
    success = continual_learning_system.rollback_to_previous_version(
        model_name, "1.0.0"
    )

    assert success is True
    assert continual_learning_system.current_versions[model_name] == "1.0.0"

    # Vérifie le statut des versions
    versions = continual_learning_system.model_versions[model_name]
    active_versions = [v for v in versions if v.status == ModelVersionStatus.ACTIVE]
    assert len(active_versions) == 1
    assert active_versions[0].version == "1.0.0"


def test_schedule_retraining(continual_learning_system):
    """Teste la planification du réentraînement."""
    # Force une faible performance pour déclencher le réentraînement
    continual_learning_system.monitoring_metrics["conversation_processing"] = {
        "current_performance": 0.7  # En dessous du seuil de 0.85
    }

    models_to_retrain = continual_learning_system.schedule_retraining()

    assert "conversation_processing" in models_to_retrain
    assert len(models_to_retrain["conversation_processing"]) > 0


def test_get_model_history(continual_learning_system, sample_conversations):
    """Teste la récupération de l'historique des modèles."""
    model_name = "conversation_processing"

    # Crée quelques versions
    continual_learning_system.update_model_incrementally(
        sample_conversations, model_name
    )
    continual_learning_system.update_model_incrementally(
        sample_conversations, model_name
    )

    history = continual_learning_system.get_model_history(model_name)

    assert len(history) == 3  # Version initiale + 2 mises à jour
    assert all(isinstance(h, dict) for h in history)
    assert all("version" in h for h in history)


def test_get_performance_trends(continual_learning_system):
    """Teste l'analyse des tendances de performance."""
    model_name = "conversation_processing"

    # Ajoute quelques métriques
    metrics = [
        PerformanceMetrics(accuracy=0.8, f1_score=0.75, loss=0.5),
        PerformanceMetrics(accuracy=0.82, f1_score=0.78, loss=0.45),
        PerformanceMetrics(accuracy=0.85, f1_score=0.80, loss=0.42)
    ]

    continual_learning_system.performance_history[model_name] = metrics

    trends = continual_learning_system.get_performance_trends(model_name, days=30)

    assert "accuracy" in trends
    assert "f1_score" in trends
    assert "loss" in trends
    assert "overall_score" in trends
    assert len(trends["accuracy"]) == 3


def test_training_batch():
    """Teste la classe TrainingBatch."""
    data = [{"input": "test", "output": "response"}]
    batch = TrainingBatch(
        task=LearningTask.CONVERSATION_PROCESSING,
        data=data
    )

    assert len(batch) == 1
    assert batch.task == LearningTask.CONVERSATION_PROCESSING

    stats = batch.get_statistics()
    assert stats["size"] == 1
    assert stats["task"] == "conversation_processing"


def test_performance_metrics():
    """Teste les métriques de performance."""
    metrics1 = PerformanceMetrics(
        accuracy=0.8, f1_score=0.75, loss=0.5
    )
    metrics2 = PerformanceMetrics(
        accuracy=0.85, f1_score=0.80, loss=0.45
    )

    # Test du score global
    score = metrics1.get_overall_score()
    assert score > 0

    # Test de comparaison
    assert metrics2.is_better_than(metrics1)
    assert not metrics1.is_better_than(metrics2)


def test_learning_schedule():
    """Teste la planification d'apprentissage."""
    schedule = LearningSchedule(
        task=LearningTask.CONVERSATION_PROCESSING,
        trigger_condition="performance_decline",
        performance_threshold=0.85
    )

    # Test des conditions de déclenchement
    assert schedule.should_trigger(0.8, 50, 3600)  # Performance faible
    assert schedule.should_trigger(0.9, 150, 3600)  # Beaucoup de données
    assert schedule.should_trigger(0.9, 50, 90000)  # Temps écoulé


def test_elastic_weight_consolidation():
    """Teste le système EWC."""
    ewc = ElasticWeightConsolidation(lambda_ewc=0.1)

    # Test de la consolidation
    model = nn.Linear(10, 2)
    ewc.consolidate_weights(model)

    assert len(ewc.importance_weights) > 0

    # Test de la perte EWC
    loss = ewc.ewc_loss(model)
    assert isinstance(loss, (float, torch.Tensor))


def test_model_version():
    """Teste la gestion des versions de modèle."""
    version = ModelVersion(
        model_name="test_model",
        version="1.0.0",
        task=LearningTask.CONVERSATION_PROCESSING
    )

    # Test de conversion dict
    data = version.to_dict()
    assert isinstance(data, dict)
    assert data["model_name"] == "test_model"
    assert data["version"] == "1.0.0"

    # Test de création depuis dict
    version2 = ModelVersion.from_dict(data)
    assert version2.model_name == version.model_name
    assert version2.version == version.version


def test_invalid_model_update(continual_learning_system):
    """Teste la gestion d'erreurs pour les modèles invalides."""
    # Modèle inexistant
    result = continual_learning_system.update_model_incrementally(
        [], "invalid_model"
    )
    assert result["success"] is False

    # Évaluation de modèle invalide
    metrics = continual_learning_system.evaluate_model_drift([], "invalid_model")
    assert metrics.accuracy == 0  # Métriques par défaut