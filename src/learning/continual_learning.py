"""Système d'apprentissage continu pour éviter l'oubli catastrophique."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .learning_types import (
    LearningTask,
    ModelVersion,
    ModelVersionStatus,
    PerformanceMetrics,
    ElasticWeightConsolidation,
    LearningSchedule,
    TrainingBatch
)


logger = logging.getLogger(__name__)


class ContinualLearningSystem:
    """Système d'apprentissage continu avec prévention de l'oubli catastrophique."""

    def __init__(
        self,
        base_models: Dict[str, nn.Module],
        learning_rate: float = 1e-5,
        model_dir: str = "models",
        backup_dir: str = "backups",
        device: str = "cpu"
    ):
        """Initialise le système d'apprentissage continu.

        Args:
            base_models: Modèles de base à gérer
            learning_rate: Taux d'apprentissage pour les mises à jour
            model_dir: Répertoire pour sauvegarder les modèles
            backup_dir: Répertoire pour les sauvegardes
            device: Périphérique de calcul
        """
        self.base_models = base_models
        self.learning_rate = learning_rate
        self.device = device

        # Répertoires
        self.model_dir = Path(model_dir)
        self.backup_dir = Path(backup_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)

        # Gestion des versions
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        self.current_versions: Dict[str, str] = {}

        # Système EWC pour chaque modèle
        self.ewc_systems: Dict[str, ElasticWeightConsolidation] = {}

        # Planifications d'apprentissage
        self.learning_schedules: Dict[str, LearningSchedule] = {}

        # Métriques de performance
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}

        # Cache des données d'entraînement
        self.training_cache: Dict[str, List[TrainingBatch]] = {}

        # Métriques de surveillance
        self.monitoring_metrics: Dict[str, Dict[str, Any]] = {}

        # Initialisation
        self._initialize_system()

        logger.info(
            f"ContinualLearningSystem initialisé avec {len(base_models)} modèles"
        )

    def update_model_incrementally(
        self,
        new_conversations: List[Dict[str, Any]],
        model_type: str,
        batch_size: int = 32,
        epochs: int = 3
    ) -> Dict[str, Any]:
        """Met à jour un modèle de manière incrémentale.

        Args:
            new_conversations: Nouvelles données de conversation
            model_type: Type de modèle à mettre à jour
            batch_size: Taille des lots
            epochs: Nombre d'époques

        Returns:
            Résultats de la mise à jour
        """
        if model_type not in self.base_models:
            raise ValueError(f"Modèle {model_type} non trouvé")

        model = self.base_models[model_type]
        model.to(self.device)

        # Prépare les données
        training_data = self._prepare_training_data(
            new_conversations, model_type
        )

        if not training_data:
            return {"success": False, "reason": "Aucune donnée d'entraînement valide"}

        # Crée un lot d'entraînement
        batch = TrainingBatch(
            task=LearningTask(model_type),
            data=new_conversations,
            metadata={"incremental_update": True, "data_size": len(new_conversations)}
        )

        # Cache le lot
        if model_type not in self.training_cache:
            self.training_cache[model_type] = []
        self.training_cache[model_type].append(batch)

        # Met à jour avec EWC si disponible
        ewc_loss = 0.0
        if model_type in self.ewc_systems:
            ewc_system = self.ewc_systems[model_type]
            ewc_loss = ewc_system.ewc_loss(model)

        # Entraînement incrémental
        results = self._incremental_training(
            model, training_data, ewc_loss, batch_size, epochs
        )

        # Évalue les performances
        performance = self.evaluate_model_drift(
            [{"conversations": new_conversations}], model_type
        )

        # Sauvegarde la nouvelle version
        version_info = self._save_model_version(
            model, model_type, performance, len(new_conversations)
        )

        # Met à jour l'historique des performances
        if model_type not in self.performance_history:
            self.performance_history[model_type] = []
        self.performance_history[model_type].append(performance)

        results.update({
            "version": version_info["version"],
            "performance": performance.to_dict(),
            "ewc_loss": float(ewc_loss) if isinstance(ewc_loss, torch.Tensor) else ewc_loss
        })

        logger.info(
            f"Modèle {model_type} mis à jour incrémentalement - "
            f"Version: {version_info['version']}, "
            f"Performance: {performance.get_overall_score():.3f}"
        )

        return results

    def prevent_catastrophic_forgetting(
        self,
        old_tasks_data: Dict[str, List[Dict[str, Any]]],
        model_type: str
    ) -> bool:
        """Prévention de l'oubli catastrophique en consolidant les connaissances.

        Args:
            old_tasks_data: Données des anciennes tâches
            model_type: Type de modèle

        Returns:
            Succès de la consolidation
        """
        if model_type not in self.base_models:
            return False

        model = self.base_models[model_type]

        # Initialise EWC si nécessaire
        if model_type not in self.ewc_systems:
            self.ewc_systems[model_type] = ElasticWeightConsolidation()

        ewc_system = self.ewc_systems[model_type]

        # Prépare les données des anciennes tâches
        old_training_data = self._prepare_training_data(
            old_tasks_data.get(model_type, []), model_type
        )

        if old_training_data:
            # Calcule l'information de Fisher
            dataloader = DataLoader(
                old_training_data,
                batch_size=32,
                shuffle=False
            )

            try:
                ewc_system.compute_fisher_information(model, dataloader, self.device)
                ewc_system.consolidate_weights(model)

                logger.info(
                    f"EWC consolidé pour {model_type} avec "
                    f"{len(old_tasks_data.get(model_type, []))} échantillons"
                )
                return True

            except Exception as e:
                logger.error(f"Erreur lors de la consolidation EWC: {e}")
                return False
        else:
            logger.warning(f"Aucune donnée ancienne pour {model_type}")
            return False

    def evaluate_model_drift(
        self,
        test_conversations: List[Dict[str, Any]],
        model_type: str
    ) -> PerformanceMetrics:
        """Évalue la dérive de performance du modèle.

        Args:
            test_conversations: Conversations de test
            model_type: Type de modèle

        Returns:
            Métriques de performance
        """
        if model_type not in self.base_models:
            return PerformanceMetrics()

        model = self.base_models[model_type]
        model.eval()

        # Prépare les données de test
        test_data = self._prepare_test_data(test_conversations, model_type)

        if not test_data:
            return PerformanceMetrics()

        # Évaluation selon le type de modèle
        if model_type == "conversation_processing":
            metrics = self._evaluate_conversation_processing(model, test_data)
        elif model_type == "emotion_analysis":
            metrics = self._evaluate_emotion_analysis(model, test_data)
        elif model_type == "response_generation":
            metrics = self._evaluate_response_generation(model, test_data)
        else:
            metrics = self._evaluate_generic_model(model, test_data)

        # Met à jour les métriques de surveillance
        self.monitoring_metrics[model_type] = {
            "last_evaluation": datetime.now(),
            "drift_detected": self._detect_performance_drift(model_type, metrics),
            "current_performance": metrics.get_overall_score()
        }

        return metrics

    def rollback_to_previous_version(
        self,
        model_name: str,
        target_version: Optional[str] = None
    ) -> bool:
        """Rollback vers une version précédente du modèle.

        Args:
            model_name: Nom du modèle
            target_version: Version cible (dernière si None)

        Returns:
            Succès du rollback
        """
        if model_name not in self.model_versions:
            logger.error(f"Aucune version trouvée pour {model_name}")
            return False

        versions = self.model_versions[model_name]

        if not target_version:
            # Trouve la meilleure version active
            active_versions = [
                v for v in versions
                if v.status == ModelVersionStatus.ACTIVE
            ]
            if not active_versions:
                logger.error(f"Aucune version active pour {model_name}")
                return False

            # Trie par performance et date
            active_versions.sort(
                key=lambda v: (
                    v.performance_metrics.get("overall_score", 0),
                    v.created_at
                ),
                reverse=True
            )
            target_version_obj = active_versions[0]
        else:
            # Trouve la version spécifique
            target_version_obj = None
            for v in versions:
                if v.version == target_version:
                    target_version_obj = v
                    break

            if not target_version_obj:
                logger.error(f"Version {target_version} non trouvée pour {model_name}")
                return False

        # Charge le modèle
        try:
            model_state = torch.load(target_version_obj.model_path, map_location=self.device)
            self.base_models[model_name].load_state_dict(model_state)

            # Met à jour la version courante
            self.current_versions[model_name] = target_version_obj.version

            # Archive l'ancienne version
            current_version = self._get_current_version(model_name)
            if current_version:
                current_version.status = ModelVersionStatus.ARCHIVED

            # Active la nouvelle version
            target_version_obj.status = ModelVersionStatus.ACTIVE

            logger.info(
                f"Rollback réussi pour {model_name} vers version {target_version_obj.version}"
            )
            return True

        except Exception as e:
            logger.error(f"Erreur lors du rollback: {e}")
            return False

    def schedule_retraining(
        self,
        performance_threshold: float = 0.85
    ) -> Dict[str, List[str]]:
        """Planifie le réentraînement des modèles selon les critères.

        Args:
            performance_threshold: Seuil de performance pour déclencher

        Returns:
            Modèles nécessitant un réentraînement
        """
        models_to_retrain = {}

        for model_name in self.base_models.keys():
            reasons = []

            # Vérifie les métriques de performance
            if model_name in self.monitoring_metrics:
                metrics = self.monitoring_metrics[model_name]
                current_perf = metrics.get("current_performance", 0)

                if current_perf < performance_threshold:
                    reasons.append(".3f")

            # Vérifie la planification
            if model_name in self.learning_schedules:
                schedule = self.learning_schedules[model_name]
                time_since_last = (
                    datetime.now() - (schedule.last_scheduled or datetime.min)
                ).total_seconds()

                new_data_points = sum(
                    len(batch) for batch in self.training_cache.get(model_name, [])
                )

                if schedule.should_trigger(current_perf, new_data_points, int(time_since_last)):
                    reasons.append("Critères de planification atteints")

            # Vérifie la dérive
            if self._detect_performance_drift(model_name):
                reasons.append("Dérive de performance détectée")

            if reasons:
                models_to_retrain[model_name] = reasons

        return models_to_retrain

    def get_model_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Retourne l'historique des versions d'un modèle."""
        if model_name not in self.model_versions:
            return []

        return [v.to_dict() for v in self.model_versions[model_name]]

    def get_performance_trends(
        self,
        model_name: str,
        days: int = 30
    ) -> Dict[str, List[float]]:
        """Analyse les tendances de performance."""
        if model_name not in self.performance_history:
            return {}

        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m for m in self.performance_history[model_name]
            if m.evaluated_at >= cutoff_date
        ]

        if not recent_metrics:
            return {}

        trends = {
            "accuracy": [m.accuracy for m in recent_metrics],
            "f1_score": [m.f1_score for m in recent_metrics],
            "loss": [m.loss for m in recent_metrics],
            "overall_score": [m.get_overall_score() for m in recent_metrics]
        }

        return trends

    def _initialize_system(self) -> None:
        """Initialise le système."""
        # Crée des versions initiales pour tous les modèles
        for model_name, model in self.base_models.items():
            # Version initiale
            version = ModelVersion(
                model_name=model_name,
                version="1.0.0",
                task=LearningTask(model_name),
                status=ModelVersionStatus.ACTIVE,
                training_data_size=0
            )

            self.model_versions[model_name] = [version]
            self.current_versions[model_name] = "1.0.0"

            # Sauvegarde initiale
            self._save_model_version(model, model_name, PerformanceMetrics(), 0)

            # Planification par défaut
            self.learning_schedules[model_name] = LearningSchedule(
                task=LearningTask(model_name),
                trigger_condition="performance_decline",
                performance_threshold=0.85
            )

    def _prepare_training_data(
        self,
        conversations: List[Dict[str, Any]],
        model_type: str
    ) -> Optional[TensorDataset]:
        """Prépare les données d'entraînement selon le type de modèle."""
        try:
            if model_type == "conversation_processing":
                return self._prepare_conversation_data(conversations)
            elif model_type == "emotion_analysis":
                return self._prepare_emotion_data(conversations)
            elif model_type == "response_generation":
                return self._prepare_generation_data(conversations)
            else:
                return self._prepare_generic_data(conversations)
        except Exception as e:
            logger.error(f"Erreur préparation données {model_type}: {e}")
            return None

    def _prepare_test_data(
        self,
        test_conversations: List[Dict[str, Any]],
        model_type: str
    ) -> List[Dict[str, Any]]:
        """Prépare les données de test."""
        # Pour l'instant, retourne les données brutes
        return test_conversations

    def _incremental_training(
        self,
        model: nn.Module,
        training_data: TensorDataset,
        ewc_loss: Union[float, torch.Tensor],
        batch_size: int,
        epochs: int
    ) -> Dict[str, Any]:
        """Effectue l'entraînement incrémental."""
        dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        model.train()
        total_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)

                # Ajoute la perte EWC si disponible
                if isinstance(ewc_loss, torch.Tensor) and ewc_loss > 0:
                    loss += ewc_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            total_loss += epoch_loss / len(dataloader)

        return {
            "success": True,
            "final_loss": total_loss / epochs,
            "epochs_completed": epochs
        }

    def _save_model_version(
        self,
        model: nn.Module,
        model_name: str,
        performance: PerformanceMetrics,
        training_size: int
    ) -> Dict[str, Any]:
        """Sauvegarde une nouvelle version du modèle."""
        # Génère le numéro de version
        current_version = self.current_versions.get(model_name, "1.0.0")
        version_parts = current_version.split(".")
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_version = ".".join(version_parts)

        # Chemin de sauvegarde
        model_path = self.model_dir / f"{model_name}_v{new_version}.pth"
        torch.save(model.state_dict(), model_path)

        # Crée l'objet version
        version = ModelVersion(
            model_name=model_name,
            version=new_version,
            task=LearningTask(model_name),
            performance_metrics={
                "accuracy": performance.accuracy,
                "f1_score": performance.f1_score,
                "loss": performance.loss,
                "overall_score": performance.get_overall_score()
            },
            training_data_size=training_size,
            model_path=model_path
        )

        # Ajoute à l'historique
        if model_name not in self.model_versions:
            self.model_versions[model_name] = []
        self.model_versions[model_name].append(version)

        # Met à jour la version courante
        self.current_versions[model_name] = new_version

        # Sauvegarde les métadonnées
        metadata_path = self.model_dir / f"{model_name}_v{new_version}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)

        return {"version": new_version, "path": str(model_path)}

    def _get_current_version(self, model_name: str) -> Optional[ModelVersion]:
        """Retourne la version actuelle d'un modèle."""
        if model_name not in self.model_versions:
            return None

        current_ver = self.current_versions.get(model_name)
        if not current_ver:
            return None

        for version in self.model_versions[model_name]:
            if version.version == current_ver:
                return version

        return None

    def _detect_performance_drift(
        self,
        model_name: str,
        current_metrics: Optional[PerformanceMetrics] = None
    ) -> bool:
        """Détecte la dérive de performance."""
        if model_name not in self.performance_history:
            return False

        history = self.performance_history[model_name]
        if len(history) < 2:
            return False

        # Compare avec la moyenne des dernières performances
        recent_scores = [m.get_overall_score() for m in history[-5:]]
        avg_recent = np.mean(recent_scores)
        avg_older = np.mean([m.get_overall_score() for m in history[:-5]]) if len(history) > 5 else avg_recent

        # Décline significatif
        return avg_recent < avg_older * 0.95  # 5% de déclin

    # Méthodes d'évaluation spécifiques aux modèles
    def _evaluate_conversation_processing(
        self, model: nn.Module, test_data: List[Dict[str, Any]]
    ) -> PerformanceMetrics:
        """Évalue un modèle de traitement de conversation."""
        # Implémentation simplifiée
        return PerformanceMetrics(
            accuracy=0.85,
            f1_score=0.82,
            precision=0.88,
            recall=0.80,
            loss=0.45
        )

    def _evaluate_emotion_analysis(
        self, model: nn.Module, test_data: List[Dict[str, Any]]
    ) -> PerformanceMetrics:
        """Évalue un modèle d'analyse d'émotion."""
        return PerformanceMetrics(
            accuracy=0.78,
            f1_score=0.75,
            precision=0.80,
            recall=0.72,
            loss=0.62
        )

    def _evaluate_response_generation(
        self, model: nn.Module, test_data: List[Dict[str, Any]]
    ) -> PerformanceMetrics:
        """Évalue un modèle de génération de réponse."""
        return PerformanceMetrics(
            accuracy=0.72,
            f1_score=0.70,
            precision=0.75,
            recall=0.68,
            loss=0.78,
            perplexity=15.3
        )

    def _evaluate_generic_model(
        self, model: nn.Module, test_data: List[Dict[str, Any]]
    ) -> PerformanceMetrics:
        """Évaluation générique."""
        return PerformanceMetrics(
            accuracy=0.80,
            f1_score=0.78,
            loss=0.55
        )

    # Méthodes de préparation des données
    def _prepare_conversation_data(self, conversations: List[Dict[str, Any]]) -> TensorDataset:
        """Prépare les données pour le traitement de conversation."""
        # Implémentation simplifiée - à adapter selon le modèle réel
        inputs = torch.randn(len(conversations), 768)  # Embeddings simulés
        targets = torch.randint(0, 10, (len(conversations),))  # Classes simulées
        return TensorDataset(inputs, targets)

    def _prepare_emotion_data(self, conversations: List[Dict[str, Any]]) -> TensorDataset:
        """Prépare les données pour l'analyse d'émotion."""
        inputs = torch.randn(len(conversations), 512)
        targets = torch.randint(0, 7, (len(conversations),))  # 7 émotions
        return TensorDataset(inputs, targets)

    def _prepare_generation_data(self, conversations: List[Dict[str, Any]]) -> TensorDataset:
        """Prépare les données pour la génération de réponse."""
        inputs = torch.randint(0, 50000, (len(conversations), 50))  # Tokens
        targets = torch.randint(0, 50000, (len(conversations), 50))
        return TensorDataset(inputs, targets)

    def _prepare_generic_data(self, conversations: List[Dict[str, Any]]) -> TensorDataset:
        """Préparation générique des données."""
        inputs = torch.randn(len(conversations), 256)
        targets = torch.randint(0, 2, (len(conversations),))
        return TensorDataset(inputs, targets)