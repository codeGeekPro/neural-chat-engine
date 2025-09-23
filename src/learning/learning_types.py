"""Types et structures pour l'apprentissage continu."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class LearningTask(Enum):
    """Types de tâches d'apprentissage."""
    CONVERSATION_PROCESSING = "conversation_processing"
    EMOTION_ANALYSIS = "emotion_analysis"
    RESPONSE_GENERATION = "response_generation"
    PERSONALITY_ADAPTATION = "personality_adaptation"
    MEMORY_MANAGEMENT = "memory_management"


class ModelVersionStatus(Enum):
    """Statuts des versions de modèle."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


@dataclass
class ModelVersion:
    """Version d'un modèle avec métadonnées."""

    model_name: str
    version: str
    task: LearningTask
    created_at: datetime = field(default_factory=datetime.now)
    status: ModelVersionStatus = ModelVersionStatus.ACTIVE
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_data_size: int = 0
    model_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "task": self.task.value,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "performance_metrics": self.performance_metrics,
            "training_data_size": self.training_data_size,
            "model_path": str(self.model_path) if self.model_path else None,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelVersion:
        """Crée une instance depuis un dictionnaire."""
        return cls(
            model_name=data["model_name"],
            version=data["version"],
            task=LearningTask(data["task"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            status=ModelVersionStatus(data["status"]),
            performance_metrics=data["performance_metrics"],
            training_data_size=data["training_data_size"],
            model_path=Path(data["model_path"]) if data.get("model_path") else None,
            metadata=data["metadata"]
        )


@dataclass
class PerformanceMetrics:
    """Métriques de performance d'un modèle."""

    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    loss: float = float('inf')
    perplexity: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "loss": self.loss,
            "perplexity": self.perplexity,
            "custom_metrics": self.custom_metrics,
            "evaluated_at": self.evaluated_at.isoformat()
        }

    def is_better_than(self, other: PerformanceMetrics, threshold: float = 0.01) -> bool:
        """Vérifie si ces métriques sont meilleures qu'une autre version."""
        # Critère principal : la perte (loss) doit être significativement meilleure
        if self.loss < other.loss * (1 - threshold):
            return True

        # Critères secondaires
        accuracy_improved = self.accuracy > other.accuracy + threshold
        f1_improved = self.f1_score > other.f1_score + threshold

        return accuracy_improved or f1_improved

    def get_overall_score(self) -> float:
        """Calcule un score global de performance."""
        base_score = (self.accuracy + self.f1_score) / 2

        # Pénalise la haute perte
        if self.loss > 1.0:
            penalty = min(0.5, self.loss / 10)
            base_score *= (1 - penalty)

        return base_score


@dataclass
class ElasticWeightConsolidation:
    """Implémentation d'EWC (Elastic Weight Consolidation)."""

    importance_weights: Dict[str, np.ndarray] = field(default_factory=dict)
    fisher_information: Dict[str, np.ndarray] = field(default_factory=dict)
    lambda_ewc: float = 0.1  # Coefficient de régularisation

    def compute_fisher_information(
        self,
        model,
        dataloader,
        device: str = "cpu"
    ) -> None:
        """Calcule l'information de Fisher pour les paramètres importants."""
        model.eval()

        # Accumulateurs pour l'information de Fisher
        fisher_accumulators = {}

        for batch in dataloader:
            model.zero_grad()

            # Forward pass
            inputs = batch["inputs"].to(device)
            outputs = model(inputs)

            # Log-likelihood (pour classification)
            if hasattr(outputs, 'log_softmax'):
                log_likelihood = outputs.log_softmax(dim=-1)
            else:
                # Pour les autres tâches, approximation
                log_likelihood = -torch.nn.functional.mse_loss(
                    outputs, batch["targets"].to(device), reduction='none'
                )

            # Calcul du gradient par rapport aux log-likelihoods
            for i in range(log_likelihood.size(0)):
                model.zero_grad()
                log_likelihood[i].sum().backward(retain_graph=True)

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if name not in fisher_accumulators:
                            fisher_accumulators[name] = param.grad.data.clone() ** 2
                        else:
                            fisher_accumulators[name] += param.grad.data.clone() ** 2

        # Normalisation
        num_samples = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else len(dataloader)
        for name in fisher_accumulators:
            fisher_accumulators[name] /= num_samples

        self.fisher_information = fisher_accumulators

    def consolidate_weights(self, model) -> None:
        """Consolide les poids actuels comme importants."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.importance_weights[name] = param.data.clone()

    def ewc_loss(self, model) -> torch.Tensor:
        """Calcule la perte EWC pour régulariser l'apprentissage."""
        loss = 0.0

        for name, param in model.named_parameters():
            if name in self.importance_weights and name in self.fisher_information:
                # Différence par rapport aux poids consolidés
                weight_diff = param - self.importance_weights[name]

                # Pénalisation basée sur l'information de Fisher
                fisher_weight = self.fisher_information[name]
                loss += (fisher_weight * weight_diff.pow(2)).sum()

        return self.lambda_ewc * loss


@dataclass
class LearningSchedule:
    """Planification de l'apprentissage."""

    task: LearningTask
    trigger_condition: str  # Condition pour déclencher l'apprentissage
    performance_threshold: float = 0.85
    min_data_points: int = 100
    max_training_time: int = 3600  # secondes
    last_scheduled: Optional[datetime] = None
    schedule_metadata: Dict[str, Any] = field(default_factory=dict)

    def should_trigger(
        self,
        current_performance: float,
        new_data_points: int,
        time_since_last: int
    ) -> bool:
        """Vérifie si l'apprentissage devrait être déclenché."""
        if current_performance < self.performance_threshold:
            return True

        if new_data_points >= self.min_data_points:
            return True

        # Vérification temporelle (toutes les 24h minimum)
        if time_since_last > 86400:  # 24 heures en secondes
            return True

        return False


@dataclass
class TrainingBatch:
    """Lot d'entraînement pour l'apprentissage continu."""

    task: LearningTask
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    batch_id: str = field(default_factory=lambda: f"batch_{datetime.now().timestamp()}")

    def __len__(self) -> int:
        """Retourne la taille du lot."""
        return len(self.data)

    def get_statistics(self) -> Dict[str, Any]:
        """Calcule des statistiques sur le lot."""
        if not self.data:
            return {}

        # Statistiques de base
        stats = {
            "size": len(self.data),
            "task": self.task.value,
            "created_at": self.created_at.isoformat()
        }

        # Statistiques spécifiques selon le type de tâche
        if self.task == LearningTask.CONVERSATION_PROCESSING:
            text_lengths = [len(item.get("text", "")) for item in self.data]
            stats.update({
                "avg_text_length": np.mean(text_lengths),
                "max_text_length": max(text_lengths),
                "min_text_length": min(text_lengths)
            })

        return stats


# Import torch ici pour éviter les imports circulaires
try:
    import torch
except ImportError:
    torch = None