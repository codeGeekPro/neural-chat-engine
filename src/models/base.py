"""
Base abstract classes for AI models used in the Neural Chat Engine.

Defines common model lifecycle and interfaces with consistent logging and
error handling across classifiers, generators, and embedding models.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar


logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when model operations are invoked before loading."""


class ModelOperationError(RuntimeError):
    """Raised when a model operation fails."""


@dataclass
class ModelInfo:
    name: str
    version: str = "0.1.0"
    provider: Optional[str] = None
    device: Optional[str] = None


I = TypeVar("I")  # Input type
O = TypeVar("O")  # Output type


class BaseModel(ABC):
    """Base abstract model with common lifecycle and metadata."""

    def __init__(self, info: ModelInfo) -> None:
        self.info = info
        self._loaded: bool = False
        logger.debug("Initialized model %s (provider=%s, device=%s)", info.name, info.provider, info.device)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        if not self._loaded:
            raise ModelNotLoadedError(f"Model '{self.info.name}' is not loaded")

    @abstractmethod
    def load(self, **kwargs: Any) -> None:
        """Load model resources (weights, tokenizer, etc.)."""

    @abstractmethod
    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        """Run inference on inputs and return outputs."""

    def train(self, data: Any, **kwargs: Any) -> Any:  # optional
        logger.info("Training not implemented for model '%s'", self.info.name)
        raise NotImplementedError("train method not implemented")

    def evaluate(self, data: Any, **kwargs: Any) -> Dict[str, Any]:  # optional
        logger.info("Evaluate not implemented for model '%s'", self.info.name)
        raise NotImplementedError("evaluate method not implemented")


class BaseClassifier(BaseModel, ABC):
    """Classifier interface for intent, routing, etc."""

    @abstractmethod
    def predict(self, inputs: str, **kwargs: Any) -> Dict[str, Any]:
        """Return label/confidence dictionary."""


class BaseGenerator(BaseModel, ABC):
    """Text generative model interface."""

    @abstractmethod
    def predict(self, inputs: Dict[str, Any], **kwargs: Any) -> str:
        """Generate a response string for the given input context."""


class BaseEmbedding(BaseModel, ABC):
    """Embedding model interface."""

    @abstractmethod
    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Return embeddings for a list of texts."""

    # For uniformity, map predict to embed when called generically
    def predict(self, inputs: List[str], **kwargs: Any) -> List[List[float]]:
        return self.embed(inputs, **kwargs)
