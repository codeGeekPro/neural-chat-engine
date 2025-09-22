from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Protocol

from .model_manager import ModelManager


logger = logging.getLogger(__name__)


class ResponseStrategy(Protocol):
    def generate(self, prompt: str, context: Dict[str, Any]) -> str:  # pragma: no cover - protocol
        ...


@dataclass
class SimpleLLMStrategy:
    model_manager: ModelManager
    model_name: str = "default-generator"

    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        gen = self.model_manager.get_generator(self.model_name)
        return gen.predict({"prompt": prompt, "context": context})


class ResponseGenerator:
    """Strategy-based response generator."""

    def __init__(self, strategy: ResponseStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: ResponseStrategy) -> None:
        self._strategy = strategy

    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        return self._strategy.generate(prompt, context)
