from __future__ import annotations

import logging
from typing import Dict, Optional

from ..models.base import BaseClassifier, BaseEmbedding, BaseGenerator, ModelInfo


logger = logging.getLogger(__name__)


class SimpleEchoGenerator(BaseGenerator):
    def load(self, **kwargs):
        self._loaded = True
        logger.debug("Loaded SimpleEchoGenerator")

    def predict(self, inputs: Dict[str, str], **kwargs) -> str:
        self.ensure_loaded()
        prompt = inputs.get("prompt", "")
        return f"[echo] {prompt}"


class SimpleRuleClassifier(BaseClassifier):
    def load(self, **kwargs):
        self._loaded = True
        logger.debug("Loaded SimpleRuleClassifier")

    def predict(self, inputs: str, **kwargs):
        self.ensure_loaded()
        label = "greeting" if any(w in inputs.lower() for w in ["bonjour", "hello", "hi"]) else "other"
        return {"label": label, "confidence": 0.9}


class SimpleHashEmbedding(BaseEmbedding):
    def load(self, **kwargs):
        self._loaded = True
        logger.debug("Loaded SimpleHashEmbedding")

    def embed(self, texts, **kwargs):
        self.ensure_loaded()
        # Dummy deterministic embedding: normalized char code sums per token size 8
        vecs = []
        for t in texts:
            s = sum(ord(c) for c in t)
            vec = [(s % (i + 7)) / (i + 7) for i in range(8)]
            vecs.append(vec)
        return vecs


class ModelManager:
    """Factory/registry for models based on config or explicit names."""

    def __init__(self) -> None:
        self._generators: Dict[str, BaseGenerator] = {}
        self._classifiers: Dict[str, BaseClassifier] = {}
        self._embeddings: Dict[str, BaseEmbedding] = {}

    def get_generator(self, name: str) -> BaseGenerator:
        if name not in self._generators:
            model = SimpleEchoGenerator(ModelInfo(name=name, provider="local", device="cpu"))
            model.load()
            self._generators[name] = model
        return self._generators[name]

    def get_classifier(self, name: str) -> BaseClassifier:
        if name not in self._classifiers:
            model = SimpleRuleClassifier(ModelInfo(name=name, provider="local", device="cpu"))
            model.load()
            self._classifiers[name] = model
        return self._classifiers[name]

    def get_embedding(self, name: str) -> BaseEmbedding:
        if name not in self._embeddings:
            model = SimpleHashEmbedding(ModelInfo(name=name, provider="local", device="cpu"))
            model.load()
            self._embeddings[name] = model
        return self._embeddings[name]
