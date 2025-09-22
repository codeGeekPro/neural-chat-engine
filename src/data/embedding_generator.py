from __future__ import annotations

from typing import List

from ..engine.model_manager import SimpleHashEmbedding
from ..models.base import ModelInfo


class EmbeddingGenerator:
    """Create sentence embeddings for texts.

    Uses a lightweight local embedding for demo. Swap with HF/SBERT in prod.
    """

    def __init__(self, model_name: str = "local-hash-embed") -> None:
        self.model = SimpleHashEmbedding(ModelInfo(name=model_name, provider="local", device="cpu"))
        self.model.load()

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed(texts)
