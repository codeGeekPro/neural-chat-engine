from __future__ import annotations

from typing import List, Optional

from ..engine.model_manager import SimpleHashEmbedding
from ..models.base import ModelInfo


class EmbeddingGenerator:
    """Create sentence embeddings for texts.

    Tries `sentence-transformers` for high-quality embeddings.
    Falls back to a lightweight local hash embedding when unavailable.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        prefer_local_fallback: bool = False,
    ) -> None:
        self._use_st: bool = False
        self._st_model = None
        self._fallback = SimpleHashEmbedding(ModelInfo(name="local-hash-embed", provider="local", device="cpu"))
        self._fallback.load()

        if prefer_local_fallback:
            # Force fallback only (useful for environments without heavy deps)
            return

        try:
            # Lazy import to avoid hard dependency when not needed
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer(model_name, device=device)
            # Run a tiny forward to ensure model is ready; catch runtime errors
            _ = self._st_model.encode(["hello"], normalize_embeddings=True)
            self._use_st = True
        except Exception:
            # Any failure -> keep fallback
            self._use_st = False
            self._st_model = None

    def encode(self, texts: List[str]) -> List[List[float]]:
        if self._use_st and self._st_model is not None:
            vecs = self._st_model.encode(texts, normalize_embeddings=True)
            # Ensure list of lists
            return [list(map(float, v)) for v in vecs]
        # Fallback
        return self._fallback.embed(texts)
