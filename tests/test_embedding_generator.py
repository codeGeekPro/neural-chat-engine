from __future__ import annotations

import importlib
import pytest

from src.data.embedding_generator import EmbeddingGenerator


def test_embedding_fallback_forced():
    eg = EmbeddingGenerator(prefer_local_fallback=True)
    vecs = eg.encode(["bonjour", "hello"])
    assert isinstance(vecs, list) and len(vecs) == 2
    assert all(isinstance(v, list) and len(v) > 0 for v in vecs)


@pytest.mark.skipif(not importlib.util.find_spec("sentence_transformers"), reason="sentence-transformers not installed")
def test_embedding_sentence_transformers_if_available():
    eg = EmbeddingGenerator()
    vecs = eg.encode(["bonjour", "hello"])
    assert isinstance(vecs, list) and len(vecs) == 2
    # MiniLM usually gives 384 dims
    assert len(vecs[0]) >= 128
