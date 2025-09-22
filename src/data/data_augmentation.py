from __future__ import annotations

import random
from typing import List, Tuple


class DataAugmentation:
    """Augment conversations with simple heuristics.

    Note: For production, consider using back-translation via APIs or
    paraphrasing models. Here we provide lightweight, deterministic options.
    """

    def __init__(self, seed: int = 42) -> None:
        self.random = random.Random(seed)
        # Minimal synonym dictionary for demo; extend per language
        self.synonyms = {
            "bonjour": ["salut", "coucou"],
            "hello": ["hi", "hey"],
            "merci": ["thanks", "thx"],
            "problème": ["souci", "bug"],
        }

    def synonym_replace(self, text: str, prob: float = 0.2) -> str:
        tokens = text.split()
        for i, tok in enumerate(tokens):
            lower = tok.lower()
            if lower in self.synonyms and self.random.random() < prob:
                cand = self.random.choice(self.synonyms[lower])
                tokens[i] = cand
        return " ".join(tokens)

    def role_masking(self, convo: List[Tuple[str, str]], prob: float = 0.1) -> List[Tuple[str, str]]:
        out = []
        for role, content in convo:
            if role == "system" and self.random.random() < prob:
                # Occasionally drop verbose system prompts
                continue
            out.append((role, content))
        return out

    def augment_conversation(self, convo: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        return [(r, self.synonym_replace(c)) for r, c in self.role_masking(convo)]
