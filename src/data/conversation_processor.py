from __future__ import annotations

import html
import re
import unicodedata
from typing import Dict, Iterable, List, Tuple


class ConversationProcessor:
    """Nettoyage et normalisation de texte multilingue pour conversations."""

    URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    INLINE_CODE_RE = re.compile(r"`[^`]+`")
    MULTISPACE_RE = re.compile(r"\s{2,}")

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Unescape HTML
        text = html.unescape(text)
        # Remove code blocks and inline code (keep placeholders)
        text = self.CODE_BLOCK_RE.sub("<code_block>", text)
        text = self.INLINE_CODE_RE.sub("<code>", text)
        # Remove URLs
        text = self.URL_RE.sub("<url>", text)
        # Strip control characters
        text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C")
        # Normalize unicode and whitespace
        text = unicodedata.normalize("NFKC", text)
        text = self.MULTISPACE_RE.sub(" ", text)
        return text.strip()

    def normalize_conversation(self, convo: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Normalize a conversation [(role, content), ...]."""
        norm: List[Tuple[str, str]] = []
        for role, content in convo:
            role = role.lower().strip()
            if role not in {"user", "assistant", "system"}:
                role = "user"
            norm.append((role, self.clean_text(content)))
        return norm

    def normalize_many(self, convos: Iterable[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
        return [self.normalize_conversation(c) for c in convos]
