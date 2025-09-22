from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple


logger = logging.getLogger(__name__)


class ContextManager:
    """In-memory short-term conversation context store."""

    def __init__(self, max_history: int = 10) -> None:
        self.max_history = max_history
        self._history: Dict[str, Deque[Tuple[str, str]]] = defaultdict(lambda: deque(maxlen=max_history))

    def add_turn(self, conv_id: str, role: str, content: str) -> None:
        self._history[conv_id].append((role, content))

    def get_context(self, conv_id: str) -> List[Tuple[str, str]]:
        return list(self._history.get(conv_id, deque()))

    def clear(self, conv_id: str) -> None:
        self._history.pop(conv_id, None)
