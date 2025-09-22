from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .events import EventBus


logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # 'user' or 'assistant' or 'system'
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    id: str
    messages: List[Message] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.messages.append(Message(role=role, content=content, meta=meta or {}))


class ConversationManager:
    """Manages conversations and emits lifecycle events via the EventBus."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._conversations: Dict[str, Conversation] = {}

    def start(self, conv_id: str, meta: Optional[Dict[str, Any]] = None) -> Conversation:
        conv = Conversation(id=conv_id, meta=meta or {})
        self._conversations[conv_id] = conv
        self.bus.publish("conversation_started", {"conversation": conv})
        logger.debug("Conversation started: %s", conv_id)
        return conv

    def get(self, conv_id: str) -> Optional[Conversation]:
        return self._conversations.get(conv_id)

    def add_user_message(self, conv_id: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        conv = self._require(conv_id)
        conv.add("user", content, meta)
        self.bus.publish("user_message", {"conversation": conv, "content": content})

    def add_assistant_message(self, conv_id: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        conv = self._require(conv_id)
        conv.add("assistant", content, meta)
        self.bus.publish("assistant_message", {"conversation": conv, "content": content})

    def end(self, conv_id: str) -> None:
        conv = self._require(conv_id)
        self.bus.publish("conversation_ended", {"conversation": conv})
        logger.debug("Conversation ended: %s", conv_id)

    def history(self, conv_id: str, limit: Optional[int] = None) -> List[Message]:
        conv = self._require(conv_id)
        msgs = conv.messages
        return msgs[-limit:] if limit else msgs

    def _require(self, conv_id: str) -> Conversation:
        conv = self._conversations.get(conv_id)
        if not conv:
            raise KeyError(f"Conversation '{conv_id}' not found")
        return conv
