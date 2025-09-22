from __future__ import annotations

import logging
from typing import Any, Dict

from .conversation import ConversationManager
from .context import ContextManager
from .events import EventBus
from .model_manager import ModelManager
from .response import ResponseGenerator, SimpleLLMStrategy


logger = logging.getLogger(__name__)


class ChatbotEngine:
    def __init__(self) -> None:
        self.events = EventBus()
        self.conversations = ConversationManager(self.events)
        self.context = ContextManager(max_history=10)
        self.models = ModelManager()
        self.responder = ResponseGenerator(SimpleLLMStrategy(self.models))

        # Wire event listeners to update context memory
        self.events.subscribe("user_message", self._on_user_message)
        self.events.subscribe("assistant_message", self._on_assistant_message)

    def _on_user_message(self, event: str, payload: Dict[str, Any]) -> None:
        conv = payload["conversation"]
        content = payload.get("content", "")
        self.context.add_turn(conv.id, "user", content)

    def _on_assistant_message(self, event: str, payload: Dict[str, Any]) -> None:
        conv = payload["conversation"]
        content = payload.get("content", "")
        self.context.add_turn(conv.id, "assistant", content)

    def start_conversation(self, conv_id: str) -> None:
        self.conversations.start(conv_id)

    def chat(self, conv_id: str, user_text: str) -> str:
        if not self.conversations.get(conv_id):
            self.start_conversation(conv_id)
        # Record user message and build context
        self.conversations.add_user_message(conv_id, user_text)
        ctx_list = self.context.get_context(conv_id)
        context = {"history": ctx_list}
        # Generate response
        reply = self.responder.generate(user_text, context)
        self.conversations.add_assistant_message(conv_id, reply)
        return reply
