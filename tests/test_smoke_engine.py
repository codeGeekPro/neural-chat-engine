from __future__ import annotations

from src.engine import ChatbotEngine


def test_smoke_chat_flow():
    engine = ChatbotEngine()
    reply = engine.chat("conv-1", "Bonjour, qui es-tu ?")
    assert isinstance(reply, str)
    assert reply.startswith("[echo]")
