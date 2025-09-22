"""
Simple event bus to support Observer pattern for conversation lifecycle events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


Callback = Callable[[str, Dict[str, Any]], None]


@dataclass
class Event:
    name: str
    payload: Dict[str, Any]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callback]] = {}

    def subscribe(self, event_name: str, callback: Callback) -> None:
        self._subscribers.setdefault(event_name, []).append(callback)

    def unsubscribe(self, event_name: str, callback: Callback) -> None:
        if event_name in self._subscribers:
            self._subscribers[event_name] = [cb for cb in self._subscribers[event_name] if cb != callback]

    def publish(self, event_name: str, payload: Dict[str, Any]) -> None:
        for cb in self._subscribers.get(event_name, []):
            cb(event_name, payload)
