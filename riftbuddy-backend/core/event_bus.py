from collections import defaultdict
from collections.abc import Callable
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        self._handlers[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        self._handlers[event] = [
            current_handler
            for current_handler in self._handlers[event]
            if current_handler != handler
        ]

    def emit(self, event: str, data: Any) -> None:
        for handler in list(self._handlers[event]):
            handler(data)
