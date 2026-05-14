from collections import defaultdict
from collections.abc import Callable
from typing import Any


class EventBus:
    """Backend modules use this small pub/sub bus to communicate by event name."""

    def __init__(self) -> None:
        """Create an empty event handler registry."""
        self._handlers: dict[str, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        """Register a handler that should run whenever the event is emitted."""
        self._handlers[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        """Remove a previously registered handler from an event."""
        self._handlers[event] = [
            current_handler
            for current_handler in self._handlers[event]
            if current_handler != handler
        ]

    def emit(self, event: str, data: Any) -> None:
        """Call all handlers registered for an event with the provided payload."""
        for handler in list(self._handlers[event]):
            handler(data)
