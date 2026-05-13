from core.event_bus import EventBus


def test_subscribe_and_emit() -> None:
    bus = EventBus()
    received = []

    bus.subscribe("test_event", lambda data: received.append(data))
    bus.emit("test_event", {"value": 42})

    assert received == [{"value": 42}]


def test_multiple_subscribers() -> None:
    bus = EventBus()
    log = []

    bus.subscribe("event", lambda data: log.append(("a", data)))
    bus.subscribe("event", lambda data: log.append(("b", data)))
    bus.emit("event", {"x": 1})

    assert ("a", {"x": 1}) in log
    assert ("b", {"x": 1}) in log


def test_emit_unknown_event_does_not_raise() -> None:
    bus = EventBus()
    bus.emit("no_subscribers", {})


def test_unsubscribe() -> None:
    bus = EventBus()
    received = []

    def handler(data: dict[str, int]) -> None:
        received.append(data)

    bus.subscribe("event", handler)
    bus.unsubscribe("event", handler)
    bus.emit("event", {"v": 1})

    assert received == []
