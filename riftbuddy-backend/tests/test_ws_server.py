import json
from unittest.mock import patch

import pytest

from core.event_bus import EventBus
from core.game_state import GameState
from core.ws_server import WsServer


def make_server() -> tuple[WsServer, EventBus, GameState]:
    bus = EventBus()
    state = GameState()
    return WsServer(event_bus=bus, game_state=state), bus, state


def test_build_payload_formats_state_update() -> None:
    server, _, state = make_server()
    state.update_mia("Zed", last_seen=1000.0)
    state.update_spell("Jinx", "Flash", available_at=1305.0)

    with patch("core.ws_server.time.time", return_value=1015.0):
        payload = json.loads(server._build_payload())

    assert payload == {
        "type": "state_update",
        "mia": [{"champion": "Zed", "elapsed_sec": 15.0}],
        "spells": [{"champion": "Jinx", "spell": "Flash", "remaining_sec": 290.0}],
    }


@pytest.mark.asyncio
async def test_handle_spell_clicked_emits_event() -> None:
    server, bus, _ = make_server()
    received = []
    bus.subscribe("spell_clicked", received.append)

    await server._handle_message(
        {"type": "spell_clicked", "champion": "Jinx", "spell": "Flash"}
    )

    assert received == [{"champion": "Jinx", "spell": "Flash"}]


@pytest.mark.asyncio
async def test_handle_unknown_message_ignores_event() -> None:
    server, bus, _ = make_server()
    received = []
    bus.subscribe("spell_clicked", received.append)

    await server._handle_message({"type": "unknown"})

    assert received == []


@pytest.mark.asyncio
async def test_send_snapshot_sends_payload() -> None:
    server, _, _ = make_server()
    sent: list[str] = []

    class FakeWebSocket:
        async def send(self, payload: str) -> None:
            sent.append(payload)

    await server._send_snapshot(FakeWebSocket())

    assert json.loads(sent[0]) == {"type": "state_update", "mia": [], "spells": []}
