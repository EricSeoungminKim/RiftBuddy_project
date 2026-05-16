import asyncio
import json
import time
from typing import Protocol

import websockets
from websockets.server import WebSocketServerProtocol

from core.event_bus import EventBus
from core.game_state import GameState

WS_HOST = "localhost"
WS_PORT = 8765
PUSH_INTERVAL_SECONDS = 0.2


class SendableWebSocket(Protocol):
    """Minimal interface needed by snapshot sending helpers."""

    async def send(self, payload: str) -> None:
        """Send a serialized payload to a websocket-like object."""
        pass


class WsServer:
    """Local WebSocket server that syncs backend state to the Electron overlay."""

    def __init__(
        self,
        event_bus: EventBus,
        game_state: GameState,
        host: str = WS_HOST,
        port: int = WS_PORT,
    ) -> None:
        """Store shared backend dependencies and connection settings."""
        self._bus = event_bus
        self._state = game_state
        self._host = host
        self._port = port
        self._clients: set[WebSocketServerProtocol] = set()

    async def start(self) -> None:
        """Start accepting overlay clients and keep pushing state updates."""
        async with websockets.serve(self._handle_client, self._host, self._port):
            await self._push_loop()

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Register one client, send initial state, and process incoming messages."""
        self._clients.add(websocket)
        await self._send_snapshot(websocket)
        try:
            async for message in websocket:
                await self._handle_message(json.loads(message))
        finally:
            self._clients.discard(websocket)

    async def _handle_message(self, message: dict[str, object]) -> None:
        """Translate frontend messages into backend events."""
        message_type = message.get("type")
        if message_type == "minimap_region_adjusted":
            self._bus.emit(message_type, message)
            return

        if message_type not in {"spell_clicked", "spell_cleared"}:
            return

        champion = message.get("champion")
        spell = message.get("spell")
        if not isinstance(champion, str) or not isinstance(spell, str):
            return

        self._bus.emit(message_type, {"champion": champion, "spell": spell})

    async def _push_loop(self) -> None:
        """Broadcast current game state to connected clients every 200 ms."""
        while True:
            await asyncio.sleep(PUSH_INTERVAL_SECONDS)
            if not self._clients:
                continue

            payload = self._build_payload()
            await asyncio.gather(
                *[
                    self._safe_send(websocket, payload)
                    for websocket in list(self._clients)
                ],
                return_exceptions=True,
            )

    def _build_payload(self) -> str:
        """Serialize GameState into the overlay's state_update message shape."""
        snapshot = self._state.snapshot()
        now = time.time()
        mia = [
            {
                "champion": champion,
                "elapsed_sec": round(now - data["last_seen"], 1),
            }
            for champion, data in snapshot["mia"].items()
        ]
        spells = []
        for champion, champion_spells in snapshot["spells"].items():
            for spell, data in champion_spells.items():
                remaining_sec = data["available_at"] - now
                if remaining_sec <= 0:
                    continue
                spells.append(
                    {
                        "champion": champion,
                        "spell": spell,
                        "remaining_sec": round(remaining_sec, 1),
                    }
                )
        return json.dumps(
            {
                "type": "state_update",
                "mia": mia,
                "spells": spells,
                "enemy_loadout": snapshot["enemy_loadout"],
                "minimap_debug": snapshot["minimap_debug"],
                "game_status": snapshot["game_status"],
            }
        )

    async def _send_snapshot(self, websocket: SendableWebSocket) -> None:
        """Send the current full state to a newly connected client."""
        await self._safe_send(websocket, self._build_payload())

    @staticmethod
    async def _safe_send(websocket: SendableWebSocket, payload: str) -> None:
        """Send a payload and ignore clients that disconnected mid-send."""
        try:
            await websocket.send(payload)
        except websockets.ConnectionClosed:
            pass
