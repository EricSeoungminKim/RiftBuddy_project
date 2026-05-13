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
    async def send(self, payload: str) -> None:
        pass


class WsServer:
    def __init__(
        self,
        event_bus: EventBus,
        game_state: GameState,
        host: str = WS_HOST,
        port: int = WS_PORT,
    ) -> None:
        self._bus = event_bus
        self._state = game_state
        self._host = host
        self._port = port
        self._clients: set[WebSocketServerProtocol] = set()

    async def start(self) -> None:
        async with websockets.serve(self._handle_client, self._host, self._port):
            await self._push_loop()

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        self._clients.add(websocket)
        await self._send_snapshot(websocket)
        try:
            async for message in websocket:
                await self._handle_message(json.loads(message))
        finally:
            self._clients.discard(websocket)

    async def _handle_message(self, message: dict[str, object]) -> None:
        if message.get("type") != "spell_clicked":
            return

        champion = message.get("champion")
        spell = message.get("spell")
        if not isinstance(champion, str) or not isinstance(spell, str):
            return

        self._bus.emit("spell_clicked", {"champion": champion, "spell": spell})

    async def _push_loop(self) -> None:
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
        snapshot = self._state.snapshot()
        now = time.time()
        mia = [
            {
                "champion": champion,
                "elapsed_sec": round(now - data["last_seen"], 1),
            }
            for champion, data in snapshot["mia"].items()
        ]
        spells = [
            {
                "champion": champion,
                "spell": spell,
                "remaining_sec": round(max(0.0, data["available_at"] - now), 1),
            }
            for champion, champion_spells in snapshot["spells"].items()
            for spell, data in champion_spells.items()
        ]
        return json.dumps({"type": "state_update", "mia": mia, "spells": spells})

    async def _send_snapshot(self, websocket: SendableWebSocket) -> None:
        await self._safe_send(websocket, self._build_payload())

    @staticmethod
    async def _safe_send(websocket: SendableWebSocket, payload: str) -> None:
        try:
            await websocket.send(payload)
        except websockets.ConnectionClosed:
            pass
