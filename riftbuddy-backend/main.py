import argparse
import asyncio
import time
from dataclasses import dataclass

import mss
import numpy as np

from core.event_bus import EventBus
from core.game_state import GameState
from core.ws_server import WsServer
from modules.asset_loader import AssetLoader
from modules.minimap_tracker import MinimapTracker
from modules.spell_timer import SpellTimer

MINIMAP_REGION = {"top": 870, "left": 1710, "width": 210, "height": 210}
SCAN_INTERVAL_SECONDS = 1.0
DEFAULT_GAME_CHAMPIONS = ["Zed", "Jinx", "Thresh", "Jayce", "LeeSin"]


@dataclass(frozen=True)
class AppComponents:
    event_bus: EventBus
    game_state: GameState
    asset_loader: AssetLoader
    minimap_tracker: MinimapTracker
    spell_timer: SpellTimer
    ws_server: WsServer


def create_app_components() -> AppComponents:
    event_bus = EventBus()
    game_state = GameState()
    asset_loader = AssetLoader()
    minimap_tracker = MinimapTracker(event_bus=event_bus, game_state=game_state)
    spell_timer = SpellTimer(event_bus=event_bus, game_state=game_state)
    ws_server = WsServer(event_bus=event_bus, game_state=game_state)

    event_bus.subscribe(
        "spell_clicked",
        lambda data: _handle_spell_clicked(spell_timer, data),
    )

    return AppComponents(
        event_bus=event_bus,
        game_state=game_state,
        asset_loader=asset_loader,
        minimap_tracker=minimap_tracker,
        spell_timer=spell_timer,
        ws_server=ws_server,
    )


def _handle_spell_clicked(spell_timer: SpellTimer, data: object) -> None:
    if not isinstance(data, dict):
        return

    champion = data.get("champion")
    spell = data.get("spell")
    if not isinstance(champion, str) or not isinstance(spell, str):
        return

    try:
        spell_timer.trigger(champion, spell, triggered_at=time.time())
    except ValueError as error:
        print(f"[warn] {error}")


async def minimap_scan_loop(
    tracker: MinimapTracker,
    game_champions: list[str],
    minimap_region: dict[str, int] | None = None,
) -> None:
    region = minimap_region or MINIMAP_REGION
    with mss.mss() as screen_capture:
        while True:
            image = np.array(screen_capture.grab(region))
            frame = image[:, :, :3]
            tracker.process_frame(
                frame,
                tracked_champions=game_champions,
                now=time.time(),
            )
            await asyncio.sleep(SCAN_INTERVAL_SECONDS)


async def run_backend(enable_minimap: bool = False) -> None:
    components = create_app_components()
    if not enable_minimap:
        await components.ws_server.start()
        return

    game_champions = DEFAULT_GAME_CHAMPIONS
    icon_paths = await components.asset_loader.load_match_assets(game_champions)
    components.minimap_tracker.load_templates(icon_paths)
    await asyncio.gather(
        components.ws_server.start(),
        minimap_scan_loop(components.minimap_tracker, game_champions),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RiftBuddy backend")
    parser.add_argument(
        "--enable-minimap",
        action="store_true",
        help="Enable DDragon asset loading and minimap screen capture",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_backend(enable_minimap=args.enable_minimap))


if __name__ == "__main__":
    main()
