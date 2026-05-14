import argparse
import asyncio
import json
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import mss
import numpy as np

from core.event_bus import EventBus
from core.game_state import GameState
from core.minimap_config import MinimapRuntimeConfig
from core.ws_server import WsServer
from modules.asset_loader import AssetLoader
from modules.live_client import LiveClient, LiveClientError
from modules.minimap_region_detector import crop_region, detect_minimap_region
from modules.minimap_tracker import MinimapTracker
from modules.minimap_training import extract_templates_from_labels
from modules.minimap_training_recorder import TrainingSampleRecorder
from modules.spell_timer import SpellTimer
from settings import SETTINGS, BackendSettings


@dataclass(frozen=True)
class AppComponents:
    """Container for the backend services created at startup."""

    event_bus: EventBus
    game_state: GameState
    asset_loader: AssetLoader
    live_client: LiveClient
    runtime_config: MinimapRuntimeConfig
    minimap_tracker: MinimapTracker
    spell_timer: SpellTimer
    ws_server: WsServer


def create_app_components(
    settings: BackendSettings = SETTINGS,
    runtime_config: MinimapRuntimeConfig | None = None,
) -> AppComponents:
    """Create backend services and wire frontend spell clicks to SpellTimer."""
    event_bus = EventBus()
    game_state = GameState()
    asset_loader = AssetLoader()
    live_client = LiveClient(settings.live_client_all_game_data_url)
    runtime_config = runtime_config or create_runtime_config(settings)
    minimap_tracker = MinimapTracker(event_bus=event_bus, game_state=game_state)
    spell_timer = SpellTimer(event_bus=event_bus, game_state=game_state)
    ws_server = WsServer(event_bus=event_bus, game_state=game_state)

    event_bus.subscribe(
        "spell_clicked",
        lambda data: _handle_spell_clicked(spell_timer, data),
    )
    event_bus.subscribe(
        "spell_cleared",
        lambda data: _handle_spell_cleared(game_state, data),
    )
    event_bus.subscribe(
        "minimap_region_adjusted",
        lambda data: _handle_minimap_region_adjusted(
            runtime_config,
            game_state,
            data,
        ),
    )

    return AppComponents(
        event_bus=event_bus,
        game_state=game_state,
        asset_loader=asset_loader,
        live_client=live_client,
        runtime_config=runtime_config,
        minimap_tracker=minimap_tracker,
        spell_timer=spell_timer,
        ws_server=ws_server,
    )


def _handle_spell_clicked(spell_timer: SpellTimer, data: object) -> None:
    """Validate a spell_clicked payload and start the matching cooldown."""
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


def _handle_spell_cleared(game_state: GameState, data: object) -> None:
    """Validate a spell_cleared payload and remove the matching cooldown."""
    if not isinstance(data, dict):
        return

    champion = data.get("champion")
    spell = data.get("spell")
    if not isinstance(champion, str) or not isinstance(spell, str):
        return

    game_state.clear_spell(champion, spell)


def _handle_minimap_region_adjusted(
    runtime_config: MinimapRuntimeConfig,
    game_state: GameState,
    data: object,
) -> None:
    """Apply frontend calibration nudges to the active minimap capture region."""
    if not isinstance(data, dict):
        return

    runtime_config.adjust_region(data)
    game_state.update_minimap_debug(runtime_config.debug_snapshot())


async def minimap_scan_loop(
    tracker: MinimapTracker,
    game_champions: list[str],
    runtime_config: MinimapRuntimeConfig,
    training_recorder: TrainingSampleRecorder | None = None,
) -> None:
    """Capture the minimap region repeatedly and feed frames to MinimapTracker."""
    with mss.mss() as screen_capture:
        while True:
            now = time.time()
            capture_display = _primary_capture_display(screen_capture)
            if runtime_config.auto_region:
                image = np.array(screen_capture.grab(capture_display))
                full_frame = image[:, :, :3]
                detected_region = detect_minimap_region(full_frame, capture_display)
                if detected_region is not None:
                    runtime_config.region = detected_region
                frame = crop_region(full_frame, runtime_config.region, capture_display)
            else:
                image = np.array(screen_capture.grab(runtime_config.region))
                frame = image[:, :, :3]

            frame_metadata = tracker.process_frame(
                frame,
                tracked_champions=game_champions,
                now=now,
                minimap_region=runtime_config.region,
                capture_display=capture_display,
                debug_mode=runtime_config.debug_mode,
                calibration_mode=runtime_config.calibration_mode,
                debug_frame_path=runtime_config.debug_frame_path,
            )
            if training_recorder is not None:
                training_recorder.maybe_record(frame, now, frame_metadata)
            await asyncio.sleep(runtime_config.scan_interval_seconds)


async def run_backend(
    enable_minimap: bool = False,
    debug_minimap: bool = False,
    calibrate_minimap: bool = False,
    test_loadout_file: str | None = None,
    collect_minimap_training: bool = False,
    minimap_training_dir: str | None = None,
) -> None:
    """Run the WebSocket backend, optionally enabling live minimap capture."""
    print("Initializing backend components...")
    settings = SETTINGS
    if test_loadout_file:
        settings = replace(SETTINGS, test_enemy_loadout_file=test_loadout_file)
    if minimap_training_dir:
        settings = replace(settings, minimap_training_capture_dir=minimap_training_dir)

    runtime_config = create_runtime_config(
        settings,
        debug_minimap=debug_minimap,
        calibrate_minimap=calibrate_minimap,
        collect_minimap_training=collect_minimap_training,
    )
    components = create_app_components(settings=settings, runtime_config=runtime_config)
    components.game_state.update_minimap_debug(runtime_config.debug_snapshot())

    print("Starting WebSocket server...")
    print("Ready and listening for frontend connections!")
    await asyncio.gather(
        components.ws_server.start(),
        active_game_lifecycle(
            components=components,
            settings=settings,
            enable_minimap=enable_minimap,
        ),
    )


def create_runtime_config(
    settings: BackendSettings,
    debug_minimap: bool = False,
    calibrate_minimap: bool = False,
    collect_minimap_training: bool = False,
) -> MinimapRuntimeConfig:
    """Create mutable minimap runtime settings from backend constants and CLI flags."""
    return MinimapRuntimeConfig(
        region=settings.minimap_region,
        scan_interval_seconds=settings.scan_interval_seconds,
        auto_region=settings.minimap_auto_region,
        debug_mode=settings.minimap_debug_mode or debug_minimap,
        calibration_mode=settings.minimap_calibration_mode or calibrate_minimap,
        debug_frame_path=settings.minimap_debug_frame_path,
        training_capture_mode=(
            settings.minimap_training_capture_mode or collect_minimap_training
        ),
        training_capture_dir=settings.minimap_training_capture_dir,
        training_capture_interval_seconds=(
            settings.minimap_training_capture_interval_seconds
        ),
    )


def create_training_recorder(
    runtime_config: MinimapRuntimeConfig,
) -> TrainingSampleRecorder | None:
    """Create a raw minimap recorder when training capture is enabled."""
    if not runtime_config.training_capture_mode:
        return None

    return TrainingSampleRecorder(
        output_dir=runtime_config.training_capture_dir,
        interval_seconds=runtime_config.training_capture_interval_seconds,
    )


def load_learned_minimap_templates(
    tracker: MinimapTracker,
    settings: BackendSettings,
) -> int:
    """Load labeled minimap capture templates when fixture labels are available."""
    image_dir = Path(settings.minimap_training_image_dir)
    labels_path = Path(settings.minimap_training_labels_path)
    if not labels_path.exists():
        return 0

    templates = extract_templates_from_labels(image_dir, labels_path)
    tracker.load_template_images(templates)
    return sum(len(images) for images in templates.values())


async def load_enemy_loadout(
    live_client: LiveClient,
    settings: BackendSettings,
) -> list[dict[str, object]]:
    """Return test loadout or fetch real enemy loadout from the active game."""
    if settings.test_mode:
        if settings.test_enemy_loadout_file:
            return load_test_enemy_loadout_file(settings.test_enemy_loadout_file)
        return deepcopy(settings.test_enemy_loadout)
    return await live_client.fetch_enemy_loadout()


async def wait_for_active_game(
    live_client: LiveClient,
    settings: BackendSettings,
    game_state: GameState,
) -> list[dict[str, object]]:
    """Poll Live Client until an active game loadout is available."""
    while True:
        try:
            enemy_loadout = await load_enemy_loadout(live_client, settings)
        except LiveClientError:
            game_state.reset_match_state()
            game_state.update_game_status(
                "waiting_for_active_game",
                "Waiting for active League game",
            )
            await asyncio.sleep(settings.live_client_poll_interval_seconds)
            continue

        game_state.update_enemy_loadout(enemy_loadout)
        game_state.update_game_status("active_game", "Active game detected")
        return enemy_loadout


async def active_game_lifecycle(
    components: AppComponents,
    settings: BackendSettings,
    enable_minimap: bool,
) -> None:
    """Keep backend alive, waiting for games and restarting tracking per match."""
    while True:
        enemy_loadout = await wait_for_active_game(
            components.live_client,
            settings,
            components.game_state,
        )

        if not enable_minimap:
            await wait_until_active_game_ends(
                components.live_client,
                settings,
                components.game_state,
            )
            continue

        await run_active_minimap_session(components, settings, enemy_loadout)


async def run_active_minimap_session(
    components: AppComponents,
    settings: BackendSettings,
    enemy_loadout: list[dict[str, object]],
) -> None:
    """Load assets and scan minimap until the current active game ends."""
    game_champions = [str(item["champion"]) for item in enemy_loadout]
    print(f"Loading champion assets for: {game_champions}")
    icon_paths = await components.asset_loader.load_match_assets(game_champions)
    components.minimap_tracker.load_templates(icon_paths)
    learned_count = load_learned_minimap_templates(
        components.minimap_tracker,
        settings,
    )
    if learned_count:
        print(f"Loaded {learned_count} learned minimap icon template(s).")

    training_recorder = create_training_recorder(components.runtime_config)
    if training_recorder is not None:
        print(
            f"Recording raw minimap training samples to {training_recorder.output_dir}"
        )

    scan_task = asyncio.create_task(
        minimap_scan_loop(
            components.minimap_tracker,
            game_champions,
            components.runtime_config,
            training_recorder=training_recorder,
        )
    )
    end_task = asyncio.create_task(
        wait_until_active_game_ends(
            components.live_client,
            settings,
            components.game_state,
        )
    )
    done, pending = await asyncio.wait(
        {scan_task, end_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    for task in done:
        task.result()


async def wait_until_active_game_ends(
    live_client: LiveClient,
    settings: BackendSettings,
    game_state: GameState,
) -> None:
    """Poll until Live Client disappears, then return backend to waiting state."""
    if settings.test_mode:
        await asyncio.Future()

    while True:
        await asyncio.sleep(settings.live_client_poll_interval_seconds)
        try:
            await live_client.fetch_enemy_loadout()
        except LiveClientError:
            game_state.reset_match_state()
            game_state.update_game_status(
                "waiting_for_active_game",
                "Waiting for active League game",
            )
            return


def load_test_enemy_loadout_file(path: str) -> list[dict[str, object]]:
    """Load a non-secret TESTMODE enemy loadout JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Test enemy loadout file must contain a list.")

    loadout: list[dict[str, object]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each test enemy loadout entry must be an object.")

        champion = item.get("champion")
        spells = item.get("spells")
        if not isinstance(champion, str) or not isinstance(spells, list):
            raise ValueError("Each entry needs champion and spells fields.")
        if not all(isinstance(spell, str) for spell in spells):
            raise ValueError("Each spells field must be a list of strings.")

        loadout.append({"champion": champion, "spells": deepcopy(spells)})
    return loadout


def _primary_capture_display(screen_capture: mss.mss) -> dict[str, int]:
    """Return the mss primary monitor rectangle in capture coordinate space."""
    monitor = screen_capture.monitors[1]
    return {
        "top": int(monitor["top"]),
        "left": int(monitor["left"]),
        "width": int(monitor["width"]),
        "height": int(monitor["height"]),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options for backend runtime mode."""
    parser = argparse.ArgumentParser(description="Run RiftBuddy backend")
    parser.add_argument(
        "--enable-minimap",
        action="store_true",
        help="Enable DDragon asset loading and minimap screen capture",
    )
    parser.add_argument(
        "--debug-minimap",
        action="store_true",
        help="Broadcast minimap confidence scores and save the latest crop",
    )
    parser.add_argument(
        "--calibrate-minimap",
        action="store_true",
        help="Show the minimap capture rectangle and enable live region nudges",
    )
    parser.add_argument(
        "--test-loadout-file",
        help="Load TESTMODE enemy champions and spells from a JSON file",
    )
    parser.add_argument(
        "--collect-minimap-training",
        action="store_true",
        help="Record raw minimap crops and metadata for later dataset curation",
    )
    parser.add_argument(
        "--minimap-training-dir",
        help="Override the raw minimap training capture output directory",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for starting the RiftBuddy backend."""
    args = parse_args()
    asyncio.run(
        run_backend(
            enable_minimap=args.enable_minimap or args.collect_minimap_training,
            debug_minimap=args.debug_minimap,
            calibrate_minimap=args.calibrate_minimap,
            test_loadout_file=args.test_loadout_file,
            collect_minimap_training=args.collect_minimap_training,
            minimap_training_dir=args.minimap_training_dir,
        )
    )


if __name__ == "__main__":
    main()
