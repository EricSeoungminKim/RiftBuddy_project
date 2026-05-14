import json
from itertools import count
from unittest.mock import patch

import cv2
import numpy as np

from ai.chat_agent import ChatAgent
from ai.rag_engine import RagEngine
from ai.vector_store import VectorStore
from main import create_app_components
from modules.live_client import LiveClient, LiveClientError
from settings import BackendSettings


def test_ai_stubs_import() -> None:
    assert isinstance(ChatAgent(), ChatAgent)
    assert isinstance(RagEngine(), RagEngine)
    assert isinstance(VectorStore(), VectorStore)


def test_create_app_components_wires_spell_clicked() -> None:
    components = create_app_components()

    with patch("main.time.time", return_value=1000.0):
        components.event_bus.emit(
            "spell_clicked",
            {"champion": "Jinx", "spell": "Flash"},
        )

    spells = components.game_state.get_spells()
    assert spells["Jinx"]["Flash"]["available_at"] == 1300.0


def test_create_app_components_wires_spell_cleared() -> None:
    components = create_app_components()
    components.game_state.update_spell("Jinx", "Flash", available_at=1300.0)

    components.event_bus.emit(
        "spell_cleared",
        {"champion": "Jinx", "spell": "Flash"},
    )

    spells = components.game_state.get_spells()
    assert "Flash" not in spells.get("Jinx", {})


def test_create_app_components_has_live_client() -> None:
    components = create_app_components()

    assert isinstance(components.live_client, LiveClient)


def test_create_app_components_wires_minimap_region_adjustment() -> None:
    components = create_app_components()

    components.event_bus.emit("minimap_region_adjusted", {"left": -10, "width": 20})

    assert components.runtime_config.region == {
        "top": 790,
        "left": 1620,
        "width": 310,
        "height": 290,
    }
    assert components.game_state.get_minimap_debug()["region"] == {
        "top": 790,
        "left": 1620,
        "width": 310,
        "height": 290,
    }


async def test_load_enemy_loadout_uses_test_mode() -> None:
    from main import load_enemy_loadout

    test_loadout = [{"champion": "Zed", "spells": ["Flash", "Ignite"]}]
    settings = BackendSettings(
        test_mode=True,
        minimap_region={"top": 0, "left": 0, "width": 1, "height": 1},
        scan_interval_seconds=1.0,
        test_enemy_loadout=test_loadout,
        live_client_all_game_data_url="https://127.0.0.1:2999/liveclientdata/allgamedata",
    )

    class FakeLiveClient:
        async def fetch_enemy_loadout(self) -> list[dict[str, object]]:
            raise AssertionError("test mode should not call Live Client API")

    loaded = await load_enemy_loadout(FakeLiveClient(), settings)
    loaded[0]["champion"] = "Changed"

    assert test_loadout == [{"champion": "Zed", "spells": ["Flash", "Ignite"]}]


async def test_load_enemy_loadout_uses_test_loadout_file(tmp_path) -> None:
    from main import load_enemy_loadout

    loadout_path = tmp_path / "loadout.json"
    loadout_path.write_text(
        '[{"champion": "Karma", "spells": ["Flash", "Ignite"]}]',
        encoding="utf-8",
    )
    settings = BackendSettings(
        test_mode=True,
        minimap_region={"top": 0, "left": 0, "width": 1, "height": 1},
        scan_interval_seconds=1.0,
        test_enemy_loadout=[],
        live_client_all_game_data_url="https://127.0.0.1:2999/liveclientdata/allgamedata",
        test_enemy_loadout_file=str(loadout_path),
    )

    class FakeLiveClient:
        async def fetch_enemy_loadout(self) -> list[dict[str, object]]:
            raise AssertionError("test mode should not call Live Client API")

    assert await load_enemy_loadout(FakeLiveClient(), settings) == [
        {"champion": "Karma", "spells": ["Flash", "Ignite"]}
    ]


async def test_load_enemy_loadout_uses_live_client_when_not_test_mode() -> None:
    from main import load_enemy_loadout

    settings = BackendSettings(
        test_mode=False,
        minimap_region={"top": 0, "left": 0, "width": 1, "height": 1},
        scan_interval_seconds=1.0,
        test_enemy_loadout=[],
        live_client_all_game_data_url="https://127.0.0.1:2999/liveclientdata/allgamedata",
    )

    class FakeLiveClient:
        async def fetch_enemy_loadout(self) -> list[dict[str, object]]:
            return [{"champion": "Garen", "spells": ["Flash", "Teleport"]}]

    assert await load_enemy_loadout(FakeLiveClient(), settings) == [
        {"champion": "Garen", "spells": ["Flash", "Teleport"]}
    ]


def test_load_learned_minimap_templates_from_settings(tmp_path) -> None:
    from main import load_learned_minimap_templates

    image_dir = tmp_path / "img"
    image_dir.mkdir()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[38:62, 38:62] = [20, 120, 220]
    cv2.imwrite(str(image_dir / "sample.png"), image)
    labels_path = image_dir / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "sample.png": {
                    "champions": [{"champion": "Nasus", "center": [50, 50], "size": 24}]
                }
            }
        ),
        encoding="utf-8",
    )
    settings = BackendSettings(
        test_mode=True,
        minimap_region={"top": 0, "left": 0, "width": 1, "height": 1},
        scan_interval_seconds=1.0,
        test_enemy_loadout=[],
        live_client_all_game_data_url="https://127.0.0.1:2999/liveclientdata/allgamedata",
        minimap_training_image_dir=str(image_dir),
        minimap_training_labels_path=str(labels_path),
    )

    class FakeTracker:
        def __init__(self) -> None:
            self.templates = {}

        def load_template_images(self, templates) -> None:
            self.templates = templates

    tracker = FakeTracker()

    assert load_learned_minimap_templates(tracker, settings) == 1
    assert list(tracker.templates) == ["Nasus"]


def test_create_runtime_config_enables_training_capture_from_cli(tmp_path) -> None:
    from main import create_runtime_config

    settings = BackendSettings(
        test_mode=True,
        minimap_region={"top": 0, "left": 0, "width": 1, "height": 1},
        scan_interval_seconds=1.0,
        test_enemy_loadout=[],
        live_client_all_game_data_url="https://127.0.0.1:2999/liveclientdata/allgamedata",
        minimap_training_capture_dir=str(tmp_path),
        minimap_training_capture_interval_seconds=4.0,
    )

    runtime_config = create_runtime_config(
        settings,
        collect_minimap_training=True,
    )

    assert runtime_config.training_capture_mode is True
    assert runtime_config.training_capture_dir == str(tmp_path)
    assert runtime_config.training_capture_interval_seconds == 4.0


def test_create_training_recorder_returns_configured_recorder(tmp_path) -> None:
    from main import create_runtime_config, create_training_recorder

    settings = BackendSettings(
        test_mode=True,
        minimap_region={"top": 0, "left": 0, "width": 1, "height": 1},
        scan_interval_seconds=1.0,
        test_enemy_loadout=[],
        live_client_all_game_data_url="https://127.0.0.1:2999/liveclientdata/allgamedata",
        minimap_training_capture_mode=True,
        minimap_training_capture_dir=str(tmp_path),
        minimap_training_capture_interval_seconds=5.0,
    )
    runtime_config = create_runtime_config(settings)

    recorder = create_training_recorder(runtime_config)

    assert recorder is not None
    assert recorder.output_dir == tmp_path
    assert recorder.interval_seconds == 5.0


async def test_wait_for_active_game_polls_until_live_client_available() -> None:
    from main import wait_for_active_game

    class FakeLiveClient:
        def __init__(self) -> None:
            self.calls = count()

        async def fetch_enemy_loadout(self) -> list[dict[str, object]]:
            if next(self.calls) == 0:
                raise LiveClientError("not in game")
            return [{"champion": "Garen", "spells": ["Flash", "Teleport"]}]

    components = create_app_components(
        settings=BackendSettings(
            test_mode=False,
            minimap_region={"top": 0, "left": 0, "width": 1, "height": 1},
            scan_interval_seconds=1.0,
            test_enemy_loadout=[],
            live_client_all_game_data_url="https://127.0.0.1:2999/liveclientdata/allgamedata",
            live_client_poll_interval_seconds=0.0,
        )
    )

    with patch("main.asyncio.sleep", return_value=None) as sleep:
        loadout = await wait_for_active_game(
            live_client=FakeLiveClient(),
            settings=BackendSettings(
                test_mode=False,
                minimap_region={"top": 0, "left": 0, "width": 1, "height": 1},
                scan_interval_seconds=1.0,
                test_enemy_loadout=[],
                live_client_all_game_data_url=(
                    "https://127.0.0.1:2999/liveclientdata/allgamedata"
                ),
                live_client_poll_interval_seconds=0.0,
            ),
            game_state=components.game_state,
        )

    assert loadout == [{"champion": "Garen", "spells": ["Flash", "Teleport"]}]
    assert components.game_state.get_enemy_loadout() == loadout
    assert components.game_state.get_game_status()["phase"] == "active_game"
    sleep.assert_awaited_once_with(0.0)
