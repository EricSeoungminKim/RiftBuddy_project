from unittest.mock import patch

import numpy as np

from core.event_bus import EventBus
from core.game_state import GameState
from modules.minimap_tracker import Detection, MinimapTracker


def make_tracker() -> tuple[MinimapTracker, EventBus, GameState]:
    bus = EventBus()
    state = GameState()
    tracker = MinimapTracker(event_bus=bus, game_state=state)
    return tracker, bus, state


def test_detection_dataclass() -> None:
    detection = Detection(champion="Zed", confidence=0.85, x=10, y=20)

    assert detection.champion == "Zed"
    assert detection.confidence == 0.85
    assert detection.x == 10
    assert detection.y == 20


def test_no_detection_emits_mia() -> None:
    tracker, bus, state = make_tracker()
    received = []
    bus.subscribe("mia_detected", received.append)
    state.update_mia("Zed", last_seen=1000.0)

    fake_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    tracker.process_frame(fake_frame, tracked_champions=["Zed"], now=1015.0)

    assert received == [{"champion": "Zed", "elapsed_sec": 15.0}]


def test_detection_clears_mia() -> None:
    tracker, _, state = make_tracker()
    state.update_mia("Zed", last_seen=1000.0)
    fake_frame = np.zeros((200, 200, 3), dtype=np.uint8)

    with patch.object(
        tracker,
        "_detect_icons",
        return_value=[Detection(champion="Zed", confidence=0.9, x=5, y=5)],
    ):
        tracker.process_frame(fake_frame, tracked_champions=["Zed"], now=1015.0)

    assert "Zed" not in state.get_mia()


def test_apply_color_prefilter_returns_mask() -> None:
    tracker, _, _ = make_tracker()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:60, 50:60] = [255, 0, 0]

    mask = tracker._apply_color_prefilter(frame)

    assert mask.shape == (200, 200)
    assert mask.dtype == np.uint8
    assert mask[55, 55] > 0
