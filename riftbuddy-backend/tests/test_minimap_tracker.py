from unittest.mock import patch

import cv2
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


def test_match_multiscale_finds_minimap_sized_icon() -> None:
    tracker, _, _ = make_tracker()
    rng = np.random.default_rng(42)
    template = rng.integers(0, 255, size=(120, 120, 3), dtype=np.uint8)
    minimap_icon = cv2.resize(template, (36, 36))
    frame = rng.integers(0, 80, size=(446, 446, 3), dtype=np.uint8)
    frame[170:206, 182:218] = minimap_icon

    detection = tracker._match_multiscale(frame, "Zed", template)

    assert detection is not None
    assert detection.confidence > 0.9
    assert abs(detection.x - 182) <= 2
    assert abs(detection.y - 170) <= 2


def test_load_template_images_scores_learned_template() -> None:
    tracker, _, _ = make_tracker()
    rng = np.random.default_rng(7)
    learned_template = rng.integers(0, 255, size=(34, 34, 3), dtype=np.uint8)
    frame = rng.integers(0, 80, size=(180, 180, 3), dtype=np.uint8)
    frame[92:126, 76:110] = learned_template

    tracker.load_template_images({"Nasus": [learned_template]})
    scores = tracker._score_icons(frame, ["Nasus"])

    assert scores["Nasus"] is not None
    assert scores["Nasus"].confidence > 0.95
    assert abs(scores["Nasus"].x - 76) <= 1
    assert abs(scores["Nasus"].y - 92) <= 1


def test_debug_mode_records_confidence_snapshot() -> None:
    tracker, _, state = make_tracker()
    fake_frame = np.zeros((20, 30, 3), dtype=np.uint8)

    with patch.object(
        tracker,
        "_detect_icons",
        return_value=[Detection(champion="Zed", confidence=0.82, x=7, y=9)],
    ):
        tracker.process_frame(
            fake_frame,
            tracked_champions=["Zed", "Jinx"],
            now=1015.0,
            minimap_region={"top": 790, "left": 1630, "width": 290, "height": 290},
            capture_display={
                "top": 0,
                "left": 0,
                "width": 4096,
                "height": 2304,
            },
            debug_mode=True,
            calibration_mode=True,
            debug_frame_path=None,
        )

    snapshot = state.get_minimap_debug()
    assert snapshot["capture_preview"].startswith("data:image/jpeg;base64,")
    snapshot["capture_preview"] = "data:image/jpeg;base64,"

    assert snapshot == {
        "enabled": True,
        "calibration_mode": True,
        "region": {"top": 790, "left": 1630, "width": 290, "height": 290},
        "capture_display": {"top": 0, "left": 0, "width": 4096, "height": 2304},
        "frame_size": {"width": 30, "height": 20},
        "capture_path": None,
        "capture_preview": "data:image/jpeg;base64,",
        "threshold": 0.75,
        "updated_at": 1015.0,
        "champions": [
            {
                "champion": "Zed",
                "confidence": 0.82,
                "detected": True,
                "x": 7,
                "y": 9,
            },
            {
                "champion": "Jinx",
                "confidence": 0.0,
                "detected": False,
                "x": None,
                "y": None,
            },
        ],
    }


def test_apply_color_prefilter_returns_mask() -> None:
    tracker, _, _ = make_tracker()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:60, 50:60] = [255, 0, 0]

    mask = tracker._apply_color_prefilter(frame)

    assert mask.shape == (200, 200)
    assert mask.dtype == np.uint8
    assert mask[55, 55] > 0
