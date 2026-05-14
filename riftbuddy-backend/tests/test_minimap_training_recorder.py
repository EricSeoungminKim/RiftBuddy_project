import json
from pathlib import Path

import cv2
import numpy as np

from modules.minimap_training_recorder import TrainingSampleRecorder


def test_training_recorder_writes_frame_and_metadata(tmp_path: Path) -> None:
    recorder = TrainingSampleRecorder(
        output_dir=tmp_path,
        interval_seconds=2.0,
        session_id="session",
    )
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    metadata = {
        "tracked_champions": ["Nasus", "Xayah"],
        "champions": [{"champion": "Nasus", "confidence": 0.91}],
        "region": {"top": 10, "left": 20, "width": 48, "height": 32},
    }

    image_path = recorder.maybe_record(frame, now=100.0, metadata=metadata)

    assert image_path == tmp_path / "session" / "frame_000001.png"
    assert cv2.imread(str(image_path)).shape == (32, 48, 3)

    saved_metadata = json.loads(image_path.with_suffix(".json").read_text())
    assert saved_metadata["image"] == "frame_000001.png"
    assert saved_metadata["recorded_at"] == 100.0
    assert saved_metadata["frame_size"] == {"width": 48, "height": 32}
    assert saved_metadata["tracked_champions"] == ["Nasus", "Xayah"]
    assert saved_metadata["quality"]["max_confidence"] == 0.91
    assert saved_metadata["quality"]["needs_review"] is True


def test_training_recorder_respects_capture_interval(tmp_path: Path) -> None:
    recorder = TrainingSampleRecorder(
        output_dir=tmp_path,
        interval_seconds=3.0,
        session_id="session",
    )
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    first = recorder.maybe_record(frame, now=10.0, metadata={})
    skipped = recorder.maybe_record(frame, now=12.0, metadata={})
    second = recorder.maybe_record(frame, now=13.0, metadata={})

    assert first == tmp_path / "session" / "frame_000001.png"
    assert skipped is None
    assert second == tmp_path / "session" / "frame_000002.png"
