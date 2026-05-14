import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class TrainingSampleRecorder:
    """Stores minimap crops and metadata for later human-reviewed training."""

    def __init__(
        self,
        output_dir: str | Path,
        interval_seconds: float,
        session_id: str | None = None,
    ) -> None:
        """Create a recorder that writes samples into one session directory."""
        self.output_dir = Path(output_dir)
        self.interval_seconds = interval_seconds
        self.session_id = session_id or _default_session_id()
        self._sample_index = 0
        self._last_recorded_at: float | None = None

    def maybe_record(
        self,
        frame: np.ndarray,
        now: float,
        metadata: dict[str, Any],
    ) -> Path | None:
        """Write a sample when the configured interval has elapsed."""
        if not self._can_record(now):
            return None

        self._sample_index += 1
        self._last_recorded_at = now
        image_path = self._image_path(self._sample_index)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), frame)
        image_path.with_suffix(".json").write_text(
            json.dumps(
                self._build_metadata(image_path, frame, now, metadata),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return image_path

    def _can_record(self, now: float) -> bool:
        """Return whether the current frame is outside the cooldown window."""
        if self._last_recorded_at is None:
            return True
        return now - self._last_recorded_at >= self.interval_seconds

    def _image_path(self, sample_index: int) -> Path:
        """Return the path for one indexed sample image."""
        return self.output_dir / self.session_id / f"frame_{sample_index:06d}.png"

    def _build_metadata(
        self,
        image_path: Path,
        frame: np.ndarray,
        now: float,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Create JSON metadata that keeps raw samples reviewable, not trusted."""
        height, width = frame.shape[:2]
        body = dict(metadata)
        body.update(
            {
                "image": image_path.name,
                "recorded_at": now,
                "frame_size": {"width": width, "height": height},
                "review": {"accepted": False, "labelled": False},
                "quality": _quality_summary(metadata),
            }
        )
        return body


def _default_session_id() -> str:
    """Create a filesystem-friendly session name for raw capture batches."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _quality_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Summarize confidence signals for later filtering and review tooling."""
    confidences = [
        champion.get("confidence")
        for champion in metadata.get("champions", [])
        if isinstance(champion, dict)
    ]
    numeric_confidences = [
        float(confidence)
        for confidence in confidences
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    ]
    max_confidence = max(numeric_confidences, default=0.0)
    return {
        "max_confidence": round(max_confidence, 3),
        "needs_review": True,
    }
