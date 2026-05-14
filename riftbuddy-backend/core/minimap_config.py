from copy import deepcopy
from dataclasses import dataclass


@dataclass
class MinimapRuntimeConfig:
    """Mutable minimap settings that can be adjusted while the backend runs."""

    region: dict[str, int]
    scan_interval_seconds: float
    auto_region: bool = True
    debug_mode: bool = False
    calibration_mode: bool = False
    debug_frame_path: str | None = None
    training_capture_mode: bool = False
    training_capture_dir: str = "data/training_sessions"
    training_capture_interval_seconds: float = 3.0

    def __post_init__(self) -> None:
        """Keep a private copy of region values supplied by settings."""
        self.region = deepcopy(self.region)

    def adjust_region(self, delta: dict[str, object]) -> None:
        """Apply numeric top/left/width/height adjustments to the capture region."""
        for key in ("top", "left", "width", "height"):
            value = delta.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue

            next_value = int(round(self.region[key] + value))
            if key in {"width", "height"}:
                self.region[key] = max(1, next_value)
            else:
                self.region[key] = next_value

    def debug_snapshot(self) -> dict[str, object]:
        """Return the debug fields that should be visible before the first frame."""
        return {
            "enabled": self.debug_mode,
            "calibration_mode": self.calibration_mode,
            "auto_region": self.auto_region,
            "training_capture_mode": self.training_capture_mode,
            "region": deepcopy(self.region),
        }
