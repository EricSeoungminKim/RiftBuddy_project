import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.event_bus import EventBus
from core.game_state import GameState

MATCH_THRESHOLD = 0.75
ICON_SIZE_RATIOS = (0.065, 0.08, 0.095, 0.11, 0.125, 0.14)
MIN_ICON_SIZE = 24
MAX_ICON_SIZE = 72


@dataclass(frozen=True)
class Detection:
    """Represents one champion icon match on the minimap frame."""

    champion: str
    confidence: float
    x: int
    y: int


class MinimapTracker:
    """Detects missing champions by matching loaded icons against minimap frames."""

    def __init__(self, event_bus: EventBus, game_state: GameState) -> None:
        """Connect minimap detection to events and shared state."""
        self._bus = event_bus
        self._state = game_state
        self._templates: dict[str, list[np.ndarray]] = {}

    def load_templates(self, icon_paths: dict[str, Path]) -> None:
        """Load champion icon image files as OpenCV templates."""
        for champion, path in icon_paths.items():
            image = cv2.imread(str(path))
            if image is not None:
                self._templates.setdefault(champion, []).append(image)

    def load_template_images(self, templates: dict[str, list[np.ndarray]]) -> None:
        """Load learned in-game minimap icon templates from labeled captures."""
        for champion, images in templates.items():
            self._templates.setdefault(champion, []).extend(images)

    def process_frame(
        self,
        frame: np.ndarray,
        tracked_champions: list[str],
        now: float,
        minimap_region: dict[str, int] | None = None,
        capture_display: dict[str, int] | None = None,
        debug_mode: bool = False,
        calibration_mode: bool = False,
        debug_frame_path: str | None = None,
    ) -> dict[str, Any]:
        """Process one minimap frame and update MIA state for tracked champions."""
        mask = self._apply_color_prefilter(frame)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        detections = self._detect_icons(masked, tracked_champions)
        scores = self._score_icons(masked, tracked_champions)
        detected_names = {detection.champion for detection in detections}

        for champion in tracked_champions:
            if champion in detected_names:
                self._state.clear_mia(champion)
                continue
            self._mark_mia(champion, now)

        frame_metadata = _frame_metadata(
            frame=frame,
            tracked_champions=tracked_champions,
            detections=detections,
            scores=scores,
            now=now,
            minimap_region=minimap_region,
            capture_display=capture_display,
            debug_mode=debug_mode,
            calibration_mode=calibration_mode,
        )
        if debug_mode or calibration_mode:
            self._record_debug_snapshot(
                frame=frame,
                frame_metadata=frame_metadata,
                debug_frame_path=debug_frame_path,
            )
        return frame_metadata

    def _mark_mia(self, champion: str, now: float) -> None:
        """Mark a champion missing and emit elapsed missing time."""
        mia = self._state.get_mia()
        last_seen = mia.get(champion, {}).get("last_seen", now)
        self._state.update_mia(champion, last_seen=last_seen)
        self._bus.emit(
            "mia_detected",
            {"champion": champion, "elapsed_sec": now - last_seen},
        )

    def _apply_color_prefilter(self, frame: np.ndarray) -> np.ndarray:
        """Create a broad HSV mask to reduce template matching work."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 30, 30], dtype=np.uint8)
        upper = np.array([180, 255, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    def _detect_icons(
        self,
        frame: np.ndarray,
        champions: list[str],
    ) -> list[Detection]:
        """Run icon matching for all champions with loaded templates."""
        detections: list[Detection] = []
        scores = self._score_icons(frame, champions)
        for detection in scores.values():
            if detection is not None and detection.confidence >= MATCH_THRESHOLD:
                detections.append(detection)
        return detections

    def _score_icons(
        self,
        frame: np.ndarray,
        champions: list[str],
    ) -> dict[str, Detection | None]:
        """Return the best match confidence for each tracked champion."""
        scores: dict[str, Detection | None] = {}
        for champion in champions:
            templates = self._templates.get(champion)
            if not templates:
                scores[champion] = None
                continue
            scores[champion] = self._match_templates(frame, champion, templates)
        return scores

    def _match_templates(
        self,
        frame: np.ndarray,
        champion: str,
        templates: list[np.ndarray],
    ) -> Detection | None:
        """Return the best score across static and learned templates."""
        best: Detection | None = None
        for template in templates:
            detection = self._match_multiscale(frame, champion, template)
            if detection is None:
                continue
            if best is None or detection.confidence > best.confidence:
                best = detection
        return best

    def _match_multiscale(
        self,
        frame: np.ndarray,
        champion: str,
        template: np.ndarray,
    ) -> Detection | None:
        """Find the best template match across the configured icon scales."""
        best: Detection | None = None
        for target_size in _target_icon_sizes(frame, template):
            resized = self._resize_template(template, target_size)
            if resized.shape[0] > frame.shape[0] or resized.shape[1] > frame.shape[1]:
                continue

            result = cv2.matchTemplate(frame, resized, cv2.TM_CCOEFF_NORMED)
            _, max_value, _, max_location = cv2.minMaxLoc(result)
            if best is None or max_value > best.confidence:
                best = Detection(
                    champion, float(max_value), max_location[0], max_location[1]
                )
        return best

    def _record_debug_snapshot(
        self,
        frame: np.ndarray,
        frame_metadata: dict[str, Any],
        debug_frame_path: str | None,
    ) -> None:
        """Publish frame metadata and confidence scores for calibration."""
        capture_path = self._write_debug_frame(frame, debug_frame_path)
        snapshot = dict(frame_metadata)
        snapshot.pop("tracked_champions", None)
        snapshot["capture_path"] = capture_path
        snapshot["capture_preview"] = _encode_debug_preview(frame)
        self._state.update_minimap_debug(snapshot)

    @staticmethod
    def _write_debug_frame(
        frame: np.ndarray, debug_frame_path: str | None
    ) -> str | None:
        """Write the latest minimap crop for manual inspection when configured."""
        if not debug_frame_path:
            return None

        path = Path(debug_frame_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        return str(path)

    @staticmethod
    def _resize_template(template: np.ndarray, target_size: int) -> np.ndarray:
        """Resize a template while keeping dimensions at least one pixel."""
        height, width = template.shape[:2]
        scale = target_size / max(width, height)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(template, size)


def _merge_detection_scores(
    detections: list[Detection],
    scores: dict[str, Detection | None],
) -> dict[str, Detection | None]:
    """Use explicit detections as fallback when tests patch detection directly."""
    merged = dict(scores)
    for detection in detections:
        merged[detection.champion] = detection
    return merged


def _frame_metadata(
    frame: np.ndarray,
    tracked_champions: list[str],
    detections: list[Detection],
    scores: dict[str, Detection | None],
    now: float,
    minimap_region: dict[str, int] | None,
    capture_display: dict[str, int] | None,
    debug_mode: bool,
    calibration_mode: bool,
) -> dict[str, Any]:
    """Build reusable per-frame metadata for debug UI and training capture."""
    detection_scores = _merge_detection_scores(detections, scores)
    height, width = frame.shape[:2]
    return {
        "enabled": debug_mode,
        "calibration_mode": calibration_mode,
        "region": minimap_region or {},
        "capture_display": capture_display or {},
        "frame_size": {"width": width, "height": height},
        "capture_path": None,
        "capture_preview": "",
        "threshold": MATCH_THRESHOLD,
        "updated_at": now,
        "tracked_champions": list(tracked_champions),
        "champions": [
            _debug_champion(champion, detection_scores.get(champion))
            for champion in tracked_champions
        ],
    }


def _target_icon_sizes(frame: np.ndarray, template: np.ndarray) -> tuple[int, ...]:
    """Estimate plausible minimap champion icon sizes from the crop size."""
    side = min(frame.shape[:2])
    sizes = {
        min(MAX_ICON_SIZE, max(MIN_ICON_SIZE, int(round(side * ratio))))
        for ratio in ICON_SIZE_RATIOS
    }
    template_size = max(template.shape[:2])
    if MIN_ICON_SIZE <= template_size <= MAX_ICON_SIZE:
        sizes.add(template_size)
    return tuple(sorted(sizes))


def _debug_champion(champion: str, detection: Detection | None) -> dict[str, Any]:
    """Format one champion's debug score for the frontend panel."""
    if detection is None:
        return {
            "champion": champion,
            "confidence": 0.0,
            "detected": False,
            "x": None,
            "y": None,
        }

    confidence = round(detection.confidence, 3)
    return {
        "champion": champion,
        "confidence": confidence,
        "detected": confidence >= MATCH_THRESHOLD,
        "x": detection.x,
        "y": detection.y,
    }


def _encode_debug_preview(frame: np.ndarray) -> str:
    """Encode the current minimap crop so the overlay can show what CV sees."""
    preview = _resize_preview(frame)
    ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return "data:image/jpeg;base64,"

    body = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{body}"


def _resize_preview(frame: np.ndarray) -> np.ndarray:
    """Keep debug previews small enough for lightweight WebSocket payloads."""
    height, width = frame.shape[:2]
    max_side = max(width, height)
    if max_side <= 220:
        return frame

    scale = 220 / max_side
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, size)
