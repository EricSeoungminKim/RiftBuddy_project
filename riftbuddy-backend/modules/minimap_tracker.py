from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core.event_bus import EventBus
from core.game_state import GameState

MATCH_THRESHOLD = 0.75
SCALES = (0.8, 1.0, 1.2)


@dataclass(frozen=True)
class Detection:
    champion: str
    confidence: float
    x: int
    y: int


class MinimapTracker:
    def __init__(self, event_bus: EventBus, game_state: GameState) -> None:
        self._bus = event_bus
        self._state = game_state
        self._templates: dict[str, np.ndarray] = {}

    def load_templates(self, icon_paths: dict[str, Path]) -> None:
        for champion, path in icon_paths.items():
            image = cv2.imread(str(path))
            if image is not None:
                self._templates[champion] = image

    def process_frame(
        self,
        frame: np.ndarray,
        tracked_champions: list[str],
        now: float,
    ) -> None:
        mask = self._apply_color_prefilter(frame)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        detections = self._detect_icons(masked, tracked_champions)
        detected_names = {detection.champion for detection in detections}

        for champion in tracked_champions:
            if champion in detected_names:
                self._state.clear_mia(champion)
                continue
            self._mark_mia(champion, now)

    def _mark_mia(self, champion: str, now: float) -> None:
        mia = self._state.get_mia()
        last_seen = mia.get(champion, {}).get("last_seen", now)
        self._state.update_mia(champion, last_seen=last_seen)
        self._bus.emit(
            "mia_detected",
            {"champion": champion, "elapsed_sec": now - last_seen},
        )

    def _apply_color_prefilter(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 30, 30], dtype=np.uint8)
        upper = np.array([180, 255, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    def _detect_icons(
        self,
        frame: np.ndarray,
        champions: list[str],
    ) -> list[Detection]:
        detections: list[Detection] = []
        for champion in champions:
            template = self._templates.get(champion)
            if template is None:
                continue
            detection = self._match_multiscale(frame, champion, template)
            if detection is not None:
                detections.append(detection)
        return detections

    def _match_multiscale(
        self,
        frame: np.ndarray,
        champion: str,
        template: np.ndarray,
    ) -> Detection | None:
        best: Detection | None = None
        for scale in SCALES:
            resized = self._resize_template(template, scale)
            if resized.shape[0] > frame.shape[0] or resized.shape[1] > frame.shape[1]:
                continue

            result = cv2.matchTemplate(frame, resized, cv2.TM_CCOEFF_NORMED)
            _, max_value, _, max_location = cv2.minMaxLoc(result)
            if max_value < MATCH_THRESHOLD:
                continue
            if best is None or max_value > best.confidence:
                best = Detection(
                    champion, float(max_value), max_location[0], max_location[1]
                )
        return best

    @staticmethod
    def _resize_template(template: np.ndarray, scale: float) -> np.ndarray:
        height, width = template.shape[:2]
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(template, size)
