from __future__ import annotations

import cv2
import numpy as np

FALLBACK_MINIMAP_HEIGHT_RATIO = 0.31


def detect_minimap_region(
    frame: np.ndarray,
    capture_display: dict[str, int],
) -> dict[str, int] | None:
    """Detect the minimap square inside a full-screen capture."""
    frame_height, frame_width = frame.shape[:2]
    search_left = int(frame_width * 0.62)
    search_top = int(frame_height * 0.55)
    search = frame[search_top:, search_left:]

    edges = _minimap_edges(search)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    search_height, search_width = search.shape[:2]
    candidate = _best_square_candidate(
        contours,
        search_width,
        search_height,
        frame_width,
        frame_height,
    )
    if candidate is None:
        return _fallback_bottom_right_region(capture_display)

    x, y, width, height = candidate
    return {
        "left": capture_display["left"] + search_left + x,
        "top": capture_display["top"] + search_top + y,
        "width": width,
        "height": height,
    }


def crop_region(
    frame: np.ndarray,
    region: dict[str, int],
    capture_display: dict[str, int],
) -> np.ndarray:
    """Crop a capture-space region from a full-screen frame."""
    x = max(0, region["left"] - capture_display["left"])
    y = max(0, region["top"] - capture_display["top"])
    width = max(1, region["width"])
    height = max(1, region["height"])
    return frame[y : y + height, x : x + width]


def _fallback_bottom_right_region(capture_display: dict[str, int]) -> dict[str, int]:
    """Return an anchored minimap estimate when visual detection is inconclusive."""
    side = max(1, int(round(capture_display["height"] * FALLBACK_MINIMAP_HEIGHT_RATIO)))
    return {
        "left": capture_display["left"] + capture_display["width"] - side,
        "top": capture_display["top"] + capture_display["height"] - side,
        "width": side,
        "height": side,
    }


def _minimap_edges(frame: np.ndarray) -> np.ndarray:
    """Highlight rectangular HUD borders in the minimap search area."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 130)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(edges, kernel, iterations=2)


def _best_square_candidate(
    contours: tuple[np.ndarray, ...],
    search_width: int,
    search_height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    """Pick the largest bottom-right square-like contour as the minimap."""
    min_side = int(min(frame_width, frame_height) * 0.14)
    max_side = int(min(frame_width, frame_height) * 0.42)
    best: tuple[int, int, int, int] | None = None
    best_score = float("-inf")

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if not _is_square_like(width, height, min_side, max_side):
            continue

        area = width * height
        distance_from_corner = abs((x + width) - search_width) + abs(
            (y + height) - search_height
        )
        score = area - distance_from_corner * 0.3
        if score > best_score:
            best = (x, y, width, height)
            best_score = score

    return best


def _is_square_like(width: int, height: int, min_side: int, max_side: int) -> bool:
    """Return whether a contour bounds a plausible minimap square."""
    if width < min_side or height < min_side:
        return False
    if width > max_side or height > max_side:
        return False
    ratio = width / height
    return 0.9 <= ratio <= 1.1
