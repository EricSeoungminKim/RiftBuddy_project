import cv2
import numpy as np

from modules.minimap_region_detector import (
    _is_square_like,
    crop_region,
    detect_minimap_region,
)


def test_detect_minimap_region_finds_bottom_right_square() -> None:
    frame = np.zeros((800, 1000, 3), dtype=np.uint8)
    frame[450:670, 700:920] = [18, 24, 21]
    cv2.rectangle(frame, (700, 450), (920, 670), (80, 170, 210), 4)

    region = detect_minimap_region(
        frame,
        {"top": 0, "left": 0, "width": 1000, "height": 800},
    )

    assert region is not None
    assert abs(region["left"] - 700) <= 8
    assert abs(region["top"] - 450) <= 8
    assert abs(region["width"] - 220) <= 10
    assert abs(region["height"] - 220) <= 10


def test_detect_minimap_region_prefers_square_over_wide_hud_panel() -> None:
    frame = np.zeros((1152, 2048, 3), dtype=np.uint8)
    cv2.rectangle(frame, (1510, 700), (2040, 1150), (48, 90, 100), 5)
    cv2.rectangle(frame, (1690, 820), (2030, 1150), (80, 170, 210), 5)

    region = detect_minimap_region(
        frame,
        {"top": 0, "left": 0, "width": 2048, "height": 1152},
    )

    assert region is not None
    assert abs(region["left"] - 1690) <= 12
    assert 790 <= region["top"] <= 825
    assert 330 <= region["width"] <= 370
    assert 325 <= region["height"] <= 370


def test_minimap_square_filter_rejects_wide_hud_panels() -> None:
    assert not _is_square_like(width=535, height=450, min_side=200, max_side=700)
    assert _is_square_like(width=450, height=442, min_side=200, max_side=700)


def test_detect_minimap_region_falls_back_to_bottom_right_square() -> None:
    frame = np.zeros((800, 1000, 3), dtype=np.uint8)

    region = detect_minimap_region(
        frame,
        {"top": 0, "left": 0, "width": 1000, "height": 800},
    )

    assert region == {"left": 752, "top": 552, "width": 248, "height": 248}


def test_crop_region_uses_capture_display_offset() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[30:50, 40:60] = [255, 255, 255]

    crop = crop_region(
        frame,
        {"top": 130, "left": 240, "width": 20, "height": 20},
        {"top": 100, "left": 200, "width": 120, "height": 100},
    )

    assert crop.shape == (20, 20, 3)
    assert crop.mean() == 255
