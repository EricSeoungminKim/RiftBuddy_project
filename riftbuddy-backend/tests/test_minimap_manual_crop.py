"""Tests for the manual single-crop helper."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tools import minimap_manual_crop as mc


def _make_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((100, 100, 3), 30, dtype=np.uint8)
    img[40:60, 40:60] = (0, 0, 200)  # marker square so we can tell crop worked
    cv2.imwrite(str(path), img)


def test_crop_writes_png_into_champion_folder(tmp_path: Path) -> None:
    samples = tmp_path / "sessions"
    out_dir = tmp_path / "auto_selected"
    frame_rel = "20260514_TEST/frame_000042.png"
    _make_frame(samples / frame_rel)

    written = mc.crop_one(
        frame_rel=frame_rel,
        champion="블리츠크랭크",
        cx=50,
        cy=50,
        size=20,
        samples_dir=samples,
        out_dir=out_dir,
    )

    assert written.exists()
    assert written.parent == out_dir / "블리츠크랭크"
    assert written.name.startswith("000042_manual_")
    assert written.suffix == ".png"
    img = cv2.imread(str(written))
    assert img.shape == (20, 20, 3)


def test_crop_clamps_to_frame_bounds(tmp_path: Path) -> None:
    samples = tmp_path / "sessions"
    out_dir = tmp_path / "auto_selected"
    frame_rel = "s/frame_000001.png"
    _make_frame(samples / frame_rel)

    # Center near the right edge so the requested 20x20 box would overflow.
    written = mc.crop_one(
        frame_rel=frame_rel,
        champion="Xerath",
        cx=98,
        cy=50,
        size=20,
        samples_dir=samples,
        out_dir=out_dir,
    )
    img = cv2.imread(str(written))
    # Width clamped (98 + 10 → 100 max), height stays 20.
    assert img.shape[0] == 20
    assert img.shape[1] <= 20 and img.shape[1] > 0


def test_crop_overwrites_same_frame_for_same_champion(tmp_path: Path) -> None:
    """Re-cropping the same (frame, champion) should replace, not duplicate."""
    samples = tmp_path / "sessions"
    out_dir = tmp_path / "auto_selected"
    frame_rel = "s/frame_000007.png"
    _make_frame(samples / frame_rel)

    mc.crop_one(
        frame_rel=frame_rel,
        champion="제라스",
        cx=50,
        cy=50,
        size=20,
        samples_dir=samples,
        out_dir=out_dir,
    )
    mc.crop_one(
        frame_rel=frame_rel,
        champion="제라스",
        cx=55,
        cy=45,
        size=20,
        samples_dir=samples,
        out_dir=out_dir,
    )

    files = list((out_dir / "제라스").glob("*.png"))
    assert len(files) == 1


def test_crop_raises_when_frame_missing(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        mc.crop_one(
            frame_rel="nope/frame_000001.png",
            champion="Xerath",
            cx=10,
            cy=10,
            size=20,
            samples_dir=tmp_path / "sessions",
            out_dir=tmp_path / "out",
        )
