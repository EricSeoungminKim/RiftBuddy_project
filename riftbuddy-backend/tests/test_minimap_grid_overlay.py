"""Tests for the grid overlay helper."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import minimap_grid_overlay as gov


def _make_frame(path: Path, side: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((side, side, 3), 30, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_overlay_writes_sibling_png(tmp_path: Path) -> None:
    src = tmp_path / "20260514_TEST" / "frame_000029.png"
    _make_frame(src, side=100)

    out = gov.write_overlay(src, spacing=25)

    assert out.exists()
    assert out.parent == src.parent
    assert out.name == "frame_000029_grid.png"
    img = cv2.imread(str(out))
    assert img.shape == (100, 100, 3)


def test_overlay_actually_draws_lines(tmp_path: Path) -> None:
    """The output must visually differ from the source (grid + labels added)."""
    src = tmp_path / "frame_000001.png"
    _make_frame(src, side=100)

    out = gov.write_overlay(src, spacing=25)

    src_img = cv2.imread(str(src))
    out_img = cv2.imread(str(out))
    diff = cv2.absdiff(src_img, out_img)
    assert diff.sum() > 0, "overlay should modify pixels"


def test_overlay_resolves_relative_path_against_samples_dir(tmp_path: Path) -> None:
    samples = tmp_path / "sessions"
    rel = "s1/frame_000007.png"
    _make_frame(samples / rel, side=100)

    out = gov.write_overlay(rel, spacing=25, samples_dir=samples)

    assert out.exists()
    assert out.parent == samples / "s1"


def test_overlay_raises_on_missing_frame(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        gov.write_overlay("nope.png", spacing=25, samples_dir=tmp_path)
