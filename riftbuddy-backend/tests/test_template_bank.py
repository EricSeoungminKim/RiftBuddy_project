"""Tests for the multi-template loader.

Loads:
  - DDragon champion icons (champion_id-keyed paths)
  - auto_selected/<champion>/*.png curated crops (named by display name)

Returns dict[display_name, list[ndarray]] for direct ingestion by
MinimapTracker.load_template_images().
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from modules.template_bank import load_templates_for_match


def _png(path: Path, color: tuple[int, int, int] = (50, 50, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((42, 42, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_loads_ddragon_icon_for_each_champion(tmp_path: Path) -> None:
    auto_dir = tmp_path / "auto_selected"
    icons = {
        "제라스": tmp_path / "ddragon" / "Xerath.png",
        "블리츠크랭크": tmp_path / "ddragon" / "Blitzcrank.png",
    }
    _png(icons["제라스"], (10, 20, 30))
    _png(icons["블리츠크랭크"], (40, 50, 60))

    templates = load_templates_for_match(icons, auto_dir)

    assert set(templates.keys()) == {"제라스", "블리츠크랭크"}
    assert len(templates["제라스"]) == 1
    assert len(templates["블리츠크랭크"]) == 1


def test_appends_auto_selected_crops_after_ddragon(tmp_path: Path) -> None:
    auto_dir = tmp_path / "auto_selected"
    icons = {"제라스": tmp_path / "ddragon" / "Xerath.png"}
    _png(icons["제라스"], (10, 20, 30))
    _png(auto_dir / "제라스" / "000005_2_conf0p623.png", (200, 0, 0))
    _png(auto_dir / "제라스" / "000010_manual_cx100cy120.png", (0, 200, 0))

    templates = load_templates_for_match(icons, auto_dir)

    # 1 DDragon + 2 auto_selected
    assert len(templates["제라스"]) == 3


def test_skips_dotfiles_in_auto_selected(tmp_path: Path) -> None:
    """.history.json / .rejected.json must not be loaded as images."""
    auto_dir = tmp_path / "auto_selected"
    icons = {"Xerath": tmp_path / "ddragon" / "Xerath.png"}
    _png(icons["Xerath"], (10, 20, 30))
    (auto_dir / "Xerath").mkdir(parents=True)
    (auto_dir / "Xerath" / ".history.json").write_text("[]")
    (auto_dir / "Xerath" / ".rejected.json").write_text("[]")
    _png(auto_dir / "Xerath" / "000001_0_conf0p700.png", (0, 0, 200))

    templates = load_templates_for_match(icons, auto_dir)

    assert len(templates["Xerath"]) == 2  # ddragon + 1 png


def test_works_when_auto_selected_missing(tmp_path: Path) -> None:
    icons = {"Xerath": tmp_path / "ddragon" / "Xerath.png"}
    _png(icons["Xerath"])
    templates = load_templates_for_match(icons, tmp_path / "does-not-exist")
    assert len(templates["Xerath"]) == 1


def test_works_when_champion_has_no_auto_selected_folder(tmp_path: Path) -> None:
    auto_dir = tmp_path / "auto_selected"
    auto_dir.mkdir()
    icons = {"Xerath": tmp_path / "ddragon" / "Xerath.png"}
    _png(icons["Xerath"])
    # Some other champion's folder exists, but Xerath's doesn't.
    (auto_dir / "Blitzcrank").mkdir()
    _png(auto_dir / "Blitzcrank" / "000001_0_conf0p700.png")

    templates = load_templates_for_match(icons, auto_dir)
    assert len(templates["Xerath"]) == 1


def test_skips_unreadable_files_silently(tmp_path: Path) -> None:
    auto_dir = tmp_path / "auto_selected"
    icons = {"Xerath": tmp_path / "ddragon" / "Xerath.png"}
    _png(icons["Xerath"])
    (auto_dir / "Xerath").mkdir(parents=True)
    # Bogus PNG (just not a real image).
    (auto_dir / "Xerath" / "000001_0_conf0p700.png").write_text("not an image")
    _png(auto_dir / "Xerath" / "000002_0_conf0p800.png")

    templates = load_templates_for_match(icons, auto_dir)
    # 1 ddragon + 1 valid auto_selected
    assert len(templates["Xerath"]) == 2


def test_max_per_champion_caps_curated_crops(tmp_path: Path) -> None:
    """When a champion has more curated crops than the cap, only `max_per_champion`
    are returned. The DDragon icon is always kept and counts toward the cap."""
    auto_dir = tmp_path / "auto_selected"
    icons = {"Xerath": tmp_path / "ddragon" / "Xerath.png"}
    _png(icons["Xerath"])
    for i in range(20):
        _png(auto_dir / "Xerath" / f"{i:06d}_0_conf0p700.png")

    templates = load_templates_for_match(icons, auto_dir, max_per_champion=10)

    assert len(templates["Xerath"]) == 10  # 1 ddragon + 9 curated


def test_under_cap_returns_all_available(tmp_path: Path) -> None:
    """If fewer crops than the cap exist, return all (no error)."""
    auto_dir = tmp_path / "auto_selected"
    icons = {"Xerath": tmp_path / "ddragon" / "Xerath.png"}
    _png(icons["Xerath"])
    for i in range(3):
        _png(auto_dir / "Xerath" / f"{i:06d}_0_conf0p700.png")

    templates = load_templates_for_match(icons, auto_dir, max_per_champion=10)

    assert len(templates["Xerath"]) == 4  # 1 ddragon + 3 curated, no padding


def test_max_per_champion_zero_or_none_disables_cap(tmp_path: Path) -> None:
    """max_per_champion=None (default) keeps existing behaviour (no cap)."""
    auto_dir = tmp_path / "auto_selected"
    icons = {"Xerath": tmp_path / "ddragon" / "Xerath.png"}
    _png(icons["Xerath"])
    for i in range(20):
        _png(auto_dir / "Xerath" / f"{i:06d}_0_conf0p700.png")

    templates = load_templates_for_match(icons, auto_dir)
    assert len(templates["Xerath"]) == 21  # 1 + 20


def test_subsampling_is_deterministic_given_seed(tmp_path: Path) -> None:
    """Same seed → same subset, so logs are reproducible."""
    auto_dir = tmp_path / "auto_selected"
    icons = {"Xerath": tmp_path / "ddragon" / "Xerath.png"}
    _png(icons["Xerath"])
    for i in range(20):
        _png(auto_dir / "Xerath" / f"{i:06d}_0_conf0p700.png", color=(i, i, i))

    a = load_templates_for_match(icons, auto_dir, max_per_champion=5, seed=42)
    b = load_templates_for_match(icons, auto_dir, max_per_champion=5, seed=42)

    assert len(a["Xerath"]) == len(b["Xerath"]) == 5
    # Compare pixel content of curated crops (index 1+) to confirm same picks.
    for x, y in zip(a["Xerath"][1:], b["Xerath"][1:]):
        assert (x == y).all()


def test_count_loaded_helper(tmp_path: Path) -> None:
    """A small helper for the bootstrap log line in main.py."""
    from modules.template_bank import count_templates

    templates = {
        "Xerath": [np.zeros((1, 1, 3), dtype=np.uint8)],
        "Blitzcrank": [np.zeros((1, 1, 3), dtype=np.uint8)] * 3,
    }
    assert count_templates(templates) == 4
