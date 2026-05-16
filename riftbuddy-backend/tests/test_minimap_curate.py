"""Tests for the minimap curate workflow tool.

Covers:
- select: per-champion top-N crops, skipping saturated champions, frame-gap diversity
- finalize: marks accepted=true on surviving crops, deletes raw + json
- prepare-next-game: clears DDragon icon cache
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import minimap_curate as curate


def _make_frame(path: Path, color: tuple[int, int, int] = (40, 40, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((100, 100, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _build_session(tmp_path: Path, n_frames: int = 30) -> tuple[Path, Path, Path]:
    """Create samples + a candidates review with two champions across N frames.

    Champion A ("제라스") gets descending confidences; B ("Blitzcrank") ascending.
    Each frame has both champions as separate candidates.
    """
    samples_dir = tmp_path / "sessions"
    session = "20260514_TEST"
    review_path = tmp_path / "curation" / "minimap_candidates.json"
    out_dir = tmp_path / "curation" / "auto_selected"

    samples = []
    for i in range(1, n_frames + 1):
        png_rel = f"{session}/frame_{i:06d}.png"
        json_rel = f"{session}/frame_{i:06d}.json"
        _make_frame(samples_dir / png_rel)
        _write(samples_dir / json_rel, {"image": f"frame_{i:06d}.png"})
        samples.append(
            {
                "frame_size": {"width": 100, "height": 100},
                "source_image": png_rel,
                "source_metadata": json_rel,
                "candidates": [
                    {
                        "champion": "제라스",
                        "confidence": round(0.9 - i * 0.01, 3),
                        "center": [40, 50],
                        "size": 20,
                        "accepted": False,
                        "notes": "",
                        "match_source": "tracker_template_match",
                    },
                    {
                        "champion": "Blitzcrank",
                        "confidence": round(0.3 + i * 0.01, 3),
                        "center": [60, 50],
                        "size": 20,
                        "accepted": False,
                        "notes": "",
                        "match_source": "tracker_template_match",
                    },
                ],
            }
        )
    _write(review_path, {"samples": samples})
    return samples_dir, review_path, out_dir


# ---------- select ----------


def test_select_auto_suggests_when_review_missing(tmp_path: Path) -> None:
    """If minimap_candidates.json doesn't exist, select must auto-build it
    from the raw training_sessions/ frames on disk."""
    samples_dir = tmp_path / "sessions"
    review_path = tmp_path / "curation" / "minimap_candidates.json"
    out_dir = tmp_path / "curation" / "auto_selected"

    # Build raw recorder-style metadata + frames, but NO review JSON.
    session = "20260515_TEST"
    for i in range(1, 6):
        png_rel = f"{session}/frame_{i:06d}.png"
        json_rel = f"{session}/frame_{i:06d}.json"
        _make_frame(samples_dir / png_rel)
        _write(
            samples_dir / json_rel,
            {
                "image": f"frame_{i:06d}.png",
                "frame_size": {"width": 100, "height": 100},
                "champions": [
                    {
                        "champion": "케인",
                        "confidence": round(0.7 + i * 0.01, 3),
                        "x": 30,
                        "y": 40,
                    },
                ],
            },
        )

    assert not review_path.exists()

    summary = curate.run_select(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
        target=3,
        batch=10,
        min_frame_gap=1,
        min_confidence=0.4,
    )

    assert review_path.exists(), "select should have generated the review JSON"
    assert summary["added"]["케인"] == 3


def test_select_errors_clearly_when_no_raw_frames(tmp_path: Path) -> None:
    """If there's no review AND no raw data, fail with a clear message instead
    of an opaque traceback."""
    import pytest

    review_path = tmp_path / "curation" / "minimap_candidates.json"
    out_dir = tmp_path / "curation" / "auto_selected"
    samples_dir = tmp_path / "sessions"  # missing entirely

    with pytest.raises(FileNotFoundError) as excinfo:
        curate.run_select(
            review_path=review_path,
            samples_dir=samples_dir,
            out_dir=out_dir,
            target=3,
            batch=10,
            min_frame_gap=1,
        )
    msg = str(excinfo.value)
    assert "training_sessions" in msg or str(samples_dir) in msg


def test_select_creates_per_champion_crops_and_marks_accepted(tmp_path: Path) -> None:
    samples_dir, review_path, out_dir = _build_session(tmp_path, n_frames=30)

    summary = curate.run_select(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
        target=10,
        batch=30,
        min_frame_gap=1,
    )

    # Both champions should each get 10 crops since target=10 and 30 frames are available.
    assert summary["added"]["제라스"] == 10
    assert summary["added"]["Blitzcrank"] == 10
    assert (out_dir / "제라스").is_dir()
    assert (out_dir / "Blitzcrank").is_dir()
    assert len(list((out_dir / "제라스").glob("*.png"))) == 10
    assert len(list((out_dir / "Blitzcrank").glob("*.png"))) == 10

    # Accepted flags reflect picked crops.
    review = json.loads(review_path.read_text())
    accepted_total = sum(
        1
        for s in review["samples"]
        for c in s["candidates"]
        if c.get("accepted")
    )
    assert accepted_total == 20


def test_select_skips_saturated_champions(tmp_path: Path) -> None:
    samples_dir, review_path, out_dir = _build_session(tmp_path, n_frames=30)
    saturated = out_dir / "제라스"
    saturated.mkdir(parents=True)
    # Pre-populate so 제라스 already has >= target.
    for i in range(10):
        _make_frame(saturated / f"00000{i}_0_conf0p900.png")

    summary = curate.run_select(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
        target=10,
        batch=30,
        min_frame_gap=1,
    )

    assert summary["skipped"] == ["제라스"]
    assert "제라스" not in summary["added"]
    assert summary["added"]["Blitzcrank"] == 10


def test_select_topup_when_under_target(tmp_path: Path) -> None:
    samples_dir, review_path, out_dir = _build_session(tmp_path, n_frames=30)
    partial = out_dir / "제라스"
    partial.mkdir(parents=True)
    # Already has 4 — needs 6 more to reach target=10.
    # Use real frame indices that exist in the session so sync sees them.
    for i in [1, 2, 3, 4]:
        _make_frame(partial / f"{i:06d}_0_conf0p900.png")

    summary = curate.run_select(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
        target=10,
        batch=30,
        min_frame_gap=1,
    )

    assert summary["added"]["제라스"] == 6
    assert len(list(partial.glob("*.png"))) == 10


def test_select_respects_frame_gap(tmp_path: Path) -> None:
    samples_dir, review_path, out_dir = _build_session(tmp_path, n_frames=30)

    summary = curate.run_select(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
        target=10,
        batch=30,
        min_frame_gap=10,
    )

    # 30 frames with gap-10 = at most 3 picks per champion.
    assert summary["added"]["제라스"] <= 3
    assert summary["added"]["Blitzcrank"] <= 3


def test_select_does_not_re_offer_user_rejected_crops(tmp_path: Path) -> None:
    """After the user deletes a crop, a follow-up select must not re-offer it."""
    samples_dir, review_path, out_dir = _build_session(tmp_path, n_frames=30)

    # First pass: pull 10 candidates for 제라스.
    curate.run_select(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
        target=10,
        batch=30,
        min_frame_gap=1,
    )
    zerath_dir = out_dir / "제라스"
    first_files = sorted(p.name for p in zerath_dir.glob("*.png"))
    assert len(first_files) == 10

    # User deletes 5 (simulating "these are bad").
    deleted_names = first_files[:5]
    for name in deleted_names:
        (zerath_dir / name).unlink()

    # Second pass: should top 제라스 back up to 10, but NOT with the same
    # (frame, idx) that the user just rejected.
    curate.run_select(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
        target=10,
        batch=30,
        min_frame_gap=1,
    )
    second_files = sorted(p.name for p in zerath_dir.glob("*.png"))
    assert len(second_files) == 10
    for name in deleted_names:
        assert name not in second_files, f"rejected crop {name} was re-offered"


def test_select_expands_pool_when_frame_gap_blocks_picks(tmp_path: Path) -> None:
    """If the top `batch` candidates are all blocked by frame-gap, the
    selector must look deeper into the candidate list rather than stop."""
    samples_dir, review_path, out_dir = _build_session(tmp_path, n_frames=30)

    # min_frame_gap=15 means at most 2 picks per champion in the top 5
    # (frames 1..5 with gap 15 → 1 pick), so a tight batch=5 used to give 1.
    # The fix should let the selector keep scanning past `batch` to honour target.
    summary = curate.run_select(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
        target=2,
        batch=5,
        min_frame_gap=15,
    )

    assert summary["added"]["제라스"] == 2
    assert summary["added"]["Blitzcrank"] == 2


def test_select_stops_when_pool_exhausted(tmp_path: Path) -> None:
    samples_dir, review_path, out_dir = _build_session(tmp_path, n_frames=5)

    summary = curate.run_select(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
        target=10,
        batch=30,
        min_frame_gap=1,
    )

    # Only 5 frames exist; target=10 unreachable but should not error.
    assert summary["added"]["제라스"] == 5
    assert summary["added"]["Blitzcrank"] == 5


# ---------- finalize ----------


def test_finalize_keeps_only_surviving_crops_and_deletes_raw(tmp_path: Path) -> None:
    samples_dir, review_path, out_dir = _build_session(tmp_path, n_frames=10)

    curate.run_select(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
        target=10,
        batch=30,
        min_frame_gap=1,
    )
    # Simulate user deleting half of 제라스 crops.
    zerath_files = sorted((out_dir / "제라스").glob("*.png"))
    for f in zerath_files[:5]:
        f.unlink()

    result = curate.run_finalize(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
    )

    # Raw session files gone.
    assert not (samples_dir / "20260514_TEST").exists()
    # Review JSON gone.
    assert not review_path.exists()
    # Auto-selected crops kept.
    assert len(list((out_dir / "제라스").glob("*.png"))) == 5
    assert len(list((out_dir / "Blitzcrank").glob("*.png"))) == 10
    assert result["kept"] == 15


def test_finalize_clears_per_champion_ledgers(tmp_path: Path) -> None:
    """history/rejected ledgers reference the just-ended session's frame
    indices. They must NOT survive finalize, otherwise next session's
    frame_000029 would collide with this session's frame_000029 in the ledger.
    """
    samples_dir, review_path, out_dir = _build_session(tmp_path, n_frames=15)

    # Run a select to generate ledgers, then delete one crop so rejected
    # actually has content.
    curate.run_select(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
        target=5,
        batch=10,
        min_frame_gap=1,
    )
    deleted = next((out_dir / "제라스").glob("*.png"))
    deleted.unlink()
    # Second select to write rejected.json.
    curate.run_select(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
        target=5,
        batch=10,
        min_frame_gap=1,
    )

    # Sanity: ledgers exist before finalize.
    assert (out_dir / "제라스" / ".history.json").exists()
    assert (out_dir / "제라스" / ".rejected.json").exists()

    curate.run_finalize(
        review_path=review_path,
        samples_dir=samples_dir,
        out_dir=out_dir,
    )

    # Ledgers gone. PNG crops stay.
    for champ_dir in out_dir.iterdir():
        assert not (champ_dir / ".history.json").exists()
        assert not (champ_dir / ".rejected.json").exists()
    assert any((out_dir / "제라스").glob("*.png"))


def test_finalize_is_safe_with_no_raw_data(tmp_path: Path) -> None:
    out_dir = tmp_path / "curation" / "auto_selected"
    out_dir.mkdir(parents=True)
    (out_dir / "Xerath").mkdir()
    _make_frame(out_dir / "Xerath" / "000001_0_conf0p900.png")

    result = curate.run_finalize(
        review_path=tmp_path / "missing.json",
        samples_dir=tmp_path / "missing_sessions",
        out_dir=out_dir,
    )

    assert result["kept"] == 1


# ---------- prepare-next-game ----------


def test_prepare_next_game_clears_ddragon_champion_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "assets"
    champ_dir = cache_dir / "champion" / "14.10.1"
    champ_dir.mkdir(parents=True)
    (champ_dir / "Xerath.png").write_bytes(b"x")
    (champ_dir / "Blitzcrank.png").write_bytes(b"y")
    # Untouched: spell cache.
    spell_dir = cache_dir / "spell"
    spell_dir.mkdir()
    (spell_dir / "SummonerFlash.png").write_bytes(b"z")

    result = curate.run_prepare_next_game(cache_dir=cache_dir)

    assert not (cache_dir / "champion").exists()
    assert (spell_dir / "SummonerFlash.png").exists()
    assert result["removed"] >= 2


def test_prepare_next_game_is_safe_with_empty_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "assets"
    result = curate.run_prepare_next_game(cache_dir=cache_dir)
    assert result["removed"] == 0
