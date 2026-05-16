"""Minimap candidate curation workflow.

Subcommands:
  select               Top up each champion folder up to --target crops, drawing
                       from the next --batch best candidates per champion.
                       Champions already at/over target are skipped.
  finalize             Mark surviving crops as accepted=true, then delete raw
                       training session files and the candidates JSON.
  prepare-next-game    Wipe the DDragon champion icon cache so the next match
                       only re-downloads icons for the champions actually in it.

Folder layout:
  data/training_sessions/<session>/frame_*.png + .json   (raw capture)
  data/curation/minimap_candidates.json                  (review doc)
  data/curation/auto_selected/<champion>/<frame>_<idx>_conf<x>.png  (curated)
  data/cache/champion/<version>/<id>.png                 (DDragon icons)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2

DEFAULT_REVIEW_PATH = Path("data/curation/minimap_candidates.json")
DEFAULT_SAMPLES_DIR = Path("data/training_sessions")
DEFAULT_OUT_DIR = Path("data/curation/auto_selected")
DEFAULT_CACHE_DIR = Path("data/cache")

CROP_NAME_RE = re.compile(r"^(?P<frame>\d+)_(?P<idx>\d+)_conf(?P<conf>[0-9p]+)\.png$")
HISTORY_FILE = ".history.json"   # every (frame, idx) ever offered for this champion
REJECTED_FILE = ".rejected.json"  # (frame, idx) the user has deleted at least once


# ---------- CLI ----------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sel = sub.add_parser("select")
    sel.add_argument("--review", type=Path, default=DEFAULT_REVIEW_PATH)
    sel.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_DIR)
    sel.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    sel.add_argument("--target", type=int, default=10)
    sel.add_argument("--batch", type=int, default=30)
    sel.add_argument("--min-frame-gap", type=int, default=10)
    sel.add_argument("--min-confidence", type=float, default=0.0)

    fin = sub.add_parser("finalize")
    fin.add_argument("--review", type=Path, default=DEFAULT_REVIEW_PATH)
    fin.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_DIR)
    fin.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)

    prep = sub.add_parser("prepare-next-game")
    prep.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR)

    args = parser.parse_args()
    if args.command == "select":
        result = run_select(
            review_path=args.review,
            samples_dir=args.samples,
            out_dir=args.out,
            target=args.target,
            batch=args.batch,
            min_frame_gap=args.min_frame_gap,
            min_confidence=args.min_confidence,
        )
        _print_select_summary(result)
        return 0
    if args.command == "finalize":
        result = run_finalize(
            review_path=args.review,
            samples_dir=args.samples,
            out_dir=args.out,
        )
        print(f"Kept {result['kept']} curated crops.")
        print(f"Deleted {result['deleted_files']} raw files.")
        return 0
    if args.command == "prepare-next-game":
        result = run_prepare_next_game(cache_dir=args.cache)
        print(f"Removed {result['removed']} cached champion icons.")
        return 0
    return 1


# ---------- public API (also used by tests) ----------


def run_select(
    *,
    review_path: Path,
    samples_dir: Path,
    out_dir: Path,
    target: int = 10,
    batch: int = 30,
    min_frame_gap: int = 10,
    min_confidence: float = 0.0,
) -> dict:
    """Top up each champion folder up to `target`, picking from the next
    `batch` highest-confidence candidates per champion that aren't already
    on disk.

    If the candidate review JSON is missing, auto-build it from raw frames
    in `samples_dir` first (uses the same threshold as `min_confidence`).
    """
    if not review_path.exists():
        if not samples_dir.exists() or not any(samples_dir.iterdir()):
            raise FileNotFoundError(
                f"No candidates JSON at {review_path} and no raw frames found "
                f"under {samples_dir} (training_sessions). Run a game with "
                "--collect-minimap-training first."
            )
        from modules.minimap_curation import write_candidate_review
        write_candidate_review(
            samples_dir,
            review_path,
            min_confidence=min_confidence,
        )
        print(f"Auto-generated {review_path} from {samples_dir}.")

    review = json.loads(review_path.read_text())
    samples = review.get("samples", [])

    existing = _existing_crops(out_dir)
    existing_count_by_champ = _count_by_champ(existing)
    existing_frames_by_champ: dict[str, list[int]] = {}
    existing_pairs_by_champ: dict[str, set[tuple[int, int]]] = {}
    for (champ, frame, idx) in existing:
        existing_frames_by_champ.setdefault(champ, []).append(frame)
        existing_pairs_by_champ.setdefault(champ, set()).add((frame, idx))

    # Group all candidates by champion (preserving identity for later mutation).
    by_champ: dict[str, list[tuple[int, int, dict, dict]]] = {}
    for sample in samples:
        frame_idx = _frame_index(sample.get("source_image", ""))
        for cand_idx, cand in enumerate(sample.get("candidates", [])):
            if cand.get("confidence", 0) < min_confidence:
                continue
            champ = cand["champion"]
            by_champ.setdefault(champ, []).append((frame_idx, cand_idx, sample, cand))

    added: dict[str, int] = {}
    skipped: list[str] = []

    all_champs = sorted(set(list(by_champ.keys()) + list(existing_count_by_champ.keys())))
    for champ in all_champs:
        champ_dir_name = _safe_dirname(champ)
        champ_dir = out_dir / champ_dir_name

        history = _load_pair_set(champ_dir / HISTORY_FILE)
        rejected = _load_pair_set(champ_dir / REJECTED_FILE)

        # Anything previously offered (history) but no longer on disk = user rejected.
        on_disk_pairs = existing_pairs_by_champ.get(champ_dir_name, set())
        newly_rejected = (history - on_disk_pairs) - rejected
        if newly_rejected:
            rejected |= newly_rejected

        have = existing_count_by_champ.get(champ_dir_name, 0)
        if have >= target:
            skipped.append(champ)
            if newly_rejected:
                champ_dir.mkdir(parents=True, exist_ok=True)
                _save_pair_set(champ_dir / REJECTED_FILE, rejected)
            continue

        need = target - have
        # `batch` caps how many crops this round produces (so the user has a
        # bounded number to review). The pool itself is scanned in full so
        # frame-gap rejections don't prematurely starve `need`.
        pool = by_champ.get(champ, [])
        pool = [
            entry for entry in pool
            if (entry[0], entry[1]) not in on_disk_pairs
            and (entry[0], entry[1]) not in rejected
        ]
        pool.sort(key=lambda t: -t[3]["confidence"])
        round_cap = min(need, batch)

        used_frames = list(existing_frames_by_champ.get(champ_dir_name, []))
        picked: list[tuple[int, int, dict, dict]] = []
        for entry in pool:
            if len(picked) >= round_cap:
                break
            frame_idx = entry[0]
            if all(abs(frame_idx - f) >= min_frame_gap for f in used_frames):
                picked.append(entry)
                used_frames.append(frame_idx)

        if not picked:
            if newly_rejected:
                champ_dir.mkdir(parents=True, exist_ok=True)
                _save_pair_set(champ_dir / REJECTED_FILE, rejected)
            continue

        champ_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for frame_idx, cand_idx, sample, cand in picked:
            crop = _crop_candidate(samples_dir, sample, cand)
            if crop is None:
                continue
            conf_tag = f"{cand['confidence']:.3f}".replace(".", "p")
            out_name = f"{frame_idx:06d}_{cand_idx}_conf{conf_tag}.png"
            cv2.imwrite(str(champ_dir / out_name), crop)
            cand["accepted"] = True
            written += 1
        if written:
            added[champ] = written

        # Update ledgers: anything we just offered joins history; carry
        # forward newly-detected rejects.
        history |= {(p[0], p[1]) for p in picked}
        _save_pair_set(champ_dir / HISTORY_FILE, history)
        if newly_rejected or rejected:
            _save_pair_set(champ_dir / REJECTED_FILE, rejected)

    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2))
    return {"added": added, "skipped": skipped}


def run_finalize(
    *,
    review_path: Path,
    samples_dir: Path,
    out_dir: Path,
) -> dict:
    """Mark surviving crops as accepted, delete raw session data and the
    review JSON. The auto_selected/ folder is the durable artifact.
    """
    surviving = _existing_crops(out_dir)

    if review_path.exists():
        review = json.loads(review_path.read_text())
        for sample in review.get("samples", []):
            frame_idx = _frame_index(sample.get("source_image", ""))
            for cand_idx, cand in enumerate(sample.get("candidates", [])):
                key = (_safe_dirname(cand["champion"]), frame_idx, cand_idx)
                cand["accepted"] = key in surviving

    deleted_files = 0
    if samples_dir.exists():
        for child in list(samples_dir.iterdir()):
            if child.is_dir():
                deleted_files += sum(1 for _ in child.rglob("*"))
                shutil.rmtree(child)
            elif child.is_file():
                child.unlink()
                deleted_files += 1

    if review_path.exists():
        review_path.unlink()

    # Clear per-champion ledgers — they reference frame indices from THIS
    # session and would collide with the next session (frame_000029 reused).
    if out_dir.exists():
        for champ_dir in out_dir.iterdir():
            if not champ_dir.is_dir():
                continue
            for ledger_name in (HISTORY_FILE, REJECTED_FILE):
                ledger = champ_dir / ledger_name
                if ledger.exists():
                    ledger.unlink()

    return {"kept": len(surviving), "deleted_files": deleted_files}


def run_prepare_next_game(*, cache_dir: Path) -> dict:
    """Clear DDragon champion icon cache so the next match re-fetches only
    the champions it needs. Spell/other caches are untouched.
    """
    champ_root = cache_dir / "champion"
    if not champ_root.exists():
        return {"removed": 0}
    removed = sum(1 for _ in champ_root.rglob("*.png"))
    shutil.rmtree(champ_root)
    return {"removed": removed}


# ---------- helpers ----------


def _existing_crops(out_dir: Path) -> set[tuple[str, int, int]]:
    found: set[tuple[str, int, int]] = set()
    if not out_dir.exists():
        return found
    for champ_dir in out_dir.iterdir():
        if not champ_dir.is_dir():
            continue
        for png in champ_dir.glob("*.png"):
            m = CROP_NAME_RE.match(png.name)
            if not m:
                continue
            found.add((champ_dir.name, int(m["frame"]), int(m["idx"])))
    return found


def _count_by_champ(existing: set[tuple[str, int, int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for (champ, _frame, _idx) in existing:
        counts[champ] = counts.get(champ, 0) + 1
    return counts


def _crop_candidate(samples_dir: Path, sample: dict, cand: dict):
    src_rel = sample.get("source_image")
    if not src_rel:
        return None
    img = cv2.imread(str(samples_dir / src_rel), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    cx, cy = cand["center"]
    size = int(cand.get("size", 42))
    half = size // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, x0 + size)
    y1 = min(h, y0 + size)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def _load_pair_set(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(data, list):
        return set()
    out: set[tuple[int, int]] = set()
    for item in data:
        if isinstance(item, list) and len(item) == 2:
            try:
                out.add((int(item[0]), int(item[1])))
            except (TypeError, ValueError):
                continue
    return out


def _save_pair_set(path: Path, pairs: set[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_pairs = sorted([list(p) for p in pairs])
    path.write_text(json.dumps(sorted_pairs))


def _frame_index(source_image: str) -> int:
    m = re.search(r"frame_(\d+)\.png$", source_image)
    return int(m.group(1)) if m else -1


def _safe_dirname(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").strip() or "unknown"


def _print_select_summary(result: dict) -> None:
    added = result.get("added", {})
    skipped = result.get("skipped", [])
    if added:
        print("Added crops:")
        for champ, n in added.items():
            print(f"  {champ}: +{n}")
    else:
        print("Added crops: (none)")
    if skipped:
        print(f"Already saturated (skipped): {', '.join(skipped)}")


if __name__ == "__main__":
    raise SystemExit(main())
