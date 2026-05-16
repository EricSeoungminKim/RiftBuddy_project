"""Auto-select balanced minimap crops per champion for review.

Workflow:
  1. select  → pick top-N candidates per champion (confidence + frame-gap diversity),
               crop them from source PNGs into data/curation/auto_selected/<champion>/,
               and mark accepted=true on those candidates in the review JSON.
  2. (human deletes bad crops from the folders)
  3. sync    → re-scan folders; any candidate whose crop file is missing gets
               accepted=false again. Survivors stay accepted=true.

After sync, run minimap_dataset.py export / prune as usual.
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

CROP_NAME_RE = re.compile(r"^(?P<frame>\d+)_(?P<idx>\d+)_conf(?P<conf>[0-9p]+)\.png$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sel = sub.add_parser("select", help="Pick top-N per champion and crop them.")
    sel.add_argument("--review", type=Path, default=DEFAULT_REVIEW_PATH)
    sel.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_DIR)
    sel.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    sel.add_argument("--per-champion", type=int, default=20)
    sel.add_argument("--min-frame-gap", type=int, default=20)
    sel.add_argument("--min-confidence", type=float, default=0.0)
    sel.add_argument("--dry-run", action="store_true")

    syn = sub.add_parser(
        "sync",
        help="Re-mark accepted based on which crop files survive in the out folder.",
    )
    syn.add_argument("--review", type=Path, default=DEFAULT_REVIEW_PATH)
    syn.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    syn.add_argument("--dry-run", action="store_true")

    more = sub.add_parser(
        "more",
        help="Add N more candidates to chosen champions, skipping ones already in the folder.",
    )
    more.add_argument("--review", type=Path, default=DEFAULT_REVIEW_PATH)
    more.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_DIR)
    more.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    more.add_argument(
        "--champion",
        action="append",
        help="Champion name (repeatable). Omit to add to ALL champions.",
    )
    more.add_argument("--add", type=int, default=20, help="How many new crops per champion.")
    more.add_argument("--min-frame-gap", type=int, default=20)
    more.add_argument("--min-confidence", type=float, default=0.0)
    more.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.command == "select":
        return cmd_select(args)
    if args.command == "sync":
        return cmd_sync(args)
    if args.command == "more":
        return cmd_more(args)
    return 1


def cmd_select(args) -> int:
    review = json.loads(args.review.read_text())
    samples = review.get("samples", [])

    # Flatten: per-champion list of (frame_idx, sample_ref, candidate_ref).
    by_champ: dict[str, list[tuple[int, dict, dict]]] = {}
    for sample in samples:
        frame_idx = _frame_index(sample.get("source_image", ""))
        for cand in sample.get("candidates", []):
            if cand.get("confidence", 0) < args.min_confidence:
                continue
            by_champ.setdefault(cand["champion"], []).append((frame_idx, sample, cand))

    # Reset all accepted flags first, then re-mark winners.
    for sample in samples:
        for cand in sample.get("candidates", []):
            cand["accepted"] = False

    selections: dict[str, list[tuple[int, dict, dict]]] = {}
    for champ, items in by_champ.items():
        items.sort(key=lambda t: -t[2]["confidence"])
        picked: list[tuple[int, dict, dict]] = []
        for entry in items:
            if len(picked) >= args.per_champion:
                break
            frame_idx = entry[0]
            if all(abs(frame_idx - p[0]) >= args.min_frame_gap for p in picked):
                picked.append(entry)
        selections[champ] = picked

    total_picked = sum(len(v) for v in selections.values())
    print(f"Selected {total_picked} crops across {len(selections)} champions:")
    for champ, picked in selections.items():
        confs = [round(c["confidence"], 3) for _, _, c in picked]
        print(f"  {champ}: {len(picked)} (conf {confs})")

    if args.dry_run:
        return 0

    # Wipe + recreate out dir to keep filesystem in sync with current selection.
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    written = 0
    for champ, picked in selections.items():
        champ_dir = args.out / _safe_dirname(champ)
        champ_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, sample, cand in picked:
            src_rel = sample.get("source_image")
            if not src_rel:
                continue
            src_path = args.samples / src_rel
            crop = _crop_candidate(src_path, cand, sample.get("frame_size"))
            if crop is None:
                print(f"  ! skip {champ} frame {frame_idx} ({src_rel} unreadable)")
                continue
            cand_idx = sample["candidates"].index(cand)
            conf_tag = f"{cand['confidence']:.3f}".replace(".", "p")
            out_name = f"{frame_idx:06d}_{cand_idx}_conf{conf_tag}.png"
            cv2.imwrite(str(champ_dir / out_name), crop)
            cand["accepted"] = True
            written += 1

    args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2))
    print(f"Wrote {written} crops to {args.out}")
    print(f"Updated {args.review} (accepted flags refreshed).")
    print("Now: open each champion folder, delete bad crops, then run `sync`.")
    return 0


def cmd_sync(args) -> int:
    review = json.loads(args.review.read_text())
    samples = review.get("samples", [])

    surviving: dict[tuple[str, int, int], Path] = {}
    if args.out.exists():
        for champ_dir in args.out.iterdir():
            if not champ_dir.is_dir():
                continue
            champ = champ_dir.name
            for png in champ_dir.glob("*.png"):
                m = CROP_NAME_RE.match(png.name)
                if not m:
                    continue
                key = (champ, int(m["frame"]), int(m["idx"]))
                surviving[key] = png

    accepted_count = 0
    rejected_count = 0
    for sample in samples:
        frame_idx = _frame_index(sample.get("source_image", ""))
        for cand_idx, cand in enumerate(sample.get("candidates", [])):
            key = (_safe_dirname(cand["champion"]), frame_idx, cand_idx)
            if key in surviving:
                cand["accepted"] = True
                accepted_count += 1
            else:
                if cand.get("accepted"):
                    rejected_count += 1
                cand["accepted"] = False

    print(f"Surviving crops: {len(surviving)}")
    print(f"Marked accepted=true: {accepted_count}")
    print(f"Reverted accepted to false: {rejected_count}")

    if args.dry_run:
        return 0
    args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2))
    print(f"Updated {args.review}.")
    print("Next: ./.venv/bin/python tools/minimap_dataset.py export")
    return 0


def cmd_more(args) -> int:
    review = json.loads(args.review.read_text())
    samples = review.get("samples", [])

    # Index existing crops in the out dir: (champ_dir_name, frame_idx, cand_idx)
    existing: set[tuple[str, int, int]] = set()
    if args.out.exists():
        for champ_dir in args.out.iterdir():
            if not champ_dir.is_dir():
                continue
            for png in champ_dir.glob("*.png"):
                m = CROP_NAME_RE.match(png.name)
                if not m:
                    continue
                existing.add((champ_dir.name, int(m["frame"]), int(m["idx"])))

    # Build per-champion candidate list excluding ones already on disk.
    by_champ: dict[str, list[tuple[int, int, dict, dict]]] = {}
    for sample in samples:
        frame_idx = _frame_index(sample.get("source_image", ""))
        for cand_idx, cand in enumerate(sample.get("candidates", [])):
            if cand.get("confidence", 0) < args.min_confidence:
                continue
            champ = cand["champion"]
            if (_safe_dirname(champ), frame_idx, cand_idx) in existing:
                continue
            by_champ.setdefault(champ, []).append((frame_idx, cand_idx, sample, cand))

    targets = args.champion if args.champion else list(by_champ.keys())

    additions: dict[str, list[tuple[int, int, dict, dict]]] = {}
    for champ in targets:
        items = by_champ.get(champ, [])
        items.sort(key=lambda t: -t[3]["confidence"])

        # Existing frame indices for this champion (so new picks stay spread out).
        champ_dir_name = _safe_dirname(champ)
        used_frames = [f for (cd, f, _i) in existing if cd == champ_dir_name]

        picked: list[tuple[int, int, dict, dict]] = []
        for entry in items:
            if len(picked) >= args.add:
                break
            frame_idx = entry[0]
            picked_frames = used_frames + [p[0] for p in picked]
            if all(abs(frame_idx - f) >= args.min_frame_gap for f in picked_frames):
                picked.append(entry)
        additions[champ] = picked

    total = sum(len(v) for v in additions.values())
    print(f"Adding {total} new crops:")
    for champ, picked in additions.items():
        confs = [round(c["confidence"], 3) for _, _, _, c in picked]
        print(f"  {champ}: +{len(picked)} (conf {confs})")

    if args.dry_run:
        return 0

    written = 0
    for champ, picked in additions.items():
        if not picked:
            continue
        champ_dir = args.out / _safe_dirname(champ)
        champ_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, cand_idx, sample, cand in picked:
            src_rel = sample.get("source_image")
            if not src_rel:
                continue
            src_path = args.samples / src_rel
            crop = _crop_candidate(src_path, cand, sample.get("frame_size"))
            if crop is None:
                print(f"  ! skip {champ} frame {frame_idx} ({src_rel} unreadable)")
                continue
            conf_tag = f"{cand['confidence']:.3f}".replace(".", "p")
            out_name = f"{frame_idx:06d}_{cand_idx}_conf{conf_tag}.png"
            cv2.imwrite(str(champ_dir / out_name), crop)
            cand["accepted"] = True
            written += 1

    args.review.write_text(json.dumps(review, ensure_ascii=False, indent=2))
    print(f"Wrote {written} new crops.")
    print("Now: review the new crops, delete bad ones, then run `sync`.")
    return 0


def _crop_candidate(src_path: Path, cand: dict, frame_size: dict | None):
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
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


def _frame_index(source_image: str) -> int:
    m = re.search(r"frame_(\d+)\.png$", source_image)
    return int(m.group(1)) if m else -1


def _safe_dirname(name: str) -> str:
    # Folder name = champion name as-is (Korean is fine on macOS); strip path separators.
    return name.replace("/", "_").replace("\\", "_").strip() or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
