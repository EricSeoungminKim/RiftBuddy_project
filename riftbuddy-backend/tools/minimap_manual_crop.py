"""Manually crop a single champion icon out of a captured minimap frame.

Usage:
  ./.venv/bin/python tools/minimap_manual_crop.py \\
      <frame_rel> <champion> <cx> <cy> [size]

Example:
  ./.venv/bin/python tools/minimap_manual_crop.py \\
      20260514_015502/frame_000029.png 블리츠크랭크 350 365

Defaults:
  size      42
  samples   data/training_sessions
  out       data/curation/auto_selected
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2

DEFAULT_SAMPLES_DIR = Path("data/training_sessions")
DEFAULT_OUT_DIR = Path("data/curation/auto_selected")
DEFAULT_SIZE = 42

FRAME_INDEX_RE = re.compile(r"frame_(\d+)\.png$")


def crop_one(
    *,
    frame_rel: str,
    champion: str,
    cx: int,
    cy: int,
    size: int = DEFAULT_SIZE,
    samples_dir: Path = DEFAULT_SAMPLES_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> Path:
    """Crop a `size x size` box centered at (cx, cy) from the given frame
    into out_dir/<champion>/. Returns the written path.

    Re-cropping the same frame for the same champion replaces the previous
    file (same frame index in the filename → glob deletes prior matches).
    """
    src = samples_dir / frame_rel
    if not src.exists():
        raise FileNotFoundError(f"frame not found: {src}")
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {src}")
    h, w = img.shape[:2]
    half = size // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, x0 + size)
    y1 = min(h, y0 + size)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"crop box collapsed at cx={cx} cy={cy} size={size}")
    crop = img[y0:y1, x0:x1]

    frame_idx = _frame_index(frame_rel)
    champ_dir = out_dir / _safe_dirname(champion)
    champ_dir.mkdir(parents=True, exist_ok=True)
    # Replace any previous manual crop for this frame+champion.
    for prior in champ_dir.glob(f"{frame_idx:06d}_manual_*.png"):
        prior.unlink()

    out_path = champ_dir / f"{frame_idx:06d}_manual_cx{cx}cy{cy}.png"
    cv2.imwrite(str(out_path), crop)
    return out_path


def _frame_index(frame_rel: str) -> int:
    m = FRAME_INDEX_RE.search(frame_rel)
    return int(m.group(1)) if m else -1


def _safe_dirname(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").strip() or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame_rel", help="Relative path under --samples (e.g. 20260514_015502/frame_000029.png)")
    parser.add_argument("champion", help="Champion name as it appears in the candidates JSON (e.g. 블리츠크랭크)")
    parser.add_argument("cx", type=int)
    parser.add_argument("cy", type=int)
    parser.add_argument("size", type=int, nargs="?", default=DEFAULT_SIZE)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    written = crop_one(
        frame_rel=args.frame_rel,
        champion=args.champion,
        cx=args.cx,
        cy=args.cy,
        size=args.size,
        samples_dir=args.samples,
        out_dir=args.out,
    )
    print(f"wrote {written}  ({args.size}x{args.size} @ cx={args.cx}, cy={args.cy})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
