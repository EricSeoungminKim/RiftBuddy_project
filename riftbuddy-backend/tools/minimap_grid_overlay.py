"""Draw a coordinate grid on top of a minimap PNG.

Usage:
  ./.venv/bin/python tools/minimap_grid_overlay.py <frame> [--spacing 25] [--no-open]

Examples:
  ./.venv/bin/python tools/minimap_grid_overlay.py 20260514_015502/frame_000029.png
  ./.venv/bin/python tools/minimap_grid_overlay.py /abs/path/to/frame.png --spacing 50

Output: <frame_dir>/<basename>_grid.png  (e.g. frame_000029_grid.png)
On macOS the output opens in Preview unless --no-open is passed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2

DEFAULT_SAMPLES_DIR = Path("data/training_sessions")
DEFAULT_SPACING = 25
GRID_COLOR = (0, 255, 255)       # BGR yellow
MAJOR_COLOR = (0, 255, 0)        # BGR green for every-Nth line
LABEL_COLOR = (255, 255, 255)
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.35
LABEL_THICK = 1


def write_overlay(
    frame: str | Path,
    *,
    spacing: int = DEFAULT_SPACING,
    samples_dir: Path = DEFAULT_SAMPLES_DIR,
) -> Path:
    src = _resolve(frame, samples_dir)
    if not src.exists():
        raise FileNotFoundError(f"frame not found: {src}")
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {src}")

    overlay = img.copy()
    h, w = overlay.shape[:2]

    for x in range(0, w, spacing):
        color = MAJOR_COLOR if x % (spacing * 4) == 0 else GRID_COLOR
        cv2.line(overlay, (x, 0), (x, h - 1), color, 1, cv2.LINE_AA)
    for y in range(0, h, spacing):
        color = MAJOR_COLOR if y % (spacing * 4) == 0 else GRID_COLOR
        cv2.line(overlay, (0, y), (w - 1, y), color, 1, cv2.LINE_AA)

    # Blend so the underlying image stays visible.
    blended = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

    # Coordinate labels along top and left, every (spacing*2).
    for x in range(0, w, spacing * 2):
        cv2.putText(blended, str(x), (x + 2, 12), LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)
    for y in range(0, h, spacing * 2):
        if y == 0:
            continue
        cv2.putText(blended, str(y), (2, y - 2), LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICK, cv2.LINE_AA)

    out_path = src.with_name(f"{src.stem}_grid.png")
    cv2.imwrite(str(out_path), blended)
    return out_path


def _resolve(frame: str | Path, samples_dir: Path) -> Path:
    p = Path(frame)
    if p.is_absolute() or p.exists():
        return p
    return samples_dir / p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", help="Frame path (relative to --samples or absolute)")
    parser.add_argument("--spacing", type=int, default=DEFAULT_SPACING)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open the result")
    args = parser.parse_args()

    out = write_overlay(args.frame, spacing=args.spacing, samples_dir=args.samples)
    print(f"wrote {out}  (spacing={args.spacing}px)")

    if not args.no_open and sys.platform == "darwin":
        try:
            subprocess.run(["open", str(out)], check=False)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
