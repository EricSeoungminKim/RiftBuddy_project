"""Multi-template loader: combine DDragon icons with curated minimap crops.

`auto_selected/<champion>/` is the durable store of human-curated good crops
produced by tools/minimap_curate.py + tools/minimap_manual_crop.py.

Returned dict feeds straight into MinimapTracker.load_template_images().
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np


def load_templates_for_match(
    icon_paths: dict[str, Path],
    auto_selected_dir: Path,
    *,
    max_per_champion: int | None = None,
    seed: int | None = None,
) -> dict[str, list[np.ndarray]]:
    """Load DDragon icon plus any curated crops for each champion.

    icon_paths: {display_name: path-to-ddragon-icon}
    auto_selected_dir: data/curation/auto_selected (may not exist)
    max_per_champion: cap on total templates per champion (DDragon counts
        toward the cap). When the curated pool exceeds the remaining slots,
        crops are randomly sub-sampled for diversity. None = no cap.
    seed: when set, sub-sampling is deterministic (useful for tests/logs).

    Returns: {display_name: [ndarray, ndarray, ...]} with the DDragon icon
    first, then up to (max_per_champion - 1) curated crops appended.
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    out: dict[str, list[np.ndarray]] = {}
    for champion, ddragon_path in icon_paths.items():
        templates: list[np.ndarray] = []
        ddragon_img = _safe_imread(ddragon_path)
        if ddragon_img is not None:
            templates.append(ddragon_img)

        curated = _load_curated_crops(auto_selected_dir / champion)

        if max_per_champion is not None and max_per_champion > 0:
            remaining = max(0, max_per_champion - len(templates))
            if len(curated) > remaining:
                curated = rng.sample(curated, remaining)
        templates.extend(curated)

        out[champion] = templates
    return out


def _load_curated_crops(champ_dir: Path) -> list[np.ndarray]:
    if not champ_dir.is_dir():
        return []
    crops: list[np.ndarray] = []
    for png in sorted(champ_dir.glob("*.png")):
        if png.name.startswith("."):
            continue
        img = _safe_imread(png)
        if img is not None:
            crops.append(img)
    return crops


def count_templates(templates: dict[str, list[np.ndarray]]) -> int:
    return sum(len(v) for v in templates.values())


def _safe_imread(path: Path):
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img
