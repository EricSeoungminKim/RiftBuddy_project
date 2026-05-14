import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ChampionLabel:
    """One labeled champion icon inside a captured minimap fixture."""

    champion: str
    center_x: int
    center_y: int
    size: int


def load_training_labels(labels_path: Path) -> dict[str, list[ChampionLabel]]:
    """Load minimap fixture labels from JSON sidecar data."""
    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Minimap labels must be a JSON object.")

    labels: dict[str, list[ChampionLabel]] = {}
    for image_name, image_data in raw.items():
        if not isinstance(image_name, str) or not isinstance(image_data, dict):
            raise ValueError("Each minimap label entry must map filename to object.")

        champions = image_data.get("champions")
        if not isinstance(champions, list):
            raise ValueError(f"{image_name} needs a champions list.")

        labels[image_name] = [_parse_label(image_name, item) for item in champions]
    return labels


def extract_templates_from_labels(
    image_dir: Path,
    labels_path: Path,
) -> dict[str, list[np.ndarray]]:
    """Extract learned champion icon templates from labeled minimap captures."""
    labels = load_training_labels(labels_path)
    templates: dict[str, list[np.ndarray]] = {}
    for image_name, champion_labels in labels.items():
        image = cv2.imread(str(image_dir / image_name))
        if image is None:
            raise ValueError(f"Could not read minimap fixture: {image_name}")

        for label in champion_labels:
            patch = _crop_center(image, label.center_x, label.center_y, label.size)
            templates.setdefault(label.champion, []).append(patch)
    return templates


def _parse_label(image_name: str, item: object) -> ChampionLabel:
    """Validate one champion label object."""
    if not isinstance(item, dict):
        raise ValueError(f"{image_name} champion labels must be objects.")

    champion = item.get("champion")
    center = item.get("center")
    size = item.get("size", 40)
    if not isinstance(champion, str) or not champion:
        raise ValueError(f"{image_name} label needs champion.")
    if not _is_point(center):
        raise ValueError(f"{image_name} {champion} label needs center [x, y].")
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError(f"{image_name} {champion} label size must be an integer.")

    return ChampionLabel(champion, int(center[0]), int(center[1]), max(1, size))


def _is_point(value: Any) -> bool:
    """Return whether a value is a two-number coordinate."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return False
    return all(
        isinstance(item, int | float) and not isinstance(item, bool) for item in value
    )


def _crop_center(
    image: np.ndarray, center_x: int, center_y: int, size: int
) -> np.ndarray:
    """Crop a square centered on a labeled champion icon."""
    half = size // 2
    left = max(0, center_x - half)
    top = max(0, center_y - half)
    right = min(image.shape[1], left + size)
    bottom = min(image.shape[0], top + size)
    left = max(0, right - size)
    top = max(0, bottom - size)
    return image[top:bottom, left:right]
