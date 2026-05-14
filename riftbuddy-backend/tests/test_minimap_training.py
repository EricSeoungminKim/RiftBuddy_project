import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.event_bus import EventBus
from core.game_state import GameState
from modules.minimap_tracker import MinimapTracker
from modules.minimap_training import extract_templates_from_labels, load_training_labels


def test_extract_templates_from_labels_crops_champion_patch(tmp_path: Path) -> None:
    image_dir = tmp_path / "img"
    image_dir.mkdir()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[38:62, 38:62] = [20, 120, 220]
    cv2.imwrite(str(image_dir / "sample.png"), image)
    labels_path = image_dir / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "sample.png": {
                    "champions": [{"champion": "Nasus", "center": [50, 50], "size": 24}]
                }
            }
        ),
        encoding="utf-8",
    )

    templates = extract_templates_from_labels(image_dir, labels_path)

    assert list(templates) == ["Nasus"]
    assert templates["Nasus"][0].shape == (24, 24, 3)
    assert templates["Nasus"][0].mean() > 100


def test_load_training_labels_validates_required_fields(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps({"sample.png": {"champions": [{"champion": "Nasus"}]}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="center"):
        load_training_labels(labels_path)


def test_repo_minimap_fixture_labels_match_existing_images() -> None:
    image_dir = Path("img")
    labels_path = image_dir / "labels.json"
    if not labels_path.exists():
        pytest.skip("No labeled minimap fixtures yet.")

    labels = load_training_labels(labels_path)

    assert labels
    for image_name in labels:
        assert (image_dir / image_name).exists()


def test_repo_minimap_fixtures_score_learned_templates() -> None:
    image_dir = Path("img")
    labels_path = image_dir / "labels.json"
    if not labels_path.exists():
        pytest.skip("No labeled minimap fixtures yet.")

    labels = load_training_labels(labels_path)
    templates = extract_templates_from_labels(image_dir, labels_path)
    tracker = MinimapTracker(EventBus(), GameState())
    tracker.load_template_images(templates)

    for image_name, champion_labels in labels.items():
        image = cv2.imread(str(image_dir / image_name))
        assert image is not None
        champions = [label.champion for label in champion_labels]
        scores = tracker._score_icons(image, champions)

        for champion in champions:
            assert scores[champion] is not None
            assert scores[champion].confidence > 0.85
