import json
from pathlib import Path

import cv2
import numpy as np

from modules.minimap_curation import (
    build_candidate_review,
    export_accepted_candidates,
)


def test_build_candidate_review_from_recorded_metadata(tmp_path: Path) -> None:
    samples_dir = tmp_path / "sessions"
    session_dir = samples_dir / "session-a"
    session_dir.mkdir(parents=True)
    _write_image(session_dir / "frame_000001.png")
    _write_json(
        session_dir / "frame_000001.json",
        {
            "image": "frame_000001.png",
            "frame_size": {"width": 100, "height": 100},
            "champions": [
                {
                    "champion": "Nasus",
                    "confidence": 0.62,
                    "x": 10,
                    "y": 12,
                },
                {
                    "champion": "Galio",
                    "confidence": 0.12,
                    "x": 40,
                    "y": 42,
                },
            ],
        },
    )

    review = build_candidate_review(samples_dir, min_confidence=0.2)

    assert review["version"] == 1
    assert review["source"]["candidate_source"] == "tracker_template_match"
    assert review["samples"] == [
        {
            "source_image": "session-a/frame_000001.png",
            "source_metadata": "session-a/frame_000001.json",
            "frame_size": {"width": 100, "height": 100},
            "candidates": [
                {
                    "champion": "Nasus",
                    "confidence": 0.62,
                    "center": [22, 24],
                    "size": 24,
                    "accepted": False,
                    "notes": "",
                    "match_source": "tracker_template_match",
                }
            ],
        }
    ]


def test_export_accepted_candidates_to_training_labels(tmp_path: Path) -> None:
    samples_dir = tmp_path / "sessions"
    session_dir = samples_dir / "session-a"
    session_dir.mkdir(parents=True)
    source_image = session_dir / "frame_000001.png"
    _write_image(source_image)

    review_path = tmp_path / "review.json"
    _write_json(
        review_path,
        {
            "version": 1,
            "samples": [
                {
                    "source_image": "session-a/frame_000001.png",
                    "candidates": [
                        {
                            "champion": "Nasus",
                            "center": [22, 24],
                            "size": 24,
                            "accepted": True,
                        },
                        {
                            "champion": "Galio",
                            "center": [60, 60],
                            "size": 24,
                            "accepted": False,
                        },
                    ],
                }
            ],
        },
    )
    image_dir = tmp_path / "img"
    image_dir.mkdir()
    labels_path = image_dir / "labels.json"
    _write_json(
        labels_path,
        {"existing.png": {"champions": [{"champion": "Jhin", "center": [5, 5]}]}},
    )

    exported_count = export_accepted_candidates(
        review_path=review_path,
        samples_dir=samples_dir,
        image_dir=image_dir,
        labels_path=labels_path,
    )

    assert exported_count == 1
    curated_image = image_dir / "curated" / "session-a__frame_000001.png"
    assert cv2.imread(str(curated_image)).shape == (64, 64, 3)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert labels["existing.png"]["champions"][0]["champion"] == "Jhin"
    assert labels["curated/session-a__frame_000001.png"] == {
        "champions": [{"champion": "Nasus", "center": [22, 24], "size": 24}]
    }


def _write_image(path: Path) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:40, 18:42] = [20, 120, 220]
    cv2.imwrite(str(path), image)


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
