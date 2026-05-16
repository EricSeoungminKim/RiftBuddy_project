import json
import shutil
from pathlib import Path
from typing import Any

DEFAULT_MIN_CONFIDENCE = 0.2
DEFAULT_ICON_SIZE_RATIO = 0.095
MIN_ICON_SIZE = 24
MAX_ICON_SIZE = 72


def build_candidate_review(
    samples_dir: Path,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Build an editable review document from raw minimap training samples."""
    samples = []
    for metadata_path in sorted(samples_dir.glob("*/*.json")):
        sample = _review_sample(samples_dir, metadata_path, min_confidence)
        if sample["candidates"]:
            samples.append(sample)

    return {
        "version": 1,
        "source": {
            "samples_dir": str(samples_dir),
            "candidate_source": "tracker_template_match",
        },
        "instructions": (
            "Set accepted=true only for candidates whose champion and center are "
            "visually correct. Edit champion/center/size before exporting if needed."
        ),
        "samples": samples,
    }


def write_candidate_review(
    samples_dir: Path,
    output_path: Path,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> int:
    """Write a candidate review JSON and return the number of candidates."""
    review = build_candidate_review(samples_dir, min_confidence=min_confidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sum(len(sample["candidates"]) for sample in review["samples"])


def export_accepted_candidates(
    review_path: Path,
    samples_dir: Path,
    image_dir: Path,
    labels_path: Path,
) -> int:
    """Export accepted review candidates into curated images and labels.json."""
    review = json.loads(review_path.read_text(encoding="utf-8"))
    labels = _load_existing_labels(labels_path)
    exported_count = 0

    for sample in review.get("samples", []):
        accepted = _accepted_candidates(sample)
        if not accepted:
            continue

        source_image = _required_str(sample, "source_image")
        source_path = samples_dir / source_image
        if not source_path.exists():
            raise ValueError(f"Missing source image: {source_image}")

        curated_name = _curated_image_name(source_image)
        destination = image_dir / curated_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        labels[curated_name] = {"champions": [_to_label(item) for item in accepted]}
        exported_count += len(accepted)

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return exported_count


def prune_unaccepted_samples(
    review_path: Path,
    samples_dir: Path,
    dry_run: bool = False,
) -> dict[str, int]:
    """Delete raw sample files that are not accepted in a review document."""
    review = json.loads(review_path.read_text(encoding="utf-8"))
    kept = _accepted_source_paths(review, samples_dir)
    deleted = 0

    for path in _review_source_paths(review, samples_dir):
        if path in kept or not path.exists():
            continue
        deleted += 1
        if not dry_run:
            path.unlink()

    return {"kept": len(kept), "deleted": deleted}


def _accepted_source_paths(review: dict[str, Any], samples_dir: Path) -> set[Path]:
    accepted: set[Path] = set()
    for sample in review.get("samples", []):
        if not _accepted_candidates(sample):
            continue
        for path in _sample_source_paths(sample, samples_dir):
            accepted.add(path)
    return accepted


def _review_source_paths(review: dict[str, Any], samples_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for sample in review.get("samples", []):
        paths.extend(_sample_source_paths(sample, samples_dir))
    return paths


def _sample_source_paths(sample: object, samples_dir: Path) -> list[Path]:
    if not isinstance(sample, dict):
        return []
    paths = []
    for key in ("source_image", "source_metadata"):
        value = sample.get(key)
        if isinstance(value, str) and value:
            paths.append(samples_dir / value)
    return paths


def _review_sample(
    samples_dir: Path,
    metadata_path: Path,
    min_confidence: float,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    image_name = _required_str(metadata, "image")
    frame_size = _frame_size(metadata)
    icon_size = _candidate_icon_size(frame_size)
    candidates = [
        _candidate_from_score(score, icon_size, frame_size)
        for score in metadata.get("champions", [])
        if _is_candidate_score(score, min_confidence)
    ]
    return {
        "source_image": str(
            (metadata_path.parent / image_name).relative_to(samples_dir)
        ),
        "source_metadata": str(metadata_path.relative_to(samples_dir)),
        "frame_size": frame_size,
        "candidates": candidates,
    }


def _is_candidate_score(value: object, min_confidence: float) -> bool:
    if not isinstance(value, dict):
        return False
    confidence = value.get("confidence")
    x = value.get("x")
    y = value.get("y")
    if not _is_number(confidence) or confidence < min_confidence:
        return False
    return _is_number(x) and _is_number(y)


def _candidate_from_score(
    score: dict[str, Any],
    icon_size: int,
    frame_size: dict[str, int],
) -> dict[str, Any]:
    champion = _required_str(score, "champion")
    x = int(round(score["x"]))
    y = int(round(score["y"]))
    center = [
        _clamp(x + icon_size // 2, 0, frame_size["width"] - 1),
        _clamp(y + icon_size // 2, 0, frame_size["height"] - 1),
    ]
    return {
        "champion": champion,
        "confidence": round(float(score["confidence"]), 3),
        "center": center,
        "size": icon_size,
        "accepted": False,
        "notes": "",
        "match_source": "tracker_template_match",
    }


def _candidate_icon_size(frame_size: dict[str, int]) -> int:
    side = min(frame_size["width"], frame_size["height"])
    guessed_size = int(round(side * DEFAULT_ICON_SIZE_RATIO))
    return _clamp(guessed_size, MIN_ICON_SIZE, MAX_ICON_SIZE)


def _accepted_candidates(sample: object) -> list[dict[str, Any]]:
    if not isinstance(sample, dict):
        return []
    candidates = sample.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [
        item for item in candidates if isinstance(item, dict) and item.get("accepted")
    ]


def _to_label(candidate: dict[str, Any]) -> dict[str, Any]:
    champion = _required_str(candidate, "champion")
    center = candidate.get("center")
    size = candidate.get("size")
    if not _is_center(center):
        raise ValueError(f"{champion} candidate needs center [x, y].")
    if not isinstance(size, int) or isinstance(size, bool):
        raise ValueError(f"{champion} candidate needs integer size.")
    return {"champion": champion, "center": center, "size": max(1, size)}


def _load_existing_labels(labels_path: Path) -> dict[str, Any]:
    if not labels_path.exists():
        return {}
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict):
        raise ValueError("Existing labels file must contain a JSON object.")
    return labels


def _curated_image_name(source_image: str) -> str:
    source_path = Path(source_image)
    session = source_path.parent.name
    stem = source_path.stem
    return f"curated/{session}__{stem}{source_path.suffix}"


def _frame_size(metadata: dict[str, Any]) -> dict[str, int]:
    frame_size = metadata.get("frame_size")
    if not isinstance(frame_size, dict):
        raise ValueError("Recorded metadata needs frame_size.")
    width = frame_size.get("width")
    height = frame_size.get("height")
    if not _is_number(width) or not _is_number(height):
        raise ValueError("Recorded frame_size needs numeric width and height.")
    return {"width": int(width), "height": int(height)}


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected non-empty string field: {key}")
    return value


def _is_center(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    return all(_is_number(item) for item in value)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))
