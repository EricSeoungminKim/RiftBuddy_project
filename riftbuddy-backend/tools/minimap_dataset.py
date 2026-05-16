import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.minimap_curation import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    export_accepted_candidates,
    prune_unaccepted_samples,
    write_candidate_review,
)

DEFAULT_SAMPLES_DIR = Path("data/training_sessions")
DEFAULT_REVIEW_PATH = Path("data/curation/minimap_candidates.json")
DEFAULT_IMAGE_DIR = Path("img")
DEFAULT_LABELS_PATH = Path("img/labels.json")


def parse_args() -> argparse.Namespace:
    """Parse minimap dataset curation commands."""
    parser = argparse.ArgumentParser(description="Curate RiftBuddy minimap datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    suggest = subparsers.add_parser(
        "suggest",
        help="Create an editable candidate review JSON from raw training samples",
    )
    suggest.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_DIR)
    suggest.add_argument("--out", type=Path, default=DEFAULT_REVIEW_PATH)
    suggest.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
    )

    export = subparsers.add_parser(
        "export",
        help="Export accepted candidates into img/labels.json",
    )
    export.add_argument("--review", type=Path, default=DEFAULT_REVIEW_PATH)
    export.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_DIR)
    export.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    export.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)

    prune = subparsers.add_parser(
        "prune",
        help="Delete raw sample files that were not accepted in a review JSON",
    )
    prune.add_argument("--review", type=Path, default=DEFAULT_REVIEW_PATH)
    prune.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES_DIR)
    prune.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many files would be deleted without deleting them",
    )
    prune.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion. Required unless --dry-run is used.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the selected dataset curation command."""
    args = parse_args()
    if args.command == "suggest":
        count = write_candidate_review(
            samples_dir=args.samples,
            output_path=args.out,
            min_confidence=args.min_confidence,
        )
        print(f"Wrote {count} candidate(s) to {args.out}")
        return

    if args.command == "export":
        count = export_accepted_candidates(
            review_path=args.review,
            samples_dir=args.samples,
            image_dir=args.image_dir,
            labels_path=args.labels,
        )
        print(f"Exported {count} accepted candidate(s) to {args.labels}")
        return

    if args.command == "prune":
        if not args.dry_run and not args.yes:
            raise SystemExit("Refusing to delete without --yes. Use --dry-run first.")
        result = prune_unaccepted_samples(
            review_path=args.review,
            samples_dir=args.samples,
            dry_run=args.dry_run,
        )
        action = "Would delete" if args.dry_run else "Deleted"
        print(f"{action} {result['deleted']} raw file(s); kept {result['kept']}.")


if __name__ == "__main__":
    main()
