"""Extract handcrafted features for every BloodMNIST split, save them and record timings.

Features go to cache/features_bloodmnist<res>.npz and timings to results/extraction_time.csv.
With --limit, only the first N images per split are used and nothing is written.

Run with: python src/extract_features.py [--resolutions 28 64 128] [--limit N]
"""

import argparse
import csv
import platform
import time
from datetime import date
from pathlib import Path

import numpy as np

from data import CLASS_SHORT_NAMES, REPO_ROOT, RESOLUTIONS, load_all
from features import extract_batch, feature_names

CACHE_DIR = REPO_ROOT / "cache"
TIMING_PATH = REPO_ROOT / "results" / "extraction_time.csv"
TIMING_COLUMNS = (
    "resolution",
    "split",
    "n_images",
    "seconds",
    "ms_per_image",
    "processor",
    "n_processes",
    "date",
)
N_PROCESSES = 1

FEATURE_NAMES = feature_names()
NUCLEUS_COLUMNS = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("shape_nucleus_")]


def features_path(resolution: int) -> Path:
    """Return the cache file holding the features of one resolution."""
    return CACHE_DIR / f"features_bloodmnist{resolution}.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        choices=RESOLUTIONS,
        default=list(RESOLUTIONS),
        help="resolutions to extract (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="use only the first N images per split, print timing only and write no files",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer.")
    return args


def missing_nucleus_counts(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return, per class, the number of images whose nucleus features are all 0."""
    missing = (features[:, NUCLEUS_COLUMNS] == 0).all(axis=1)
    return np.bincount(labels[missing], minlength=len(CLASS_SHORT_NAMES))


def print_split(split: str, n_images: int, seconds: float, missing: np.ndarray) -> None:
    print(
        f"  {split:<5} {n_images:>6} images  {seconds:>8.1f} s  "
        f"{seconds / n_images * 1000:>6.2f} ms/image  nucleus not found: {missing.sum()}"
    )
    for name, count in zip(CLASS_SHORT_NAMES, missing):
        if count:
            print(f"        {name:<22} {count}")


def remove_timing_rows(resolutions: list[int]) -> None:
    """Drop existing timing rows of the given resolutions so a rerun does not duplicate them."""
    if not TIMING_PATH.is_file():
        return
    with TIMING_PATH.open(newline="") as file:
        rows = [row for row in csv.DictReader(file) if int(row["resolution"]) not in resolutions]
    with TIMING_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TIMING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def append_timing_rows(rows: list[dict[str, object]]) -> None:
    """Append timing rows, writing the header if the file is new."""
    TIMING_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not TIMING_PATH.is_file()
    with TIMING_PATH.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TIMING_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def extract_resolution(
    resolution: int, limit: int | None
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Extract all splits of one resolution; return the arrays to save and the timing rows."""
    print(f"{resolution} px")
    arrays: dict[str, np.ndarray] = {}
    timing_rows: list[dict[str, object]] = []
    for split, (images, labels, indices) in load_all(resolution).items():
        images, labels, indices = images[:limit], labels[:limit], indices[:limit]
        start = time.perf_counter()
        features = extract_batch(images).astype(np.float32)
        seconds = time.perf_counter() - start
        print_split(split, len(images), seconds, missing_nucleus_counts(features, labels))
        arrays |= {
            f"{split}_features": features,
            f"{split}_labels": labels,
            f"{split}_indices": indices,
        }
        timing_rows.append({
            "resolution": resolution,
            "split": split,
            "n_images": len(images),
            "seconds": round(seconds, 3),
            "ms_per_image": round(seconds / len(images) * 1000, 3),
            "processor": platform.processor(),
            "n_processes": N_PROCESSES,
            "date": date.today().isoformat(),
        })
    return arrays, timing_rows


def main() -> None:
    args = parse_args()
    resolutions = sorted(set(args.resolutions))
    save = args.limit is None
    if save:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        remove_timing_rows(resolutions)

    saved: list[Path] = []
    for resolution in resolutions:
        arrays, timing_rows = extract_resolution(resolution, args.limit)
        if save:
            path = features_path(resolution)
            np.savez(path, **arrays, feature_names=np.array(FEATURE_NAMES))
            append_timing_rows(timing_rows)
            saved.append(path)

    if save:
        print("Saved:")
        for path in [*saved, TIMING_PATH]:
            print(f"  {path}")
    else:
        print(f"Timing test with --limit {args.limit}: no files written.")


if __name__ == "__main__":
    main()
