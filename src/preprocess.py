"""Per-channel normalization for the CNN models, fitted on the training split only.

Run with: python src/preprocess.py
"""

import json
from collections.abc import Iterable, Iterator

import numpy as np

from data import REPO_ROOT, RESOLUTIONS, load_split

STATS_PATH = REPO_ROOT / "cache" / "normalization_stats.json"
CHUNK_SIZE = 256  # images per chunk; about 100 MB of float64 at 128 px
SANITY_TOLERANCE = 1e-4


def _check_images(images: np.ndarray) -> None:
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"Expected uint8 images of shape (N, H, W, 3), got {images.dtype} {images.shape}."
        )


def _chunks(images: np.ndarray) -> Iterator[np.ndarray]:
    for start in range(0, len(images), CHUNK_SIZE):
        yield images[start:start + CHUNK_SIZE]


def _moments(chunks: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population std per column of float64 (M, 3) chunks."""
    count = 0
    total = np.zeros(3)
    total_sq = np.zeros(3)
    for values in chunks:
        count += len(values)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
    if count == 0:
        raise ValueError("Cannot compute statistics of an empty image array.")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 0.0))
    return mean, std


def channel_stats(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel mean and std (ddof=0) of uint8 RGB images scaled to [0, 1]."""
    _check_images(images)
    return _moments(chunk.reshape(-1, 3).astype(np.float64) / 255 for chunk in _chunks(images))


def normalize(images: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Scale to [0, 1], standardize per channel and return float32 (N, 3, H, W)."""
    _check_images(images)
    scaled = images.astype(np.float32) / 255
    standardized = (scaled - mean.astype(np.float32)) / std.astype(np.float32)
    return np.ascontiguousarray(standardized.transpose(0, 3, 1, 2))


def load_stats(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the saved (mean, std) for one resolution."""
    if not STATS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {STATS_PATH}. Run python src/preprocess.py first."
        )
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    entry = stats.get(str(resolution))
    if entry is None:
        raise ValueError(
            f"No normalization stats for {resolution} px in {STATS_PATH}. "
            "Run python src/preprocess.py."
        )
    return np.array(entry["mean"], dtype=np.float64), np.array(entry["std"], dtype=np.float64)


def _normalized_stats(
    images: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return _moments(
        normalize(chunk, mean, std).transpose(0, 2, 3, 1).reshape(-1, 3).astype(np.float64)
        for chunk in _chunks(images)
    )


def _format(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.6f}" for v in values) + "]"


def main() -> None:
    stats = {}
    for resolution in RESOLUTIONS:
        images, _, _ = load_split(resolution, "train")
        mean, std = channel_stats(images)
        check_mean, check_std = _normalized_stats(images, mean, std)
        if not (
            np.allclose(check_mean, 0.0, atol=SANITY_TOLERANCE)
            and np.allclose(check_std, 1.0, atol=SANITY_TOLERANCE)
        ):
            raise RuntimeError(
                f"Sanity check failed at {resolution} px: normalized mean "
                f"{_format(check_mean)}, std {_format(check_std)}."
            )
        stats[str(resolution)] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "n_images": len(images),
            "split": "train",
        }
        print(f"{resolution} px")
        print(f"  n_images:        {len(images)}")
        print(f"  mean:            {_format(mean)}")
        print(f"  std:             {_format(std)}")
        print(f"  normalized mean: {_format(check_mean)}")
        print(f"  normalized std:  {_format(check_std)}")

    STATS_PATH.parent.mkdir(exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {STATS_PATH}")


if __name__ == "__main__":
    main()
