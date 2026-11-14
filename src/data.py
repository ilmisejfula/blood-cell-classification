"""Load BloodMNIST splits at any supported resolution.

Images are returned as stored. By default, val and test images that exactly
duplicate an earlier split are dropped, and each split comes with the original
indices of the images it keeps. Preprocessing belongs elsewhere and must be
fitted on the training split only.
"""

import hashlib
from functools import cache
from pathlib import Path

import numpy as np
from medmnist import INFO

REPO_ROOT = Path(__file__).resolve().parent.parent  # parent of src/
RAW_DIR = REPO_ROOT / "data" / "raw"

RESOLUTIONS = (28, 64, 128)
SPLITS = ("train", "val", "test")

FILE_NAMES = {
    28: "bloodmnist.npz",
    64: "bloodmnist_64.npz",
    128: "bloodmnist_128.npz",
}

_LABELS = INFO["bloodmnist"]["label"]
CLASS_NAMES = [_LABELS[str(i)] for i in range(len(_LABELS))]
# Display-only names for plots and tables.
CLASS_SHORT_NAMES = [*CLASS_NAMES[:3], "immature granulocyte", *CLASS_NAMES[4:]]

Split = tuple[np.ndarray, np.ndarray, np.ndarray]


def _npz_path(resolution: int) -> Path:
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Unknown resolution {resolution!r}; expected one of {RESOLUTIONS}.")
    path = RAW_DIR / FILE_NAMES[resolution]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing data file {path}. Run python src/download_data.py first."
        )
    return path


def _image_hashes(images: np.ndarray) -> list[str]:
    return [hashlib.md5(image.tobytes()).hexdigest() for image in images]


def _matching_indices(hashes: list[str], reference: set[str]) -> np.ndarray:
    indices = np.array([i for i, h in enumerate(hashes) if h in reference], dtype=np.int64)
    indices.flags.writeable = False  # shared by the cache
    return indices


@cache
def duplicate_indices() -> dict[str, np.ndarray]:
    """Return sorted original indices of val and test images to drop.

    val: images identical to a train image. test: images identical to a train
    or val image. Computed once on the 28 px file; the splits are aligned
    across resolutions.
    """
    with np.load(_npz_path(28)) as npz:
        hashes = {split: _image_hashes(npz[f"{split}_images"]) for split in SPLITS}
    train, val = set(hashes["train"]), set(hashes["val"])
    return {
        "val": _matching_indices(hashes["val"], train),
        "test": _matching_indices(hashes["test"], train | val),
    }


def _read_split(npz: np.lib.npyio.NpzFile, split: str, deduplicate: bool) -> Split:
    images = npz[f"{split}_images"]
    labels = npz[f"{split}_labels"].reshape(-1).astype(np.int64)
    indices = np.arange(len(labels), dtype=np.int64)
    if deduplicate and split != "train":
        keep = np.ones(len(labels), dtype=bool)
        keep[duplicate_indices()[split]] = False
        images, labels, indices = images[keep], labels[keep], indices[keep]
    return images, labels, indices


def load_split(resolution: int, split: str, deduplicate: bool = True) -> Split:
    """Return (images, labels, indices) for one split.

    images: uint8 array of shape (N, H, W, 3); labels: int64 array of shape (N,);
    indices: int64 array of shape (N,) with each image's position in the official split.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")
    with np.load(_npz_path(resolution)) as npz:
        return _read_split(npz, split, deduplicate)


def load_all(resolution: int, deduplicate: bool = True) -> dict[str, Split]:
    """Return {"train", "val", "test"} -> (images, labels, indices), reading the file once."""
    with np.load(_npz_path(resolution)) as npz:
        return {split: _read_split(npz, split, deduplicate) for split in SPLITS}
