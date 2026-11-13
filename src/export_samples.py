"""Export a random sample of BloodMNIST training images as PNG files.

The same image indices are used at 28, 64 and 128 px so the same cells can be
compared across resolutions. Run with: python src/export_samples.py
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from medmnist import INFO
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
SAMPLES_DIR = REPO_ROOT / "data" / "samples"

RESOLUTIONS = {
    28: RAW_DIR / "bloodmnist.npz",
    64: RAW_DIR / "bloodmnist_64.npz",
    128: RAW_DIR / "bloodmnist_128.npz",
}

CLASS_NAMES = {int(k): v for k, v in INFO["bloodmnist"]["label"].items()}

GRID_PADDING = 4
GRID_BACKGROUND = (255, 255, 255)
GRID_TEXT_COLOR = (0, 0, 0)


def folder_name(class_index: int) -> str:
    return f"{class_index}_{CLASS_NAMES[class_index].replace(' ', '_')}"


def grid_label(class_index: int) -> str:
    # Class 3 has a long parenthetical; keep only the main name in the grid.
    return f"{class_index} {CLASS_NAMES[class_index].split('(')[0].strip()}"


def select_indices(labels: np.ndarray, per_class: int, seed: int) -> dict[int, np.ndarray]:
    """Pick `per_class` random train indices for each class, sorted for readability."""
    rng = np.random.default_rng(seed)
    selection = {}
    for class_index in sorted(CLASS_NAMES):
        candidates = np.flatnonzero(labels == class_index)
        if len(candidates) < per_class:
            raise ValueError(
                f"Class {class_index} has only {len(candidates)} train images, "
                f"fewer than --per-class {per_class}."
            )
        selection[class_index] = np.sort(rng.choice(candidates, size=per_class, replace=False))
    return selection


def load_train(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["train_images"], data["train_labels"].ravel()


def save_samples(images: np.ndarray, selection: dict[int, np.ndarray], out_dir: Path) -> list[Path]:
    # Start clean so files from an earlier seed or sample size do not linger.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    written = []
    for class_index, indices in selection.items():
        class_dir = out_dir / folder_name(class_index)
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx in indices:
            path = class_dir / f"train_{idx}.png"
            Image.fromarray(images[idx]).save(path)
            written.append(path)
    return written


def save_grid(images: np.ndarray, selection: dict[int, np.ndarray], res: int, path: Path) -> None:
    """One row per class (label on the left), one column per sample."""
    font = ImageFont.load_default(size=max(10, min(18, res // 3)))
    labels = {c: grid_label(c) for c in selection}
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_width = max(measure.textlength(text, font=font) for text in labels.values())
    label_width = int(label_width) + 2 * GRID_PADDING

    n_cols = max(len(indices) for indices in selection.values())
    cell = res + GRID_PADDING
    width = label_width + n_cols * cell + GRID_PADDING
    height = len(selection) * cell + GRID_PADDING

    grid = Image.new("RGB", (width, height), GRID_BACKGROUND)
    draw = ImageDraw.Draw(grid)
    for row, (class_index, indices) in enumerate(selection.items()):
        top = GRID_PADDING + row * cell
        draw.text((GRID_PADDING, top + res / 2), labels[class_index],
                  fill=GRID_TEXT_COLOR, font=font, anchor="lm")
        for col, idx in enumerate(indices):
            grid.paste(Image.fromarray(images[idx]), (label_width + col * cell, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-class", type=int, default=10)
    args = parser.parse_args()

    reference_labels = None
    selection = None
    for res, npz_path in RESOLUTIONS.items():
        images, labels = load_train(npz_path)

        # Indices are chosen once and reused, which requires identical label order.
        if selection is None:
            reference_labels = labels
            selection = select_indices(labels, args.per_class, args.seed)
        elif not np.array_equal(labels, reference_labels):
            raise ValueError(f"Train labels in {npz_path.name} differ from {RESOLUTIONS[28].name}.")

        written = save_samples(images, selection, SAMPLES_DIR / f"{res}px")
        grid_path = SAMPLES_DIR / f"grid_{res}px.png"
        save_grid(images, selection, res, grid_path)

        total_bytes = sum(p.stat().st_size for p in written)
        print(f"{res} px ({npz_path.name})")
        print(f"  train_images: shape {images.shape}, dtype {images.dtype}")
        print(f"  PNGs written: {len(written)}")
        print(f"  total size:   {total_bytes} bytes ({total_bytes / 1024:.2f} KB)")
        print(f"  average size: {total_bytes / len(written):.0f} bytes")
        print(f"  grid:         {grid_path.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
