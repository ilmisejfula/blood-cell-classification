"""Test metrics and plots for the final runs.

Plots are drawn on matplotlib Figure objects directly, so no GUI backend is needed.
"""

import math
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from data import CLASS_SHORT_NAMES

N_CLASSES = len(CLASS_SHORT_NAMES)
CLASS_KEYS = [name.replace(" ", "_") for name in CLASS_SHORT_NAMES]
RECALL_KEYS = [f"recall_{key}" for key in CLASS_KEYS]
METRIC_NAMES = ("accuracy", "macro_f1", *RECALL_KEYS)
GRID_COLUMNS = 6
DPI = 150


def compute_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return accuracy, macro F1 and per-class recall (keys recall_<class>)."""
    recalls = recall_score(labels, predictions, labels=np.arange(N_CLASSES), average=None)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        **{key: float(recall) for key, recall in zip(RECALL_KEYS, recalls)},
    }


def draw_confusion_matrix(ax: Axes, labels: np.ndarray, predictions: np.ndarray) -> None:
    """Draw a confusion matrix on ax: counts in the cells, colour by row-normalized values."""
    counts = confusion_matrix(labels, predictions, labels=np.arange(N_CLASSES))
    totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)

    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.figure.colorbar(image, ax=ax, label="fraction of true class")
    ticks = np.arange(N_CLASSES)
    ax.set_xticks(ticks, CLASS_SHORT_NAMES, rotation=45, ha="right")
    ax.set_yticks(ticks, CLASS_SHORT_NAMES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    for row, column in np.ndindex(counts.shape):
        ax.text(
            column,
            row,
            str(counts[row, column]),
            ha="center",
            va="center",
            color="white" if normalized[row, column] > 0.5 else "black",
        )


def save_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, path: Path) -> None:
    """Save a confusion matrix PNG drawn by draw_confusion_matrix."""
    figure = Figure(figsize=(8, 7), layout="constrained")
    draw_confusion_matrix(figure.subplots(), labels, predictions)
    figure.savefig(path, dpi=DPI)


def save_misclassified_grid(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    indices: np.ndarray,
    path: Path,
    max_images: int = 36,
) -> None:
    """Save a PNG grid of wrongly predicted images in dataset index order.

    Each image is titled with its dataset index, true class and predicted class.
    """
    wrong = np.flatnonzero(labels != predictions)
    wrong = wrong[np.argsort(indices[wrong], kind="stable")]
    shown = wrong[:max_images]

    columns = min(GRID_COLUMNS, max(len(shown), 1))
    rows = max(math.ceil(len(shown) / columns), 1)
    figure = Figure(figsize=(columns * 2.2, rows * 2.6 + 0.4), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, shown):
        ax.imshow(images[i], interpolation="nearest")
        ax.set_title(
            f"index {indices[i]}\ntrue: {CLASS_SHORT_NAMES[labels[i]]}\n"
            f"predicted: {CLASS_SHORT_NAMES[predictions[i]]}",
            fontsize=7,
        )
    if not len(shown):
        axes[0].text(0.5, 0.5, "No misclassified images", ha="center", va="center")
    figure.suptitle(f"Misclassified images: {len(shown)} of {len(wrong)}, in index order")
    figure.savefig(path, dpi=DPI)
