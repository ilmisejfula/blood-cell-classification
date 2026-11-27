"""CNN models, data loading and training steps for BloodMNIST.

Each split is moved to the device once as uint8; batches are sliced from it and
scaled and normalized there.
The command-line check trains one model on the training split and scores it on the
validation split after every epoch. It never touches the test split and writes no
files unless --history is given.

Run with: python src/cnn.py --model {SmallCNN,ResNet18} --resolution {28,64,128}
          [--epochs N] [--lr LR] [--weight-decay WD] [--batch-size N] [--dropout P]
          [--history CSV]
"""

import argparse
import csv
import json
import random
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from data import CLASS_NAMES, RESOLUTIONS, load_split
from evaluate import compute_metrics
from preprocess import STATS_PATH, load_stats

MODELS = ("SmallCNN", "ResNet18")
NUM_CLASSES = len(CLASS_NAMES)
SMALL_CNN_CHANNELS = (32, 64, 128)
RESNET18_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1

SEED = 0
EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3


class SmallCNN(nn.Module):
    """Three conv blocks and global average pooling; parameters do not depend on image size."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in SMALL_CNN_CHANNELS:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def resnet18_classifier() -> nn.Module:
    """Return ImageNet-pretrained ResNet-18 with a new NUM_CLASSES output layer."""
    model = resnet18(weights=RESNET18_WEIGHTS)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_model(model: str, **params: Any) -> nn.Module:
    """Return an untrained model; params go to its constructor."""
    if model == "SmallCNN":
        return SmallCNN(**params)
    if model == "ResNet18":
        return resnet18_classifier(**params)
    raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}.")


def normalization_stats(model: str, resolution: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-channel (mean, std) as float32 tensors of shape (3,).

    SmallCNN uses the training-split statistics saved by preprocess.py; ResNet18 uses
    the statistics of its ImageNet weights.
    """
    if model == "SmallCNN":
        entry = json.loads(STATS_PATH.read_text(encoding="utf-8")).get(str(resolution), {})
        if entry.get("split") != "train":
            raise ValueError(
                f"Normalization stats for {resolution} px in {STATS_PATH} were not fitted "
                f"on the train split (split: {entry.get('split')!r})."
            )
        mean, std = load_stats(resolution)
    elif model == "ResNet18":
        transforms = RESNET18_WEIGHTS.transforms()
        mean, std = transforms.mean, transforms.std
    else:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}.")
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


class DeviceBatches:
    """Mini-batches of uint8 (N, H, W, 3) images and int64 labels held on the device.

    With a generator, each pass draws a new permutation from it; without one,
    batches come in stored order.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> None:
        self.images = torch.from_numpy(images).to(device)
        self.labels = torch.from_numpy(labels).to(device)
        self.batch_size = batch_size
        self.generator = generator

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        count = len(self.labels)
        if self.generator is None:
            for start in range(0, count, self.batch_size):
                stop = start + self.batch_size
                yield self.images[start:stop], self.labels[start:stop]
            return
        order = torch.randperm(count, generator=self.generator).to(self.labels.device)
        for start in range(0, count, self.batch_size):
            batch = order[start:start + self.batch_size]
            yield self.images[batch], self.labels[batch]


def prepare_batch(
    images: torch.Tensor, device: torch.device, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Return uint8 NHWC images as normalized float32 NCHW on the device."""
    x = images.permute(0, 3, 1, 2).float().div(255)
    mean = mean.to(device).view(1, -1, 1, 1)
    std = std.to(device).view(1, -1, 1, 1)
    return (x - mean) / std


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DeviceBatches,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[float, float]:
    """Train for one pass over loader and return (mean loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        x = prepare_batch(images, device, mean, std)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += len(labels)
    return total_loss / count, correct / count


class EvalResult(NamedTuple):
    loss: float
    accuracy: float
    macro_f1: float
    predictions: np.ndarray
    probabilities: np.ndarray


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DeviceBatches,
    criterion: nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> EvalResult:
    """Score model on loader; probabilities are softmax outputs of shape (N, NUM_CLASSES)."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probabilities = []
    for images, labels in loader:
        logits = model(prepare_batch(images, device, mean, std))
        total_loss += criterion(logits, labels).item() * len(labels)
        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(logits.softmax(dim=1).cpu().numpy())
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    predictions = probabilities.argmax(axis=1)
    metrics = compute_metrics(labels, predictions)
    return EvalResult(
        loss=total_loss / len(labels),
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        predictions=predictions,
        probabilities=probabilities,
    )


class EpochRecord(NamedTuple):
    epoch: int
    lr: float
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    train_time_s: float
    val_time_s: float


class FitResult(NamedTuple):
    history: list[EpochRecord]
    best_epoch: int
    best_state: dict[str, torch.Tensor]


def fit(
    model: nn.Module,
    train_batches: DeviceBatches,
    val_batches: DeviceBatches,
    criterion: nn.Module,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    on_epoch: Callable[[EpochRecord], None] | None = None,
) -> FitResult:
    """Train with AdamW and cosine annealing, validating after every epoch.

    The best epoch (1-based) has the highest val macro F1, then the highest val
    accuracy; exact ties keep the earlier epoch. best_state is a CPU copy of the
    model's state_dict at that epoch. on_epoch is called with each epoch's record.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[EpochRecord] = []
    best_key: tuple[float, float] | None = None
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        start = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model, train_batches, optimizer, criterion, device, mean, std
        )
        train_time = time.perf_counter() - start
        scheduler.step()

        start = time.perf_counter()
        result = evaluate(model, val_batches, criterion, device, mean, std)
        val_time = time.perf_counter() - start

        record = EpochRecord(
            epoch=epoch,
            lr=epoch_lr,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=result.loss,
            val_accuracy=result.accuracy,
            val_macro_f1=result.macro_f1,
            train_time_s=train_time,
            val_time_s=val_time,
        )
        history.append(record)
        key = (result.macro_f1, result.accuracy)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {name: t.detach().cpu().clone() for name, t in model.state_dict().items()}
        if on_epoch is not None:
            on_epoch(record)
    return FitResult(history, best_epoch, best_state)


def write_history(history: list[EpochRecord], path: Path) -> None:
    """Write one CSV row per epoch, with EpochRecord's fields as columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EpochRecord._fields)
        writer.writeheader()
        writer.writerows(record._asdict() for record in history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--resolution", type=int, choices=RESOLUTIONS, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"(default: {EPOCHS})")
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE, help=f"(default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=WEIGHT_DECAY, help=f"(default: {WEIGHT_DECAY})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help=f"(default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--dropout", type=float, default=DROPOUT, help=f"SmallCNN only (default: {DROPOUT})"
    )
    parser.add_argument("--history", type=Path, help="optional CSV path for the per-epoch history")
    return parser.parse_args()


def _print_epoch(record: EpochRecord) -> None:
    print(
        f"epoch {record.epoch:3d}  lr {record.lr:.2e}  "
        f"train loss {record.train_loss:.4f} acc {record.train_accuracy:.4f}  "
        f"val loss {record.val_loss:.4f} acc {record.val_accuracy:.4f} "
        f"F1 {record.val_macro_f1:.4f}  train {record.train_time_s:.1f} s"
    )


def _format(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{v:.6f}" for v in values.tolist()) + "]"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this check needs a CUDA build of PyTorch.")
    device = torch.device("cuda")
    set_seed(SEED)

    mean, std = normalization_stats(args.model, args.resolution)
    train_images, train_labels, _ = load_split(args.resolution, "train")
    val_images, val_labels, _ = load_split(args.resolution, "val")
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DeviceBatches(train_images, train_labels, args.batch_size, device, generator)
    val_loader = DeviceBatches(val_images, val_labels, args.batch_size, device)

    params = {"dropout": args.dropout} if args.model == "SmallCNN" else {}
    model = build_model(args.model, **params).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Shape check on a val batch, so the shuffled train loader's random state is untouched.
    images, _ = next(iter(val_loader))
    x = prepare_batch(images, device, mean, std)
    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"GPU:              {torch.cuda.get_device_name(device)}")
    print(f"model:            {args.model}")
    print(f"resolution:       {args.resolution}")
    print(f"mean:             {_format(mean)}")
    print(f"std:              {_format(std)}")
    print(f"parameters:       {total_params:,} ({trainable_params:,} trainable)")
    print(f"input batch:      {tuple(x.shape)}")
    print(f"output:           {tuple(output.shape)}")

    start = time.perf_counter()
    result = fit(
        model,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(),
        device,
        mean,
        std,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        on_epoch=_print_epoch,
    )
    seconds = time.perf_counter() - start

    best = result.history[result.best_epoch - 1]
    print(
        f"best epoch:       {best.epoch} (val macro F1 {best.val_macro_f1:.4f}, "
        f"val accuracy {best.val_accuracy:.4f})"
    )
    print(f"total time:       {seconds:.2f} s")
    if args.history is not None:
        write_history(result.history, args.history)
        print(f"Saved {args.history}")


if __name__ == "__main__":
    main()
