"""Tune a CNN with W&B random sweeps, training on train and scoring on val after every epoch.

The test split is never loaded. Each try keeps its best epoch by val macro F1, with val
accuracy as the tie-breaker, and is logged to W&B and appended to
results/tuning_<Model>_bloodmnist<res>.csv as soon as it finishes. Each sweep also writes
config.json, log.txt and one history_try<NN>.csv per try to its own folder,
runs/BloodCellClassification_Tuning_<Model>_bloodmnist<res>_<YYYYMMDD>_<HHMMSS>/.
With --count below the config budget, "_test" is added to the sweep, CSV and run folder names.
No weights are saved.

Run with: python src/tune_cnn.py --model {SmallCNN,ResNet18} [--resolution {28,64,128}] [--count N]
"""

import argparse
import csv
import gc
import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import wandb
import yaml
from torch import nn

from cnn import (
    MODELS,
    DeviceBatches,
    EpochRecord,
    build_model,
    fit,
    normalization_stats,
    set_seed,
    write_history,
)
from data import REPO_ROOT, RESOLUTIONS, load_split
from tune_classical import (
    METRIC,
    RESULTS_DIR,
    RUNS_DIR,
    TIMESTAMP_FORMAT,
    TUNING_SPLITS,
    Config,
    append_row,
    best_row,
    check_settings,
    close_logger,
    format_settings,
    load_wandb_env,
    make_logger,
    read_rows,
    tuning_csv,
    tuning_stem,
)

CONFIG_PATHS = {
    "SmallCNN": REPO_ROOT / "configs" / "cnn_smallcnn.yaml",
    "ResNet18": REPO_ROOT / "configs" / "cnn_resnet18.yaml",
}
CONFIG_KEYS = ("model", "resolutions", "splits", "epochs", "tuning_seed", "budget", "sweep")
# Settings used by fit() and the batches; every other sampled setting goes to build_model.
TRAINING_SETTINGS = ("lr", "weight_decay", "batch_size")
EPOCH_KEYS = ("epoch", "lr", "train_loss", "train_accuracy", "val_loss", "val_accuracy", METRIC)
RESULT_COLUMNS = ("best_epoch", METRIC, "val_accuracy", "train_time_s", "val_time_s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--resolution",
        type=int,
        choices=RESOLUTIONS,
        help="resolution to tune (default: every resolution in the config)",
    )
    parser.add_argument(
        "--count",
        type=int,
        metavar="N",
        help="number of tries per sweep for a short test run (default: the config budget)",
    )
    args = parser.parse_args()
    if args.count is not None and args.count < 1:
        parser.error("--count must be a positive integer.")
    return args


def load_config(path: Path, model: str) -> Config:
    """Read a CNN tuning config and check it against the model, resolutions and splits."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} must hold a YAML mapping.")
    missing = [key for key in CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path} is missing keys {missing}.")
    if config["model"] != model:
        raise ValueError(f"{path} is for {config['model']!r}, not {model!r}.")
    unknown = [r for r in config["resolutions"] if r not in RESOLUTIONS]
    if unknown:
        raise ValueError(f"Unknown resolutions {unknown} in {path}; expected {RESOLUTIONS}.")
    if config["splits"] != TUNING_SPLITS:
        raise ValueError(f"splits in {path} must be {TUNING_SPLITS}; tuning never uses test.")
    for key in ("epochs", "budget"):
        value = config[key]
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} in {path} must be a positive integer, got {value!r}.")
    metric = (config["sweep"].get("metric") or {}).get("name")
    if metric != METRIC:
        raise ValueError(f"The sweep metric in {path} must be {METRIC!r}, got {metric!r}.")
    parameters = config["sweep"].get("parameters") or {}
    absent = [name for name in TRAINING_SETTINGS if name not in parameters]
    if absent:
        raise ValueError(f"The sweep in {path} does not define {absent}.")
    return config


def write_run_config(
    path: Path,
    config: Config,
    resolution: int,
    sweep_name: str,
    sweep_id: str,
    count: int,
    device: torch.device,
) -> None:
    """Write the config, sweep identity, library versions and GPU of one sweep as JSON."""
    record = {
        "config": config,
        "resolution": resolution,
        "sweep_name": sweep_name,
        "sweep_id": sweep_id,
        "count": count,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "wandb": wandb.__version__,
        },
        "gpu": torch.cuda.get_device_name(device),
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def csv_columns(param_names: list[str]) -> list[str]:
    return ["timestamp", "try", "run_id", *param_names, *RESULT_COLUMNS]


def csv_row(
    number: int,
    run_id: str,
    settings: dict[str, Any],
    best: EpochRecord,
    train_time: float,
    val_time: float,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().strftime(TIMESTAMP_FORMAT),
        "try": number,
        "run_id": run_id,
        **settings,
        "best_epoch": best.epoch,
        METRIC: round(best.val_macro_f1, 4),
        "val_accuracy": round(best.val_accuracy, 4),
        "train_time_s": round(train_time, 3),
        "val_time_s": round(val_time, 3),
    }


def format_epoch(record: EpochRecord) -> str:
    return (
        f"epoch {record.epoch:03d} | lr {record.lr:.2e} | "
        f"train loss {record.train_loss:.4f} acc {record.train_accuracy:.4f} | "
        f"val loss {record.val_loss:.4f} acc {record.val_accuracy:.4f} "
        f"F1 {record.val_macro_f1:.4f} | train {record.train_time_s:.2f} s"
    )


@dataclass
class Sweep:
    """State shared by the tries of one sweep; run_try is the W&B agent function."""

    model: str
    resolution: int
    name: str
    seed: int
    epochs: int
    count: int
    parameters: Config
    train: tuple[np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray]
    mean: torch.Tensor
    std: torch.Tensor
    device: torch.device
    run_dir: Path
    csv_path: Path
    columns: list[str]
    logger: logging.Logger
    started: int = 0

    def run_try(self) -> None:
        """Train and validate one sampled setting; log it to W&B, the log file and the CSVs."""
        self.started += 1
        number = self.started
        run_config = {
            "model": self.model,
            "resolution": self.resolution,
            "epochs": self.epochs,
            "tuning_seed": self.seed,
        }
        try:
            with wandb.init(group=self.name, job_type="tuning", config=run_config) as run:
                settings = {name: run.config[name] for name in self.parameters}
                check_settings(settings, self.parameters)
                self.logger.info(f"try {number:02d} | {format_settings(settings)}")
                history, best_epoch = self.train_model(number, settings, run)

                best = history[best_epoch - 1]
                train_time = sum(record.train_time_s for record in history)
                val_time = sum(record.val_time_s for record in history)
                # The sweep reads val_macro_f1 from the summary, so it must hold the best epoch.
                run.summary.update({
                    "best_epoch": best_epoch,
                    **{key: getattr(best, key) for key in EPOCH_KEYS},
                    "train_time_s": train_time,
                    "val_time_s": val_time,
                })
                write_history(history, self.run_dir / f"history_try{number:02d}.csv")
                row = csv_row(number, run.id, settings, best, train_time, val_time)
                append_row(self.csv_path, self.columns, row)
        except Exception as exc:
            self.logger.error(f"try {number:02d} | failed: {type(exc).__name__}: {exc}")
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        self.logger.info(
            f"try {number:02d} | best epoch {best_epoch:03d} | "
            f"val macro F1 {best.val_macro_f1:.4f} | val accuracy {best.val_accuracy:.4f} | "
            f"train {train_time:.1f} s"
        )

    def train_model(
        self, number: int, settings: dict[str, Any], run: wandb.Run
    ) -> tuple[list[EpochRecord], int]:
        """Train one model; its weights and batches are freed when this returns."""
        set_seed(self.seed)
        model_params = {k: v for k, v in settings.items() if k not in TRAINING_SETTINGS}
        model = build_model(self.model, **model_params).to(self.device)
        generator = torch.Generator().manual_seed(self.seed)
        batch_size = settings["batch_size"]
        train_batches = DeviceBatches(*self.train, batch_size, self.device, generator)
        val_batches = DeviceBatches(*self.val, batch_size, self.device)

        def on_epoch(record: EpochRecord) -> None:
            self.logger.info(f"try {number:02d} | {format_epoch(record)}")
            run.log({key: getattr(record, key) for key in EPOCH_KEYS})

        result = fit(
            model,
            train_batches,
            val_batches,
            nn.CrossEntropyLoss(),
            self.device,
            self.mean,
            self.std,
            epochs=self.epochs,
            lr=settings["lr"],
            weight_decay=settings["weight_decay"],
            on_epoch=on_epoch,
        )
        return result.history, result.best_epoch


def run_sweep(
    config: Config, resolution: int, count: int, entity: str, project: str, device: torch.device
) -> None:
    """Create and run one W&B sweep for one resolution, then log the best try from the CSV."""
    model = config["model"]
    stem = tuning_stem(model, resolution, count < config["budget"])
    sweep_name = f"tune_{stem}"
    csv_path = tuning_csv(stem)
    train_images, train_labels, _ = load_split(resolution, TUNING_SPLITS["fit"])
    val_images, val_labels, _ = load_split(resolution, TUNING_SPLITS["score"])
    mean, std = normalization_stats(model, resolution)

    started = datetime.now()
    sweep_id = wandb.sweep({**config["sweep"], "name": sweep_name}, entity=entity, project=project)
    run_dir = RUNS_DIR / f"BloodCellClassification_Tuning_{stem}_{started:%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    write_run_config(
        run_dir / "config.json", config, resolution, sweep_name, sweep_id, count, device
    )

    param_names = list(config["sweep"]["parameters"])
    columns = csv_columns(param_names)
    RESULTS_DIR.mkdir(exist_ok=True)
    with csv_path.open("x", newline="") as file:
        csv.DictWriter(file, fieldnames=columns).writeheader()

    log_path = run_dir / "log.txt"
    logger = make_logger(
        f"tune_cnn.{model}.{resolution}", f"{model} | bloodmnist{resolution}", log_path
    )
    try:
        logger.info(
            f"sweep {sweep_name} started (id {sweep_id}), budget {config['budget']}, "
            f"tries {count}, epochs {config['epochs']}"
        )
        sweep = Sweep(
            model=model,
            resolution=resolution,
            name=sweep_name,
            seed=config["tuning_seed"],
            epochs=config["epochs"],
            count=count,
            parameters=config["sweep"]["parameters"],
            train=(train_images, train_labels),
            val=(val_images, val_labels),
            mean=mean,
            std=std,
            device=device,
            run_dir=run_dir,
            csv_path=csv_path,
            columns=columns,
            logger=logger,
        )
        wandb.agent(sweep_id, function=sweep.run_try, entity=entity, project=project, count=count)

        rows = read_rows(csv_path)
        if rows:
            best = best_row(rows)
            settings = ", ".join(f"{name}={best[name]}" for name in param_names)
            logger.info(
                f"best try {int(best['try']):02d} | run {best['run_id']} | {settings} | "
                f"best epoch {best['best_epoch']} | val macro F1 {best[METRIC]} | "
                f"val accuracy {best['val_accuracy']}"
            )
        if len(rows) < count:
            raise RuntimeError(f"Only {len(rows)} of {count} tries finished; see {log_path}.")
    finally:
        close_logger(logger)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; tuning needs a CUDA build of PyTorch.")
    device = torch.device("cuda")
    config = load_config(CONFIG_PATHS[args.model], args.model)
    budget = config["budget"]
    count = budget if args.count is None else args.count
    if count > budget:
        raise ValueError(f"--count {count} exceeds the config budget of {budget}.")
    if args.resolution is None:
        resolutions = sorted(set(config["resolutions"]))
    elif args.resolution in config["resolutions"]:
        resolutions = [args.resolution]
    else:
        raise ValueError(
            f"{args.model} is tuned at {config['resolutions']} px, not {args.resolution} px."
        )

    for resolution in resolutions:
        csv_path = tuning_csv(tuning_stem(args.model, resolution, count < budget))
        if csv_path.exists():
            raise FileExistsError(
                f"{csv_path} already exists; move or delete it before rerunning this sweep."
            )

    entity, project = load_wandb_env()
    print(f"W&B entity: {entity}, project: {project}")
    # wandb.agent adds its own DEBUG console handler, which WANDB_SILENT does not cover.
    logging.getLogger("wandb.wandb_agent").addFilter(lambda record: record.levelno >= logging.ERROR)
    for resolution in resolutions:
        run_sweep(config, resolution, count, entity, project, device)


if __name__ == "__main__":
    main()
