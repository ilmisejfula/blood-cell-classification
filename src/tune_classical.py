"""Tune a classical model with W&B random sweeps, fitting on train and scoring on val.

The test split is never loaded. Each try is logged to W&B and appended to
results/tuning_<Model>_bloodmnist<res>.csv as soon as it finishes. Each sweep also writes
config.json and log.txt to its own folder,
runs/BloodCellClassification_Tuning_<Model>_bloodmnist<res>_<YYYYMMDD>_<HHMMSS>/.
With --count below the config budget, "_test" is added to the sweep, CSV and run folder names.

Run with: python src/tune_classical.py --config CONFIG [--resolution {28,64,128}] [--count N]
"""

import argparse
import csv
import json
import logging
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import wandb
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline

from classical import LR_MAX_ITER, MODELS, build_pipeline, load_features
from data import CLASS_SHORT_NAMES, REPO_ROOT, RESOLUTIONS, Split
from extract_features import features_path

ENV_PATH = REPO_ROOT / ".env"
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = REPO_ROOT / "runs"
WANDB_VARIABLES = ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY")
CONFIG_KEYS = ("model", "resolutions", "features", "splits", "tuning_seed", "budget", "sweep")
TUNING_SPLITS = {"fit": "train", "score": "val"}
METRIC = "val_macro_f1"
TEST_SUFFIX = "_test"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
RECALL_KEYS = [f"val_recall_{name.replace(' ', '_')}" for name in CLASS_SHORT_NAMES]
SCORE_COLUMNS = ("train_accuracy", "val_accuracy", "val_macro_f1", *RECALL_KEYS)
TIME_COLUMNS = ("fit_seconds", "val_predict_ms_per_image")
LR_COLUMNS = ("n_iter", "converged")
DISTRIBUTION_TYPES = {"log_uniform_values": float, "uniform": float, "int_uniform": int}

Config = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="tuning config (YAML)")
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


def load_config(path: Path) -> Config:
    """Read a tuning config and check it against the models, resolutions and splits in code."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} must hold a YAML mapping.")
    missing = [key for key in CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path} is missing keys {missing}.")
    if config["model"] not in MODELS:
        raise ValueError(f"Unknown model {config['model']!r} in {path}; expected one of {MODELS}.")
    unknown = [r for r in config["resolutions"] if r not in RESOLUTIONS]
    if unknown:
        raise ValueError(f"Unknown resolutions {unknown} in {path}; expected {RESOLUTIONS}.")
    if config["splits"] != TUNING_SPLITS:
        raise ValueError(f"splits in {path} must be {TUNING_SPLITS}; tuning never uses test.")
    budget = config["budget"]
    if not isinstance(budget, int) or budget < 1:
        raise ValueError(f"budget in {path} must be a positive integer, got {budget!r}.")
    metric = (config["sweep"].get("metric") or {}).get("name")
    if metric != METRIC:
        raise ValueError(f"The sweep metric in {path} must be {METRIC!r}, got {metric!r}.")
    if not config["sweep"].get("parameters"):
        raise ValueError(f"The sweep in {path} defines no parameters.")
    return config


def check_features_path(config: Config, resolution: int) -> None:
    """Raise if the config's features entry does not resolve to features_path(resolution)."""
    configured = (REPO_ROOT / config["features"].format(resolution=resolution)).resolve()
    expected = features_path(resolution).resolve()
    if configured != expected:
        raise ValueError(
            f"features in the config resolves to {configured} at {resolution} px, "
            f"but the feature cache is {expected}."
        )


def load_wandb_env() -> tuple[str, str]:
    """Load the W&B settings from .env into this process only; return (entity, project).

    Also makes W&B print errors only and keep its run files in the git-ignored wandb/.
    Call it before any other W&B call.
    """
    load_dotenv(ENV_PATH, override=True)
    missing = [name for name in WANDB_VARIABLES if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            f"Set them in {ENV_PATH} (see .env.example)."
        )
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_DIR"] = str(REPO_ROOT)
    return os.environ["WANDB_ENTITY"], os.environ["WANDB_PROJECT"]


def tuning_stem(model: str, resolution: int, test: bool) -> str:
    """Return <Model>_bloodmnist<res>, plus the test suffix for short test runs."""
    return f"{model}_bloodmnist{resolution}{TEST_SUFFIX if test else ''}"


def tuning_csv(stem: str) -> Path:
    return RESULTS_DIR / f"tuning_{stem}.csv"


def csv_columns(model: str, param_names: list[str]) -> list[str]:
    columns = ["timestamp", "run_id", *param_names, *SCORE_COLUMNS, *TIME_COLUMNS]
    if model == "LogisticRegression":
        columns += LR_COLUMNS
    return columns


def make_logger(name: str, prefix: str, path: Path) -> logging.Logger:
    """Return a logger writing "<time> | <prefix> | <message>" lines to path and the console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(f"%(asctime)s | {prefix} | %(message)s", datefmt=TIMESTAMP_FORMAT)
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def write_run_config(
    path: Path, config: Config, resolution: int, sweep_name: str, sweep_id: str, count: int
) -> None:
    """Write the config, sweep identity and library versions of one sweep as JSON."""
    record = {
        "config": config,
        "resolution": resolution,
        "sweep_name": sweep_name,
        "sweep_id": sweep_id,
        "count": count,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "wandb": wandb.__version__,
        },
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def check_settings(settings: dict[str, Any], parameters: Config) -> None:
    """Raise if a sampled value is not a listed value of the same type, or has the wrong type.

    W&B can return a listed 1.0 as the int 1, which sklearn reads as a different setting.
    """
    for name, value in settings.items():
        spec = parameters[name]
        if "values" in spec:
            valid = any(type(value) is type(v) and value == v for v in spec["values"])
            expected = f"one of {spec['values']} with the same type"
        else:
            expected_type = DISTRIBUTION_TYPES.get(spec.get("distribution"))
            if expected_type is None:
                raise ValueError(f"No type check defined for the distribution of {name}: {spec}.")
            valid = type(value) is expected_type
            expected = f"of type {expected_type.__name__}"
        if not valid:
            raise TypeError(f"Sampled {name}={value!r} ({type(value).__name__}) is not {expected}.")


def format_settings(settings: dict[str, Any]) -> str:
    return ", ".join(
        f"{name}={value:.4g}" if isinstance(value, float) else f"{name}={value}"
        for name, value in settings.items()
    )


def evaluate(pipeline: Pipeline, train: Split, val: Split) -> tuple[dict[str, Any], np.ndarray]:
    """Fit on train, score train and val; return the metrics and the val predictions."""
    train_features, train_labels, _ = train
    val_features, val_labels, _ = val
    start = time.perf_counter()
    pipeline.fit(train_features, train_labels)
    fit_seconds = time.perf_counter() - start
    train_predictions = pipeline.predict(train_features)
    start = time.perf_counter()
    val_predictions = pipeline.predict(val_features)
    predict_seconds = time.perf_counter() - start
    recalls = recall_score(
        val_labels, val_predictions, labels=np.arange(len(CLASS_SHORT_NAMES)), average=None
    )
    metrics = {
        "train_accuracy": float(accuracy_score(train_labels, train_predictions)),
        "train_macro_f1": float(f1_score(train_labels, train_predictions, average="macro")),
        "val_accuracy": float(accuracy_score(val_labels, val_predictions)),
        "val_macro_f1": float(f1_score(val_labels, val_predictions, average="macro")),
        **{key: float(recall) for key, recall in zip(RECALL_KEYS, recalls)},
        "fit_seconds": fit_seconds,
        "val_predict_ms_per_image": predict_seconds / len(val_labels) * 1000,
    }
    return metrics, val_predictions


def csv_row(run_id: str, settings: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    row = {"timestamp": datetime.now().strftime(TIMESTAMP_FORMAT), "run_id": run_id}
    row |= {name: "None" if value is None else value for name, value in settings.items()}
    row |= {key: round(metrics[key], 4) for key in SCORE_COLUMNS}
    row |= {key: round(metrics[key], 3) for key in TIME_COLUMNS}
    if "n_iter" in metrics:
        row |= {"n_iter": metrics["n_iter"], "converged": not metrics["reached_max_iter"]}
    return row


def append_row(path: Path, columns: list[str], row: dict[str, Any]) -> None:
    with path.open("a", newline="") as file:
        csv.DictWriter(file, fieldnames=columns).writerow(row)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    """Return the tuning try with the highest val macro F1; val accuracy breaks ties."""
    return max(rows, key=lambda row: (float(row[METRIC]), float(row["val_accuracy"])))


@dataclass
class Sweep:
    """State shared by the tries of one sweep; run_try is the W&B agent function."""

    model: str
    resolution: int
    name: str
    seed: int
    count: int
    parameters: Config
    train: Split
    val: Split
    csv_path: Path
    columns: list[str]
    logger: logging.Logger
    started: int = 0

    def run_try(self) -> None:
        """Fit and score one sampled setting; log it to W&B, the log file and the CSV."""
        self.started += 1
        number = self.started
        run_config = {"model": self.model, "resolution": self.resolution, "tuning_seed": self.seed}
        try:
            with wandb.init(group=self.name, job_type="tuning", config=run_config) as run:
                settings = {name: run.config[name] for name in self.parameters}
                check_settings(settings, self.parameters)
                pipeline = build_pipeline(self.model, settings, self.seed)
                metrics, val_predictions = evaluate(pipeline, self.train, self.val)
                if self.model == "LogisticRegression":
                    n_iter = int(pipeline.named_steps["model"].n_iter_.max())
                    metrics |= {"n_iter": n_iter, "reached_max_iter": n_iter >= LR_MAX_ITER}
                run.log({
                    **metrics,
                    "val_confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=self.val[1].tolist(),
                        preds=val_predictions.tolist(),
                        class_names=list(CLASS_SHORT_NAMES),
                    ),
                })
                append_row(self.csv_path, self.columns, csv_row(run.id, settings, metrics))
        except Exception as exc:
            self.logger.error(f"try {number}/{self.count} failed: {type(exc).__name__}: {exc}")
            raise
        message = (
            f"try {number}/{self.count}: {format_settings(settings)}, "
            f"fit {metrics['fit_seconds']:.3f} s, "
            f"train accuracy {metrics['train_accuracy']:.4f}, "
            f"val accuracy {metrics['val_accuracy']:.4f}, "
            f"val macro F1 {metrics['val_macro_f1']:.4f}"
        )
        if "n_iter" in metrics:
            message += f", iterations {metrics['n_iter']}"
        self.logger.info(message)


def run_sweep(config: Config, resolution: int, count: int, entity: str, project: str) -> None:
    """Create and run one W&B sweep for one resolution, then log the best try from the CSV."""
    model = config["model"]
    stem = tuning_stem(model, resolution, count < config["budget"])
    sweep_name = f"tune_{stem}"
    csv_path = tuning_csv(stem)
    data, _ = load_features(resolution, (TUNING_SPLITS["fit"], TUNING_SPLITS["score"]))

    started = datetime.now()
    sweep_id = wandb.sweep({**config["sweep"], "name": sweep_name}, entity=entity, project=project)
    run_dir = RUNS_DIR / f"BloodCellClassification_Tuning_{stem}_{started:%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    write_run_config(run_dir / "config.json", config, resolution, sweep_name, sweep_id, count)

    param_names = list(config["sweep"]["parameters"])
    columns = csv_columns(model, param_names)
    RESULTS_DIR.mkdir(exist_ok=True)
    with csv_path.open("x", newline="") as file:
        csv.DictWriter(file, fieldnames=columns).writeheader()

    log_path = run_dir / "log.txt"
    logger = make_logger(
        f"tune_classical.{model}.{resolution}", f"{model} | bloodmnist{resolution}", log_path
    )
    try:
        logger.info(
            f"sweep {sweep_name} started (id {sweep_id}), budget {config['budget']}, "
            f"tries {count}"
        )
        sweep = Sweep(
            model=model,
            resolution=resolution,
            name=sweep_name,
            seed=config["tuning_seed"],
            count=count,
            parameters=config["sweep"]["parameters"],
            train=data[TUNING_SPLITS["fit"]],
            val=data[TUNING_SPLITS["score"]],
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
                f"best try: run {best['run_id']}, {settings}, "
                f"val macro F1 {best[METRIC]}, val accuracy {best['val_accuracy']}"
            )
        if len(rows) < count:
            raise RuntimeError(f"Only {len(rows)} of {count} tries finished; see {log_path}.")
    finally:
        close_logger(logger)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    budget = config["budget"]
    count = budget if args.count is None else args.count
    if count > budget:
        raise ValueError(f"--count {count} exceeds the config budget of {budget}.")
    if args.resolution is None:
        resolutions = sorted(set(config["resolutions"]))
    else:
        resolutions = [args.resolution]

    for resolution in resolutions:
        check_features_path(config, resolution)
        csv_path = tuning_csv(tuning_stem(config["model"], resolution, count < budget))
        if csv_path.exists():
            raise FileExistsError(
                f"{csv_path} already exists; move or delete it before rerunning this sweep."
            )

    entity, project = load_wandb_env()
    print(f"W&B entity: {entity}, project: {project}")
    # wandb.agent adds its own DEBUG console handler, which WANDB_SILENT does not cover.
    logging.getLogger("wandb.wandb_agent").addFilter(lambda record: record.levelno >= logging.ERROR)
    for resolution in resolutions:
        run_sweep(config, resolution, count, entity, project)


if __name__ == "__main__":
    main()
