"""Final runs of the classical models: best tuned settings, several seeds, one test evaluation.

Settings come from the best try in results/tuning_<Model>_bloodmnist<res>.csv. Each run fits on
train, evaluates once on test and writes its files to
runs/BloodCellClassification_<Model>_bloodmnist<res>_<YYYYMMDD>_<HHMMSS>/. Every run is appended
to results/classical_runs.csv as it finishes; results/classical_summary.csv is written at the end.
With --test, "_test" is added to the run folder and results CSV names.

Run with: python src/run_classical.py [--model MODEL] [--resolution RES] [--seeds 0 1 2 3 4]
                                      [--wandb] [--test]
"""

import argparse
import csv
import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import wandb
from sklearn.pipeline import Pipeline

from classical import MODELS, build_pipeline, load_features
from data import CLASS_SHORT_NAMES, REPO_ROOT, RESOLUTIONS, Split, load_split
from evaluate import (
    CLASS_KEYS,
    METRIC_NAMES,
    compute_metrics,
    save_confusion_matrix,
    save_misclassified_grid,
)
from tune_classical import (
    METRIC,
    RESULTS_DIR,
    RUNS_DIR,
    TEST_SUFFIX,
    TIMESTAMP_FORMAT,
    append_row,
    best_row,
    close_logger,
    format_settings,
    load_config,
    load_wandb_env,
    make_logger,
    read_rows,
    tuning_csv,
    tuning_stem,
)

DATASET = "bloodmnist"
SEEDS = (0, 1, 2, 3, 4)
CONFIG_PATHS = {
    "LogisticRegression": REPO_ROOT / "configs" / "classical_logreg.yaml",
    "RandomForest": REPO_ROOT / "configs" / "classical_rf.yaml",
}
SETTING_TYPES: dict[str, Callable[[str], Any]] = {
    "C": float,
    "n_estimators": int,
    "min_samples_leaf": int,
    "max_depth": lambda value: None if value == "None" else int(value),
    "max_features": lambda value: (
        None if value == "None" else value if value in ("sqrt", "log2") else float(value)
    ),
}
METRIC_COLUMNS = [f"test_{name}" for name in METRIC_NAMES]
TIME_COLUMNS = ["fit_seconds", "predict_seconds", "predict_us_per_image"]
SUMMARY_VALUES = [*METRIC_COLUMNS, *TIME_COLUMNS, "model_bytes"]
RUN_COLUMNS = ["run_name", "timestamp", "model", "resolution", "seed", "settings", *SUMMARY_VALUES]
PREDICTION_COLUMNS = ["index", "true_class", "predicted_class", *(f"prob_{k}" for k in CLASS_KEYS)]
PROBABILITY_DIGITS = 6

WandbTarget = tuple[str, str]  # (entity, project)


@dataclass(frozen=True)
class Tuned:
    """The best tuning try of one model and resolution."""

    settings: dict[str, Any]
    csv_path: Path
    run_id: str
    val_macro_f1: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, help="model to run (default: all)")
    parser.add_argument(
        "--resolution", type=int, choices=RESOLUTIONS, help="resolution to run (default: all)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS), help="seeds (default: 0 1 2 3 4)"
    )
    parser.add_argument("--wandb", action="store_true", help="log each run to W&B")
    parser.add_argument(
        "--test", action="store_true", help=f'short test run: add "{TEST_SUFFIX}" to output names'
    )
    return parser.parse_args()


def results_csvs(test_run: bool) -> tuple[Path, Path]:
    """Return the paths of the per-run CSV and the summary CSV."""
    suffix = TEST_SUFFIX if test_run else ""
    return (
        RESULTS_DIR / f"classical_runs{suffix}.csv",
        RESULTS_DIR / f"classical_summary{suffix}.csv",
    )


def load_tuned(model: str, resolution: int) -> Tuned:
    """Return the best try of the full tuning CSV, with settings converted to their types."""
    config = load_config(CONFIG_PATHS[model])
    path = tuning_csv(tuning_stem(model, resolution, test=False))
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing tuning results {path}. "
            f"Run python src/tune_classical.py --config {CONFIG_PATHS[model].name} first."
        )
    rows = read_rows(path)
    if len(rows) != config["budget"]:
        raise ValueError(f"{path} has {len(rows)} tries; expected {config['budget']}.")
    best = best_row(rows)
    unknown = [name for name in config["sweep"]["parameters"] if name not in SETTING_TYPES]
    if unknown:
        raise ValueError(f"No type conversion defined for tuned settings {unknown}.")
    settings = {name: SETTING_TYPES[name](best[name]) for name in config["sweep"]["parameters"]}
    return Tuned(settings, path, best["run_id"], best[METRIC])


def load_test_images(resolution: int, test: Split) -> np.ndarray:
    """Return the test images, checking they match the cached test labels and indices."""
    images, labels, indices = load_split(resolution, "test")
    _, cached_labels, cached_indices = test
    if not (np.array_equal(indices, cached_indices) and np.array_equal(labels, cached_labels)):
        raise ValueError(
            f"Test images at {resolution} px do not match the cached test indices and labels. "
            "Rerun python src/extract_features.py."
        )
    return images


def make_run_dir(stem: str) -> Path:
    """Create the run folder (with plots/) named after the start time, waiting for a free second."""
    while True:
        started = datetime.now()
        path = RUNS_DIR / f"BloodCellClassification_{stem}_{started:%Y%m%d_%H%M%S}"
        if not path.exists():
            (path / "plots").mkdir(parents=True)
            return path
        time.sleep(1 - started.microsecond / 1_000_000)


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def run_config(
    model: str, resolution: int, seed: int, tuned: Tuned, pipeline: Pipeline
) -> dict[str, Any]:
    return {
        "model": model,
        "dataset": DATASET,
        "resolution": resolution,
        "seed": seed,
        "tuned_settings": tuned.settings,
        "model_params": pipeline.named_steps["model"].get_params(),
        "tuning_csv": tuned.csv_path.relative_to(REPO_ROOT).as_posix(),
        "tuning_run_id": tuned.run_id,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "wandb": wandb.__version__,
            "joblib": joblib.__version__,
        },
        "processor": platform.processor(),
    }


def write_predictions(
    path: Path,
    indices: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Write one row per test image: dataset index, true and predicted class, probabilities."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(PREDICTION_COLUMNS)
        for index, label, prediction, row in zip(indices, labels, predictions, probabilities):
            writer.writerow([
                index,
                CLASS_SHORT_NAMES[label],
                CLASS_SHORT_NAMES[prediction],
                *(round(float(p), PROBABILITY_DIGITS) for p in row),
            ])


def log_to_wandb(
    target: WandbTarget,
    run_name: str,
    group: str,
    config: dict[str, Any],
    record: dict[str, Any],
    labels: np.ndarray,
    predictions: np.ndarray,
) -> None:
    entity, project = target
    with wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        group=group,
        job_type="final",
        config=config,
    ) as run:
        run.log({
            **record,
            "test_confusion_matrix": wandb.plot.confusion_matrix(
                y_true=labels.tolist(),
                preds=predictions.tolist(),
                class_names=list(CLASS_SHORT_NAMES),
            ),
        })


def run_one(
    model: str,
    resolution: int,
    seed: int,
    tuned: Tuned,
    data: dict[str, Split],
    test_images: np.ndarray,
    test_run: bool,
    wandb_target: WandbTarget | None,
) -> dict[str, Any]:
    """Fit one seed on train, evaluate it once on test, write its files; return its CSV row."""
    stem = tuning_stem(model, resolution, test_run)
    run_dir = make_run_dir(stem)
    run_name = run_dir.name
    prefix = f"{model} | {DATASET}{resolution} | seed {seed}"
    log = make_logger(f"run_classical.{run_name}", prefix, run_dir / "log.txt")
    test_log = make_logger(f"run_classical.{run_name}.test", prefix, run_dir / "test_log.txt")
    try:
        train_features, train_labels, _ = data["train"]
        test_features, test_labels, test_indices = data["test"]
        log.info(f"run {run_name} started")
        pipeline = build_pipeline(model, tuned.settings, seed)
        config = run_config(model, resolution, seed, tuned, pipeline)
        write_json(run_dir / "config.json", config)
        log.info(
            f"settings: {format_settings(tuned.settings)} (best tuning try {tuned.run_id}, "
            f"val macro F1 {tuned.val_macro_f1})"
        )

        start = time.perf_counter()
        pipeline.fit(train_features, train_labels)
        fit_seconds = time.perf_counter() - start
        log.info(f"fit done in {fit_seconds:.3f} s")
        if not np.array_equal(pipeline.classes_, np.arange(len(CLASS_SHORT_NAMES))):
            raise RuntimeError(f"Fitted classes {pipeline.classes_} do not cover every class.")

        start = time.perf_counter()
        predictions = pipeline.predict(test_features)
        predict_seconds = time.perf_counter() - start
        probabilities = pipeline.predict_proba(test_features)

        model_path = run_dir / "model.joblib"
        joblib.dump(pipeline, model_path)
        metrics = compute_metrics(test_labels, predictions)
        record = {
            **{f"test_{name}": value for name, value in metrics.items()},
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "predict_us_per_image": predict_seconds / len(test_labels) * 1_000_000,
            "model_bytes": model_path.stat().st_size,
        }
        write_json(run_dir / "metrics.json", record)
        write_predictions(
            run_dir / "predictions.csv", test_indices, test_labels, predictions, probabilities
        )
        save_confusion_matrix(test_labels, predictions, run_dir / "plots" / "confusion_matrix.png")
        save_misclassified_grid(
            test_images,
            test_labels,
            predictions,
            test_indices,
            run_dir / "plots" / "misclassified.png",
        )

        test_log.info(f"run {run_name}")
        test_log.info(f"test images: {len(test_labels)}")
        test_log.info(f"accuracy {metrics['accuracy']:.4f}")
        test_log.info(f"macro F1 {metrics['macro_f1']:.4f}")
        recalls = ", ".join(
            f"{name} {metrics[f'recall_{key}']:.4f}"
            for name, key in zip(CLASS_SHORT_NAMES, CLASS_KEYS)
        )
        test_log.info(f"recall: {recalls}")
        test_log.info(
            f"prediction time {predict_seconds:.3f} s, "
            f"{record['predict_us_per_image']:.3f} us per image"
        )
        test_log.info(f"model size {record['model_bytes']} bytes")

        if wandb_target is not None:
            log_to_wandb(
                wandb_target, run_name, f"final_{stem}", config, record, test_labels, predictions
            )
    finally:
        close_logger(log)
        close_logger(test_log)

    return {
        "run_name": run_name,
        "timestamp": datetime.now().strftime(TIMESTAMP_FORMAT),
        "model": model,
        "resolution": resolution,
        "seed": seed,
        "settings": json.dumps(tuned.settings),
        **{column: round(record[column], 4) for column in METRIC_COLUMNS},
        **{column: round(record[column], 3) for column in TIME_COLUMNS},
        "model_bytes": record["model_bytes"],
    }


def write_summary(runs_path: Path, summary_path: Path) -> None:
    """Write mean and sample std (ddof=1) of every value per model and resolution."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in read_rows(runs_path):
        groups.setdefault((row["model"], row["resolution"]), []).append(row)
    columns = ["model", "resolution", "n_seeds"]
    columns += [f"{value}_{stat}" for value in SUMMARY_VALUES for stat in ("mean", "std")]
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for (model, resolution), rows in groups.items():
            summary: dict[str, Any] = {
                "model": model,
                "resolution": resolution,
                "n_seeds": len(rows),
            }
            for value in SUMMARY_VALUES:
                values = np.array([float(row[value]) for row in rows])
                std = values.std(ddof=1) if len(values) > 1 else np.nan
                digits = 4 if value in METRIC_COLUMNS else 3
                summary[f"{value}_mean"] = round(float(values.mean()), digits)
                summary[f"{value}_std"] = round(float(std), digits)
            writer.writerow(summary)


def main() -> None:
    args = parse_args()
    models = list(MODELS) if args.model is None else [args.model]
    resolutions = list(RESOLUTIONS) if args.resolution is None else [args.resolution]
    seeds = sorted(set(args.seeds))
    runs_path, summary_path = results_csvs(args.test)
    if runs_path.exists():
        raise FileExistsError(f"{runs_path} already exists; move or delete it before rerunning.")
    tuned = {
        (model, resolution): load_tuned(model, resolution)
        for resolution in resolutions
        for model in models
    }

    wandb_target = None
    if args.wandb:
        wandb_target = load_wandb_env()
        print(f"W&B entity: {wandb_target[0]}, project: {wandb_target[1]}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with runs_path.open("x", newline="") as file:
        csv.DictWriter(file, fieldnames=RUN_COLUMNS).writeheader()
    for resolution in resolutions:
        data, _ = load_features(resolution, ("train", "test"))
        test_images = load_test_images(resolution, data["test"])
        for model in models:
            for seed in seeds:
                row = run_one(
                    model,
                    resolution,
                    seed,
                    tuned[model, resolution],
                    data,
                    test_images,
                    args.test,
                    wandb_target,
                )
                append_row(runs_path, RUN_COLUMNS, row)

    write_summary(runs_path, summary_path)
    print("Saved:")
    for path in (runs_path, summary_path):
        print(f"  {path}")


if __name__ == "__main__":
    main()
