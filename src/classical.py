"""Classical classifiers on the cached handcrafted features.

The command-line check fits one model with default settings on the training split
and scores it on the validation split. It never touches the test split and writes no files.

Run with: python src/classical.py --model MODEL --resolution {28,64,128} [--seed N]
"""

import argparse
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import RESOLUTIONS, SPLITS, Split
from extract_features import features_path

MODELS = ("LogisticRegression", "RandomForest")
LR_MAX_ITER = 1000
RF_N_JOBS = -1


def load_features(
    resolution: int, splits: Sequence[str] = SPLITS
) -> tuple[dict[str, Split], list[str]]:
    """Return {split: (features, labels, indices)} for the given splits and the feature names."""
    unknown = [split for split in splits if split not in SPLITS]
    if unknown:
        raise ValueError(f"Unknown splits {unknown}; expected a subset of {SPLITS}.")
    path = features_path(resolution)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing feature file {path}. Run python src/extract_features.py first."
        )
    with np.load(path) as npz:
        kinds = ("features", "labels", "indices")
        required = [f"{split}_{kind}" for split in splits for kind in kinds] + ["feature_names"]
        missing = [key for key in required if key not in npz.files]
        if missing:
            raise KeyError(f"{path} is missing arrays {missing}.")
        data: dict[str, Split] = {}
        for split in splits:
            features = npz[f"{split}_features"]
            labels = npz[f"{split}_labels"]
            indices = npz[f"{split}_indices"]
            if not len(features) == len(labels) == len(indices):
                raise ValueError(
                    f"Length mismatch in {split} of {path}: {len(features)} features, "
                    f"{len(labels)} labels, {len(indices)} indices."
                )
            data[split] = (features, labels, indices)
        names = npz["feature_names"].tolist()
    return data, names


def build_pipeline(model: str, params: dict[str, Any], seed: int) -> Pipeline:
    """Return an unfitted pipeline; params override the model step's defaults.

    Unknown parameter names raise a ValueError.
    """
    if model == "LogisticRegression":
        # l1_ratio=0.0 is plain L2; penalty is deprecated since 1.8.
        estimator = LogisticRegression(
            l1_ratio=0.0, solver="lbfgs", max_iter=LR_MAX_ITER, random_state=seed
        )
        steps = [("scaler", StandardScaler())]
    elif model == "RandomForest":
        estimator = RandomForestClassifier(n_jobs=RF_N_JOBS, random_state=seed)
        steps = []  # trees do not depend on feature scaling
    else:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}.")
    estimator.set_params(**params)
    return Pipeline([*steps, ("model", estimator)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--resolution", type=int, choices=RESOLUTIONS, required=True)
    parser.add_argument("--seed", type=int, default=0, help="random state (default: 0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, names = load_features(args.resolution, ("train", "val"))
    train_features, train_labels, _ = data["train"]
    val_features, val_labels, _ = data["val"]

    pipeline = build_pipeline(args.model, {}, args.seed)
    start = time.perf_counter()
    pipeline.fit(train_features, train_labels)
    seconds = time.perf_counter() - start
    predictions = pipeline.predict(val_features)

    print(f"model:        {args.model}")
    print(f"resolution:   {args.resolution}")
    print(f"seed:         {args.seed}")
    print(f"train images: {len(train_labels)}")
    print(f"val images:   {len(val_labels)}")
    print(f"features:     {len(names)}")
    print(f"fit time:     {seconds:.2f} s")
    print(f"val accuracy: {accuracy_score(val_labels, predictions):.4f}")
    print(f"val macro F1: {f1_score(val_labels, predictions, average='macro'):.4f}")
    if args.model == "LogisticRegression":
        n_iter = int(pipeline.named_steps["model"].n_iter_.max())
        print(f"iterations:   {n_iter} (reached LR_MAX_ITER: {n_iter >= LR_MAX_ITER})")


if __name__ == "__main__":
    main()
