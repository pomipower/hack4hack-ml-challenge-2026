"""Camera-grouped feature ablations for transfer-risk review.

Example:
    python -m planet_hunters.ablation --data-dir .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from .modeling import LABELS, _global_probabilities, _lightgbm_classifier


DEFAULT_FEATURE_SETS = {
    "baseline": (),
    "drop_bls_period": ("bls_period",),
    "drop_bls_duration": ("bls_duration",),
    "drop_bls_dump_overlap_excess": ("bls_dump_overlap_excess",),
    "drop_orbit_level_diff": ("orbit_level_diff",),
    "drop_all_four": ("bls_period", "bls_duration", "bls_dump_overlap_excess", "orbit_level_diff"),
}


def _score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    predicted = np.asarray(LABELS, dtype=object)[np.argmax(probabilities, axis=1)]
    return float(f1_score(labels, predicted, labels=LABELS, average="macro", zero_division=0))


def run(data_dir: Path, cache_dir: Path, output_dir: Path, n_estimators: int = 500) -> Path:
    stars = pd.read_csv(data_dir / "train_stars.csv")
    y = stars["label"].to_numpy(dtype=str)
    camera = stars["camera"].to_numpy()
    features = pd.read_pickle(cache_dir / "train_features.pkl").drop(columns="id")
    splits = list(GroupKFold(n_splits=min(4, len(np.unique(camera)))).split(features, y, groups=camera))
    report = {"baseline_features": int(features.shape[1]), "sets": {}}
    for name, dropped in DEFAULT_FEATURE_SETS.items():
        unavailable = sorted(set(dropped) - set(features.columns))
        if unavailable:
            raise ValueError(f"Missing requested ablation features: {unavailable}")
        x = features.drop(columns=list(dropped))
        folds = []
        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            model = _lightgbm_classifier(8000 + 100 * len(dropped) + fold, n_estimators)
            weights = compute_sample_weight(class_weight="balanced", y=y[train_idx]).astype(np.float32)
            model.fit(x.iloc[train_idx], y[train_idx], sample_weight=weights)
            folds.append(_score(y[val_idx], _global_probabilities(model, x.iloc[val_idx])))
        report["sets"][name] = {
            "dropped": list(dropped), "feature_count": int(x.shape[1]),
            "mean_camera_macro_f1": float(np.mean(folds)), "std_camera_macro_f1": float(np.std(folds)),
            "folds": [float(value) for value in folds],
        }
        print(f"{name}: {np.mean(folds):.4f} ± {np.std(folds):.4f}")
    baseline = report["sets"]["baseline"]["mean_camera_macro_f1"]
    for values in report["sets"].values():
        values["delta_vs_baseline"] = float(values["mean_camera_macro_f1"] - baseline)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "camera_feature_ablation.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", default="artifacts_cnn")
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args()
    run(Path(args.data_dir).resolve(), Path(args.cache_dir).resolve(), Path(args.output_dir).resolve(), args.n_estimators)


if __name__ == "__main__":
    main()
