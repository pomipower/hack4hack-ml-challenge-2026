"""Post-training diagnostics for saved LightGBM/CNN candidate artifacts.

Run with:
    python -m planet_hunters.audit --data-dir . --artifacts-dir artifacts_cnn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from .modeling import LABELS, _global_probabilities, _tune_class_multipliers, make_submission


def _macro(y: np.ndarray, probabilities: np.ndarray, multipliers: np.ndarray | None = None) -> float:
    adjusted = probabilities if multipliers is None else probabilities * multipliers[None, :]
    prediction = np.asarray(LABELS, dtype=object)[np.argmax(adjusted, axis=1)]
    return float(f1_score(y, prediction, labels=LABELS, average="macro", zero_division=0))


def _per_class(y: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(LABELS, dtype=object)[np.argmax(probabilities, axis=1)]
    report = classification_report(y, prediction, labels=LABELS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y, prediction, labels=LABELS)
    quiet_column = LABELS.index("quiet_star")
    return {
        "macro_f1": _macro(y, probabilities),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": matrix.tolist(),
        "non_quiet_predicted_quiet": int(matrix[:, quiet_column].sum() - matrix[quiet_column, quiet_column]),
        "predicted_quiet_by_true_class": {
            label: int(matrix[index, quiet_column]) for index, label in enumerate(LABELS) if label != "quiet_star"
        },
    }


def _recover_cnn_camera_oof(
    sequences: np.ndarray, labels: np.ndarray, camera: np.ndarray, checkpoint: Path, batch_size: int, *, shifted: bool = False
) -> np.ndarray:
    """Recreate grouped CNN OOF predictions from the saved four fold states."""
    import torch
    from .cnn import _build_model, _predict, _shifted_sequences

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    states = saved["state_dicts"]
    splitter = GroupKFold(n_splits=min(4, len(np.unique(camera))))
    splits = list(splitter.split(np.zeros(len(labels)), labels, groups=camera))
    if len(states) < len(splits):
        raise ValueError("CNN checkpoint does not contain all camera-fold models.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = np.zeros((len(labels), len(LABELS)), dtype=np.float32)
    for fold, (state, (_, val_idx)) in enumerate(zip(states, splits), start=1):
        model = _build_model(torch).to(device)
        model.load_state_dict(state)
        source = _shifted_sequences(sequences, val_idx, 42 + 100 + fold) if shifted else sequences
        indices = np.arange(len(val_idx)) if shifted else val_idx
        output[val_idx] = _predict(model, source, indices, device, batch_size)
        del model
    return output


def _planet_slices(labels: np.ndarray, probabilities: np.ndarray, features: pd.DataFrame) -> dict[str, Any]:
    """Planet recall stratified by magnitude and BLS strength, using stratified OOF."""
    planet = labels == "planet"
    prediction = np.asarray(LABELS, dtype=object)[np.argmax(probabilities, axis=1)]
    result: dict[str, Any] = {}
    for feature in ("tmag", "bls_snr"):
        values = features[feature].to_numpy(dtype=float)
        valid_planet_values = values[planet & np.isfinite(values)]
        edges = np.unique(np.quantile(valid_planet_values, [0.0, 0.25, 0.5, 0.75, 1.0]))
        rows = []
        for low, high in zip(edges[:-1], edges[1:]):
            include_high = high == edges[-1]
            membership = planet & (values >= low) & ((values <= high) if include_high else (values < high))
            n = int(membership.sum())
            if n:
                rows.append({
                    "range": f"[{low:.4g}, {high:.4g}{']' if include_high else ')'}",
                    "support": n,
                    "planet_recall": float(np.mean(prediction[membership] == "planet")),
                    "predicted_starspots": int(np.sum(prediction[membership] == "starspots")),
                    "predicted_quiet": int(np.sum(prediction[membership] == "quiet_star")),
                })
        result[feature] = rows
    return result


def _blend_audit(
    y: np.ndarray, camera: np.ndarray, lgb: np.ndarray, cnn: np.ndarray,
    lgb_shifted: np.ndarray, cnn_shifted: np.ndarray,
) -> dict[str, Any]:
    alphas = np.round(np.arange(0.0, 0.51, 0.05), 2)
    splitter = GroupKFold(n_splits=min(4, len(np.unique(camera))))
    splits = list(splitter.split(np.zeros(len(y)), y, groups=camera))
    fixed_scores = {}
    for alpha in alphas:
        values = []
        for _, val_idx in splits:
            values.append(_macro(y[val_idx], (1.0 - alpha) * lgb[val_idx] + alpha * cnn[val_idx]))
        fixed_scores[str(alpha)] = {"mean": float(np.mean(values)), "folds": [float(score) for score in values]}

    crossfit_alphas, crossfit_scores, crossfit_probs = [], [], np.zeros_like(lgb)
    crossfit_shifted_scores, baseline_shifted_scores = [], []
    for train_idx, val_idx in splits:
        scores = [_macro(y[train_idx], (1.0 - alpha) * lgb[train_idx] + alpha * cnn[train_idx]) for alpha in alphas]
        alpha = float(alphas[int(np.argmax(scores))])
        probabilities = (1.0 - alpha) * lgb[val_idx] + alpha * cnn[val_idx]
        crossfit_alphas.append(alpha)
        crossfit_scores.append(_macro(y[val_idx], probabilities))
        crossfit_probs[val_idx] = probabilities
        baseline_shifted_scores.append(_macro(y[val_idx], lgb_shifted[val_idx]))
        crossfit_shifted_scores.append(_macro(
            y[val_idx], (1.0 - alpha) * lgb_shifted[val_idx] + alpha * cnn_shifted[val_idx]
        ))

    unguarded, before, after = _tune_class_multipliers(y, crossfit_probs)
    nested_before, nested_after, nested_values = [], [], []
    for train_idx, val_idx in splits:
        multipliers, base, _ = _tune_class_multipliers(y[train_idx], crossfit_probs[train_idx])
        nested_before.append(_macro(y[val_idx], crossfit_probs[val_idx]))
        nested_after.append(_macro(y[val_idx], crossfit_probs[val_idx], multipliers))
        nested_values.append({label: float(value) for label, value in zip(LABELS, multipliers)})
    weighted = crossfit_probs * unguarded[None, :]
    weighted /= np.maximum(weighted.sum(axis=1, keepdims=True), 1e-12)
    return {
        "fixed_alpha_camera_grouped_scores": fixed_scores,
        "cross_fitted_alpha_by_camera": crossfit_alphas,
        "cross_fitted_blend_macro_f1": float(np.mean(crossfit_scores)),
        "cross_fitted_blend_fold_scores": [float(score) for score in crossfit_scores],
        "cross_fitted_shifted_blend_macro_f1": float(np.mean(crossfit_shifted_scores)),
        "cross_fitted_shifted_blend_delta": float(np.mean(crossfit_shifted_scores) - np.mean(baseline_shifted_scores)),
        "cross_fitted_blend_per_class": _per_class(y, crossfit_probs),
        "ungated_full_oof_multipliers": {label: float(value) for label, value in zip(LABELS, unguarded)},
        "ungated_full_oof_macro_f1_before": float(before),
        "ungated_full_oof_macro_f1_after": float(after),
        "nested_multiplier_macro_f1_before": float(np.mean(nested_before)),
        "nested_multiplier_macro_f1_after": float(np.mean(nested_after)),
        "nested_multiplier_gain": float(np.mean(nested_after) - np.mean(nested_before)),
        "nested_multiplier_by_camera": nested_values,
        "ungated_full_oof_per_class_after_multipliers": _per_class(y, weighted),
    }


def run(data_dir: Path, artifacts_dir: Path, cache_dir: Path, batch_size: int = 256) -> Path:
    train = pd.read_csv(data_dir / "train_stars.csv")
    y = train["label"].to_numpy(dtype=str)
    camera = train["camera"].to_numpy()
    features = pd.read_pickle(cache_dir / "train_features.pkl").drop(columns="id")
    lgb_grouped = np.load(artifacts_dir / "camera_grouped_probabilities.npy")
    lgb_shifted = np.load(artifacts_dir / "camera_grouped_shifted_probabilities.npy")
    lgb_stratified = np.load(artifacts_dir / "oof_probabilities.npy")
    sequences = np.load(cache_dir / "train_cnn_sequences.npy", mmap_mode="r")
    cnn_grouped = _recover_cnn_camera_oof(sequences, y, camera, artifacts_dir / "cnn_models.pt", batch_size)
    cnn_shifted = _recover_cnn_camera_oof(
        sequences, y, camera, artifacts_dir / "cnn_models.pt", batch_size, shifted=True
    )
    audit = {
        "labels": LABELS,
        "source_artifacts": str(artifacts_dir),
        "lightgbm_stratified_oof": _per_class(y, lgb_stratified),
        "planet_recall_by_tmag_and_bls_snr": _planet_slices(y, lgb_stratified, features),
        "camera_grouped_lightgbm": _per_class(y, lgb_grouped),
        "camera_grouped_cnn": _per_class(y, cnn_grouped),
        "blend": _blend_audit(y, camera, lgb_grouped, cnn_grouped, lgb_shifted, cnn_shifted),
    }
    output = artifacts_dir / "model_audit.json"
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Camera-grouped LightGBM: {audit['camera_grouped_lightgbm']['macro_f1']:.4f}")
    print(f"Camera-grouped CNN: {audit['camera_grouped_cnn']['macro_f1']:.4f}")
    print(f"Cross-fitted dense blend: {audit['blend']['cross_fitted_blend_macro_f1']:.4f}")
    print(f"Nested multiplier gain: {audit['blend']['nested_multiplier_gain']:+.4f}")
    return output


def write_alpha_submission(data_dir: Path, artifacts_dir: Path, cache_dir: Path, alpha: float) -> Path:
    """Create a separately named test candidate from the saved ensemble models."""
    import joblib

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    test = pd.read_csv(data_dir / "test_stars.csv")
    template = pd.read_csv(data_dir / "sample_submission.csv")
    features = pd.read_pickle(cache_dir / "test_features.pkl").drop(columns="id").astype(np.float32)
    ensemble = joblib.load(artifacts_dir / "lightgbm_ensemble.joblib")
    if len(ensemble) % 6:
        raise ValueError("Expected five fold models plus one full-data model per LightGBM seed.")
    n_seeds = len(ensemble) // 6
    lgb = np.zeros((len(test), len(LABELS)), dtype=np.float32)
    for seed in range(n_seeds):
        models = ensemble[seed * 6:(seed + 1) * 6]
        fold_average = np.mean([_global_probabilities(model, features) for model in models[:5]], axis=0)
        lgb += 0.5 * (fold_average + _global_probabilities(models[5], features)) / n_seeds
    cnn = np.load(artifacts_dir / "cnn_test_probabilities.npy")
    probabilities = (1.0 - alpha) * lgb + alpha * cnn
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    labels = np.asarray(LABELS, dtype=object)[np.argmax(probabilities, axis=1)]
    target = artifacts_dir / f"submission_alpha_{alpha:.2f}.csv"
    make_submission(test, labels, template, target)
    np.save(artifacts_dir / f"test_probabilities_alpha_{alpha:.2f}.npy", probabilities)
    print(f"Wrote {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--artifacts-dir", default="artifacts_cnn")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--write-alpha-submission", type=float, default=None,
                        help="Write a separate test candidate using this CNN blend weight.")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    run(data_dir, artifacts_dir, cache_dir, args.batch_size)
    if args.write_alpha_submission is not None:
        write_alpha_submission(data_dir, artifacts_dir, cache_dir, args.write_alpha_submission)


if __name__ == "__main__":
    main()
