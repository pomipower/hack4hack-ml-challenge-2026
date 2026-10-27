"""Cross-validated LightGBM modeling and competition submission helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight


LABELS = [
    "planet", "eclipsing_binary", "background_binary", "pulsating_star",
    "flare_star", "starspots", "instrument_glitch", "quiet_star",
]


def _lightgbm_classifier(seed: int, n_estimators: int = 500):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError("LightGBM is required. Install dependencies with `python -m pip install -r requirements.txt`.") from exc
    return LGBMClassifier(
        objective="multiclass", num_class=len(LABELS), n_estimators=n_estimators,
        learning_rate=0.04, num_leaves=31, max_depth=-1, min_child_samples=25,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        reg_lambda=2.0, class_weight=None, random_state=seed,
        n_jobs=-1, verbosity=-1,
    )


def _align_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if list(train.columns) != list(test.columns):
        missing = sorted(set(train.columns) ^ set(test.columns))
        if missing:
            raise ValueError(f"Train/test feature columns differ: {missing}")
        test = test[train.columns]
    return train.astype(np.float32), test.astype(np.float32)


def _synthetic_shift(features: pd.DataFrame) -> pd.DataFrame:
    """Apply a bounded feature-level sector-shift stress case to validation rows."""
    shifted = features.copy()
    for name in ("sigma_star", "ptp_noise", "noise_lag3", "noise_lag6", "noise_lag12"):
        if name in shifted:
            shifted[name] *= 1.25
    for name in ("centroid_column_std", "centroid_row_std", "centroid_column_range", "centroid_row_range"):
        if name in shifted:
            shifted[name] *= 1.20
    for name, delta in (("bls_dump_overlap_excess", 0.10),):
        if name in shifted:
            shifted[name] = np.clip(shifted[name] + delta, -1, 1)
    band_names = ("power_spot", "power_gdor", "power_rr", "power_delta")
    if all(name in shifted for name in band_names):
        original = shifted.loc[:, band_names].to_numpy(dtype=np.float32)
        low = np.clip(original[:, 0] + 0.08, 0, 1)
        rest = original[:, 1:]
        rest_sum = rest.sum(axis=1, keepdims=True)
        remaining = 1.0 - low
        shifted.loc[:, band_names[0]] = low
        shifted.loc[:, list(band_names[1:])] = np.divide(rest * remaining[:, None], rest_sum, out=np.zeros_like(rest), where=rest_sum > 0)
    return shifted


def _global_probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)
    result = np.zeros((len(features), len(LABELS)), dtype=np.float32)
    for j, label in enumerate(model.classes_):
        result[:, LABELS.index(label)] = raw[:, j]
    return result


def _macro_score(y: np.ndarray, probabilities: np.ndarray, multipliers: np.ndarray | None = None) -> float:
    adjusted = probabilities if multipliers is None else probabilities * multipliers[None, :]
    pred = np.asarray(LABELS, dtype=object)[np.argmax(adjusted, axis=1)]
    return float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0))


def _tune_class_multipliers(y: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Conservative coordinate search, bounded to avoid extreme class thresholds."""
    multipliers = np.ones(len(LABELS), dtype=np.float32)
    before = _macro_score(y, probabilities)
    candidates = np.asarray([0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.45, 1.7, 2.0], dtype=np.float32)
    for _ in range(2):
        for c in range(len(LABELS)):
            best_value, best_score = float(multipliers[c]), _macro_score(y, probabilities, multipliers)
            for value in candidates:
                trial = multipliers.copy()
                trial[c] = value
                score = _macro_score(y, probabilities, trial)
                if score > best_score + 1e-8 or (abs(score - best_score) <= 1e-8 and abs(value - 1.0) < abs(best_value - 1.0)):
                    best_value, best_score = float(value), score
            multipliers[c] = best_value
    after = _macro_score(y, probabilities, multipliers)
    return multipliers, before, after


def _sector_adversarial_validation(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Diagnose feature-table sector signatures; this does not enter the label model."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError("LightGBM is required. Install dependencies with `python -m pip install -r requirements.txt`.") from exc
    combined = pd.concat([train, test], ignore_index=True)
    domain = np.r_[np.zeros(len(train), dtype=np.int8), np.ones(len(test), dtype=np.int8)]
    fit_idx, val_idx = train_test_split(np.arange(len(domain)), test_size=0.25, random_state=seed, stratify=domain)
    model = LGBMClassifier(
        objective="binary", n_estimators=250, learning_rate=0.04, num_leaves=31,
        min_child_samples=30, colsample_bytree=0.85, reg_lambda=2.0,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )
    model.fit(combined.iloc[fit_idx], domain[fit_idx])
    probability = model.predict_proba(combined.iloc[val_idx])[:, 1]
    importance = sorted(zip(combined.columns, model.feature_importances_), key=lambda item: item[1], reverse=True)
    return {
        "heldout_sector_auc": float(roc_auc_score(domain[val_idx], probability)),
        "top_sector_discriminators": [{"feature": str(name), "gain": int(value)} for name, value in importance[:20]],
        "interpretation": "Diagnostic only: a high AUC identifies sector-signature features for review; it is not a classification score.",
    }


@dataclass
class FitResult:
    probabilities: np.ndarray
    labels: np.ndarray
    oof_probabilities: np.ndarray
    report: dict[str, Any]
    model: Any
    camera_grouped_probabilities: np.ndarray | None = None
    camera_grouped_shifted_probabilities: np.ndarray | None = None


def fit_predict(
    train_features: pd.DataFrame,
    labels: pd.Series | np.ndarray,
    test_features: pd.DataFrame,
    *,
    camera: pd.Series | np.ndarray,
    seed: int = 42,
    folds: int = 5,
    n_estimators: int = 500,
    ensemble_seeds: int = 3,
    run_camera_cv: bool = True,
) -> FitResult:
    """Fit a cross-sector multiclass model and return probabilities plus diagnostics."""
    x, xt = _align_features(train_features, test_features)
    y = np.asarray(labels, dtype=str)
    cam = np.asarray(camera)
    if set(np.unique(y)) != set(LABELS):
        unknown = sorted(set(np.unique(y)) - set(LABELS))
        missing = sorted(set(LABELS) - set(np.unique(y)))
        raise ValueError(f"Unexpected label set; unknown={unknown}, missing={missing}")
    if len(x) != len(y) or len(y) != len(cam):
        raise ValueError("Training features, labels, and camera groups must have equal lengths.")
    domain_report = _sector_adversarial_validation(x, xt, seed)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros((len(x), len(LABELS)), dtype=np.float32)
    fold_scores: list[float] = []
    for fold, (tr, va) in enumerate(splitter.split(x, y), start=1):
        model = _lightgbm_classifier(seed + fold, n_estimators)
        fold_weights = compute_sample_weight(class_weight="balanced", y=y[tr]).astype(np.float32)
        model.fit(x.iloc[tr], y[tr], sample_weight=fold_weights)
        probs = _global_probabilities(model, x.iloc[va])
        oof[va] = probs
        fold_scores.append(_macro_score(y[va], probs))

    oof_labels = np.asarray(LABELS, dtype=object)[np.argmax(oof, axis=1)]
    oof_score = float(f1_score(y, oof_labels, average="macro", labels=LABELS, zero_division=0))
    suspect = (oof[np.arange(len(y)), [LABELS.index(label) for label in y]] < 0.05) & (oof.max(axis=1) > 0.90)
    # Cap label-noise down-weighting at 2% of the training set.
    cap = int(np.floor(0.02 * len(y)))
    suspect_idx = np.flatnonzero(suspect)
    if suspect_idx.size > cap:
        confidence = oof[suspect_idx].max(axis=1)
        suspect_idx = suspect_idx[np.argsort(confidence)[-cap:]] if cap else np.empty(0, dtype=int)
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float32)
    if suspect_idx.size:
        weights[suspect_idx] *= 0.25
    tuning_suspect = np.zeros(len(y), dtype=bool)
    tuning_suspect[suspect_idx] = True

    camera_scores: list[float] = []
    shifted_camera_scores: list[float] = []
    camera_fold_probabilities: list[np.ndarray] = []
    camera_fold_shifted_probabilities: list[np.ndarray] = []
    camera_fold_indices: list[np.ndarray] = []
    grouped_oof: np.ndarray | None = None
    shifted_grouped_oof: np.ndarray | None = None
    camera_report: dict[str, Any] = {}
    if run_camera_cv:
        unique_camera = np.unique(cam)
        if len(unique_camera) >= 2:
            grouped = GroupKFold(n_splits=min(4, len(unique_camera)))
            for fold, (tr, va) in enumerate(grouped.split(x, y, groups=cam), start=1):
                model = _lightgbm_classifier(seed + 100 + fold, n_estimators)
                fold_weights = compute_sample_weight(class_weight="balanced", y=y[tr]).astype(np.float32)
                model.fit(x.iloc[tr], y[tr], sample_weight=fold_weights)
                probs = _global_probabilities(model, x.iloc[va])
                pred = np.asarray(LABELS, dtype=object)[np.argmax(probs, axis=1)]
                score = float(f1_score(y[va], pred, average="macro", labels=LABELS, zero_division=0))
                camera_scores.append(score)
                camera_fold_probabilities.append(probs)
                camera_fold_indices.append(va)
                shifted_probs = _global_probabilities(model, _synthetic_shift(x.iloc[va]))
                shifted_pred = np.asarray(LABELS, dtype=object)[np.argmax(shifted_probs, axis=1)]
                shifted_camera_scores.append(float(f1_score(y[va], shifted_pred, average="macro", labels=LABELS, zero_division=0)))
                camera_fold_shifted_probabilities.append(shifted_probs)
            grouped_oof = np.zeros((len(x), len(LABELS)), dtype=np.float32)
            shifted_grouped_oof = np.zeros_like(grouped_oof)
            for indices, probs, shifted_probs in zip(camera_fold_indices, camera_fold_probabilities, camera_fold_shifted_probabilities):
                grouped_oof[indices] = probs
                shifted_grouped_oof[indices] = shifted_probs
            # Cross-fit the calibration itself: each camera fold is evaluated
            # with multipliers tuned on the other camera-held-out predictions.
            calibrated_fold_scores: list[float] = []
            clean_baseline_scores: list[float] = []
            for indices, probs in zip(camera_fold_indices, camera_fold_probabilities):
                calibration_rows = np.ones(len(x), dtype=bool)
                calibration_rows[indices] = False
                calibration_rows &= ~tuning_suspect
                evaluation_rows = ~tuning_suspect[indices]
                if not evaluation_rows.any() or not calibration_rows.any():
                    calibrated_fold_scores.append(camera_scores[len(calibrated_fold_scores)])
                    clean_baseline_scores.append(camera_scores[len(clean_baseline_scores)])
                    continue
                fold_multipliers, _, _ = _tune_class_multipliers(
                    y[calibration_rows], grouped_oof[calibration_rows]
                )
                # Shrink noisy calibration estimates toward neutral weighting.
                fold_multipliers = 1.0 + 0.75 * (fold_multipliers - 1.0)
                calibrated_fold_scores.append(_macro_score(y[indices][evaluation_rows], probs[evaluation_rows], fold_multipliers))
                clean_baseline_scores.append(_macro_score(y[indices][evaluation_rows], probs[evaluation_rows]))

            clean_rows = ~tuning_suspect
            multipliers, _, _ = _tune_class_multipliers(y[clean_rows], grouped_oof[clean_rows])
            multipliers = 1.0 + 0.75 * (multipliers - 1.0)
            grouped_before = float(np.mean(clean_baseline_scores))
            grouped_after = float(np.mean(calibrated_fold_scores))
            if grouped_after < grouped_before + 0.0075 or any(after < before - 0.02 for after, before in zip(calibrated_fold_scores, clean_baseline_scores)):
                multipliers = np.ones(len(LABELS), dtype=np.float32)
                grouped_after = grouped_before
                calibrated_fold_scores = clean_baseline_scores.copy()
            camera_report = {
                "mean": float(np.mean(camera_scores)), "std": float(np.std(camera_scores)), "folds": camera_scores,
                "cross_fitted_clean_baseline_macro_f1": grouped_before,
                "cross_fitted_macro_f1_after_multipliers": grouped_after,
                "cross_fitted_macro_f1_by_camera_after_multipliers": calibrated_fold_scores,
                "multiplier_gain_gate": 0.0075,
                "suspect_labels_excluded_from_multiplier_tuning": int(suspect_idx.size),
                "final_test_class_multipliers": {label: float(value) for label, value in zip(LABELS, multipliers)},
                "synthetic_shift_mean": float(np.mean(shifted_camera_scores)),
                "synthetic_shift_folds": shifted_camera_scores,
                "synthetic_shift_definition": "feature-level stress: +25% noise, +20% centroid scale, +0.08 low-frequency power, and stronger dump overlap",
            }
        else:
            multipliers = np.ones(len(LABELS), dtype=np.float32)
    else:
        multipliers = np.ones(len(LABELS), dtype=np.float32)

    # Average held-out-fold models across several seeds; down-weighted labels are
    # learned only from the independent stratified OOF predictions above.
    ensemble: list[Any] = []
    raw_test_probs = np.zeros((len(xt), len(LABELS)), dtype=np.float32)
    seed_scores: list[float] = []
    for seed_idx in range(max(1, ensemble_seeds)):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 1000 * seed_idx)
        seed_oof = np.zeros_like(oof)
        fold_test_probs = np.zeros_like(raw_test_probs)
        for fold, (tr, va) in enumerate(splitter.split(x, y), start=1):
            model = _lightgbm_classifier(seed + 1000 * seed_idx + fold, n_estimators)
            model.fit(x.iloc[tr], y[tr], sample_weight=weights[tr])
            seed_oof[va] = _global_probabilities(model, x.iloc[va])
            fold_test_probs += _global_probabilities(model, xt) / folds
            ensemble.append(model)
        seed_scores.append(_macro_score(y, seed_oof))
        full_model = _lightgbm_classifier(seed + 500 + 1000 * seed_idx, n_estimators)
        full_model.fit(x, y, sample_weight=weights)
        full_test_probs = _global_probabilities(full_model, xt)
        raw_test_probs += 0.5 * (fold_test_probs + full_test_probs) / max(1, ensemble_seeds)
        ensemble.append(full_model)
    weighted = raw_test_probs * multipliers[None, :]
    probs = weighted / np.maximum(weighted.sum(axis=1, keepdims=True), 1e-12)
    if grouped_oof is not None:
        grouped_weighted = grouped_oof * multipliers[None, :]
        grouped_oof = grouped_weighted / np.maximum(grouped_weighted.sum(axis=1, keepdims=True), 1e-12)
    if shifted_grouped_oof is not None:
        shifted_weighted = shifted_grouped_oof * multipliers[None, :]
        shifted_grouped_oof = shifted_weighted / np.maximum(shifted_weighted.sum(axis=1, keepdims=True), 1e-12)
    pred = np.asarray(LABELS, dtype=object)[np.argmax(probs, axis=1)]
    report = {
        "labels": LABELS,
        "oof_macro_f1": oof_score,
        "stratified_fold_macro_f1": fold_scores,
        "ensemble_oof_macro_f1_by_seed": seed_scores,
        "camera_grouped_macro_f1": camera_report,
        "sector_adversarial_validation": domain_report,
        "class_multipliers": {label: float(value) for label, value in zip(LABELS, multipliers)},
        "label_noise_candidates_downweighted": int(suspect_idx.size),
        "classification_report": classification_report(y, oof_labels, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, oof_labels, labels=LABELS).tolist(),
        "pair_confusion_counts": {
            f"{a}<->{b}": int(confusion_matrix(y, oof_labels, labels=LABELS)[LABELS.index(a), LABELS.index(b)] + confusion_matrix(y, oof_labels, labels=LABELS)[LABELS.index(b), LABELS.index(a)])
            for a, b in (("planet", "background_binary"), ("planet", "eclipsing_binary"), ("planet", "instrument_glitch"), ("pulsating_star", "starspots"))
        },
        "transit_duration_at_search_cap_fraction_by_class": {
            label: float(np.mean(x.loc[np.asarray(y) == label, "bls_duration"].to_numpy() >= 14.0 / 48.0))
            for label in LABELS if "bls_duration" in x
        },
        "feature_count": int(x.shape[1]),
        "training_rows": int(len(x)),
    }
    return FitResult(probs, pred, oof, report, ensemble, grouped_oof, shifted_grouped_oof)


def make_submission(
    test_stars: pd.DataFrame,
    predictions: np.ndarray,
    sample_submission: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    """Validate IDs, order, labels and required two-column upload format."""
    pred = np.asarray(predictions, dtype=str)
    expected_ids = sample_submission["id"].astype(str).to_numpy()
    test_ids = test_stars["id"].astype(str).to_numpy()
    if not np.array_equal(expected_ids, test_ids):
        raise ValueError("Test IDs/order do not exactly match sample_submission.csv.")
    if pred.shape != (len(expected_ids),):
        raise ValueError(f"Expected {len(expected_ids)} labels, got shape {pred.shape}.")
    invalid = sorted(set(pred) - set(LABELS))
    if invalid:
        raise ValueError(f"Invalid predicted labels: {invalid}")
    submission = pd.DataFrame({"id": expected_ids, "label": pred})
    if submission["id"].duplicated().any() or list(submission.columns) != ["id", "label"]:
        raise ValueError("Submission format validation failed.")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def save_fit_artifacts(result: FitResult, output_dir: str | Path) -> None:
    import json
    import joblib

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "test_probabilities.npy", result.probabilities)
    np.save(root / "oof_probabilities.npy", result.oof_probabilities)
    if result.camera_grouped_probabilities is not None:
        np.save(root / "camera_grouped_probabilities.npy", result.camera_grouped_probabilities)
    if result.camera_grouped_shifted_probabilities is not None:
        np.save(root / "camera_grouped_shifted_probabilities.npy", result.camera_grouped_shifted_probabilities)
    with (root / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(result.report, handle, indent=2)
    joblib.dump(result.model, root / "lightgbm_ensemble.joblib")
