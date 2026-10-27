"""CLI entry point: cache features, train, validate, and write submission.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import extract_features
from .modeling import fit_predict, make_submission, save_fit_artifacts
from .preprocessing import preprocess_sector

FEATURE_VERSION = "2026-10-22-v5"
CNN_SEQUENCE_VERSION = "cnn-sequence-v1"


def _source_signature(data_dir: Path, split: str, include_bls: bool) -> dict[str, Any]:
    paths = [data_dir / f"{split}_{suffix}" for suffix in ("flux.npy", "centroid.npy", "stars.csv", "cadences.csv")]
    return {
        "version": FEATURE_VERSION,
        "split": split,
        "include_bls": include_bls,
        "files": [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths],
    }


def _load_or_build_features(data_dir: Path, cache_dir: Path, split: str, *, include_bls: bool, rebuild: bool) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_features.pkl"
    meta_path = cache_dir / f"{split}_features.json"
    signature = _source_signature(data_dir, split, include_bls)
    if cache_path.exists() and meta_path.exists() and not rebuild:
        try:
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
            if saved == signature:
                print(f"Using cached {split} features: {cache_path}")
                return pd.read_pickle(cache_path)
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    print(f"Preprocessing {split} sector and extracting features...")
    processed, stars, _ = preprocess_sector(data_dir, split)
    features = extract_features(processed, stars, include_bls=include_bls)
    features.insert(0, "id", stars["id"].astype(str).to_numpy())
    features.to_pickle(cache_path)
    meta_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    print(f"Saved {len(features)} {split} feature rows to {cache_path}")
    return features


def _load_or_build_sequences(data_dir: Path, cache_dir: Path, split: str, *, rebuild: bool):
    from .cnn import write_sequence_cache

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{split}_cnn_sequences.npy"
    meta_path = cache_dir / f"{split}_cnn_sequences.json"
    signature = {"sequence_version": CNN_SEQUENCE_VERSION, **_source_signature(data_dir, split, False)}
    if target.exists() and meta_path.exists() and not rebuild:
        try:
            if json.loads(meta_path.read_text(encoding="utf-8")) == signature:
                import numpy as np
                return np.load(target, mmap_mode="r")
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    print(f"Preparing {split} cadence sequences for the CNN...")
    processed, _, _ = preprocess_sector(data_dir, split)
    sequences = write_sequence_cache(processed, target)
    del processed
    meta_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return sequences


def _macro_prob_score(labels, probabilities) -> float:
    import numpy as np
    from sklearn.metrics import f1_score

    y = np.asarray(labels, dtype=str)
    pred = np.asarray([
        "planet", "eclipsing_binary", "background_binary", "pulsating_star",
        "flare_star", "starspots", "instrument_glitch", "quiet_star",
    ], dtype=object)[np.argmax(probabilities, axis=1)]
    return float(f1_score(y, pred, average="macro", labels=[
        "planet", "eclipsing_binary", "background_binary", "pulsating_star",
        "flare_star", "starspots", "instrument_glitch", "quiet_star",
    ], zero_division=0))


def _evaluate_cnn_blend(result, cnn_result, labels, camera) -> dict[str, Any]:
    import numpy as np
    from sklearn.model_selection import GroupKFold

    if result.camera_grouped_probabilities is None or result.camera_grouped_shifted_probabilities is None:
        raise ValueError("CNN validation requires camera-grouped LightGBM probabilities.")
    y = np.asarray(labels, dtype=str)
    groups = np.asarray(camera)
    lb = result.camera_grouped_probabilities
    cnn = cnn_result.grouped_oof_probabilities
    lb_shift = result.camera_grouped_shifted_probabilities
    cnn_shift = cnn_result.shifted_oof_probabilities
    splits = list(GroupKFold(n_splits=min(4, len(np.unique(groups)))).split(np.zeros(len(y)), y, groups))
    # The first experiment used a coarse grid and accepted 0.50.  The CNN is a
    # complementary but weaker model, so use a fine, conservative range rather
    # than allowing it to dominate the tabular model.
    candidates = tuple(np.round(np.arange(0.0, 0.51, 0.05), 2))
    crossfit_alphas, baseline_scores, blended_scores = [], [], []
    shifted_baseline_scores, shifted_blended_scores = [], []
    for train_idx, val_idx in splits:
        best_alpha, best_score = 0.0, _macro_prob_score(y[train_idx], lb[train_idx])
        for alpha in candidates[1:]:
            score = _macro_prob_score(y[train_idx], (1.0 - alpha) * lb[train_idx] + alpha * cnn[train_idx])
            if score > best_score + 1e-8:
                best_alpha, best_score = alpha, score
        crossfit_alphas.append(best_alpha)
        baseline_scores.append(_macro_prob_score(y[val_idx], lb[val_idx]))
        blended_scores.append(_macro_prob_score(y[val_idx], (1.0 - best_alpha) * lb[val_idx] + best_alpha * cnn[val_idx]))
        shifted_baseline_scores.append(_macro_prob_score(y[val_idx], lb_shift[val_idx]))
        shifted_blended_scores.append(_macro_prob_score(y[val_idx], (1.0 - best_alpha) * lb_shift[val_idx] + best_alpha * cnn_shift[val_idx]))

    final_alpha = 0.0
    final_score = _macro_prob_score(y, lb)
    for alpha in candidates[1:]:
        score = _macro_prob_score(y, (1.0 - alpha) * lb + alpha * cnn)
        if score > final_score + 1e-8:
            final_alpha, final_score = alpha, score
    clean_gain = float(np.mean(blended_scores) - np.mean(baseline_scores))
    shift_delta = float(np.mean(shifted_blended_scores) - np.mean(shifted_baseline_scores))
    accepted = clean_gain >= 0.003 and all(
        after >= before - 0.02 for after, before in zip(blended_scores, baseline_scores)
    ) and shift_delta >= -0.02
    if accepted and final_alpha > 0:
        test = (1.0 - final_alpha) * result.probabilities + final_alpha * cnn_result.test_probabilities
        result.probabilities = test / np.maximum(test.sum(axis=1, keepdims=True), 1e-12)
        from .modeling import LABELS
        result.labels = np.asarray(LABELS, dtype=object)[np.argmax(result.probabilities, axis=1)]
    else:
        final_alpha = 0.0
    return {
        "cnn_camera_grouped_macro_f1_by_fold": cnn_result.fold_scores,
        "cnn_camera_grouped_macro_f1_mean": float(np.mean(cnn_result.fold_scores)),
        "cnn_shifted_macro_f1_by_fold": cnn_result.shifted_fold_scores,
        "cnn_best_epoch_by_fold": cnn_result.best_epochs,
        "cnn_device": cnn_result.device,
        "cross_fitted_blend_alpha_by_camera": crossfit_alphas,
        "cross_fitted_blend_macro_f1_by_camera": blended_scores,
        "cross_fitted_blend_gain": clean_gain,
        "cross_fitted_shifted_blend_delta": shift_delta,
        "blend_accepted": bool(accepted),
        "final_test_blend_alpha": float(final_alpha),
        "blend_gate": "accept only if cross-fitted grouped macro-F1 improves by at least 0.003, no camera fold drops by over 0.02, and shifted macro-F1 drops by at most 0.02",
    }


def run(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.ensemble_seeds < 1:
        raise ValueError("--ensemble-seeds must be at least 1")
    if args.n_estimators < 1:
        raise ValueError("--n-estimators must be at least 1")
    root = Path(args.data_dir).resolve()
    output = Path(args.output_dir).resolve()
    cache = Path(args.cache_dir).resolve()
    train_stars = pd.read_csv(root / "train_stars.csv")
    test_stars = pd.read_csv(root / "test_stars.csv")
    template = pd.read_csv(root / "sample_submission.csv")
    if len(train_stars) != 18000 or len(test_stars) != 7000:
        print(f"Dataset rows: train={len(train_stars)}, test={len(test_stars)}")
    if not test_stars["id"].astype(str).equals(template["id"].astype(str)):
        raise ValueError("test_stars.csv IDs/order differ from sample_submission.csv")
    if train_stars["label"].isna().any() or not set(train_stars["label"].unique()).issubset({
        "planet", "eclipsing_binary", "background_binary", "pulsating_star", "flare_star", "starspots", "instrument_glitch", "quiet_star"
    }):
        raise ValueError("Training labels contain missing or unknown categories.")

    train_all = _load_or_build_features(root, cache, "train", include_bls=not args.no_bls, rebuild=args.rebuild_features)
    test_all = _load_or_build_features(root, cache, "test", include_bls=not args.no_bls, rebuild=args.rebuild_features)
    if not train_all["id"].astype(str).equals(train_stars["id"].astype(str)) or not test_all["id"].astype(str).equals(test_stars["id"].astype(str)):
        raise ValueError("Cached feature rows are not aligned with the star metadata CSVs.")
    x = train_all.drop(columns=["id"])
    xt = test_all.drop(columns=["id"])
    result = fit_predict(
        x, train_stars["label"], xt, camera=train_stars["camera"],
        seed=args.seed, folds=args.folds, n_estimators=args.n_estimators,
        ensemble_seeds=args.ensemble_seeds,
        run_camera_cv=not args.no_camera_cv,
    )
    output.mkdir(parents=True, exist_ok=True)
    if args.train_cnn:
        if args.no_camera_cv:
            raise ValueError("--train-cnn requires camera-grouped validation so the blend can be gated safely.")
        try:
            from .cnn import CNN_CHANNELS, fit_cnn_grouped
            import torch
        except ImportError as exc:
            raise RuntimeError("CNN training requires PyTorch. Install the optional `torch` dependency first.") from exc
        train_sequences = _load_or_build_sequences(root, cache, "train", rebuild=args.rebuild_features)
        test_sequences = _load_or_build_sequences(root, cache, "test", rebuild=args.rebuild_features)
        cnn_result = fit_cnn_grouped(
            train_sequences, train_stars["label"], test_sequences, train_stars["camera"],
            seed=args.seed, epochs=args.cnn_epochs, batch_size=args.cnn_batch_size,
        )
        result.report["cnn_experiment"] = _evaluate_cnn_blend(
            result, cnn_result, train_stars["label"], train_stars["camera"]
        )
        np.save(output / "cnn_test_probabilities.npy", cnn_result.test_probabilities)
        np.save(output / "cnn_camera_grouped_probabilities.npy", cnn_result.grouped_oof_probabilities)
        np.save(output / "cnn_camera_grouped_shifted_probabilities.npy", cnn_result.shifted_oof_probabilities)
        torch.save({
            "channels": CNN_CHANNELS,
            "labels": ["planet", "eclipsing_binary", "background_binary", "pulsating_star", "flare_star", "starspots", "instrument_glitch", "quiet_star"],
            "state_dicts": [{k: v.detach().cpu() for k, v in model.state_dict().items()} for model in cnn_result.models],
        }, output / "cnn_models.pt")
        print(f"CNN camera-grouped macro-F1: {result.report['cnn_experiment']['cnn_camera_grouped_macro_f1_mean']:.4f}")
        print(f"CNN blend accepted: {result.report['cnn_experiment']['blend_accepted']}")
    save_fit_artifacts(result, output)
    sub = make_submission(test_stars, result.labels, template, output / "submission.csv")
    print(f"Stratified OOF macro-F1: {result.report['oof_macro_f1']:.4f}")
    if result.report["camera_grouped_macro_f1"]:
        camera = result.report["camera_grouped_macro_f1"]
        print(f"Camera-grouped macro-F1: {camera['mean']:.4f} ± {camera['std']:.4f}")
    print(f"Down-weighted noisy-label candidates: {result.report['label_noise_candidates_downweighted']}")
    print(f"Wrote {len(sub)} predictions to {output / 'submission.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".", help="Directory containing the supplied CSV and NPY files")
    parser.add_argument("--cache-dir", default="cache", help="Feature cache directory")
    parser.add_argument("--output-dir", default="artifacts", help="Model, validation, and submission output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--ensemble-seeds", type=int, default=3, help="Number of random seeds for the fold-averaged LightGBM ensemble")
    parser.add_argument("--train-cnn", action="store_true", help="Train and validate the optional cadence-level CNN; blend only if grouped and shift gates pass")
    parser.add_argument("--cnn-epochs", type=int, default=25, help="Maximum CNN epochs per camera fold")
    parser.add_argument("--cnn-batch-size", type=int, default=128, help="CNN training and inference batch size")
    parser.add_argument("--no-bls", action="store_true", help="Skip the native-cadence transit search for a quick baseline")
    parser.add_argument("--no-camera-cv", action="store_true", help="Skip camera-grouped validation")
    parser.add_argument("--rebuild-features", action="store_true", help="Discard cached features and recompute them")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
