"""CNN experiment that combines raw cadence and period-folded transit views."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from .cnn import CNN_CHANNELS, _SequenceDataset, _predict, _shifted_sequences, write_sequence_cache
from .modeling import LABELS, _global_probabilities, make_submission
from .preprocessing import preprocess_sector


FOLDED_CHANNELS = (
    "global_flux", "global_centroid_col", "global_centroid_row", "global_occupancy",
    "local_flux", "odd_event_flux", "even_event_flux", "secondary_flux",
    "local_centroid_col", "local_occupancy",
)
FOLD_BINS = 256


def _best_fold_center(flux: np.ndarray, time: np.ndarray, valid: np.ndarray, period: float, duration: float) -> float:
    cadence = float(np.nanmedian(np.diff(time)))
    if not np.isfinite(period) or period <= 0 or not np.isfinite(cadence) or cadence <= 0:
        return 0.0
    bins = max(16, int(round(period / cadence)))
    phase = np.floor(np.mod(time - time[0], period) / period * bins).astype(np.int32)
    phase = np.clip(phase, 0, bins - 1)
    count = np.bincount(phase[valid], minlength=bins).astype(np.float64)
    sums = np.bincount(phase[valid], weights=flux[valid], minlength=bins).astype(np.float64)
    width = int(np.clip(round(duration * bins / period), 2, min(15, bins // 4)))
    double_sum = np.r_[sums, sums]
    double_count = np.r_[count, count]
    cs = np.r_[0.0, np.cumsum(double_sum)]
    cc = np.r_[0.0, np.cumsum(double_count)]
    inside_n = cc[np.arange(bins) + width] - cc[np.arange(bins)]
    inside_s = cs[np.arange(bins) + width] - cs[np.arange(bins)]
    means = np.divide(inside_s, inside_n, out=np.full_like(inside_s, np.inf), where=inside_n > 0)
    start = int(np.argmin(means))
    return float(((start + width / 2.0) / bins) % 1.0)


def _bin_mean(values: np.ndarray, bin_ids: np.ndarray, valid: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    count = np.bincount(bin_ids[valid], minlength=n_bins).astype(np.float32)
    total = np.bincount(bin_ids[valid], weights=values[valid], minlength=n_bins).astype(np.float32)
    means = np.divide(total, count, out=np.zeros(n_bins, dtype=np.float32), where=count > 0)
    return means, count


def build_folded_cache(processed: Any, features: pd.DataFrame, path: Path, *, chunk_size: int = 256) -> np.ndarray:
    """Build global, local, odd/even, secondary and centroid folded views."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n, t = processed.f_bls_ready.shape
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, len(FOLDED_CHANNELS), FOLD_BINS))
    periods = features["bls_period"].to_numpy(dtype=np.float64)
    durations = features["bls_duration"].to_numpy(dtype=np.float64)
    time = processed.time
    noise = np.maximum(processed.sigma_star, 1e-7)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        for i in range(start, stop):
            period = float(periods[i])
            duration = float(durations[i])
            row = processed.f_bls_ready[i].astype(np.float32, copy=False)
            valid = np.isfinite(row) & ~processed.gap_mask
            if valid.sum() < 30:
                continue
            center = _best_fold_center(row, time, valid, period, duration)
            flux = np.clip(np.nan_to_num(row / noise[i]), -15, 15)
            col = np.nan_to_num(processed.cen_ccd[i, 0], nan=0.0)
            row_c = np.nan_to_num(processed.cen_ccd[i, 1], nan=0.0)
            phase = np.mod(time - time[0], period) / period
            global_bins = np.minimum((phase * FOLD_BINS).astype(np.int32), FOLD_BINS - 1)
            global_flux, global_count = _bin_mean(flux, global_bins, valid, FOLD_BINS)
            global_col, _ = _bin_mean(col, global_bins, valid, FOLD_BINS)
            global_row, _ = _bin_mean(row_c, global_bins, valid, FOLD_BINS)
            out[i, 0] = np.clip(global_flux, -10, 10)
            out[i, 1] = np.clip(global_col / max(float(np.nanstd(col[valid])), 1e-6), -10, 10)
            out[i, 2] = np.clip(global_row / max(float(np.nanstd(row_c[valid])), 1e-6), -10, 10)
            out[i, 3] = np.clip(global_count / max(float(np.mean(global_count[global_count > 0])) if np.any(global_count > 0) else 1.0, 1.0), 0, 10)

            # Transit-centered local view spans +/- three candidate durations.
            phase_delta = np.mod(phase - center + 0.5, 1.0) - 0.5
            width = max(duration / period, 1.0 / FOLD_BINS)
            local_coordinate = (phase_delta / (6.0 * width) + 0.5) * FOLD_BINS
            local_mask = valid & (np.abs(phase_delta) <= 3.0 * width)
            local_bins = np.clip(local_coordinate.astype(np.int32), 0, FOLD_BINS - 1)
            local_flux, local_count = _bin_mean(flux, local_bins, local_mask, FOLD_BINS)
            local_col, _ = _bin_mean(col, local_bins, local_mask, FOLD_BINS)
            event_number = np.floor((time - time[0]) / period + 0.5).astype(np.int64)
            odd_flux, _ = _bin_mean(flux, local_bins, local_mask & ((event_number & 1) == 1), FOLD_BINS)
            even_flux, _ = _bin_mean(flux, local_bins, local_mask & ((event_number & 1) == 0), FOLD_BINS)

            secondary_delta = np.mod(phase - center - 0.5 + 0.5, 1.0) - 0.5
            secondary_mask = valid & (np.abs(secondary_delta) <= 3.0 * width)
            secondary_coordinate = (secondary_delta / (6.0 * width) + 0.5) * FOLD_BINS
            secondary_bins = np.clip(secondary_coordinate.astype(np.int32), 0, FOLD_BINS - 1)
            secondary_flux, _ = _bin_mean(flux, secondary_bins, secondary_mask, FOLD_BINS)

            out[i, 4] = np.clip(local_flux, -10, 10)
            out[i, 5] = np.clip(odd_flux, -10, 10)
            out[i, 6] = np.clip(even_flux, -10, 10)
            out[i, 7] = np.clip(secondary_flux, -10, 10)
            out[i, 8] = np.clip(local_col / max(float(np.nanstd(col[valid])), 1e-6), -10, 10)
            out[i, 9] = np.clip(local_count / max(float(np.mean(local_count[local_count > 0])) if np.any(local_count > 0) else 1.0, 1.0), 0, 10)
    out.flush()
    del out
    return np.load(path, mmap_mode="r")


class _DualDataset:
    def __init__(self, raw: np.ndarray, folded: np.ndarray, labels: np.ndarray | None = None):
        self.raw, self.folded, self.labels = raw, folded, labels

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int):
        import torch
        raw = torch.from_numpy(np.array(self.raw[idx], dtype=np.float32, copy=True))
        folded = torch.from_numpy(np.array(self.folded[idx], dtype=np.float32, copy=True))
        if self.labels is None:
            return raw, folded
        return raw, folded, int(self.labels[idx])


def _build_model(torch: Any):
    nn = torch.nn

    class DualViewCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.raw_encoder = nn.Sequential(
                # Reduce cadence resolution immediately. The phase-folded branch
                # retains transit timing at full phase resolution, while this
                # branch focuses on broader flare, spot, and ramp morphology.
                nn.Conv1d(len(CNN_CHANNELS), 32, 11, stride=4, padding=5, bias=False),
                nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(3, stride=2, padding=1),
                nn.Conv1d(32, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 96, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(96), nn.GELU(),
                nn.Conv1d(96, 128, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(128), nn.GELU(),
            )
            self.fold_encoder = nn.Sequential(
                nn.Conv1d(len(FOLDED_CHANNELS), 32, 9, padding=4, bias=False), nn.BatchNorm1d(32), nn.GELU(),
                nn.MaxPool1d(2), nn.Conv1d(32, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 96, 5, padding=2, bias=False), nn.BatchNorm1d(96), nn.GELU(),
            )
            self.pool = nn.ModuleList([nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)])
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear((128 + 96) * 2, 160), nn.GELU(),
                                      nn.Dropout(0.2), nn.Linear(160, len(LABELS)))

        def forward(self, raw, folded):
            a, b = self.raw_encoder(raw), self.fold_encoder(folded)
            return self.head(torch.cat([self.pool[0](a), self.pool[1](a), self.pool[0](b), self.pool[1](b)], dim=1))

    return DualViewCNN()


def _dual_predict(model: Any, raw: np.ndarray, folded: np.ndarray, indices: np.ndarray, device: Any, batch_size: int) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(_DualDataset(raw[indices], folded[indices]), batch_size=batch_size, shuffle=False, num_workers=0)
    outputs = []
    model.eval()
    with torch.no_grad():
        for raw_batch, fold_batch in loader:
            logits = model(raw_batch.to(device), fold_batch.to(device))
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _dual_fit(raw: np.ndarray, folded: np.ndarray, y: np.ndarray, train_idx: np.ndarray, *, seed: int,
              device: Any, epochs: int, batch_size: int, patience: int = 5):
    import torch
    from torch.utils.data import DataLoader
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    ids = np.asarray([LABELS.index(label) for label in y], dtype=np.int64)
    sub_train, sub_val = next(StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=seed).split(train_idx, ids[train_idx]))
    fit_idx, val_idx = train_idx[sub_train], train_idx[sub_val]
    class_weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=ids[fit_idx])
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    model = _build_model(torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=2e-4)
    loader = DataLoader(_DualDataset(raw[fit_idx], folded[fit_idx], ids[fit_idx]), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(_DualDataset(raw[val_idx], folded[val_idx], ids[val_idx]), batch_size=batch_size, shuffle=False, num_workers=0)
    best_loss, best_epoch, stale = float("inf"), 1, 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        for raw_batch, fold_batch, target in loader:
            raw_batch, fold_batch, target = raw_batch.to(device), fold_batch.to(device), target.to(device)
            # Small multiplicative amplitude changes and noise regularize the
            # sequence branch against sector-level scale/noise changes.
            if np.random.random() < 0.5:
                raw_batch[:, :3] = raw_batch[:, :3] * torch.empty((len(raw_batch), 1, 1), device=device).uniform_(0.9, 1.1)
                raw_batch[:, :2] += torch.randn_like(raw_batch[:, :2]) * 0.025
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(raw_batch, fold_batch), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for raw_batch, fold_batch, target in val_loader:
                raw_batch, fold_batch, target = raw_batch.to(device), fold_batch.to(device), target.to(device)
                total += float(loss_fn(model(raw_batch, fold_batch), target).item()) * len(target)
                count += len(target)
        val_loss = total / max(1, count)
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def train_grouped(raw: np.ndarray, folded: np.ndarray, labels: np.ndarray, cameras: np.ndarray,
                  test_raw: np.ndarray, test_folded: np.ndarray, *, seed: int = 42,
                  epochs: int = 5, batch_size: int = 128,
                  checkpoint_dir: Path | None = None) -> dict[str, Any]:
    import torch
    y = np.asarray(labels, dtype=str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = list(GroupKFold(n_splits=min(4, len(np.unique(cameras)))).split(raw, y, groups=cameras))
    grouped = np.zeros((len(y), len(LABELS)), dtype=np.float32)
    test_probs = np.zeros((len(test_raw), len(LABELS)), dtype=np.float32)
    models, scores, best_epochs = [], [], []
    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        model, best_epoch = _dual_fit(raw, folded, y, train_idx, seed=seed + fold, device=device,
                                      epochs=epochs, batch_size=batch_size)
        grouped[val_idx] = _dual_predict(model, raw, folded, val_idx, device, batch_size)
        prediction = np.asarray(LABELS, dtype=object)[np.argmax(grouped[val_idx], axis=1)]
        score = float(f1_score(y[val_idx], prediction, labels=LABELS, average="macro", zero_division=0))
        scores.append(score)
        best_epochs.append(best_epoch)
        test_probs += _dual_predict(model, test_raw, test_folded, np.arange(len(test_raw)), device, batch_size) / len(splits)
        models.append(model)
        print(f"Folded CNN camera fold {fold}/{len(splits)}: {score:.4f} (epoch {best_epoch})", flush=True)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            np.save(checkpoint_dir / "camera_grouped_probabilities_partial.npy", grouped)
            np.save(checkpoint_dir / "test_probabilities_partial.npy", test_probs * (len(splits) / fold))
            import torch
            torch.save({"state_dicts": [{k: v.detach().cpu() for k, v in item.state_dict().items()} for item in models],
                        "completed_folds": fold}, checkpoint_dir / "models_partial.pt")
    return {"grouped_oof": grouped, "test_probs": test_probs, "fold_scores": scores,
            "best_epochs": best_epochs, "models": models, "device": str(device)}


def _make_caches(data_dir: Path, cache_dir: Path, split: str, rebuild: bool = False) -> tuple[np.ndarray, np.ndarray]:
    raw_path = cache_dir / f"{split}_cnn_sequences.npy"
    fold_path = cache_dir / f"{split}_folded_sequences.npy"
    feat_path = cache_dir / f"{split}_features.pkl"
    if not raw_path.exists() or rebuild:
        processed, stars, _ = preprocess_sector(data_dir, split)
        raw = write_sequence_cache(processed, raw_path)
    else:
        processed = None
        raw = np.load(raw_path, mmap_mode="r")
        stars = pd.read_csv(data_dir / f"{split}_stars.csv")
    features = pd.read_pickle(feat_path)
    if not fold_path.exists() or rebuild:
        if processed is None:
            processed, _, _ = preprocess_sector(data_dir, split)
        folded = build_folded_cache(processed, features, fold_path)
    else:
        folded = np.load(fold_path, mmap_mode="r")
    return raw, folded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", default="artifacts_folded_cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    data_dir, cache_dir, output_dir = Path(args.data_dir).resolve(), Path(args.cache_dir).resolve(), Path(args.output_dir).resolve()
    train_raw, train_folded = _make_caches(data_dir, cache_dir, "train", args.rebuild_cache)
    test_raw, test_folded = _make_caches(data_dir, cache_dir, "test", args.rebuild_cache)
    stars = pd.read_csv(data_dir / "train_stars.csv")
    test_stars = pd.read_csv(data_dir / "test_stars.csv")
    template = pd.read_csv(data_dir / "sample_submission.csv")
    result = train_grouped(train_raw, train_folded, stars["label"].to_numpy(), stars["camera"].to_numpy(),
                           test_raw, test_folded, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                           checkpoint_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "camera_grouped_probabilities.npy", result["grouped_oof"])
    np.save(output_dir / "test_probabilities.npy", result["test_probs"])
    torch_payload = {"raw_channels": CNN_CHANNELS, "folded_channels": FOLDED_CHANNELS,
                     "labels": LABELS, "state_dicts": [{k: v.detach().cpu() for k, v in model.state_dict().items()} for model in result["models"]]}
    import torch
    torch.save(torch_payload, output_dir / "models.pt")
    report = {"model": "raw-cadence plus folded-view 1D CNN", "fold_bins": FOLD_BINS,
              "folded_channels": FOLDED_CHANNELS, "device": result["device"],
              "camera_grouped_macro_f1_by_fold": result["fold_scores"],
              "camera_grouped_macro_f1_mean": float(np.mean(result["fold_scores"])),
              "best_epochs": result["best_epochs"]}
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    labels = np.asarray(LABELS, dtype=object)[np.argmax(result["test_probs"], axis=1)]
    make_submission(test_stars, labels, template, output_dir / "cnn_only_submission.csv")
    print(f"Folded CNN camera-grouped macro-F1: {report['camera_grouped_macro_f1_mean']:.4f}")


if __name__ == "__main__":
    main()
