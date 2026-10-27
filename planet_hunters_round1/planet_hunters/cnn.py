"""Optional multi-view 1D CNN for cadence-level light-curve experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from .modeling import LABELS
from .preprocessing import PreprocessedSector


CNN_CHANNELS = (
    "flat_flux_over_noise", "transit_flux_over_noise", "flat_flux_milli_relative",
    "centroid_column_robust_scaled", "centroid_row_robust_scaled", "valid_cadence_mask",
    "log_noise_scale",
)


def write_sequence_cache(processed: PreprocessedSector, path: str | Path, *, chunk_size: int = 256) -> np.ndarray:
    """Write compact model inputs to a memory-mappable .npy file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    n, t = processed.f_flat.shape
    out = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(n, len(CNN_CHANNELS), t))
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        flat = processed.f_flat[start:stop].astype(np.float32, copy=True)
        transit = processed.f_transit[start:stop].astype(np.float32, copy=True)
        valid = np.isfinite(flat) & np.isfinite(transit)
        noise = np.maximum(processed.sigma_star[start:stop, None], 1e-6).astype(np.float32)
        flat_safe = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        transit_safe = np.nan_to_num(transit, nan=0.0, posinf=0.0, neginf=0.0)
        out[start:stop, 0] = np.clip(flat_safe / noise, -15, 15)
        out[start:stop, 1] = np.clip(transit_safe / noise, -15, 15)
        out[start:stop, 2] = np.clip(flat_safe * 1000.0, -10, 10)
        out[start:stop, 5] = valid.astype(np.float32)
        out[start:stop, 6] = np.clip((np.log10(noise) + 5.0) / 4.0, -1, 1)
        for axis in range(2):
            cent = processed.cen_ccd[start:stop, axis].astype(np.float32, copy=True)
            med = np.nanmedian(cent, axis=1, keepdims=True)
            scale = 1.4826 * np.nanmedian(np.abs(cent - med), axis=1, keepdims=True)
            scale = np.maximum(scale, 1e-6)
            out[start:stop, 3 + axis] = np.clip(np.nan_to_num((cent - med) / scale), -10, 10)
    out.flush()
    del out
    return np.load(target, mmap_mode="r")


class _SequenceDataset:
    def __init__(self, sequences: np.ndarray, labels: np.ndarray | None = None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        import torch

        x = torch.from_numpy(np.array(self.sequences[index], dtype=np.float32, copy=True))
        if self.labels is None:
            return x
        return x, int(self.labels[index])


def _build_model(torch: Any):
    nn = torch.nn

    class LightCurveCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(len(CNN_CHANNELS), 32, kernel_size=11, stride=2, padding=5, bias=False),
                nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(3, stride=2, padding=1),
                nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(96), nn.GELU(),
                nn.Conv1d(96, 128, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(128), nn.GELU(),
            )
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)
            self.head = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.30), nn.Linear(256, 128), nn.GELU(),
                nn.Dropout(0.20), nn.Linear(128, len(LABELS)),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.head(torch.cat([self.avg_pool(z), self.max_pool(z)], dim=1))

    return LightCurveCNN()


def _predict(model: Any, sequences: np.ndarray, indices: np.ndarray, device: Any, batch_size: int) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(_SequenceDataset(sequences[indices]), batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device, non_blocking=True))
            values.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float32)


def _fit_model(
    sequences: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    *,
    seed: int,
    device: Any,
    epochs: int,
    batch_size: int,
    validation_indices: np.ndarray | None = None,
    patience: int = 5,
) -> tuple[Any, int]:
    import copy
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    label_ids = np.asarray([LABELS.index(label) for label in y], dtype=np.int64)
    if validation_indices is None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=seed)
        fit_indices, val_indices = next(splitter.split(train_indices, label_ids[train_indices]))
        fit_indices, val_indices = train_indices[fit_indices], train_indices[val_indices]
    else:
        fit_indices, val_indices = train_indices, validation_indices
    classes = np.arange(len(LABELS))
    weights = compute_class_weight("balanced", classes=classes, y=label_ids[fit_indices])
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    model = _build_model(torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    train_loader = DataLoader(_SequenceDataset(sequences[fit_indices], label_ids[fit_indices]), batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(_SequenceDataset(sequences[val_indices], label_ids[val_indices]), batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    stale = 0
    for epoch in range(epochs):
        model.train()
        for batch, target in train_loader:
            batch = batch.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        total_loss, n_rows = 0.0, 0
        with torch.no_grad():
            for batch, target in val_loader:
                batch, target = batch.to(device, non_blocking=True), target.to(device, non_blocking=True)
                total_loss += float(loss_fn(model(batch), target).item()) * len(target)
                n_rows += len(target)
        val_loss = total_loss / max(n_rows, 1)
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def _shifted_sequences(sequences: np.ndarray, indices: np.ndarray, seed: int) -> np.ndarray:
    """Bounded raw-input stress: noisier flux and larger centroid motion."""
    rng = np.random.default_rng(seed)
    shifted = np.array(sequences[indices], dtype=np.float32, copy=True)
    shifted[:, 0:2] += rng.normal(0, 0.08, size=shifted[:, 0:2].shape).astype(np.float32)
    shifted[:, 3:5] *= 1.2
    return shifted


@dataclass
class CNNFitResult:
    test_probabilities: np.ndarray
    grouped_oof_probabilities: np.ndarray
    shifted_oof_probabilities: np.ndarray
    fold_indices: list[np.ndarray]
    fold_scores: list[float]
    shifted_fold_scores: list[float]
    best_epochs: list[int]
    models: list[Any]
    device: str


def fit_cnn_grouped(
    train_sequences: np.ndarray,
    labels: pd.Series | np.ndarray,
    test_sequences: np.ndarray,
    camera: pd.Series | np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 25,
    batch_size: int = 128,
    n_splits: int = 4,
) -> CNNFitResult:
    """Train group-held-out CNN models, then a full-data model for test inference."""
    import torch

    y = np.asarray(labels, dtype=str)
    groups = np.asarray(camera)
    if len(train_sequences) != len(y) or len(y) != len(groups):
        raise ValueError("CNN sequence rows, labels, and camera groups must align.")
    if train_sequences.ndim != 3 or train_sequences.shape[1] != len(CNN_CHANNELS):
        raise ValueError(f"Expected sequence input shaped (stars, {len(CNN_CHANNELS)}, cadences).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_count = min(n_splits, len(np.unique(groups)))
    if fold_count < 2:
        raise ValueError("CNN camera-grouped validation needs at least two camera groups.")
    splitter = GroupKFold(n_splits=fold_count)
    grouped_oof = np.zeros((len(y), len(LABELS)), dtype=np.float32)
    shifted_oof = np.zeros_like(grouped_oof)
    test_probs = np.zeros((len(test_sequences), len(LABELS)), dtype=np.float32)
    fold_indices, fold_scores, shifted_scores, best_epochs, models = [], [], [], [], []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(train_sequences, y, groups=groups), start=1):
        model, best_epoch = _fit_model(train_sequences, y, train_idx, seed=seed + fold, device=device,
                                       epochs=epochs, batch_size=batch_size)
        probs = _predict(model, train_sequences, val_idx, device, batch_size)
        shifted = _shifted_sequences(train_sequences, val_idx, seed + 100 + fold)
        import torch.utils.data
        loader = torch.utils.data.DataLoader(_SequenceDataset(shifted), batch_size=batch_size, shuffle=False, num_workers=0)
        model.eval()
        chunks = []
        with torch.no_grad():
            for batch in loader:
                chunks.append(torch.softmax(model(batch.to(device)), dim=1).cpu().numpy())
        shifted_probs = np.concatenate(chunks, axis=0).astype(np.float32)
        grouped_oof[val_idx] = probs
        shifted_oof[val_idx] = shifted_probs
        fold_indices.append(val_idx)
        fold_scores.append(float(f1_score(y[val_idx], np.asarray(LABELS)[np.argmax(probs, axis=1)], average="macro", labels=LABELS, zero_division=0)))
        shifted_scores.append(float(f1_score(y[val_idx], np.asarray(LABELS)[np.argmax(shifted_probs, axis=1)], average="macro", labels=LABELS, zero_division=0)))
        best_epochs.append(best_epoch)
        test_probs += _predict(model, test_sequences, np.arange(len(test_sequences)), device, batch_size) / n_splits
        models.append(model)
        print(f"CNN camera fold {fold}/{fold_count}: macro-F1={fold_scores[-1]:.4f}, best_epoch={best_epoch}")

    full_epochs = max(1, int(np.median(best_epochs)))
    all_indices = np.arange(len(y))
    full_model, _ = _fit_model(train_sequences, y, all_indices, seed=seed + 500, device=device,
                               epochs=full_epochs, batch_size=batch_size, validation_indices=all_indices, patience=full_epochs + 1)
    test_probs = 0.5 * (test_probs + _predict(full_model, test_sequences, np.arange(len(test_sequences)), device, batch_size))
    models.append(full_model)
    return CNNFitResult(test_probs, grouped_oof, shifted_oof, fold_indices, fold_scores, shifted_scores, best_epochs, models, str(device))
