"""Memory-conscious, sector-specific light-curve preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


@dataclass
class PreprocessedSector:
    rel_flux: np.ndarray
    f_ccd: np.ndarray
    f_flat: np.ndarray
    f_transit: np.ndarray
    f_bls_ready: np.ndarray
    cen_rel: np.ndarray
    cen_ccd: np.ndarray
    orbit_id: np.ndarray
    dump_mask: np.ndarray
    scat_mask: np.ndarray
    gap_mask: np.ndarray
    sigma_star: np.ndarray
    time: np.ndarray


def _robust_scale(x: np.ndarray, axis: int = -1) -> np.ndarray:
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
    return np.maximum(mad, 1e-7)


def find_orbits(time: np.ndarray, quality: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mark flagged/missing cadences and split at the main internal gap."""
    n, t = flux.shape
    frac_nan = np.mean(~np.isfinite(flux), axis=0)
    gap = ((quality.astype(np.int64) & 4) != 0) | (frac_nan > 0.90) | ~np.isfinite(time)
    dt = np.diff(time)
    typical = np.nanmedian(dt[dt > 0]) if np.any(dt > 0) else 1.0 / 48.0
    jump_idx = np.flatnonzero(np.isfinite(dt) & (dt > max(typical * 3.0, 0.08)))
    # Prefer the largest timestamp discontinuity; otherwise use the widest central
    # quality/missing-data run, which is how the provided downlink gap is encoded.
    split = int(t // 2)
    if jump_idx.size:
        split = int(jump_idx[np.argmax(dt[jump_idx])] + 1)
    else:
        transitions = np.diff(np.r_[False, gap, False].astype(np.int8))
        starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
        runs = [(a, b) for a, b in zip(starts, ends) if b - a >= 3 and 0.2 * t < a < 0.8 * t]
        if runs:
            a, b = max(runs, key=lambda ab: ab[1] - ab[0])
            split = (a + b) // 2
    orbit = np.zeros(t, dtype=np.int8)
    orbit[split:] = 1
    orbit[gap] = -1
    return orbit, gap


def _ccd_basis(group_flux: np.ndarray, valid: np.ndarray, n_pca: int = 2) -> np.ndarray:
    """Build a robust shared median plus up to two clipped co-trending modes."""
    m, t = group_flux.shape
    med = np.nanmedian(group_flux, axis=0).astype(np.float32)
    med = np.nan_to_num(med, nan=0.0)
    basis = [med]
    residual = group_flux - med[None, :]
    scale = _robust_scale(residual, axis=1)
    normalized = np.clip(np.nan_to_num(residual / scale, nan=0.0, posinf=0.0, neginf=0.0), -4, 4)
    normalized[:, ~valid] = 0.0
    # Randomized power iteration avoids a large SVD while capturing common CCD modes.
    k = min(n_pca, max(0, min(m - 1, int(valid.sum()) - 1)))
    if k:
        rng = np.random.default_rng(17)
        q = rng.standard_normal((int(valid.sum()), k), dtype=np.float32)
        q, _ = np.linalg.qr(q, mode="reduced")
        for _ in range(4):
            z = normalized[:, valid].T @ (normalized[:, valid] @ q)
            q, _ = np.linalg.qr(z, mode="reduced")
        expanded = np.zeros((q.shape[1], t), dtype=np.float32)
        expanded[:, valid] = q.T.astype(np.float32)
        basis.extend(expanded)
    B = np.asarray(basis, dtype=np.float32)
    # Remove linear dependence and keep unit-norm basis vectors on valid cadences.
    q, _ = np.linalg.qr(B[:, valid].T, mode="reduced")
    out = np.zeros((q.shape[1], t), dtype=np.float32)
    out[:, valid] = q.T.astype(np.float32)
    return out


def _fit_basis(y: np.ndarray, basis: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Vectorized two-pass sigma-clipped least squares for rows against a small basis."""
    x = basis[:, valid].astype(np.float32)
    values = y[:, valid]
    present = np.isfinite(values)
    weights = present.astype(np.float32)
    coeff = np.zeros((y.shape[0], x.shape[0]), dtype=np.float32)
    for _ in range(2):
        gram = np.einsum("nkt,nlt->nkl", x[None, :, :] * weights[:, None, :], np.broadcast_to(x, (y.shape[0],) + x.shape))
        rhs = np.einsum("nkt,nt->nk", x[None, :, :] * weights[:, None, :], np.nan_to_num(values))
        # Small ridge term handles cadence masks that make a row's design singular.
        gram += np.eye(x.shape[0], dtype=np.float32)[None, :, :] * 1e-6
        coeff = np.linalg.solve(gram, rhs[..., None])[..., 0]
        residual = values - coeff @ x
        scale = _robust_scale(residual, axis=1)
        weights = (present & (np.abs(residual) <= 3.5 * scale)).astype(np.float32)
    return coeff @ basis


def _preprocess_arrays(
    flux: np.ndarray,
    centroid: np.ndarray,
    stars: pd.DataFrame,
    cadences: pd.DataFrame,
    *,
    flatten_window: int = 33,
) -> PreprocessedSector:
    flux = np.asarray(flux, dtype=np.float32)
    centroid = np.asarray(centroid, dtype=np.float32)
    time = cadences["time_btjd"].to_numpy(dtype=np.float64)
    quality = cadences["quality"].to_numpy(dtype=np.int32)
    if flux.ndim != 2 or centroid.shape != (flux.shape[0], 2, flux.shape[1]):
        raise ValueError("Flux must be (stars, cadences) and centroid (stars, 2, cadences).")
    if len(stars) != flux.shape[0] or len(cadences) != flux.shape[1]:
        raise ValueError("Metadata lengths do not match the flux arrays.")

    median = np.nanmedian(flux, axis=1, keepdims=True)
    median[~np.isfinite(median) | (np.abs(median) < 1e-12)] = 1.0
    rel = flux / median - 1.0
    orbit, gap = find_orbits(time, quality, rel)
    rel[:, gap] = np.nan

    f_ccd = rel.copy()
    for _, indices in stars.groupby(["camera", "ccd"], sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=np.int64)
        for orb in (0, 1):
            cols = np.flatnonzero(orbit == orb)
            if cols.size < 20:
                continue
            sub = rel[np.ix_(idx, cols)]
            valid = np.mean(np.isfinite(sub), axis=0) > 0.5
            if valid.sum() < 20:
                continue
            basis = _ccd_basis(sub, valid)
            trend = _fit_basis(sub, basis, valid)
            clean = sub - trend
            clean -= np.nanmedian(clean, axis=1, keepdims=True)
            f_ccd[np.ix_(idx, cols)] = clean

    f_flat = np.full_like(f_ccd, np.nan)
    for orb in (0, 1):
        cols = np.flatnonzero(orbit == orb)
        if not cols.size:
            continue
        chunk_size = 256
        for start in range(0, len(f_ccd), chunk_size):
            stop = min(start + chunk_size, len(f_ccd))
            sub = f_ccd[start:stop, cols]
            filled = sub.copy()
            row_med = np.nanmedian(filled, axis=1, keepdims=True)
            row_med[~np.isfinite(row_med)] = 0
            row_scale = _robust_scale(sub, axis=1)
            upper = row_med + 2.5 * row_scale
            # Clip only the trend-estimation view. Keep the original positive
            # excursions in `sub` so flare features retain their amplitudes.
            clipped = np.minimum(filled, upper)
            filled = np.where(np.isfinite(clipped), clipped, row_med)
            trend = median_filter(filled, size=(1, flatten_window), mode="nearest")
            flat = sub - trend
            f_flat[start:stop, cols] = flat

    # Keep the general-purpose 33-cadence view, and make a broader transit view
    # whose baseline window is at least three times the longest searched box.
    transit_window = max(65, flatten_window)
    f_transit = np.full_like(f_ccd, np.nan)
    for orb in (0, 1):
        cols = np.flatnonzero(orbit == orb)
        if not cols.size:
            continue
        for start in range(0, len(f_ccd), 256):
            stop = min(start + 256, len(f_ccd))
            sub = f_ccd[start:stop, cols]
            row_med = np.nanmedian(sub, axis=1, keepdims=True)
            row_med[~np.isfinite(row_med)] = 0
            row_scale = _robust_scale(sub, axis=1)
            clipped = np.minimum(sub, row_med + 2.5 * row_scale)
            filled = np.where(np.isfinite(clipped), clipped, row_med)
            trend = median_filter(filled, size=(1, transit_window), mode="nearest")
            f_transit[start:stop, cols] = sub - trend

    f_bls_ready = prewhiten_fast_sinusoids(f_transit, time, orbit)

    cen_rel = centroid - np.nanmedian(centroid, axis=2, keepdims=True)
    cen_ccd = cen_rel.copy()
    for _, indices in stars.groupby(["camera", "ccd"], sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=np.int64)
        common = np.nanmedian(cen_rel[idx], axis=0, keepdims=True)
        cen_ccd[idx] = cen_rel[idx] - common

    diff = np.diff(f_flat, axis=1)
    sigma = (1.4826 * np.nanmedian(np.abs(diff - np.nanmedian(diff, axis=1, keepdims=True)), axis=1) / np.sqrt(2.0)).astype(np.float32)
    sigma = np.maximum(sigma, 1e-7)
    q = quality.astype(np.int64)
    raw_dump = (q & 1) != 0
    dump = np.convolve(raw_dump.astype(np.int8), np.ones(3, dtype=np.int8), mode="same") > 0
    scatter = (q & 2) != 0
    return PreprocessedSector(rel, f_ccd, f_flat, f_transit, f_bls_ready, cen_rel, cen_ccd, orbit, dump, scatter, gap, sigma, time)


def prewhiten_fast_sinusoids(
    f_flat: np.ndarray,
    time: np.ndarray,
    orbit_id: np.ndarray | None = None,
    *,
    threshold: float = 0.22,
) -> np.ndarray:
    """Subtract the strongest coherent peak above 2 cycles/day for BLS only."""
    n, t = f_flat.shape
    cadence = np.nanmedian(np.diff(time))
    if not np.isfinite(cadence) or cadence <= 0:
        cadence = 1.0 / 48.0
    freq = np.fft.rfftfreq(t, d=cadence)
    if orbit_id is None:
        orbit_id = np.zeros(t, dtype=np.int8)
        orbit_id[t // 2:] = 1
    high = np.flatnonzero(freq >= 2.0)
    ready = f_flat.copy()
    if not high.size:
        return ready
    for start in range(0, n, 256):
        stop = min(n, start + 256)
        for local, row in enumerate(f_flat[start:stop]):
            valid = np.isfinite(row)
            if valid.sum() < 40:
                continue
            good = np.flatnonzero(valid)
            filled = np.interp(np.arange(t), good, row[good]).astype(np.float32)
            filled -= np.mean(filled)
            spec = np.fft.rfft(filled)
            power = spec.real**2 + spec.imag**2
            total = float(np.sum(power[1:])) + 1e-12
            high_power = power[high]
            high_share = float(np.sum(high_power) / total)
            peak_index = int(high[np.argmax(high_power)])
            peak_share = float(power[peak_index] / total)
            if high_share < 0.45 or peak_share < threshold:
                continue
            # Sharp periodic eclipses have strong harmonics; preserve those for BLS.
            harmonic_ratios = []
            for harmonic in (2, 3):
                index = int(round(peak_index * harmonic))
                if index < len(power):
                    harmonic_ratios.append(float(power[index] / (power[peak_index] + 1e-12)))
            if max(harmonic_ratios, default=0.0) > 0.18:
                continue
            candidate_freq = float(freq[peak_index])
            orbit_coefficients = []
            for orb in (0, 1):
                orbit_cols = np.flatnonzero(valid & (orbit_id == orb))
                if orbit_cols.size < 20:
                    continue
                basis = np.exp(-2j * np.pi * candidate_freq * time[orbit_cols])
                orbit_coefficients.append(np.sum(filled[orbit_cols] * basis) / orbit_cols.size)
            if len(orbit_coefficients) == 2:
                amplitudes = [abs(value) for value in orbit_coefficients]
                phase_coherence = np.cos(np.angle(orbit_coefficients[0]) - np.angle(orbit_coefficients[1]))
                if min(amplitudes) / max(amplitudes + [1e-12]) < 0.35 or phase_coherence < 0.0:
                    continue
            sinusoid_spec = np.zeros_like(spec)
            sinusoid_spec[peak_index] = spec[peak_index]
            model = np.fft.irfft(sinusoid_spec, n=t).astype(np.float32)
            dest = ready[start + local]
            dest[valid] -= model[valid]
    return ready


def preprocess_sector(data_dir: str | Path, split: str, *, flatten_window: int = 33) -> tuple[PreprocessedSector, pd.DataFrame, pd.DataFrame]:
    """Load and process one sector. `split` is `train` or `test`; sectors stay isolated."""
    root = Path(data_dir)
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    flux = np.load(root / f"{split}_flux.npy", mmap_mode="r")
    centroid = np.load(root / f"{split}_centroid.npy", mmap_mode="r")
    stars = pd.read_csv(root / f"{split}_stars.csv")
    cadences = pd.read_csv(root / f"{split}_cadences.csv")
    return _preprocess_arrays(flux, centroid, stars, cadences, flatten_window=flatten_window), stars, cadences
