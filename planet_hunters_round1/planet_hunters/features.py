"""Astronomy-informed tabular features for the Planet Hunters challenge."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from .preprocessing import PreprocessedSector


def _row_skew_kurt(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmean(x, axis=1, keepdims=True)
    centered = x - med
    var = np.nanmean(centered * centered, axis=1)
    scale = np.sqrt(np.maximum(var, 1e-20))
    skew = np.nanmean(centered**3, axis=1) / scale**3
    kurt = np.nanmean(centered**4, axis=1) / scale**4 - 3.0
    return skew.astype(np.float32), kurt.astype(np.float32)


def _mad_1d(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return float(1.4826 * np.nanmedian(np.abs(x - med)))


def _tmag_relative(values: np.ndarray, stars: pd.DataFrame) -> np.ndarray:
    """Normalize a per-star scale against stars of similar apparent brightness."""
    values = np.asarray(values, dtype=np.float32)
    if "tmag" not in stars:
        return values.copy()
    tmag = pd.to_numeric(stars["tmag"], errors="coerce").to_numpy(dtype=np.float64)
    edges = np.unique(np.nanquantile(tmag, np.linspace(0, 1, 17)))
    bins = np.clip(np.searchsorted(edges[1:-1], tmag), 0, max(0, len(edges) - 2))
    expected = np.full(len(values), np.nan, dtype=np.float32)
    global_value = float(np.nanmedian(values))
    for group in np.unique(bins):
        rows = bins == group
        expected[rows] = np.nanmedian(values[rows])
    expected[~np.isfinite(expected) | (expected < 1e-8)] = max(global_value, 1e-8)
    return values / expected


def _basic_features(p: PreprocessedSector, batch_size: int, stars: pd.DataFrame) -> dict[str, np.ndarray]:
    n, t = p.f_flat.shape
    result: dict[str, np.ndarray] = {}
    names = ("std", "mad", "min", "max", "p01", "p05", "p25", "p75", "p95", "p99", "skew", "kurt", "ptp_noise")
    for name in names:
        result[name] = np.full(n, np.nan, dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        x = p.f_flat[start:stop]
        med = np.nanmedian(x, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(x - med), axis=1)
        sk, ku = _row_skew_kurt(x)
        q = np.nanpercentile(x, [1, 5, 25, 75, 95, 99], axis=1)
        dx = np.diff(x, axis=1)
        dx_med = np.nanmedian(dx, axis=1, keepdims=True)
        ptp = 1.4826 * np.nanmedian(np.abs(dx - dx_med), axis=1) / np.sqrt(2.0)
        vals = {
            "std": np.nanstd(x, axis=1), "mad": mad, "min": np.nanmin(x, axis=1), "max": np.nanmax(x, axis=1),
            "p01": q[0], "p05": q[1], "p25": q[2], "p75": q[3], "p95": q[4], "p99": q[5],
            "skew": sk, "kurt": ku, "ptp_noise": ptp,
        }
        for name, value in vals.items():
            result[name][start:stop] = np.asarray(value, dtype=np.float32)
    for k in (3, 6, 12):
        # Difference noise at approximately 1.5, 3, and 6 hour scales.
        d = p.f_flat[:, k:] - p.f_flat[:, :-k]
        med = np.nanmedian(d, axis=1, keepdims=True)
        result[f"noise_lag{k}"] = (1.4826 * np.nanmedian(np.abs(d - med), axis=1) / np.sqrt(2.0)).astype(np.float32)
    result["sigma_star"] = p.sigma_star.astype(np.float32)
    # Normalize each star's noise against the noise floor of stars with similar T magnitude.
    if "tmag" in stars:
        tmag = pd.to_numeric(stars["tmag"], errors="coerce").to_numpy(dtype=np.float64)
        edges = np.unique(np.nanquantile(tmag, np.linspace(0, 1, 17)))
        bins = np.clip(np.searchsorted(edges[1:-1], tmag), 0, max(0, len(edges) - 2))
        expected = np.full(n, np.nan, dtype=np.float32)
        global_mad = float(np.nanmedian(result["mad"]))
        for b in np.unique(bins):
            rows = bins == b
            expected[rows] = np.nanmedian(result["mad"][rows])
        expected[~np.isfinite(expected) | (expected < 1e-10)] = max(global_mad, 1e-10)
        result["mad_vs_tmag_expected"] = result["mad"] / expected
    # These ratios describe each star's response during flagged intervals, not how
    # many such cadences happened to occur in a particular sector.
    for name, mask in (("dump", p.dump_mask), ("scattered", p.scat_mask)):
        ratio = np.ones(n, dtype=np.float32)
        for i, row in enumerate(p.f_ccd):
            flagged = row[mask]
            clean = row[~mask & ~p.gap_mask]
            flagged = flagged[np.isfinite(flagged)]
            clean = clean[np.isfinite(clean)]
            if flagged.size >= 4 and clean.size >= 20:
                ratio[i] = _mad_1d(flagged) / max(_mad_1d(clean), 1e-8)
        result[f"{name}_response_mad_ratio_vs_tmag"] = _tmag_relative(ratio, stars)
    return result


def _fill_row(x: np.ndarray) -> np.ndarray:
    good = np.flatnonzero(np.isfinite(x))
    if not good.size:
        return np.zeros_like(x, dtype=np.float32)
    return np.interp(np.arange(x.size), good, x[good]).astype(np.float32)


def _spectral_features(p: PreprocessedSector, batch_size: int, transit_mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    n, t = p.f_ccd.shape
    cadence = np.nanmedian(np.diff(p.time))
    if not np.isfinite(cadence) or cadence <= 0:
        cadence = 1.0 / 48.0
    freq = np.fft.rfftfreq(t, d=cadence)
    out = {f"fft_peak_{i}_{field}": np.zeros(n, dtype=np.float32) for i in range(1, 4) for field in ("freq", "power")}
    names = (
        "power_spot", "power_gdor", "power_rr", "power_delta", "dominant_freq", "dominant_period",
        "harmonic_2_ratio", "harmonic_3_ratio", "spectral_entropy", "orbit_peak_freq_1",
        "orbit_peak_freq_2", "orbit_peak_freq_delta", "orbit_amplitude_ratio", "orbit_phase_coherence",
    )
    for name in names:
        out[name] = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        orbit_columns = [np.flatnonzero(p.orbit_id == orb) for orb in (0, 1)]
        power = np.zeros((stop - start, len(freq)), dtype=np.float32)
        for i in range(start, stop):
            for cols in orbit_columns:
                if cols.size < 20:
                    continue
                row = p.f_ccd[i, cols].copy()
                if transit_mask is not None:
                    row[transit_mask[i, cols]] = np.nan
                segment = _fill_row(row)
                segment -= np.mean(segment)
                window = np.hanning(len(segment)).astype(np.float32)
                spec = np.fft.rfft(segment * window, n=t)
                power[i - start] += (spec.real**2 + spec.imag**2).astype(np.float32)
        power[:, 0] = 0
        total = np.sum(power, axis=1) + 1e-20
        bands = (
            (freq >= 0.03) & (freq < 0.5),
            (freq >= 0.5) & (freq < 2.0),
            (freq >= 2.0) & (freq < 8.0),
            (freq >= 8.0),
        )
        for name, mask in zip(("power_spot", "power_gdor", "power_rr", "power_delta"), bands):
            out[name][start:stop] = np.sum(power[:, mask], axis=1) / total
        top = np.argpartition(power, kth=max(1, power.shape[1] - 3), axis=1)[:, -3:]
        order = np.take_along_axis(power, top, axis=1).argsort(axis=1)[:, ::-1]
        top = np.take_along_axis(top, order, axis=1)
        for j in range(3):
            f = freq[top[:, j]]
            out[f"fft_peak_{j+1}_freq"][start:stop] = f
            out[f"fft_peak_{j+1}_power"][start:stop] = np.take_along_axis(power, top[:, j:j+1], axis=1)[:, 0] / total
        dom = top[:, 0]
        dominant = freq[dom]
        out["dominant_freq"][start:stop] = dominant
        out["dominant_period"][start:stop] = np.where(dominant > 0, 1.0 / dominant, 0.0)
        df = freq[1] - freq[0]
        for harmonic in (2, 3):
            indices = np.rint(dominant * harmonic / df).astype(int)
            usable = indices < power.shape[1]
            hpower = np.zeros(len(indices), dtype=np.float32)
            hpower[usable] = power[np.flatnonzero(usable), indices[usable]]
            dpower = power[np.arange(len(indices)), dom] + 1e-20
            out[f"harmonic_{harmonic}_ratio"][start:stop] = hpower / dpower
        probability = power / total[:, None]
        out["spectral_entropy"][start:stop] = -np.sum(probability * np.log(probability + 1e-20), axis=1) / np.log(max(2, probability.shape[1]))

    # Compare dominant frequency/amplitude in the two independently observed orbits.
    # Phase is evaluated at the full-curve peak frequency with actual cadence times.
    cadence = float(cadence)
    for i in range(n):
        f0 = float(out["dominant_freq"][i])
        if f0 <= 0:
            continue
        orbit_coefficients: list[complex] = []
        orbit_peaks: list[float] = []
        for orb in (0, 1):
            cols = np.flatnonzero(p.orbit_id == orb)
            if cols.size < 20:
                orbit_coefficients.append(0j)
                orbit_peaks.append(0.0)
                continue
            row = p.f_ccd[i, cols].copy()
            if transit_mask is not None:
                row[transit_mask[i, cols]] = np.nan
            filled = _fill_row(row)
            filled -= np.mean(filled)
            spec = np.fft.rfft(filled)
            freq_orbit = np.fft.rfftfreq(len(filled), d=cadence)
            valid_freq = freq_orbit >= 0.03
            peak_index = np.argmax(np.abs(spec[valid_freq])) if np.any(valid_freq) else 0
            local_freq = freq_orbit[valid_freq][peak_index] if np.any(valid_freq) else 0.0
            orbit_peaks.append(float(local_freq))
            phase_basis = np.exp(-2j * np.pi * f0 * p.time[cols])
            orbit_coefficients.append(complex(np.sum(filled * phase_basis) / len(filled)))
        f1, f2 = orbit_peaks
        out["orbit_peak_freq_1"][i] = f1
        out["orbit_peak_freq_2"][i] = f2
        out["orbit_peak_freq_delta"][i] = abs(f1 - f2)
        amps = [abs(value) for value in orbit_coefficients]
        out["orbit_amplitude_ratio"][i] = min(amps) / max(amps) if max(amps) > 0 else 0.0
        if amps[0] > 0 and amps[1] > 0:
            out["orbit_phase_coherence"][i] = np.cos(np.angle(orbit_coefficients[0]) - np.angle(orbit_coefficients[1]))
    return out


def _flare_features(p: PreprocessedSector) -> dict[str, np.ndarray]:
    n, _ = p.f_flat.shape
    count = np.zeros(n, dtype=np.float32)
    max_snr = np.zeros(n, dtype=np.float32)
    max_decay = np.zeros(n, dtype=np.float32)
    equiv = np.zeros(n, dtype=np.float32)
    single_spikes = np.zeros(n, dtype=np.float32)
    for i, row in enumerate(p.f_flat):
        valid = np.isfinite(row)
        if valid.sum() < 10:
            continue
        noise = max(float(p.sigma_star[i]), 1e-7)
        snr = row / noise
        event = valid & (snr > 3.5)
        edges = np.diff(np.r_[False, event, False].astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for a, b in zip(starts, ends):
            peak = float(np.nanmax(snr[a:b]))
            if b - a == 1:
                single_spikes[i] += 1
                continue
            count[i] += 1
            max_snr[i] = max(max_snr[i], peak)
            equiv[i] += float(np.nansum(np.maximum(row[a:b], 0.0)))
            # Positive value means the post-peak decay spans more samples than rise.
            peak_idx = a + int(np.nanargmax(row[a:b]))
            max_decay[i] = max(max_decay[i], float(max(0, b - 1 - peak_idx) - max(0, peak_idx - a)))
    return {"flare_count": count, "flare_max_snr": max_snr, "flare_decay_asym": max_decay, "flare_equiv_duration": equiv, "single_spike_count": single_spikes}


def _glitch_features(p: PreprocessedSector) -> dict[str, np.ndarray]:
    """Measure abrupt flux and centroid changes without relying on BLS events."""
    n = p.f_ccd.shape[0]
    jump_max = np.zeros(n, dtype=np.float32)
    jump_count = np.zeros(n, dtype=np.float32)
    slope_delta = np.zeros(n, dtype=np.float32)
    orbit_tail_ratio = np.zeros(n, dtype=np.float32)
    recovery_tau = np.zeros(n, dtype=np.float32)
    centroid_derivative = np.zeros(n, dtype=np.float32)
    for i in range(n):
        jumps_by_orbit, slopes, tails = [], [], []
        for orb in (0, 1):
            cols = np.flatnonzero(p.orbit_id == orb)
            row = p.f_ccd[i, cols]
            valid = np.isfinite(row)
            if valid.sum() < 20:
                continue
            values = row[valid]
            diffs = np.diff(row)
            pair_valid = np.isfinite(row[1:]) & np.isfinite(row[:-1])
            diffs = diffs[pair_valid]
            scale = max(float(p.sigma_star[i]), 1e-7)
            snr = np.abs(diffs) / scale
            jumps_by_orbit.append(float(np.max(snr)) if snr.size else 0.0)
            jump_count[i] += float(np.sum(snr > 8.0))
            slopes.append(float(np.nanmedian(diffs)) if diffs.size else 0.0)
            med = float(np.nanmedian(values))
            tails.append(max(0.0, med - float(np.nanpercentile(values, 1))) / scale)
            if snr.size:
                pair_indices = np.flatnonzero(pair_valid)
                k = int(pair_indices[int(np.argmax(snr))]) + 1
                # Fit a small bank of exponential recovery templates after the
                # largest discontinuity; report the best normalized correlation.
                post = row[k:min(len(row), k + 48)]
                post = post[np.isfinite(post)]
                if post.size >= 8:
                    centered = post - np.nanmedian(post[-min(12, len(post)):])
                    centered -= np.nanmean(centered)
                    best_corr = 0.0
                    for tau in (2, 4, 8, 16, 32):
                        template = np.exp(-np.arange(len(centered)) / tau)
                        denom = np.linalg.norm(centered) * np.linalg.norm(template) + 1e-9
                        best_corr = max(best_corr, abs(float(np.dot(centered, template) / denom)))
                    recovery_tau[i] = max(recovery_tau[i], best_corr)
        jump_max[i] = max(jumps_by_orbit, default=0.0)
        if len(slopes) == 2:
            slope_delta[i] = abs(slopes[0] - slopes[1]) / max(float(p.sigma_star[i]), 1e-7)
        if len(tails) == 2:
            orbit_tail_ratio[i] = max(tails) / max(min(tails), 1e-3)
        c = p.cen_ccd[i]
        dc = np.diff(c, axis=1)
        centroid_derivative[i] = float(np.nanmax(np.sqrt(np.nansum(dc * dc, axis=0)))) if np.isfinite(dc).any() else 0.0
    return {
        "glitch_jump_max_snr": jump_max,
        "glitch_jump_count_snr8": jump_count,
        "glitch_orbit_slope_delta": slope_delta,
        "glitch_orbit_tail_ratio": orbit_tail_ratio,
        "glitch_recovery_template_corr": recovery_tau,
        "centroid_derivative_max": centroid_derivative,
    }


def _masked_transit_flatten(curve: np.ndarray, orbit_id: np.ndarray, mask: np.ndarray, window: int = 65) -> np.ndarray:
    """Re-estimate a transit-view baseline without candidate transit cadences."""
    result = curve.copy()
    for orb in (0, 1):
        cols = np.flatnonzero(orbit_id == orb)
        if cols.size < 20:
            continue
        source = curve[cols].astype(np.float64, copy=True)
        source[mask[cols]] = np.nan
        good = np.flatnonzero(np.isfinite(source))
        if good.size < 20:
            continue
        filled = np.interp(np.arange(cols.size), good, source[good])
        trend = median_filter(filled, size=window, mode="nearest")
        result[cols] = curve[cols] - trend
    return result


def _best_box(y: np.ndarray, period: float, cadence: float, time: np.ndarray, valid: np.ndarray, variance: float | None = None) -> tuple[float, float, float, int]:
    """Native-cadence folded box search for one trial period."""
    bins = max(16, int(round(period / cadence)))
    phase = np.floor(np.mod(time - time[0], period) / period * bins).astype(np.int32)
    phase = np.clip(phase, 0, bins - 1)
    w = valid.astype(np.float64)
    count = np.bincount(phase[valid], weights=w[valid], minlength=bins).astype(np.float64)
    sums = np.bincount(phase[valid], weights=y[valid], minlength=bins).astype(np.float64)
    doubled_sum = np.r_[sums, sums]
    doubled_count = np.r_[count, count]
    cs = np.r_[0.0, np.cumsum(doubled_sum)]
    cc = np.r_[0.0, np.cumsum(doubled_count)]
    best_snr, best_depth, best_width, best_start = 0.0, 0.0, 1, 0
    total_count, total_sum = count.sum(), sums.sum()
    if total_count < 20:
        return best_snr, best_depth, cadence, 0
    max_width = min(15, max(2, bins // 4))
    widths = np.arange(2, max_width + 1, dtype=np.int32)[:, None]
    starts = np.arange(bins, dtype=np.int32)[None, :]
    inside_n = cc[widths + starts] - cc[starts]
    inside_s = cs[widths + starts] - cs[starts]
    outside_n = total_count - inside_n
    depth = total_sum / total_count - inside_s / np.maximum(inside_n, 1.0)
    # Matched-filter SNR for the mean-level contrast, robustly scaled outside transit.
    var = (float(np.nanvar(y[valid])) if variance is None else variance) + 1e-14
    snr = depth * np.sqrt(np.maximum(inside_n * outside_n / total_count, 0.0)) / np.sqrt(var)
    wi, j = np.unravel_index(int(np.nanargmax(snr)), snr.shape)
    if snr[wi, j] > best_snr:
        best_snr, best_depth, best_width, best_start = float(snr[wi, j]), float(depth[wi, j]), int(widths[wi, 0]), int(j)
    duration = period * best_width / bins
    return best_snr, best_depth, duration, best_start


def _transit_features(p: PreprocessedSector, stars: pd.DataFrame, coarse_frequencies: int = 48) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Coarse-to-fine BLS-like search over the full 30-minute cadence series."""
    n, t = p.f_flat.shape
    cadence = float(np.nanmedian(np.diff(p.time)))
    if not np.isfinite(cadence) or cadence <= 0:
        cadence = 1.0 / 48.0
    span = float(np.nanmax(p.time) - np.nanmin(p.time))
    fmin, fmax = 1.0 / 13.7, 1.0 / 0.5
    coarse_n = max(coarse_frequencies, int((fmax - fmin) * span))
    coarse = np.linspace(fmin, fmax, coarse_n)
    keys = ("bls_snr", "bls_period", "bls_depth", "bls_duration", "bls_odd_even", "bls_secondary", "bls_centroid_shift", "bls_dump_overlap_excess", "bls_orbit_consistency", "bls_expected_duration_ratio", "bls_harmonic_gain", "bls_inner_full_depth_ratio", "bls_edge_scatter_ratio")
    out = {key: np.zeros(n, dtype=np.float32) for key in keys}
    transit_mask = np.zeros((n, t), dtype=bool)
    out["bls_period"][:] = 13.7
    out["bls_expected_duration_ratio"][:] = 0.0
    time = p.time
    dump = p.dump_mask
    for i in range(n):
        y = p.f_bls_ready[i].astype(np.float64, copy=False)
        valid = np.isfinite(y) & ~p.dump_mask & ~p.gap_mask
        if valid.sum() < 80:
            continue
        y = y - np.nanmedian(y[valid])
        variance = float(np.nanvar(y[valid]))
        scores = np.zeros(coarse.size, dtype=np.float32)
        for j, freq in enumerate(coarse):
            snr, _, _, _ = _best_box(y, 1.0 / freq, cadence, time, valid, variance)
            scores[j] = snr
        step = float(coarse[1] - coarse[0]) if len(coarse) > 1 else 1.0 / span
        top = np.argpartition(scores, -min(4, len(scores)))[-min(4, len(scores)):]
        # Resolve the strongest coarse basins, then repeatedly narrow around the
        # best frequency. The final spacing is much finer than 1/T, which limits
        # transit phase drift across the sector baseline without a huge dense grid.
        refine = np.unique(np.clip(np.concatenate([np.linspace(coarse[j] - step, coarse[j] + step, 9) for j in top]), fmin, fmax))
        best = (0.0, 0.0, cadence, 0, 13.7)
        for freq in refine:
            period = 1.0 / freq
            snr, depth, duration, phase_bin = _best_box(y, period, cadence, time, valid, variance)
            if snr > best[0]:
                best = (snr, depth, duration, phase_bin, period)
        center = 1.0 / best[4]
        for radius in (step / 8.0, step / 64.0):
            local = np.linspace(max(fmin, center - radius), min(fmax, center + radius), 9)
            for freq in local:
                period = 1.0 / freq
                snr, depth, duration, phase_bin = _best_box(y, period, cadence, time, valid, variance)
                if snr > best[0]:
                    best = (snr, depth, duration, phase_bin, period)
            center = 1.0 / best[4]
        snr, depth, duration, phase_bin, period = best
        out["bls_snr"][i] = snr
        out["bls_period"][i] = period
        out["bls_depth"][i] = depth
        out["bls_duration"][i] = duration
        if snr <= 0:
            continue
        harmonic_scores = []
        for harmonic in (period / 2.0, period * 2.0):
            if 0.5 <= harmonic <= 13.7:
                harmonic_scores.append(_best_box(y, harmonic, cadence, time, valid, variance)[0])
        out["bls_harmonic_gain"][i] = max(harmonic_scores, default=snr) - snr
        phase = np.mod(time - time[0], period) / period
        half = max(duration / (2 * period), cadence / period)
        phase_center = (phase_bin / max(16, round(period / cadence)) + half) % 1.0
        phase_delta = np.abs(phase - phase_center)
        circular_delta = np.minimum(phase_delta, 1.0 - phase_delta)
        event_zone = circular_delta <= half
        in_dip = valid & event_zone
        central = in_dip & (circular_delta <= 0.5 * half)
        ingress_egress = in_dip & (circular_delta >= 0.7 * half)
        # Reflatten with every predicted event masked out of the trend estimate.
        # This keeps long boxes from being absorbed by the broad running median.
        shape_y = _masked_transit_flatten(p.f_ccd[i], p.orbit_id, event_zone)
        baseline = valid & ~event_zone
        if baseline.sum() >= 10 and in_dip.sum() >= 2:
            baseline_level = float(np.nanmedian(shape_y[baseline]))
            full_depth = baseline_level - float(np.nanmedian(shape_y[in_dip]))
            if full_depth > 0:
                out["bls_depth"][i] = full_depth
            if central.sum() >= 2 and full_depth > 0:
                inner_depth = baseline_level - float(np.nanmedian(shape_y[central]))
                out["bls_inner_full_depth_ratio"][i] = np.clip(inner_depth / full_depth, 0.0, 3.0)
            if ingress_egress.sum() >= 2 and full_depth > 0:
                edge_scatter = _mad_1d(shape_y[ingress_egress])
                out["bls_edge_scatter_ratio"][i] = edge_scatter / (full_depth + 1e-8)
        if snr > 10 and depth > 0:
            transit_mask[i] = in_dip
        event_num = np.floor((time - time[0]) / period + 0.5).astype(int)
        odd, even = in_dip & (event_num % 2 == 1), in_dip & (event_num % 2 == 0)
        if odd.sum() >= 2 and even.sum() >= 2:
            do, de = -float(np.nanmedian(shape_y[odd])), -float(np.nanmedian(shape_y[even]))
            out["bls_odd_even"][i] = abs(do - de) / (float(p.sigma_star[i]) + 1e-7)
        secondary_center = (phase_center + 0.5) % 1.0
        secondary_delta = np.abs(phase - secondary_center)
        secondary = valid & (np.minimum(secondary_delta, 1.0 - secondary_delta) <= half)
        if secondary.sum() >= 2:
            out["bls_secondary"][i] = max(0.0, -float(np.nanmedian(shape_y[secondary])))
        dip_cad = np.flatnonzero(in_dip)
        event_cad = np.flatnonzero(event_zone)
        if event_cad.size:
            observed_overlap = float(np.mean(dump[event_cad]))
            expected = []
            for offset in (np.arange(32, dtype=np.float64) + 0.5) / 32.0:
                shifted_phase = np.mod(phase - offset, 1.0)
                delta = np.abs(shifted_phase - phase_center)
                zone = np.minimum(delta, 1.0 - delta) <= half
                if zone.any():
                    expected.append(float(np.mean(dump[zone])))
            out["bls_dump_overlap_excess"][i] = observed_overlap - float(np.mean(expected)) if expected else 0.0
        base_cad = valid & ~in_dip
        if dip_cad.size >= 2 and base_cad.sum() >= 10:
            shifts = p.cen_ccd[i][:, dip_cad]
            baseline = np.nanmedian(p.cen_ccd[i][:, base_cad], axis=1)
            shift = np.nanmedian(shifts, axis=1) - baseline
            out["bls_centroid_shift"][i] = float(np.linalg.norm(shift))
            orbit_values = p.orbit_id[dip_cad]
            out["bls_orbit_consistency"][i] = float(len(np.unique(orbit_values[orbit_values >= 0])))
        row = stars.iloc[i]
        radius = float(row.get("radius", np.nan))
        logg = float(row.get("logg", np.nan))
        if np.isfinite(radius) and radius > 0 and np.isfinite(logg):
            rho = 10 ** (logg - 4.438) / radius
            expected = 0.13 * period ** (1 / 3) * max(rho, 1e-5) ** (-1 / 3)
            out["bls_expected_duration_ratio"][i] = duration / max(expected, cadence)
    return out, transit_mask


def extract_features(p: PreprocessedSector, stars: pd.DataFrame, *, batch_size: int = 256, include_bls: bool = True) -> pd.DataFrame:
    """Return one astronomy-aware feature row per star, preserving input order."""
    blocks: dict[str, np.ndarray] = {}
    blocks.update(_basic_features(p, batch_size, stars))
    transit_mask = None
    if include_bls:
        transit_blocks, transit_mask = _transit_features(p, stars)
        blocks.update(transit_blocks)
    blocks.update(_spectral_features(p, batch_size, transit_mask))
    blocks.update(_flare_features(p))
    blocks.update(_glitch_features(p))
    # Cross-centroid diagnostics independent of candidate BLS events.
    for axis, name in enumerate(("column", "row")):
        blocks[f"centroid_{name}_std"] = np.nanstd(p.cen_ccd[:, axis], axis=1).astype(np.float32)
        blocks[f"centroid_{name}_range"] = (np.nanpercentile(p.cen_ccd[:, axis], 95, axis=1) - np.nanpercentile(p.cen_ccd[:, axis], 5, axis=1)).astype(np.float32)
        blocks[f"flux_centroid_{name}_corr"] = np.asarray([np.corrcoef(np.nan_to_num(p.f_flat[i]), np.nan_to_num(p.cen_ccd[i, axis]))[0, 1] for i in range(len(stars))], dtype=np.float32)
    blocks["orbit_level_diff"] = np.zeros(len(stars), dtype=np.float32)
    for orb in (0, 1):
        cols = p.orbit_id == orb
        med = np.nanmedian(p.rel_flux[:, cols], axis=1)
        if orb == 0:
            first = med
        else:
            blocks["orbit_level_diff"] = (med - first).astype(np.float32)
    feat = pd.DataFrame(blocks)
    for col in ("tmag", "teff", "logg", "radius", "contamination"):
        if col in stars:
            feat[col] = pd.to_numeric(stars[col], errors="coerce").to_numpy(dtype=np.float32)
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feat
