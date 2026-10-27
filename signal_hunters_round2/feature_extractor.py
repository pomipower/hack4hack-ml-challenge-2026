import numpy as np
import pandas as pd
import time
from scipy.ndimage import median_filter, uniform_filter1d
from joblib import Parallel, delayed

def extract_cadence_features(cad, slew, tsamp, ch_w):
    cad = cad.astype(np.float32)
    # Mask dropped rows (value 0)
    cad[cad == 0] = np.nan
    
    # Mask center-channel spike (present in GBT, absent in PKS)
    cad[:, :, 63:66] = np.nan
    
    # 1. Panel Level Normalization (remove panel gain offsets)
    p_med = np.nanmedian(cad, axis=(1, 2), keepdims=True)
    cad_centered = cad - p_med
    
    # 2. Cadence Noise Standardization (Eliminates GBT vs PKS receiver noise scale shift!)
    cad_std = np.nanstd(cad_centered)
    norm = cad_centered / (cad_std + 1e-4)
    norm_zero = np.nan_to_num(norm, nan=0.0)
    
    f = {}
    on_idx = [0, 2, 4]
    off_idx = [1, 3, 5]
    
    # Per-panel standardized stats
    p_max = np.nanmax(norm, axis=(1, 2))
    p_std = np.nanstd(norm, axis=(1, 2))
    p_mean = np.nanmean(norm, axis=(1, 2))
    
    for p in range(6):
        f[f'p{p}_max'] = float(p_max[p])
        f[f'p{p}_std'] = float(p_std[p])
        
    f['on_max_mean'] = float(np.mean(p_max[on_idx]))
    f['off_max_mean'] = float(np.mean(p_max[off_idx]))
    f['on_off_max_diff'] = f['on_max_mean'] - f['off_max_mean']
    f['on_off_max_ratio'] = f['on_max_mean'] / (f['off_max_mean'] + 1e-3)
    
    f['on_std_mean'] = float(np.mean(p_std[on_idx]))
    f['off_std_mean'] = float(np.mean(p_std[off_idx]))
    f['on_off_std_diff'] = f['on_std_mean'] - f['off_std_mean']
    
    # Satellite metrics: energy concentrated in 1 or 2 panels + steep row shift
    sorted_p_max = np.sort(p_max)[::-1]
    f['sat_top1_ratio'] = float(sorted_p_max[0] / (sorted_p_max[2] + 1e-3))
    f['sat_top2_ratio'] = float((sorted_p_max[0] + sorted_p_max[1]) / (np.sum(sorted_p_max[2:]) + 1e-3))
    f['sat_gini'] = float((sorted_p_max[0] - sorted_p_max[-1]) / (np.mean(p_max) + 1e-3))
    p_best = int(np.argmax(p_max))
    best_row_peaks = np.argmax(norm_zero[p_best], axis=1)
    f['sat_row_shift'] = float(abs(best_row_peaks[-1] - best_row_peaks[0]))
    
    # 3. Bandpass-Flattened Spectrum (Sharp Narrow Lines)
    prof = np.nanmean(norm, axis=1) # (6, 128)
    prof = np.nan_to_num(prof, nan=0.0)
    bandpass = median_filter(prof, size=(1, 15), mode='nearest')
    lines = prof - bandpass # pure lines above local bandpass
    
    peak_lines = np.max(lines, axis=1) # (6,)
    peak_chs = np.argmax(lines, axis=1) # (6,)
    for p in range(6):
        f[f'p{p}_line_peak'] = float(peak_lines[p])
        
    f['on_line_mean'] = float(np.mean(peak_lines[on_idx]))
    f['off_line_mean'] = float(np.mean(peak_lines[off_idx]))
    f['on_off_line_diff'] = f['on_line_mean'] - f['off_line_mean']
    f['on_min_line'] = float(np.min(peak_lines[on_idx]))
    f['off_max_line'] = float(np.max(peak_lines[off_idx]))
    f['eti_line_contrast'] = f['on_min_line'] - f['off_max_line']
    
    # Persistent RFI: peak line at consensus channel across ALL 6 panels
    consensus_ch = int(np.argmax(np.sum(lines, axis=0)))
    consensus_vals = [lines[p, consensus_ch] for p in range(6)]
    f['rfi_consensus_min'] = float(np.min(consensus_vals))
    f['rfi_consensus_mean'] = float(np.mean(consensus_vals))
    f['rfi_consensus_std'] = float(np.std(consensus_vals))
    f['rfi_line_ch_std'] = float(np.std(peak_chs))
    
    # 4. Broadband Row Bursts (Pulsar & FRB)
    row_prof = np.nanmean(norm, axis=2) # (6, 16)
    row_prof = np.nan_to_num(row_prof, nan=0.0)
    # Subtract median along rows for each panel to isolate sharp bursts
    row_med = np.median(row_prof, axis=1, keepdims=True)
    row_bursts = row_prof - row_med # (6, 16)
    
    p_burst_max = np.max(row_bursts, axis=1) # (6,)
    for p in range(6):
        f[f'p{p}_burst_max'] = float(p_burst_max[p])
        
    on_burst_maxes = p_burst_max[on_idx]
    sorted_on_bursts = np.sort(on_burst_maxes)[::-1]
    f['frb_on_ratio'] = float(sorted_on_bursts[0] / (sorted_on_bursts[1] + 1e-2))
    f['frb_on_minus_off'] = float(sorted_on_bursts[0] - np.max(p_burst_max[off_idx]))
    
    # Pulsar: multiple bursts across ON panels
    on_all_bursts = row_bursts[on_idx].flatten() # 48 rows
    on_burst_med = float(np.median(on_all_bursts))
    on_burst_std = float(np.std(on_all_bursts))
    f['pulsar_burst_count'] = int(np.sum(on_all_bursts > (on_burst_med + 2.5 * on_burst_std)))
    f['pulsar_burst_sum'] = float(np.sum(np.maximum(0, on_all_bursts - (on_burst_med + 2.0 * on_burst_std))))
    f['pulsar_on_mean'] = float(np.mean(on_burst_maxes))
    f['pulsar_off_mean'] = float(np.mean(p_burst_max[off_idx]))
    f['pulsar_contrast'] = f['pulsar_on_mean'] - f['pulsar_off_mean']
    
    # 5. Physical Doppler Drift Search
    kappa = tsamp / ch_w
    dot_f_grid = np.array([-0.20, -0.15, -0.10, -0.06, -0.03, -0.015,
                           0.0,
                           0.015, 0.03, 0.06, 0.10, 0.15, 0.20], dtype=np.float32)
    best_eti_contrast = -999.0
    best_eti_on_min = -999.0
    best_eti_off_max = 999.0
    best_eti_drift = 0.0
    zero_drift_contrast = 0.0
    
    for dot_f in dot_f_grid:
        on_p_profs = []
        off_p_profs = []
        for p in range(6):
            p_rows = []
            for r in range(16):
                t_glob = p * (16 + slew) + r
                shift = int(round(dot_f * kappa * t_glob))
                row = norm_zero[p, r]
                if shift != 0:
                    row_s = np.roll(row, -shift)
                    if shift > 0: row_s[-shift:] = 0
                    else: row_s[:-shift] = 0
                else:
                    row_s = row
                p_rows.append(row_s)
            p_sum = np.mean(p_rows, axis=0)
            if p % 2 == 0: on_p_profs.append(p_sum)
            else: off_p_profs.append(p_sum)
            
        on_min_prof = np.minimum(np.minimum(on_p_profs[0], on_p_profs[1]), on_p_profs[2])
        off_max_prof = np.maximum(np.maximum(off_p_profs[0], off_p_profs[1]), off_p_profs[2])
        contrast_prof = on_min_prof - off_max_prof
        max_idx = np.argmax(contrast_prof)
        c_val = contrast_prof[max_idx]
        
        if abs(dot_f) < 1e-4:
            zero_drift_contrast = c_val
            f['zero_drift_on_min'] = float(on_min_prof[max_idx])
            f['zero_drift_off_max'] = float(off_max_prof[max_idx])
            f['zero_drift_contrast'] = float(c_val)
        else:
            if c_val > best_eti_contrast:
                best_eti_contrast = c_val
                best_eti_on_min = on_min_prof[max_idx]
                best_eti_off_max = off_max_prof[max_idx]
                best_eti_drift = dot_f
                
    f['eti_best_contrast'] = float(best_eti_contrast)
    f['eti_best_on_min'] = float(best_eti_on_min)
    f['eti_best_off_max'] = float(best_eti_off_max)
    f['eti_best_drift_hz_s'] = float(best_eti_drift)
    f['eti_vs_zero_drift_diff'] = float(best_eti_contrast - zero_drift_contrast)
    
    # 6. Scintillation (Patchy Texture across channels)
    on_smooth = uniform_filter1d(norm_zero[on_idx], size=5, axis=2)
    off_smooth = uniform_filter1d(norm_zero[off_idx], size=5, axis=2)
    f['on_smooth_var'] = float(np.var(on_smooth))
    f['off_smooth_var'] = float(np.var(off_smooth))
    f['scint_var_diff'] = f['on_smooth_var'] - f['off_smooth_var']
    f['scint_var_ratio'] = f['on_smooth_var'] / (f['off_smooth_var'] + 1e-4)
    f['on_off_flux_diff'] = float(np.mean(norm_zero[on_idx]) - np.mean(norm_zero[off_idx]))
    
    # 8 sub-bands of 16 channels
    sub_on = np.mean(norm_zero[on_idx].reshape(3, 16, 8, 16), axis=(1, 3))
    sub_off = np.mean(norm_zero[off_idx].reshape(3, 16, 8, 16), axis=(1, 3))
    sub_diffs = np.mean(sub_on, axis=0) - np.mean(sub_off, axis=0)
    f['scint_subband_max'] = float(np.max(sub_diffs))
    f['scint_subband_std'] = float(np.std(sub_diffs))
    
    # 7. Radar Chirp: Sawtooth gradients normalized by panel std
    grad_all = np.diff(norm_zero, axis=1) # (6, 15, 128)
    p_grad_stds = np.std(grad_all, axis=(1, 2))
    f['radar_grad_mean'] = float(np.mean(p_grad_stds))
    f['radar_grad_cv'] = float(np.std(p_grad_stds) / (np.mean(p_grad_stds) + 1e-4))
    
    return f

def extract_dataset_features(X, meta, n_jobs=14):
    n = len(X)
    print(f'Extracting {n} cadences with {n_jobs} cores...')
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(extract_cadence_features)(
            X[i],
            meta.loc[i, 'slew_gap_rows'],
            meta.loc[i, 'tsamp_s'],
            meta.loc[i, 'channel_width_hz']
        )
        for i in range(n)
    )
    df = pd.DataFrame(results)
    print(f'Extracted {n} samples in {time.time() - t0:.1f}s, shape: {df.shape}')
    return df
