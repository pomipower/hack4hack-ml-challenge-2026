import numpy as np
import pandas as pd
import time
from scipy.ndimage import median_filter, uniform_filter1d
from joblib import Parallel, delayed

def extract_cadence_v3(cad, slew, tsamp, ch_w):
    cad = cad.astype(np.float32)
    # Mask dropped rows (value 0)
    cad[cad == 0] = np.nan
    # Zero out GBT center-channel spike (channels 63..65)
    cad[:, :, 63:66] = np.nan
    
    # 1. Robust Panel Centering
    p_med = np.nanmedian(cad, axis=(1, 2), keepdims=True)
    cad_centered = cad - p_med
    
    # 2. Robust MAD Scale (Median Absolute Deviation is 100% immune to spikes/glitches!)
    # Standard deviation can be inflated by a single glitch; MAD cannot!
    abs_dev = np.abs(cad_centered)
    cad_mad = float(np.nanmedian(abs_dev)) * 1.4826
    norm = cad_centered / (cad_mad + 1e-4)
    norm_zero = np.nan_to_num(norm, nan=0.0)
    
    f = {}
    on_idx = [0, 2, 4]
    off_idx = [1, 3, 5]
    
    # Standardized panel max & std
    p_max = np.nanmax(norm, axis=(1, 2))
    p_std = np.nanstd(norm, axis=(1, 2))
    p_mean = np.nanmean(norm, axis=(1, 2))
    
    # ALL FEATURES ARE STRICT CONTRASTS (ON vs OFF differences / ratios)
    # Never expose raw single-panel powers to prevent glitch-based splits!
    f['on_max_mean'] = float(np.mean(p_max[on_idx]))
    f['off_max_mean'] = float(np.mean(p_max[off_idx]))
    f['on_off_max_diff'] = f['on_max_mean'] - f['off_max_mean']
    f['on_off_max_ratio'] = (f['on_max_mean'] + 1e-2) / (f['off_max_mean'] + 1e-2)
    
    f['on_std_mean'] = float(np.mean(p_std[on_idx]))
    f['off_std_mean'] = float(np.mean(p_std[off_idx]))
    f['on_off_std_diff'] = f['on_std_mean'] - f['off_std_mean']
    
    f['on_mean_mean'] = float(np.mean(p_mean[on_idx]))
    f['off_mean_mean'] = float(np.mean(p_mean[off_idx]))
    f['on_off_mean_diff'] = f['on_mean_mean'] - f['off_mean_mean']
    
    # 3. Broadband Row Bursts (Pulsar & FRB with GLITCH IMMUNITY)
    row_prof = np.nanmean(norm, axis=2) # (6, 16)
    row_prof = np.nan_to_num(row_prof, nan=0.0)
    # Subtract panel median row to isolate sharp bursts
    row_med = np.median(row_prof, axis=1, keepdims=True)
    row_bursts = row_prof - row_med # (6, 16)
    
    p_burst_max = np.max(row_bursts, axis=1) # (6,)
    b_on = p_burst_max[on_idx] # [b0, b2, b4]
    b_off = p_burst_max[off_idx] # [b1, b3, b5]
    
    b_on_sorted = np.sort(b_on)[::-1]
    b_off_sorted = np.sort(b_off)[::-1]
    
    # Pulsar signature: ALL 3 ON panels must have pulses, and OFF panels must be quiet!
    f['pulsar_on_min'] = float(b_on_sorted[-1]) # min burst among all 3 ON panels
    f['pulsar_off_max'] = float(b_off_sorted[0]) # max burst in OFF panels
    f['pulsar_contrast'] = float(f['pulsar_on_min'] - f['pulsar_off_max'])
    f['pulsar_ratio'] = float((f['pulsar_on_min'] + 1e-2) / (f['pulsar_off_max'] + 1e-2))
    f['pulsar_mean_contrast'] = float(np.mean(b_on) - np.mean(b_off))
    
    # Count pulses exceeding absolute SNR threshold (e.g. 1.0 MAD sigma above row background)
    on_pulse_count = int(np.sum(row_bursts[on_idx] > 1.2))
    off_pulse_count = int(np.sum(row_bursts[off_idx] > 1.2))
    f['on_pulse_count'] = on_pulse_count
    f['off_pulse_count'] = off_pulse_count
    f['net_pulse_count'] = on_pulse_count - off_pulse_count
    
    # FRB signature: Exactly ONE ON panel has a burst, others are quiet
    f['frb_isolation'] = float(b_on_sorted[0] - max(b_on_sorted[1], b_off_sorted[0]))
    f['frb_ratio'] = float((b_on_sorted[0] + 1e-2) / (max(b_on_sorted[1], b_off_sorted[0]) + 1e-2))
    
    # 4. Bandpass-Flattened Spectrum (Narrowband Lines: RFI & ETI)
    prof = np.nanmean(norm, axis=1) # (6, 128)
    prof = np.nan_to_num(prof, nan=0.0)
    bandpass = median_filter(prof, size=(1, 15), mode='nearest')
    lines = prof - bandpass # pure lines above local bandpass
    
    peak_lines = np.max(lines, axis=1) # (6,)
    peak_chs = np.argmax(lines, axis=1) # (6,)
    
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
    
    # 5. Exact Doppler Drift Search (ETI CANDIDATE)
    # Search over integer channel shifts k in [-30, 30] from Panel 0 to Panel 2
    best_eti_contrast = -999.0
    best_eti_on_min = -999.0
    best_eti_off_max = 999.0
    best_k = 0
    zero_drift_contrast = 0.0
    
    for k in range(-30, 31):
        c_min = max(0, -k, -2 * k)
        c_max = min(128, 128 - k, 128 - 2 * k)
        if c_min >= c_max:
            continue
        c_range = np.arange(c_min, c_max)
        
        on0 = lines[0, c_range]
        on2 = lines[2, c_range + k]
        on4 = lines[4, c_range + 2 * k]
        on_min = np.minimum(np.minimum(on0, on2), on4)
        
        c1 = np.clip(np.round(c_range + 0.5 * k).astype(int), 0, 127)
        c3 = np.clip(np.round(c_range + 1.5 * k).astype(int), 0, 127)
        c5 = np.clip(np.round(c_range + 2.5 * k).astype(int), 0, 127)
        off_max = np.maximum(np.maximum(lines[1, c1], lines[3, c3]), lines[5, c5])
        
        contrast = on_min - off_max
        max_idx = np.argmax(contrast)
        c_val = contrast[max_idx]
        
        if k == 0:
            zero_drift_contrast = c_val
            f['zero_drift_on_min'] = float(on_min[max_idx])
            f['zero_drift_off_max'] = float(off_max[max_idx])
            f['zero_drift_contrast'] = float(c_val)
        else:
            if c_val > best_eti_contrast:
                best_eti_contrast = c_val
                best_eti_on_min = on_min[max_idx]
                best_eti_off_max = off_max[max_idx]
                best_k = k
                
    f['eti_best_contrast'] = float(best_eti_contrast)
    f['eti_best_on_min'] = float(best_eti_on_min)
    f['eti_best_off_max'] = float(best_eti_off_max)
    # Convert k channels per 2*(16+slew) rows into physical drift rate [Hz/s]
    # Total time from P0 to P2 is 2*(16+slew)*tsamp seconds
    elapsed_time_s = 2 * (16 + slew) * tsamp
    f['eti_best_drift_hz_s'] = float(best_k * ch_w / (elapsed_time_s + 1e-4))
    f['eti_vs_zero_drift_diff'] = float(best_eti_contrast - zero_drift_contrast)
    
    # 6. Scintillating Star (Patchy 2D blobs across channels)
    on_smooth = uniform_filter1d(norm_zero[on_idx], size=7, axis=2)
    off_smooth = uniform_filter1d(norm_zero[off_idx], size=7, axis=2)
    f['on_smooth_var'] = float(np.var(on_smooth))
    f['off_smooth_var'] = float(np.var(off_smooth))
    f['scint_var_diff'] = f['on_smooth_var'] - f['off_smooth_var']
    f['scint_var_ratio'] = (f['on_smooth_var'] + 1e-4) / (f['off_smooth_var'] + 1e-4)
    f['scint_flux_diff'] = float(np.mean(norm_zero[on_idx]) - np.mean(norm_zero[off_idx]))
    
    # 8 sub-bands of 16 channels
    sub_on = np.mean(norm_zero[on_idx].reshape(3, 16, 8, 16), axis=(1, 3))
    sub_off = np.mean(norm_zero[off_idx].reshape(3, 16, 8, 16), axis=(1, 3))
    sub_diffs = np.mean(sub_on, axis=0) - np.mean(sub_off, axis=0)
    f['scint_subband_max'] = float(np.max(sub_diffs))
    f['scint_subband_std'] = float(np.std(sub_diffs))
    
    # 7. Radar Chirp (Sawtooth gradients & diagonal correlation)
    grad_all = np.diff(norm_zero, axis=1) # (6, 15, 128)
    p_grad_stds = np.std(grad_all, axis=(1, 2))
    f['radar_grad_mean'] = float(np.mean(p_grad_stds))
    f['radar_grad_cv'] = float(np.std(p_grad_stds) / (np.mean(p_grad_stds) + 1e-4))
    
    # 8. Satellite Pass (Energy concentration + steep row drift)
    sorted_p_max = np.sort(p_max)[::-1]
    f['sat_top1_ratio'] = float(sorted_p_max[0] / (sorted_p_max[2] + 1e-3))
    f['sat_top2_ratio'] = float((sorted_p_max[0] + sorted_p_max[1]) / (np.sum(sorted_p_max[2:]) + 1e-3))
    p_best = int(np.argmax(p_max))
    best_row_peaks = np.argmax(norm_zero[p_best], axis=1)
    f['sat_row_shift'] = float(abs(best_row_peaks[-1] - best_row_peaks[0]))
    
    # 9. Pure Noise Metric
    f['noise_symmetry'] = float(abs(f['on_mean_mean'] - f['off_mean_mean']) + abs(f['on_std_mean'] - f['off_std_mean']))
    
    return f

def extract_features_v3_parallel(X, meta, n_jobs=14):
    n = len(X)
    print(f'Extracting {n} cadences with {n_jobs} cores...')
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(extract_cadence_v3)(
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

if __name__ == '__main__':
    train_meta = pd.read_csv('train_meta.csv')
    test_meta = pd.read_csv('test_meta.csv')
    X_train = np.load('train_cadences.npy', mmap_mode='r')
    X_test = np.load('test_cadences.npy', mmap_mode='r')
    
    print('Testing v3 feature extractor on 200 train and 200 test samples...')
    f_tr = [extract_cadence_v3(X_train[i], train_meta.loc[i, 'slew_gap_rows'], train_meta.loc[i, 'tsamp_s'], train_meta.loc[i, 'channel_width_hz']) for i in range(200)]
    f_te = [extract_cadence_v3(X_test[i], test_meta.loc[i, 'slew_gap_rows'], test_meta.loc[i, 'tsamp_s'], test_meta.loc[i, 'channel_width_hz']) for i in range(200)]
    
    df_tr = pd.DataFrame(f_tr)
    df_te = pd.DataFrame(f_te)
    
    print('\nKey feature comparison (Train vs Test):')
    for col in ['on_max_mean', 'pulsar_contrast', 'frb_isolation', 'scint_var_diff', 'rfi_consensus_min', 'eti_best_contrast', 'noise_symmetry']:
        tr_m = df_tr[col].median()
        te_m = df_te[col].median()
        print(f'{col:20s}: Train={tr_m:7.3f} | Test={te_m:7.3f} | Ratio={te_m/(tr_m+1e-4):5.2f}')
