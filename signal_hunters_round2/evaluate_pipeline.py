import numpy as np
import pandas as pd
import time
from scipy.ndimage import uniform_filter1d
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def extract_single_cadence(cad, slew, tsamp, ch_w):
    cad = cad.astype(np.float32)
    cad[cad == 0] = np.nan
    
    # Bandpass flattening: subtract median across rows per channel
    ch_med = np.nanmedian(cad, axis=1, keepdims=True) # (6, 1, 128)
    norm = cad - ch_med
    norm_zero = np.nan_to_num(norm, nan=0.0)
    
    f = {}
    on_idx = [0, 2, 4]
    off_idx = [1, 3, 5]
    
    # 1. Per-panel basic stats
    p_std = np.nanstd(norm, axis=(1, 2))
    p_max = np.nanmax(norm, axis=(1, 2))
    p_mean = np.nanmean(norm, axis=(1, 2))
    
    for p in range(6):
        f[f'p{p}_std'] = float(p_std[p])
        f[f'p{p}_max'] = float(p_max[p])
        f[f'p{p}_mean'] = float(p_mean[p])
        
    f['on_max_mean'] = float(np.mean(p_max[on_idx]))
    f['off_max_mean'] = float(np.mean(p_max[off_idx]))
    f['on_off_max_diff'] = f['on_max_mean'] - f['off_max_mean']
    f['on_off_max_ratio'] = f['on_max_mean'] / (f['off_max_mean'] + 1e-3)
    
    f['on_std_mean'] = float(np.mean(p_std[on_idx]))
    f['off_std_mean'] = float(np.mean(p_std[off_idx]))
    f['on_off_std_diff'] = f['on_std_mean'] - f['off_std_mean']
    
    f['on_mean_mean'] = float(np.mean(p_mean[on_idx]))
    f['off_mean_mean'] = float(np.mean(p_mean[off_idx]))
    f['on_off_mean_diff'] = f['on_mean_mean'] - f['off_mean_mean']
    
    # Satellite metrics
    sorted_p_max = np.sort(p_max)[::-1]
    f['sat_top1_ratio'] = float(sorted_p_max[0] / (sorted_p_max[2] + 1e-3))
    f['sat_top2_ratio'] = float((sorted_p_max[0] + sorted_p_max[1]) / (np.sum(sorted_p_max[2:]) + 1e-3))
    f['sat_gini'] = float((sorted_p_max[0] - sorted_p_max[-1]) / (np.mean(p_max) + 1e-3))
    
    # 2. Spectrum and Channel Profiles (Time-Averaged)
    ch_prof = np.nanmean(norm, axis=1) # (6, 128)
    ch_prof_zero = np.nan_to_num(ch_prof, nan=0.0)
    
    peak_chs = np.argmax(ch_prof_zero, axis=1) # (6,)
    peak_vals = np.max(ch_prof_zero, axis=1) # (6,)
    
    for p in range(6):
        f[f'p{p}_peak_val'] = float(peak_vals[p])
        
    f['on_peak_mean'] = float(np.mean(peak_vals[on_idx]))
    f['off_peak_mean'] = float(np.mean(peak_vals[off_idx]))
    f['on_off_peak_diff'] = f['on_peak_mean'] - f['off_peak_mean']
    f['on_min_peak'] = float(np.min(peak_vals[on_idx]))
    f['off_max_peak'] = float(np.max(peak_vals[off_idx]))
    f['eti_peak_contrast'] = f['on_min_peak'] - f['off_max_peak']
    
    # Persistent RFI: peak at same channel across all panels
    f['rfi_ch_std'] = float(np.std(peak_chs))
    f['rfi_all_min'] = float(np.min(peak_vals))
    f['rfi_all_mean'] = float(np.mean(peak_vals))
    
    # Consensus channel across all 6 panels
    consensus_ch = int(np.argmax(np.sum(ch_prof_zero, axis=0)))
    consensus_vals = [ch_prof_zero[p, consensus_ch] for p in range(6)]
    f['rfi_consensus_min'] = float(np.min(consensus_vals))
    f['rfi_consensus_mean'] = float(np.mean(consensus_vals))
    f['rfi_consensus_std'] = float(np.std(consensus_vals))
    
    # Number of panels where peak exceeds 3.5
    f['on_lines_count'] = int(np.sum(peak_vals[on_idx] > 3.5))
    f['off_lines_count'] = int(np.sum(peak_vals[off_idx] > 3.5))
    
    # 3. Row Profiles (Broadband Bursts: FRB & Pulsar)
    row_prof = np.nanmean(norm, axis=2) # (6, 16)
    row_prof_zero = np.nan_to_num(row_prof, nan=0.0)
    
    p_row_max = np.max(row_prof_zero, axis=1) # (6,)
    for p in range(6):
        f[f'p{p}_rowmax'] = float(p_row_max[p])
        
    on_row_maxes = p_row_max[on_idx]
    sorted_on_row_maxes = np.sort(on_row_maxes)[::-1]
    f['frb_on_ratio'] = float(sorted_on_row_maxes[0] / (sorted_on_row_maxes[1] + 1e-2))
    f['frb_on_minus_off'] = float(sorted_on_row_maxes[0] - np.max(p_row_max[off_idx]))
    
    # Total row profiles for ON
    on_all_rows = row_prof_zero[on_idx].flatten() # 48 rows
    on_all_sorted = np.sort(on_all_rows)[::-1]
    f['on_row_top1'] = float(on_all_sorted[0])
    f['on_row_top2'] = float(on_all_sorted[1])
    f['on_row_top3'] = float(on_all_sorted[2])
    f['on_row_top4'] = float(on_all_sorted[3])
    
    on_row_med = float(np.median(on_all_rows))
    on_row_std = float(np.std(on_all_rows))
    f['pulsar_burst_count'] = int(np.sum(on_all_rows > (on_row_med + 2.5 * on_row_std)))
    f['pulsar_burst_sum'] = float(np.sum(np.maximum(0, on_all_rows - (on_row_med + 2.0 * on_row_std))))
    
    # 4. Physical Doppler Drift Search
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
            if p % 2 == 0:
                on_p_profs.append(p_sum)
            else:
                off_p_profs.append(p_sum)
                
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
    
    # 5. Scintillation (Patchy 2D blobs across channels)
    on_smooth = uniform_filter1d(norm_zero[on_idx], size=5, axis=2)
    off_smooth = uniform_filter1d(norm_zero[off_idx], size=5, axis=2)
    f['on_smooth_var'] = float(np.var(on_smooth))
    f['off_smooth_var'] = float(np.var(off_smooth))
    f['scint_var_diff'] = f['on_smooth_var'] - f['off_smooth_var']
    f['scint_var_ratio'] = f['on_smooth_var'] / (f['off_smooth_var'] + 1e-4)
    
    # 8 sub-band fluxes (16 channels each)
    sub_bands_on = np.mean(norm_zero[on_idx].reshape(3, 16, 8, 16), axis=(1, 3)) # (3, 8)
    sub_bands_off = np.mean(norm_zero[off_idx].reshape(3, 16, 8, 16), axis=(1, 3)) # (3, 8)
    sub_band_diffs = np.mean(sub_bands_on, axis=0) - np.mean(sub_bands_off, axis=0) # (8,)
    f['scint_subband_max_diff'] = float(np.max(sub_band_diffs))
    f['scint_subband_std_diff'] = float(np.std(sub_band_diffs))
    
    # 6. Radar Chirp (Sawtooth gradients)
    grad_all = np.diff(norm_zero, axis=1) # (6, 15, 128)
    p_grad_stds = np.std(grad_all, axis=(1, 2))
    f['radar_grad_mean'] = float(np.mean(p_grad_stds))
    f['radar_grad_std'] = float(np.std(p_grad_stds))
    
    return f

def extract_all(X, meta, n_jobs=14):
    n = len(X)
    print(f'Extracting features for {n} cadences using {n_jobs} cores...')
    t0 = time.time()
    
    results = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(extract_single_cadence)(
            X[i],
            meta.loc[i, 'slew_gap_rows'],
            meta.loc[i, 'tsamp_s'],
            meta.loc[i, 'channel_width_hz']
        )
        for i in range(n)
    )
    
    df = pd.DataFrame(results)
    print(f'Done! {n} cadences processed in {time.time() - t0:.1f}s. Shape: {df.shape}')
    return df

if __name__ == '__main__':
    train_meta = pd.read_csv('train_meta.csv')
    X_train = np.load('train_cadences.npy', mmap_mode='r')
    
    # Extract features for all 18,000 train samples
    F_train = extract_all(X_train, train_meta, n_jobs=14)
    F_train.to_pickle('train_features.pkl')
    print('Saved train_features.pkl!')
    
    # Run 5-fold Stratified CV
    print('\nRunning 5-Fold Stratified Cross-Validation with HistGradientBoostingClassifier...')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y = train_meta.label
    
    oof_preds = pd.Series(index=y.index, dtype=object)
    
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(F_train, y)):
        X_tr, y_tr = F_train.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = F_train.iloc[val_idx], y.iloc[val_idx]
        
        clf = HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.06,
            class_weight='balanced',
            random_state=42 + fold
        )
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_va)
        oof_preds.iloc[val_idx] = preds
        score = f1_score(y_va, preds, average='macro')
        fold_scores.append(score)
        print(f'Fold {fold+1} Macro-F1: {score:.4f}')
        
    overall_score = f1_score(y, oof_preds, average='macro')
    print(f'\n>>> Overall 5-Fold Macro-F1: {overall_score:.4f} <<<')
    print('\nDetailed Classification Report:')
    print(classification_report(y, oof_preds))
