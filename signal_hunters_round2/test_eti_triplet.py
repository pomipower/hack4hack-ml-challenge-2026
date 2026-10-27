import numpy as np
import pandas as pd
import time

train = pd.read_csv('train_meta.csv')
X = np.load('train_cadences.npy', mmap_mode='r')

def fast_eti_features(cad, slew, tsamp, ch_w):
    ch_med = np.nanmedian(cad, axis=1, keepdims=True)
    norm = np.nan_to_num(cad - ch_med, nan=0.0)
    
    prof = np.mean(norm, axis=1) # (6, 128)
    
    # In each ON panel (0, 2, 4), find top 5 peak channels
    top_k = 6
    p0_top = np.argsort(prof[0])[-top_k:]
    p2_top = np.argsort(prof[2])[-top_k:]
    p4_top = np.argsort(prof[4])[-top_k:]
    
    best_lin_err = 999.0
    best_on_min = -999.0
    best_off_max = 999.0
    best_drift_hz_s = 0.0
    best_contrast = -999.0
    
    delta_rows_02 = 2 * (16 + slew)
    delta_rows_24 = 2 * (16 + slew)
    
    for c0 in p0_top:
        for c2 in p2_top:
            d02 = (c2 - c0)
            for c4 in p4_top:
                d24 = (c4 - c2)
                err = abs(d24 - d02)
                on_vals = [prof[0, c0], prof[2, c2], prof[4, c4]]
                on_min = min(on_vals)
                
                c1 = int(round((c0 + c2) / 2.0))
                c3 = int(round((c2 + c4) / 2.0))
                c5 = int(round(c4 + d24 / 2.0))
                
                off_vals = []
                if 0 <= c1 < 128: off_vals.append(prof[1, c1])
                if 0 <= c3 < 128: off_vals.append(prof[3, c3])
                if 0 <= c5 < 128: off_vals.append(prof[5, c5])
                off_max = max(off_vals) if off_vals else 0.0
                
                contrast = on_min - off_max
                total_drift_hz_s = (c4 - c0) * ch_w / (4 * (16 + slew) * tsamp)
                
                penalized = contrast - 2.0 * err - (6.0 if abs(c4 - c0) == 0 else 0.0)
                if penalized > best_contrast:
                    best_contrast = penalized
                    best_lin_err = err
                    best_on_min = on_min
                    best_off_max = off_max
                    best_drift_hz_s = total_drift_hz_s
                    
    return {
        'eti_best_contrast': best_contrast,
        'eti_lin_err': best_lin_err,
        'eti_on_min': best_on_min,
        'eti_off_max': best_off_max,
        'eti_drift_hz_s': best_drift_hz_s,
    }

t0 = time.time()
print('Testing fast ETI feature extraction:')
for label in ['eti_candidate', 'rfi_intermittent', 'rfi_persistent', 'noise']:
    idx = train.index[train.label == label][:4]
    for i in idx:
        cid = train.loc[i, 'id']
        slew = train.loc[i, 'slew_gap_rows']
        tsamp = train.loc[i, 'tsamp_s']
        ch_w = train.loc[i, 'channel_width_hz']
        res = fast_eti_features(X[i].astype(np.float32), slew, tsamp, ch_w)
        c = res['eti_best_contrast']
        e = res['eti_lin_err']
        on_m = res['eti_on_min']
        off_m = res['eti_off_max']
        d = res['eti_drift_hz_s']
        print(f'{label:16s} ID {cid}: contrast={c:5.1f}, err={e}, ON_min={on_m:5.1f}, OFF_max={off_m:5.1f}, drift={d:+.4f} Hz/s')
print(f'16 samples took {time.time() - t0:.2f}s ({1000*(time.time() - t0)/16:.1f}ms/sample)')
