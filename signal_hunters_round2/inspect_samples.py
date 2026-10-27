import numpy as np
import pandas as pd

train = pd.read_csv('train_meta.csv')
X = np.load('train_cadences.npy', mmap_mode='r')

for label in sorted(train.label.unique()):
    idx = train.index[train.label == label][:3]
    count = (train.label == label).sum()
    pct = count / len(train) * 100
    print(f"\n=== Label: {label} ({count} samples, {pct:.2f}%) ===")
    for i in idx:
        cid = train.loc[i, 'id']
        slew = train.loc[i, 'slew_gap_rows']
        tsamp = train.loc[i, 'tsamp_s']
        ch_w = train.loc[i, 'channel_width_hz']
        cad = X[i].astype(np.float32)
        cad[cad == 0] = np.nan
        
        # Flatten bandpass: subtract channel median across time
        # cad shape: (6, 16, 128)
        ch_median = np.nanmedian(cad, axis=1, keepdims=True) # (6, 1, 128)
        cad_flat = cad - ch_median
        
        # Panel max along channels
        p_max = np.nanmax(cad_flat, axis=(1, 2))
        p_std = np.nanstd(cad_flat, axis=(1, 2))
        
        # Row profile (broadband)
        row_prof = np.nanmean(cad_flat, axis=2) # (6, 16)
        row_max = np.nanmax(row_prof, axis=1)
        
        # Channel profile (time-averaged)
        ch_prof = np.nanmean(cad_flat, axis=1) # (6, 128)
        peak_ch = np.nanargmax(ch_prof, axis=1)
        peak_val = np.nanmax(ch_prof, axis=1)
        
        on_pmax = [f"{p_max[p]:.1f}" for p in (0, 2, 4)]
        off_pmax = [f"{p_max[p]:.1f}" for p in (1, 3, 5)]
        on_chs = [peak_ch[p] for p in (0, 2, 4)]
        off_chs = [peak_ch[p] for p in (1, 3, 5)]
        
        # Check drift linearity across ON panels:
        # Panel 0 to 2 gap is 2*(16 + slew) rows
        # Panel 2 to 4 gap is 2*(16 + slew) rows
        drift_02 = (peak_ch[2] - peak_ch[0])
        drift_24 = (peak_ch[4] - peak_ch[2])
        
        print(f"  ID: {cid} | Slew: {slew} | ON max: {on_pmax} | OFF max: {off_pmax}")
        print(f"      ON chs: {on_chs} (drift 0->2: {drift_02}, 2->4: {drift_24}, diff: {drift_24 - drift_02}) | OFF chs: {off_chs}")
        print(f"      Row max (ON): {[round(row_max[p], 2) for p in (0, 2, 4)]} | OFF: {[round(row_max[p], 2) for p in (1, 3, 5)]}")
