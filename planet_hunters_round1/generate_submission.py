"""
Planet Hunters Challenge - Hack4Hack 2026 Round 1
High-Performing Calibrated Ensemble Pipeline
- Combines domain-engineered features on light-curve statistics and centroids
- 5-Fold Stratified Tree Ensemble (ExtraTrees + RandomForest with balanced class weights)
- Blended with precomputed LightGBM + CNN multi-model probabilities
- Calibrated class multipliers to eliminate the 'Quiet Star Trap'
- Enforces competition priority hierarchy tie-breaking
"""

import pickle
import numpy as np
import pandas as pd
from pandas.core.arrays.string_ import StringArray
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


class CompatibleStringArray(StringArray):
    """Compatibility handler for pandas string array unpickling."""
    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            super().__init__(np.asarray(state[1], dtype=object))
        else:
            super().__setstate__(state)


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'StringArray':
            return CompatibleStringArray
        return super().find_class(module, name)


def load_features():
    print("Loading precomputed feature caches...")
    with open('planet_hunters_round1/cache/train_features.pkl', 'rb') as f:
        df_train = CustomUnpickler(f).load()
    with open('planet_hunters_round1/cache/test_features.pkl', 'rb') as f:
        df_test = CustomUnpickler(f).load()
    return df_train, df_test


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inject astronomy-specific physical interaction features."""
    f = df.copy()
    
    # 1. Background binary identifier: centroid shift scaled by contamination and noise
    if 'bls_centroid_shift' in f and 'contamination' in f:
        f['centroid_shift_x_contam'] = f['bls_centroid_shift'] * f['contamination']
        cen_noise = f.get('centroid_column_std', 0.0) + f.get('centroid_row_std', 0.0) + 1e-6
        f['centroid_shift_snr'] = f['bls_centroid_shift'] / cen_noise
    
    # 2. Planet vs Binary shape & depth
    if 'bls_depth' in f and 'ptp_noise' in f:
        f['depth_snr_ratio'] = f['bls_depth'] / (f['ptp_noise'] + 1e-6)
    if 'bls_depth' in f and 'tmag' in f:
        f['bls_depth_x_tmag'] = f['bls_depth'] * (f['tmag'] / 10.0)
    
    # 3. Flare asymmetry and duration
    if 'flare_max_snr' in f and 'flare_decay_asym' in f:
        f['flare_sig_strength'] = f['flare_max_snr'] * (1.0 + f['flare_decay_asym'])
    
    # 4. Spot vs Pulsator frequency sharpness
    if 'fft_peak_1_power' in f and 'spectral_entropy' in f:
        f['fft_sharpness'] = f['fft_peak_1_power'] / (f['spectral_entropy'] + 1e-4)
    
    # 5. Glitch thruster correlation
    if 'glitch_jump_max_snr' in f and 'dump_response_mad_ratio_vs_tmag' in f:
        f['glitch_thruster_score'] = f['glitch_jump_max_snr'] * f['dump_response_mad_ratio_vs_tmag']
        
    return f


def build_calibrated_ensemble():
    labels = [
        'planet', 'eclipsing_binary', 'background_binary', 'pulsating_star', 
        'flare_star', 'starspots', 'instrument_glitch', 'quiet_star'
    ]
    label2idx = {l: i for i, l in enumerate(labels)}
    idx2label = {i: l for i, l in enumerate(labels)}
    
    train_stars = pd.read_csv('train_stars.csv')
    test_stars = pd.read_csv('test_stars.csv')
    sample_sub = pd.read_csv('sample_submission.csv')
    y = np.array([label2idx[l] for l in train_stars['label']])
    
    df_train, df_test = load_features()
    df_train_ext = add_domain_features(df_train)
    df_test_ext = add_domain_features(df_test)
    
    feature_cols = [c for c in df_train_ext.columns if c not in ['id', 'label']]
    median_vals = df_train_ext[feature_cols].median()
    X_train = df_train_ext[feature_cols].fillna(median_vals)
    X_test = df_test_ext[feature_cols].fillna(median_vals)
    
    print(f"Features: {len(feature_cols)} | Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # Train 5-fold tree ensemble and predict test set
    test_probs_rf = np.zeros((len(X_test), len(labels)), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (idx_tr, idx_va) in enumerate(skf.split(X_train, y)):
        print(f"  Training fold {fold+1}/5...")
        et = ExtraTreesClassifier(n_estimators=160, max_depth=24, min_samples_leaf=2,
                                  class_weight='balanced', random_state=42 + fold, n_jobs=1)
        rf = RandomForestClassifier(n_estimators=160, max_depth=22, min_samples_leaf=2,
                                    class_weight='balanced', random_state=100 + fold, n_jobs=1)
        et.fit(X_train.iloc[idx_tr], y[idx_tr])
        rf.fit(X_train.iloc[idx_tr], y[idx_tr])
        
        test_probs_rf += (0.5 * et.predict_proba(X_test) + 0.5 * rf.predict_proba(X_test)) / 5.0
        
    # Load teammate's precomputed LightGBM + CNN model probabilities
    p_lgb_cnn = np.load('planet_hunters_round1/artifacts_cnn/test_probabilities_alpha_0.40.npy')
    
    # Multi-model Blend: 55% LightGBM+CNN + 45% Tree ensemble
    p_blend = 0.55 * p_lgb_cnn + 0.45 * test_probs_rf
    
    # Calibrated decision multipliers (counteracting the 'Quiet Star Trap')
    multipliers = np.array([
        1.25, # planet: boost shallow transits
        1.15, # eclipsing_binary
        1.15, # background_binary
        1.00, # pulsating_star
        1.05, # flare_star
        1.00, # starspots
        0.95, # instrument_glitch
        0.72  # quiet_star: penalty to counteract noise-bias
    ])
    
    p_cal = p_blend * multipliers
    preds = np.argmax(p_cal, axis=1)
    
    # Apply astrophysical priority hierarchy:
    # planet > eclipsing_binary > background_binary > pulsating_star > flare_star > starspots > instrument_glitch > quiet_star
    p_planet = p_blend[:, 0]
    p_flare = p_blend[:, 4]
    bls_snr = df_test['bls_snr'].values if 'bls_snr' in df_test else np.zeros(len(df_test))
    
    # 1. Spotted star hosting a planet -> planet
    mask_planet_from_spots = (preds == 5) & (p_planet > 0.20) & (bls_snr > 6.0)
    preds[mask_planet_from_spots] = 0
    
    # 2. Quiet star with legitimate transit dip -> planet
    mask_planet_from_quiet = (preds == 7) & (p_planet > 0.22) & (bls_snr > 5.0)
    preds[mask_planet_from_quiet] = 0
    
    # 3. Flaring red dwarf with spots -> flare_star
    mask_flare_from_spots = (preds == 5) & (p_flare > 0.20)
    preds[mask_flare_from_spots] = 4
    
    pred_labels = [labels[i] for i in preds]
    sub = pd.DataFrame({'id': test_stars['id'], 'label': pred_labels})
    
    # Safety checks
    assert len(sub) == len(sample_sub)
    assert (sub['id'] == sample_sub['id']).all()
    assert set(sub['label']).issubset(set(labels))
    assert sub.isnull().sum().sum() == 0
    
    out_file = 'submission_calibrated_ensemble.csv'
    sub.to_csv(out_file, index=False)
    print(f"\nSaved verified submission to: {out_file}")
    print("\nClass distribution:")
    print(sub['label'].value_counts())
    return sub


if __name__ == '__main__':
    build_calibrated_ensemble()
