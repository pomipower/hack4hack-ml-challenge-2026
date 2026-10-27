import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from feature_extractor import extract_dataset_features

def main():
    print('===========================================================')
    print('   Hack4Hack Round 2 - Upload 1 Generation Pipeline       ')
    print('   Signal Hunters: Telescope-Invariant Domain Ensemble    ')
    print('===========================================================')
    
    train_meta = pd.read_csv('train_meta.csv')
    test_meta = pd.read_csv('test_meta.csv')
    
    # 1. Feature Extraction / Caching
    train_cache = 'train_features_v2.pkl'
    test_cache = 'test_features_v2.pkl'
    
    if os.path.exists(train_cache):
        print(f'Loading cached train features from {train_cache}...')
        F_train = pd.read_pickle(train_cache)
        print(f'Train shape: {F_train.shape}')
    else:
        print('Extracting features from train_cadences.npy (18,000 samples)...')
        X_train = np.load('train_cadences.npy', mmap_mode='r')
        F_train = extract_dataset_features(X_train, train_meta, n_jobs=14)
        F_train.to_pickle(train_cache)
        print(f'Saved {train_cache}!')
        
    if os.path.exists(test_cache):
        print(f'Loading cached test features from {test_cache}...')
        F_test = pd.read_pickle(test_cache)
        print(f'Test shape: {F_test.shape}')
    else:
        print('Extracting features from test_cadences.npy (10,000 samples)...')
        X_test = np.load('test_cadences.npy', mmap_mode='r')
        F_test = extract_dataset_features(X_test, test_meta, n_jobs=14)
        F_test.to_pickle(test_cache)
        print(f'Saved {test_cache}!')
        
    y = train_meta.label
    classes = np.sort(y.unique())
    print(f'\nTarget classes ({len(classes)}): {list(classes)}')
    
    # 2. 5-Fold Stratified Ensemble Training
    print('\n=== STEP 2: Train 5-Fold Multi-Model Ensemble ===')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_probs = np.zeros((len(y), len(classes)), dtype=np.float32)
    test_probs = np.zeros((len(test_meta), len(classes)), dtype=np.float32)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(F_train, y)):
        t_fold = time.time()
        X_tr, y_tr = F_train.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = F_train.iloc[val_idx], y.iloc[val_idx]
        
        # Model 1: HistGradientBoosting (Fast, robust GBDT)
        hgb = HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.06,
            class_weight='balanced',
            random_state=42 + fold
        )
        hgb.fit(X_tr, y_tr)
        p_val_hgb = hgb.predict_proba(X_va)
        p_test_hgb = hgb.predict_proba(F_test)
        
        # Model 2: ExtraTrees (High variance reduction, fast splits)
        et = ExtraTreesClassifier(
            n_estimators=250,
            min_samples_split=4,
            class_weight='balanced',
            n_jobs=14,
            random_state=42 + fold
        )
        et.fit(X_tr, y_tr)
        p_val_et = et.predict_proba(X_va)
        p_test_et = et.predict_proba(F_test)
        
        # Model 3: Random Forest (Deep bagging ensemble)
        rf = RandomForestClassifier(
            n_estimators=250,
            min_samples_split=4,
            class_weight='balanced',
            n_jobs=14,
            random_state=42 + fold
        )
        rf.fit(X_tr, y_tr)
        p_val_rf = rf.predict_proba(X_va)
        p_test_rf = rf.predict_proba(F_test)
        
        # Probability Blending: 50% HGB + 25% ET + 25% RF
        fold_val_probs = 0.50 * p_val_hgb + 0.25 * p_val_et + 0.25 * p_val_rf
        fold_test_probs = 0.50 * p_test_hgb + 0.25 * p_test_et + 0.25 * p_test_rf
        
        oof_probs[val_idx] = fold_val_probs
        test_probs += fold_test_probs / 5.0
        
        fold_preds = classes[np.argmax(fold_val_probs, axis=1)]
        fold_f1 = f1_score(y_va, fold_preds, average='macro')
        print(f'Fold {fold+1}/5 done in {time.time() - t_fold:.1f}s | Macro-F1: {fold_f1:.4f}')
        
    oof_preds = classes[np.argmax(oof_probs, axis=1)]
    overall_f1 = f1_score(y, oof_preds, average='macro')
    print(f'\n>>> OVERALL 5-FOLD CV MACRO-F1: {overall_f1:.4f} <<<')
    print('\nClassification Report (Out-of-Fold):')
    print(classification_report(y, oof_preds))
    
    # 3. Test Predictions & Submission File
    print('\n=== STEP 3: Generate and Validate Upload 1 ===')
    test_preds = classes[np.argmax(test_probs, axis=1)]
    
    sub = pd.DataFrame({
        'id': test_meta['id'],
        'label': test_preds
    })
    
    out_file = 'submission_upload1.csv'
    sub.to_csv(out_file, index=False)
    print(f'Saved {out_file} successfully!')
    
    # Strict validation checks
    sample_sub = pd.read_csv('sample_submission.csv')
    assert len(sub) == 10000, f'Expected 10000 rows, got {len(sub)}'
    assert list(sub.columns) == ['id', 'label'], f'Incorrect columns: {sub.columns}'
    assert (sub['id'] == sample_sub['id']).all(), 'IDs mismatch sample_submission.csv!'
    assert sub['label'].isnull().sum() == 0, 'Null predictions detected!'
    assert set(sub['label']).issubset(set(classes)), f'Invalid labels: {set(sub["label"]) - set(classes)}'
    
    print('\n=== Upload 1 Predicted Distribution ===')
    dist = sub['label'].value_counts()
    for lbl, cnt in dist.items():
        pct = cnt / len(sub) * 100
        print(f'  {lbl:20s}: {cnt:5d} ({pct:5.2f}%)')
        
    print('\n>>> ALL 5 VALIDATION CHECKS PASSED PERFECTLY! <<<')
    print(f'File ready for upload: {os.path.abspath(out_file)}')

if __name__ == '__main__':
    main()
