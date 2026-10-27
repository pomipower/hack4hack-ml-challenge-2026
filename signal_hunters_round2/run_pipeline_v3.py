import numpy as np
import pandas as pd
import time
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from feature_extractor_v3 import extract_features_v3_parallel

def main():
    print('===========================================================')
    print('   Hack4Hack Round 2 - Upload 2 Pipeline (Glitch-Proof)   ')
    print('   Ensemble: LightGBM + XGBoost + HistGradientBoosting    ')
    print('===========================================================')
    
    train_meta = pd.read_csv('train_meta.csv')
    test_meta = pd.read_csv('test_meta.csv')
    
    train_cache = 'train_features_v3.pkl'
    test_cache = 'test_features_v3.pkl'
    
    # 1. Feature Extraction
    if os.path.exists(train_cache):
        print(f'Loading cached train features from {train_cache}...')
        F_train = pd.read_pickle(train_cache)
        print(f'Train features shape: {F_train.shape}')
    else:
        print('Extracting features from train_cadences.npy (18,000 samples)...')
        X_train = np.load('train_cadences.npy', mmap_mode='r')
        F_train = extract_features_v3_parallel(X_train, train_meta, n_jobs=14)
        F_train.to_pickle(train_cache)
        print(f'Saved {train_cache}!')
        
    if os.path.exists(test_cache):
        print(f'Loading cached test features from {test_cache}...')
        F_test = pd.read_pickle(test_cache)
        print(f'Test features shape: {F_test.shape}')
    else:
        print('Extracting features from test_cadences.npy (10,000 samples)...')
        X_test = np.load('test_cadences.npy', mmap_mode='r')
        F_test = extract_features_v3_parallel(X_test, test_meta, n_jobs=14)
        F_test.to_pickle(test_cache)
        print(f'Saved {test_cache}!')
        
    y = train_meta.label
    classes = np.sort(y.unique())
    print(f'\nTarget classes ({len(classes)}): {list(classes)}')
    
    # Map labels to integer codes 0..8
    label_to_code = {c: i for i, c in enumerate(classes)}
    code_to_label = {i: c for i, c in enumerate(classes)}
    y_encoded = y.map(label_to_code).values
    
    # 2. 5-Fold Stratified Ensemble Training
    print('\n=== STEP 2: 5-Fold Stratified Training (LGBM + XGB + HGB) ===')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_probs = np.zeros((len(y), len(classes)), dtype=np.float32)
    test_probs = np.zeros((len(test_meta), len(classes)), dtype=np.float32)
    
    # Compute class sample weights for XGBoost
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights_all = compute_sample_weight('balanced', y_encoded)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(F_train, y_encoded)):
        t_fold = time.time()
        X_tr, y_tr = F_train.iloc[train_idx], y_encoded[train_idx]
        X_va, y_va = F_train.iloc[val_idx], y_encoded[val_idx]
        sw_tr = sample_weights_all[train_idx]
        
        # Model 1: LightGBM (State of the art GBDT)
        model_lgb = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )
        model_lgb.fit(X_tr, y_tr)
        p_val_lgb = model_lgb.predict_proba(X_va)
        p_test_lgb = model_lgb.predict_proba(F_test)
        
        # Model 2: XGBoost (Exact greedy tree boosting)
        model_xgb = XGBClassifier(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
            verbosity=0
        )
        model_xgb.fit(X_tr, y_tr, sample_weight=sw_tr)
        p_val_xgb = model_xgb.predict_proba(X_va)
        p_test_xgb = model_xgb.predict_proba(F_test)
        
        # Model 3: HistGradientBoosting (Fast histogram binning)
        model_hgb = HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42 + fold
        )
        model_hgb.fit(X_tr, y_tr)
        p_val_hgb = model_hgb.predict_proba(X_va)
        p_test_hgb = model_hgb.predict_proba(F_test)
        
        # Blend probabilities: 45% LightGBM + 35% XGBoost + 20% HGB
        fold_val_probs = 0.45 * p_val_lgb + 0.35 * p_val_xgb + 0.20 * p_val_hgb
        fold_test_probs = 0.45 * p_test_lgb + 0.35 * p_test_xgb + 0.20 * p_test_hgb
        
        oof_probs[val_idx] = fold_val_probs
        test_probs += fold_test_probs / 5.0
        
        fold_preds = np.argmax(fold_val_probs, axis=1)
        fold_f1 = f1_score(y_va, fold_preds, average='macro')
        print(f'Fold {fold+1}/5 completed in {time.time() - t_fold:.1f}s | Macro-F1: {fold_f1:.4f}')
        
    oof_preds_code = np.argmax(oof_probs, axis=1)
    oof_preds_labels = [code_to_label[c] for c in oof_preds_code]
    overall_f1 = f1_score(y, oof_preds_labels, average='macro')
    print(f'\n>>> OVERALL 5-FOLD ENSEMBLE CV MACRO-F1: {overall_f1:.4f} <<<')
    print('\nPer-class CV Classification Report:')
    print(classification_report(y, oof_preds_labels))
    
    # Save probability caches for ensembling with CadenceNet CNN
    np.save('oof_gbdt_probs.npy', oof_probs)
    np.save('test_gbdt_probs.npy', test_probs)
    print('Saved oof_gbdt_probs.npy and test_gbdt_probs.npy!')
    
    # 3. Test Predictions & Submission Generation
    print('\n=== STEP 3: Generate Upload 2 Submission ===')
    test_preds_code = np.argmax(test_probs, axis=1)
    test_preds_labels = [code_to_label[c] for c in test_preds_code]
    
    sub = pd.DataFrame({
        'id': test_meta['id'],
        'label': test_preds_labels
    })
    
    out_file = 'submission_upload2.csv'
    sub.to_csv(out_file, index=False)
    print(f'Saved {out_file} successfully!')
    
    # Validation checks
    sample_sub = pd.read_csv('sample_submission.csv')
    assert len(sub) == 10000, f'Expected 10000 rows, got {len(sub)}'
    assert list(sub.columns) == ['id', 'label'], f'Incorrect columns: {sub.columns}'
    assert (sub['id'] == sample_sub['id']).all(), 'IDs mismatch sample_submission.csv!'
    assert sub['label'].isnull().sum() == 0, 'Null predictions detected!'
    assert set(sub['label']).issubset(set(classes)), f'Invalid labels: {set(sub["label"]) - set(classes)}'
    
    print('\n=== Upload 2 Predicted Test Distribution ===')
    dist = sub['label'].value_counts()
    for lbl, cnt in dist.items():
        pct = cnt / len(sub) * 100
        print(f'  {lbl:20s}: {cnt:5d} ({pct:5.2f}%)')
        
    print('\n>>> ALL 5 VALIDATION CHECKS PASSED PERFECTLY! <<<')
    print(f'Upload 2 File: {os.path.abspath(out_file)}')

if __name__ == '__main__':
    main()
