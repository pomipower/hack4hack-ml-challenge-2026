import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
import os
import time

def main():
    print('=================================================================')
    print('   Hack4Hack Round 2 - Blended Ensemble (GBDT + CadenceNet CNN)  ')
    print('=================================================================')
    
    train_meta = pd.read_csv('train_meta.csv')
    test_meta = pd.read_csv('test_meta.csv')
    sample_sub = pd.read_csv('sample_submission.csv')
    
    classes = np.sort(train_meta['label'].unique())
    label_to_code = {c: i for i, c in enumerate(classes)}
    code_to_label = {i: c for i, c in enumerate(classes)}
    y = train_meta['label'].map(label_to_code).values
    
    # 1. Load OOF and Test probabilities
    assert os.path.exists('oof_gbdt_probs.npy'), 'Missing oof_gbdt_probs.npy!'
    assert os.path.exists('test_gbdt_probs.npy'), 'Missing test_gbdt_probs.npy!'
    assert os.path.exists('oof_cnn_probs.npy'), 'Missing oof_cnn_probs.npy!'
    assert os.path.exists('test_cnn_probs.npy'), 'Missing test_cnn_probs.npy!'
    
    p_gbdt_oof = np.load('oof_gbdt_probs.npy')
    p_gbdt_test = np.load('test_gbdt_probs.npy')
    
    p_cnn_oof = np.load('oof_cnn_probs.npy')
    p_cnn_test = np.load('test_cnn_probs.npy')
    
    # 2. Individual Model Scores
    preds_gbdt = np.argmax(p_gbdt_oof, axis=1)
    f1_gbdt = f1_score(y, preds_gbdt, average='macro')
    print(f'1. GBDT Ensemble (LGBM + XGB + HGB) 5-Fold CV Macro-F1: {f1_gbdt:.4f}')
    
    preds_cnn = np.argmax(p_cnn_oof, axis=1)
    f1_cnn = f1_score(y, preds_cnn, average='macro')
    print(f'2. CadenceNet 2D CNN 5-Fold CV Macro-F1:                 {f1_cnn:.4f}')
    
    # 3. Grid Search Blending Weight w: P = w * P_gbdt + (1 - w) * P_cnn
    print('\nEvaluating linear blend weights w * GBDT + (1 - w) * CNN:')
    best_w = 0.5
    best_blend_f1 = 0.0
    
    for w in np.linspace(0.1, 0.9, 17):
        p_blend = w * p_gbdt_oof + (1.0 - w) * p_cnn_oof
        preds = np.argmax(p_blend, axis=1)
        score = f1_score(y, preds, average='macro')
        print(f'  w = {w:.2f} (GBDT) + {1-w:.2f} (CNN) -> CV Macro-F1: {score:.4f}')
        if score > best_blend_f1:
            best_blend_f1 = score
            best_w = w
            
    print(f'\n>>> OPTIMAL BLEND WEIGHT: {best_w:.2f} GBDT + {1-best_w:.2f} CNN <<<')
    print(f'>>> BEST BLENDED CV MACRO-F1: {best_blend_f1:.4f} <<<')
    
    # Classification Report of the Best Blend
    best_p_oof = best_w * p_gbdt_oof + (1.0 - best_w) * p_cnn_oof
    best_oof_preds = np.argmax(best_p_oof, axis=1)
    print('\nPer-class CV Classification Report for Optimal Blend:')
    print(classification_report(y, best_oof_preds, target_names=classes))
    
    # 4. Generate Upload 3 Submission
    best_p_test = best_w * p_gbdt_test + (1.0 - best_w) * p_cnn_test
    test_preds = np.argmax(best_p_test, axis=1)
    test_labels = [code_to_label[c] for c in test_preds]
    
    sub = pd.DataFrame({
        'id': test_meta['id'],
        'label': test_labels
    })
    
    out_file = 'submission_upload3.csv'
    sub.to_csv(out_file, index=False)
    print(f'\nSaved {out_file} successfully!')
    
    # Validation checks
    assert len(sub) == 10000, f'Expected 10000 rows, got {len(sub)}'
    assert list(sub.columns) == ['id', 'label'], f'Incorrect columns: {sub.columns}'
    assert (sub['id'] == sample_sub['id']).all(), 'IDs mismatch sample_submission.csv!'
    assert sub['label'].isnull().sum() == 0, 'Null predictions detected!'
    assert set(sub['label']).issubset(set(classes)), f'Invalid labels: {set(sub["label"]) - set(classes)}'
    
    print('\n=== Upload 3 Predicted Test Distribution ===')
    dist = sub['label'].value_counts()
    for lbl, cnt in dist.items():
        pct = cnt / len(sub) * 100
        print(f'  {lbl:20s}: {cnt:5d} ({pct:5.2f}%)')
        
    print('\n>>> ALL 5 VALIDATION CHECKS PASSED PERFECTLY! <<<')

if __name__ == '__main__':
    main()
