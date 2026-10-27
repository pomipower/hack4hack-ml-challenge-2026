# Planet Hunters: Current Cross-Sector Classifier Architecture

**Status:** Current candidate is a feature-based LightGBM ensemble with a gated 1D CNN blend.  
**Dataset:** 18,000 labeled training stars and 7,000 test stars; 1,315 cadences per light curve.  
**Target:** Eight-class classification, evaluated with macro-F1.  
**Current recommended candidate:** `artifacts_cnn/submission_alpha_0.40.csv`, a separately written 40% CNN blend. The earlier LightGBM-only artifacts remain in `artifacts/`.

An experimental dual-view CNN is implemented in `planet_hunters/folded_cnn.py`. Its local GTX 1650 training was stopped before completing a camera fold because it remained at full GPU use for over 20 minutes without producing a validation score. It is not part of the current candidate. The portable [Planet_Hunters_Colab_CNN.ipynb](Planet_Hunters_Colab_CNN.ipynb) runs this experiment on Colab and saves intermediate fold checkpoints.

> Scores below come from the saved local reports as of 2026-10-22. The 0.7006 blend score is camera-grouped cross-validation, not a competition leaderboard score. The previously reported leaderboard score of 0.7192 belongs to the earlier LightGBM-only submission; no leaderboard result for the CNN blend is recorded here.

## 1. System overview

```text
train/test flux + centroids + star and cadence metadata
                         │
                         ▼
       independent, sector-specific preprocessing
     normalization → quality masks → CCD co-trending
                         │
             ┌───────────┴────────────┐
             ▼                        ▼
    76-row tabular features     7-channel cadence sequences
             │                        │
             ▼                        ▼
  LightGBM: stratified OOF,      optional 1D CNN:
  camera CV and 3-seed          camera-grouped training
  fold/full-data ensemble       and test inference
             └───────────┬────────────┘
                         ▼
    cross-fitted probability blend (α = 0.5 here)
                         │
                         ▼
           argmax → template-ordered submission
```

Train and test sectors are preprocessed independently using their own cadence metadata and detector groups. Test features may contribute to unsupervised preprocessing and the train-vs-test domain diagnostic. Test labels are never available; no pseudo-labeling is used.

## 2. Input and preprocessing

### Inputs and alignment

- `train_flux.npy`: `(18000, 1315)`; `test_flux.npy`: `(7000, 1315)`.
- Centroid arrays have two channels (column and row) per star and cadence.
- Each sector has its own stars and cadence CSV files. Array rows are checked against metadata order; test IDs are checked against `sample_submission.csv`.
- Flux and centroid arrays are opened with NumPy memory mapping and processed in `float32` where practical.

### Cleaning and views

1. Divide each star's flux by its median and center it around zero.
2. Build masks for momentum dump (`quality & 1`), scattered light (`quality & 2`), and gaps (`quality & 4`, invalid times, and mostly missing cadences). The main orbit split is inferred from time discontinuities or the internal gap.
3. For each camera/CCD and orbit, estimate a shared median trend and up to two co-trending modes. Fit each star robustly and subtract the shared component. Remove shared camera/CCD centroid motion too.
4. Create a 33-cadence flattened view for general morphology. Positive outliers above 2.5 robust sigma are clipped only in the trend-estimation copy so flare amplitudes remain in the signal view.
5. Create a broader, at-least-65-cadence transit view and a BLS-ready view with a coherent fast sinusoid removed where the heuristic identifies one. Transit-like harmonic structure is protected from that pre-whitening.
6. Preserve invalid cadences as masks; compute per-star noise scale from robust point-to-point differences.

Raw camera/CCD IDs, raw sector quality-flag rates, and raw missing-data fraction are not model inputs. Star-specific responses during flagged vs clean cadences are normalized relative to stars of similar T magnitude in the same sector.

## 3. Feature model: LightGBM

`extract_features(...)` produces **76 numeric features per star** in the saved candidate run. Feature families:

- **Robust variability/noise:** range and percentiles, MAD/spread, skew/kurtosis, lagged point-to-point noise and noise scale relative to typical stars of similar T magnitude.
- **Flag response:** relative flux-MAD response during momentum dumps and scattered-light intervals, normalized against similar-magnitude stars.
- **Frequency:** orbit-windowed FFT to avoid bridging the downlink gap; peak frequencies and powers, dominant period, normalized spectral entropy, harmonic ratios, and power fractions in 0.03–0.5, 0.5–2, 2–8, and 8–Nyquist cycles/day bands. Orbit-to-orbit peak frequency, amplitude ratio, and phase coherence are included.
- **Transit candidates:** custom native-cadence folded box search over 0.5–13.7 days, sampled in frequency and locally refined. Candidate features include SNR, period, depth, duration, odd/even difference, secondary dip, centroid shift, orbit coverage, expected stellar-density duration ratio, harmonic gain, inner/full depth, edge scatter, and dump overlap above phase-shifted chance. Candidate events are masked for baseline re-estimation and, for SNR > 10, excluded from selected spectral features.
- **Flares:** event counts, maximum SNR, decay asymmetry, integrated positive excess, and isolated single-cadence spikes.
- **Instrument glitches:** abrupt jump magnitude/counts, orbit slope and negative-tail differences, exponential-recovery template similarity, and centroid derivative maximum.
- **Centroid/context:** centroid variability/range, flux-centroid correlation, orbit-level difference, and stellar catalog properties `tmag`, `teff`, `logg`, `radius`, `contamination`.

### LightGBM configuration and fitting

- Multiclass LightGBM, 500 estimators, learning rate 0.04, 31 leaves, minimum child samples 25, row/column subsampling 0.85, L2 regularization 2.0.
- Balanced sample weights are calculated within each training fold.
- Five-fold stratified OOF predictions are generated for the ordinary validation report.
- Four-fold camera-grouped validation holds out camera groups to estimate transfer robustness.
- Final tabular test probabilities average 5 stratified fold models plus a full-data model for each of 3 seeds: **18 models total**.
- OOF-suspect labels are identified conservatively (given-label probability <0.05 and another label >0.90), capped at 2% and down-weighted in fitting. The run marked 284 candidates (1.58% of training data).
- Class multipliers are tuned cross-fitted by camera and shrunk toward 1. They are accepted only with at least +0.0075 macro-F1 and no camera fold regression greater than 0.02. In this run all multipliers stayed at 1.0.

## 4. Optional sequence model: 1D CNN

The optional CNN is a compact 1D convolutional classifier over the ordered cadence sequence. Each star has 1,315 timesteps and seven input channels:

1. 33-cadence flattened flux divided by noise scale.
2. Transit-view flux divided by noise scale.
3. Flattened flux in milli-relative units.
4. Robust-scaled centroid column.
5. Robust-scaled centroid row.
6. Valid-cadence mask.
7. Log noise-scale channel.

The network uses four Conv1D blocks (32, 64, 96, 128 channels), global average and max pooling, dropout and a dense eight-class head. It uses class-weighted cross entropy, AdamW, gradient clipping, early stopping on a stratified slice from the training cameras, and no time reversal augmentation. Camera-grouped folds evaluate transfer to held-out cameras; the CNN is trained on GPU when CUDA is available.

### Blend gate and current blend

CNN and LightGBM probabilities are blended as `(1 − α) × LightGBM + α × CNN`. Candidate α values are 0.25, 0.5, 0.75, and 1.0; cross-fitting chooses the value without evaluating a camera fold on itself. The blend is accepted only if grouped macro-F1 improves by at least 0.003, no fold drops by more than 0.02, and synthetic-shift macro-F1 decreases by no more than 0.02.

The original saved run used a coarse grid and selected **α = 0.5**. A later audit used 0.05 increments between 0 and 0.5; **α = 0.40** was the best fixed blend and is the recommended separately written candidate. Future pipeline runs use this denser conservative grid.

## 5. Validation and metrics

### Headline scores

| Model/measure | Result | Validation meaning |
|---|---:|---|
| LightGBM stratified OOF macro-F1 | **0.7005** | Pooled out-of-fold training predictions |
| LightGBM camera-grouped macro-F1 | **0.6927 ± 0.0079** | Four folds, each holding out camera group(s) |
| CNN camera-grouped macro-F1 | **0.5551** | CNN alone; four held-out-camera folds |
| Original α=0.50 LightGBM + CNN blend | **0.7006** | Camera-grouped score from the original coarse blend search |
| Recommended α=0.40 LightGBM + CNN blend | **0.7035** | Fixed camera-grouped score; +0.0108 vs LightGBM alone |
| Nested cross-fitted dense blend | **0.7031** | α selected on other camera groups (0.40, 0.40, 0.40, 0.45) |
| Recommended blend shifted-score change | **+0.0088** | Difference vs LightGBM under synthetic feature-level stress |
| LightGBM synthetic-shift macro-F1 | **0.6845** | Noise/centroid/low-frequency/dump-overlap feature stress |
| Train-vs-test adversarial AUC | **0.8488** | Domain distinguishability diagnostic, not classifier score |
| Historical leaderboard score | **0.7192** | Earlier LightGBM-only upload; not the current blend |

Camera-grouped LightGBM fold scores: **0.7042, 0.6819, 0.6931, 0.6915**.  
Recommended α=0.40 blend fold scores: **0.7062, 0.7003, 0.7058, 0.7018**.  
CNN-only camera folds: **0.5538, 0.5817, 0.5325, 0.5524**.  
CNN-only synthetic-shift folds: **0.5402, 0.5493, 0.4787, 0.5240**.

Stratified LightGBM fold scores: **0.6930, 0.6974, 0.6976, 0.7105, 0.7033**. Three seed-level ensemble OOF scores: **0.7022, 0.7014, 0.6991**.

The blend's cross-fitted camera fold scores are computed by selecting α on the other camera folds and evaluating on the held-out fold. They are the appropriate local score for the blend comparison. Per-class diagnostics for LightGBM, CNN, and blend are stored in `artifacts_cnn/model_audit.json`. Ungated multiplier fitting looked promising in-sample (+0.0028), but failed nested camera validation (−0.0008), so multipliers remain disabled. The leaderboard result for the blend is not yet recorded.

### LightGBM stratified OOF per-class report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| planet | 0.6263 | 0.4993 | 0.5556 | 2,145 |
| eclipsing_binary | 0.8571 | 0.6778 | 0.7570 | 2,247 |
| background_binary | 0.7009 | 0.5873 | 0.6391 | 1,604 |
| pulsating_star | 0.8442 | 0.7496 | 0.7941 | 1,973 |
| flare_star | 0.8256 | 0.6746 | 0.7425 | 1,804 |
| starspots | 0.5925 | 0.6145 | 0.6033 | 2,882 |
| instrument_glitch | 0.8412 | 0.8158 | 0.8283 | 1,721 |
| quiet_star | 0.5763 | 0.8405 | 0.6838 | 3,624 |
| **Macro average** | **0.7330** | **0.6824** | **0.7005** | **18,000** |
| Accuracy | — | — | 0.6918 | 18,000 |
| Weighted F1 | — | — | 0.6926 | 18,000 |

### Confusion matrix (LightGBM stratified OOF)

Rows are true labels and columns predicted labels, in this order: `planet`, `eclipsing_binary`, `background_binary`, `pulsating_star`, `flare_star`, `starspots`, `instrument_glitch`, `quiet_star`.

```text
[[1071,  43,  86,  28,  31, 355,  77, 454],
 [ 168,1523, 237,  23,  14, 122,  13, 147],
 [ 148, 172, 942,  32,  20, 112,  10, 168],
 [  27,   4,   7,1479,  12,  72,  37, 335],
 [  22,   9,   7,  20,1217, 248,  14, 267],
 [ 106,  18,  36,  61, 113,1771,  58, 719],
 [  52,   3,  13,  18,  15,  67,1404, 149],
 [ 116,   5,  16,  91,  52, 242,  56,3046]]
```

Selected two-way confusion totals (both directions combined): planet↔background_binary **234**; planet↔eclipsing_binary **211**; planet↔instrument_glitch **129**; pulsating_star↔starspots **133**.

### Other diagnostics

- Candidate transit duration reached the search cap for: planet 12.54%; eclipsing_binary 19.27%; background_binary 17.02%; pulsating_star 12.98%; flare_star 25.94%; starspots 28.66%; instrument_glitch 26.03%; quiet_star 13.74%.
- The largest train/test domain feature importances were `bls_period` (720), `bls_dump_overlap_excess` (620), `orbit_level_diff` (507), `bls_duration` (313), and `mad_vs_tmag_expected` (205). Importance indicates what distinguishes these two sectors; it does not show that a feature harms class prediction. The high AUC remains a transfer-risk diagnostic to investigate with grouped ablations.
- Synthetic feature stress applied +25% selected noise measures, +20% centroid scales, +0.08 low-frequency power, and stronger dump-overlap excess. This approximates a perturbation; it is not real held-out-sector validation.

## 6. Submission and reproducibility

The recommended candidate is `artifacts_cnn/submission_alpha_0.40.csv`, with 7,000 rows and `id,label` columns ordered to match `sample_submission.csv`; the pipeline checks ID order and permitted labels before writing. The original α=0.50 candidate remains `artifacts_cnn/submission.csv`. LightGBM-only submission and metrics remain under `artifacts/`.

To regenerate the CNN candidate (with dependencies installed):

```powershell
python -m planet_hunters.pipeline --data-dir . --train-cnn --output-dir artifacts_cnn
```

To reproduce the post-training audit or write another separately named blend candidate:

```powershell
python -m planet_hunters.audit --data-dir . --artifacts-dir artifacts_cnn --write-alpha-submission 0.40
```

Key implementation files: `planet_hunters/preprocessing.py`, `planet_hunters/features.py`, `planet_hunters/modeling.py`, `planet_hunters/cnn.py`, and `planet_hunters/pipeline.py`.

## 7. Known limitations and next measurement gaps

- The strongest transfer estimate available is camera-grouped validation within the training sector plus a synthetic feature/input stress. Neither replaces a genuinely separate sector validation set.
- Adversarial AUC is high (0.8488); the top signatures should be tested via camera-grouped feature ablations before changes are accepted.
- CNN-only macro-F1 is well below LightGBM. Its value here is complementary probability information. The audit shows the 40% blend improves flare, starspot, eclipsing-binary, and background-binary F1, while planet F1 changes only slightly.
- The saved classification report/confusion matrix is for LightGBM OOF, so do not attribute its class-level values to the blend.
- Transit injection/recovery checks and real raw-level flag/ramp stress evaluation have not been completed.
- The raw-plus-folded CNN is an unscored experiment until the Colab camera-grouped evaluation completes; keep the α=0.40 submission as the fallback.
- The public score 0.7192 applies to the earlier uploaded LightGBM submission. The new blend's competition score is pending submission/evaluation.
