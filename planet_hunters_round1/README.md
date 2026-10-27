# Planet Hunters cross-sector classifier

This project classifies each light curve into one of the eight Planet Hunters labels. The training stars come from one observing sector and the test stars from another, so the pipeline removes sector-specific instrument trends and uses features that describe the shape and timing of the stellar signals.

The supplied `starter_notebook.ipynb` remains as a quick statistical-feature baseline. The implementation described here is the reproducible command-line pipeline in `planet_hunters/`.

## Setup and run

Use Python 3.10 or newer. From this folder:

```powershell
python -m pip install -r requirements.txt
python -m planet_hunters.pipeline --data-dir .
```

The first run processes both sectors, caches their feature tables in `cache/`, trains and validates LightGBM, and writes `artifacts/submission.csv`. Later runs reuse cached features. Use `--rebuild-features` to recompute them after changing feature code or data; the feature-cache version is also bumped when the schema changes.

To run the optional cadence-level CNN experiment, install PyTorch for your Python/CUDA setup and add `--train-cnn`:

```powershell
python -m planet_hunters.pipeline --data-dir . --train-cnn
```

The CNN run reuses the tabular feature cache, prepares memory-mapped sequence caches, and compares camera-grouped predictions with the LightGBM baseline. It only replaces the submission with a blend when the cross-fitted grouped score and synthetic input-shift checks pass.

For the experimental raw-plus-phase-folded CNN, use [Planet_Hunters_Colab_CNN.ipynb](Planet_Hunters_Colab_CNN.ipynb) in Colab with a GPU runtime. It expects the precomputed sequence and feature caches plus the saved LightGBM ensemble on Drive; the notebook evaluates held-out camera folds before writing a separate candidate.

Useful options:

- `--no-bls` skips the transit search for a quicker baseline.
- `--no-camera-cv` skips camera-grouped validation.
- `--seed`, `--folds`, `--n-estimators`, and `--ensemble-seeds` control the training run. The default is three seeds and five folds, plus one full-data fit per seed (18 final LightGBM models).
- `--train-cnn`, `--cnn-epochs`, and `--cnn-batch-size` enable and configure the optional CNN experiment. Camera-grouped validation is required for this option.

## Pipeline overview

```text
train/test flux + centroids + star/cadence metadata
                         │
                         ▼
     sector-specific normalization and quality masks
                         │
                         ▼
 camera/CCD co-trending → 33-cadence general view + 65-cadence transit view
                         │
                         ▼
 robust stats | orbit-windowed FFT | glitches | flares | transit vetting
                         │
                         ▼
 stratified OOF + camera-group checks → 5×3 LightGBM baseline
                                      └ optional cadence CNN → gated blend
                         │
                         ▼
 probabilities → validated, template-ordered submission.csv
```

## What each stage does

### 1. Load and align the two sectors

`preprocess_sector(data_dir, split)` opens each `.npy` flux array with memory mapping and reads the matching centroid, star, and cadence files. It checks the dimensions against the metadata and keeps the CSV row order aligned with the array rows. Train and test are processed separately, each using its own cadence flags and camera/CCD groups.

### 2. Normalize and remove shared instrument trends

Each star's flux is divided by its median and centered around zero. The pipeline marks momentum-dump cadences (`quality & 1`), scattered-light cadences (`quality & 2`), and data gaps (`quality & 4`, missing times, or mostly missing flux). It finds the main orbit split from the time series and gap flags rather than assuming a fixed cadence index.

For each camera/CCD and orbit, it builds a shared median trend plus up to two clipped co-trending modes. A robust, per-star fit removes those shared modes while limiting the influence of transits, flares, and outliers. Before estimating either running-median trend, positive excursions above 2.5 robust sigma are clipped in a trend-estimation copy; original excursions remain available to flare features. The general feature view uses a 33-cadence window. The transit view uses at least 65 cadences (32.5 hours), then pre-whitens the strongest coherent fast sinusoid while preserving sharp, harmonic-rich dips. Centroids are zero-centered and their common camera/CCD trend is removed as well. Original missing cadences remain masked.

### 3. Extract light-curve features

`extract_features(...)` returns one numeric feature row per star. The feature groups are:

- **Variability and noise:** spread, percentiles, skew/kurtosis, point-to-point noise at several lags, and each star's MAD relative to the sector's typical MAD at similar T magnitude.
- **Flag response:** each star's flux MAD during momentum-dump and scattered-light intervals divided by its clean-cadence MAD, then normalized by the sector median among stars in a similar T-magnitude bin. Raw sector flag rates, missing fraction, and camera/CCD identifiers are excluded from model features because they can encode sector-specific conditions.
- **Frequency:** windowed FFT power is computed separately within each orbit, avoiding interpolation across the long mid-sector gap. Features include peak frequencies and powers, dominant period, spectral entropy, second/third harmonic ratios, and power bands for slow variability (0.03–0.5 cycles/day), 0.5–2, 2–8, and 8–Nyquist. Orbit-specific peak frequency and amplitude/phase stability help separate persistent pulsations from evolving spots. Remaining missing cadences and transit cadences are interpolated within each orbit, so these features remain approximate under extensive masking.
- **Flares:** multi-cadence positive excursions, peak signal-to-noise, decay asymmetry, integrated positive flux excess, and isolated single-point spikes.
- **Transit candidates:** a coarse-to-fine, native-cadence box search over 0.5–13.7 days. Frequencies are sampled uniformly; the four strongest coarse basins are searched locally, followed by two progressively narrower refinements around the best candidate. Each phase bin corresponds to about one native cadence; tested box widths span 1–7.5 hours, not a fixed fraction of phase. After selecting a candidate, the pipeline masks its predicted events while re-estimating the broader transit baseline; depth, shape, odd/even, and secondary-dip features use that reflattened view. Dump overlap is represented as observed overlap minus mean overlap over 32 phase shifts, controlling for the sector's dump rhythm. Other features include centroid displacement, orbit coverage, a stellar-density duration ratio, and inner-to-full depth / ingress-edge shape measures.
- **Instrument glitches:** robust flux-jump maxima/counts, orbit-to-orbit slope and negative-tail differences, exponential-recovery template correlation, and maximum centroid derivative. These features are independent of BLS candidates so jumps and ramps outside candidate transit windows can contribute.
- **Centroid and context:** centroid range and variability, flux-centroid correlation, relative orbit-level flux change, stellar parameters, and the star-specific flag-response ratios above.

The transit search excludes flagged dump cadences when fitting boxes and records overlap relative to chance. It runs before spectral extraction. When it finds a positive transit candidate with SNR above 10, those in-transit cadences are masked from FFT and orbit-stability features. This SNR mask is a first-pass heuristic, not a fully vetted planet decision. The code uses a custom folded box-search feature extractor; it does not call Astropy's BLS implementation.

### 4. Validate and train the classifier

The default run produces five-fold stratified out-of-fold (OOF) predictions and macro-F1 diagnostics, then four-fold `GroupKFold` predictions that hold out a camera at a time. It also trains a train-versus-test domain classifier and reports held-out AUC plus its strongest feature discriminators; these are diagnostics, not class-model features. A bounded feature-level stress case increases noise, centroid scale, low-frequency power, and dump-overlap excess; its score is recorded beside the camera-group score. Class multipliers are clipped to [0.5, 2.0] and shrunk toward 1. Their validation score is cross-fitted by tuning on three camera folds and measuring the fourth, excluding capped suspect labels from calibration. Multipliers are retained only when cross-fitted macro-F1 improves by at least 0.0075 without a camera fold dropping by more than 0.02.

Each fit uses inverse-frequency sample weights for class balance. Training rows are down-weighted only when their OOF probability for the supplied label is below 0.05 while another label is above 0.90; this is capped at 2% of the training set. Final test predictions average five stratified-fold models and a full-training-data model for each of three random seeds, then apply accepted class multipliers. Test-sector flux is used only for unsupervised sector-specific preprocessing, feature extraction, and domain-shift diagnostics; no test labels are inferred or pseudo-labeled. CatBoost is not included in this baseline because its gain has not been measured.

### Optional cadence-level CNN

The `--train-cnn` experiment trains a compact 1D CNN on ordered cadence views: general flattened flux, broad-window transit flux, relative flux scale, robust-scaled column/row centroid curves, a valid-cadence mask, and a noise-scale channel. It uses camera-grouped folds with a stratified early-stopping split inside the training cameras. No time reversal is used. The experiment reports CNN-only and shifted-input fold scores, then cross-fits the blend weight across held-out camera groups. The CNN is blended into test probabilities only when grouped macro-F1 improves by at least 0.003, no camera fold loses more than 0.02, and the synthetic input-shift score drops by no more than 0.02. Otherwise the original LightGBM submission is kept.

The public leaderboard is intended only to break ties between locally validated candidates. A difference under 0.01 macro-F1 is treated as too small to drive a model change.

The validation report includes per-class precision/recall/F1, confusion and selected pair-confusion counts, transit-duration-at-search-cap rates by class, camera-group scores, synthetic feature-shift scores, and sector adversarial-validation AUC. A high domain AUC is a prompt to inspect its strongest features and compare ablations under camera-grouped validation, not by itself a reason to drop a feature. Transit injection/recovery against Astropy BLS and raw-level flag/ramp stress tests remain additional validation work before treating small leaderboard differences as reliable.

### 5. Write the submission

`make_submission(...)` checks that predictions match all test IDs in the exact order of `sample_submission.csv`, contain only the permitted label names, and produce the required `id,label` columns. It writes `artifacts/submission.csv`.

## Outputs

- `cache/train_features.pkl`, `cache/test_features.pkl` and JSON cache signatures
- `artifacts/validation_report.json` with OOF, per-class/pair, camera-grouped, shifted-feature, transit-cap, and sector-domain diagnostics
- `artifacts/oof_probabilities.npy` and `artifacts/test_probabilities.npy`
- `artifacts/lightgbm_ensemble.joblib` (18 fold/full-data seed models by default)
- `artifacts/submission.csv`
- Optional CNN outputs: `cache/{train,test}_cnn_sequences.npy`, `artifacts/cnn_models.pt`, and `artifacts/cnn_test_probabilities.npy`

The synthetic-shift score and adversarial domain classifier diagnose only these two sectors; they do not establish generalization to future sectors. The CNN remains opt-in and does not replace the baseline unless its validation gates pass.
