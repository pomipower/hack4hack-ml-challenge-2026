# 🚀 Hack4Hack ML Challenge 2026

Repository containing end-to-end solutions, pipelines, architectures, and submission histories for the **Hack4Hack 2026 Machine Learning Challenge**.

---

## 📁 Repository Structure

```
├── planet_hunters_round1/       # Round 1: Exoplanet Detection from Light Curves & Centroids
│   ├── planet_hunters/          # Modular Python package (CNN, features, modeling, pipeline)
│   ├── artifacts_cnn/           # Trained models, audits, validation reports
│   ├── ARCHITECTURE.md          # Multi-branch CNN & folded light curve architecture docs
│   ├── PROBLEM_STATEMENT.md     # Round 1 problem details & metrics
│   ├── generate_submission.py   # Round 1 ensemble submission generator
│   └── submission_*.csv         # Round 1 submission history
│
├── signal_hunters_round2/       # Round 2 (Finals): SETI 6-Panel ON/OFF Cadence Classification
│   ├── feature_extractor.py     # Telescope-invariant robust normalization & domain extractor
│   ├── generate_upload1.py      # 5-fold multi-model ensemble (HGB + ExtraTrees + RF)
│   ├── evaluate_pipeline.py     # Parallel cross-validation & diagnostic pipeline
│   ├── submission_upload1.csv   # Upload 1 submission file (CV Macro-F1: 0.6994)
│   ├── PROBLEM_STATEMENT.md     # Round 2 problem details & physics rules
│   └── starter_notebook.ipynb   # Official starter notebook
│
└── .gitignore                   # Large data files (*.npy, *.pkl) excluded from Git
```

---

## 🪐 Round 1: Planet Hunters
- **Task:** Binary classification of Kepler/TESS light curves and centroid offsets into planet transit candidates vs false positives (eclipsing binaries, stellar variability, background artifacts).
- **Core Approach:** Multi-branch 1D CNN + Folded Transit CNN + Gradient Boosted Trees on physics features (transit depth, duration, SNR, centroid motion).

---

## 🛸 Round 2: Signal Hunters
- **Task:** 9-class classification of 6-panel ON/OFF radio telescope cadences (`ON, OFF, ON, OFF, ON, OFF`).
- **Target Classes:** `eti_candidate`, `frb`, `pulsar`, `scintillating_star`, `satellite_pass`, `radar_chirp`, `rfi_intermittent`, `rfi_persistent`, `noise`.
- **Domain Shift Challenge:** Train data comes from **Green Bank Telescope (GBT)**, while test data comes from **Parkes Telescope (PKS)** with a noisier receiver, no center spike, and different sampling rates.
- **Solution:** Per-cadence robust receiver noise standardization ($\sigma = 1.00$), local bandpass median filtering, physical Doppler drift rate tracking in Hz/s ($d_{\text{Hz/s}}$), and a 5-fold ensemble (`HistGradientBoosting` + `ExtraTrees` + `RandomForest`).
