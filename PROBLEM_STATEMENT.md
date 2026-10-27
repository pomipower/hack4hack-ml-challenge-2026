# Planet Hunters: find new worlds in starlight

**ML Challenge Hack4Hack 2026 · Round 1 (24-hour ranking round)**
**22 Oct 2026, 10:00 AM IST → 23 Oct 2026, 10:00 AM IST**

---

## 1. The story

Space telescopes like NASA's TESS stare at the same patch of sky for 27 days at a time and measure how bright hundreds of thousands of stars are, every 30 minutes. When a planet passes in front of its star, the starlight dips by a tiny amount, often less than 0.1%.

The catch: **most dips are not planets.** Two stars eclipsing each other, a faint star hiding behind the target, dark spots on a spinning star, a star that pulses, a flare, or the telescope itself wobbling can all look like a planet to a careless eye. Astronomers spend thousands of hours vetting candidates by hand.

Your job: **build a model that looks at a star's light and tells us what is really going on.**

---

## 2. The task

For every star in the test set, predict **one of 8 labels**:

| Label | What it is | What it tends to look like |
|---|---|---|
| `planet` | A planet transiting its star | Small, regular, flat-bottomed (U-shaped) dips; equal depth every time; no dip halfway between; the star's position on the camera does not move |
| `eclipsing_binary` | Two stars eclipsing each other | Deeper dips, often V-shaped; a second, different-depth dip between them; sometimes smooth wobbles between eclipses |
| `background_binary` | A faint eclipsing binary **behind or next to** the target, blended into its light | Shallow, planet-sized dips — the hardest trap. Clues: V-shape, odd and even dips of different depths, and **the light centroid moves during the dip** |
| `starspots` | Dark spots on a rotating star | Smooth, quasi-periodic dimming over days; changes slowly as spots evolve |
| `pulsating_star` | A star that pulsates (δ Scuti, γ Doradus, RR Lyrae…) | Regular oscillations; can be very fast (hours) or very large |
| `flare_star` | A star with magnetic flares (usually a cool red dwarf) | Sudden bright spikes with a fast rise and a slower decay |
| `instrument_glitch` | Nothing astrophysical — the telescope or detector misbehaves | Sudden drops and slow recoveries, jumps, short sharp dips with a big centroid jump, ramps, stripes in one orbit only |
| `quiet_star` | Nothing to see | Noise and common camera trends only |

**If a star shows more than one thing, the label is the most important one**, in this order:

`planet` › `eclipsing_binary` › `background_binary` › `pulsating_star` › `flare_star` › `starspots` › `instrument_glitch` › `quiet_star`

For example, a spotted star with a transiting planet is a `planet`; a flaring red dwarf that also has spots is a `flare_star`.

---

## 3. The data

One TESS-like observing **sector**: 27.4 days, one reading every 30 minutes = **1,315 cadences** per star.

**Training and test stars come from different sectors.** The pointing, the rhythm of the telescope's momentum dumps and the scattered light are different in the test sector. Models that memorise the training sector's quirks will lose places on the private leaderboard.

| File | Shape | What's inside |
|---|---|---|
| `train_flux.npy` | 18,000 × 1,315, float32 | Raw brightness of each training star, in electrons per second. `NaN` = no data (the mid-sector data downlink) |
| `train_centroid.npy` | 18,000 × 2 × 1,315, float16 | Where the star's light sits on the camera: column and row offsets in **millipixels**. `NaN` = no data |
| `train_stars.csv` | 18,000 rows | One row per star, **in the same order as the .npy rows**, with the label |
| `train_cadences.csv` | 1,315 rows | Time of each cadence and its quality flags |
| `test_flux.npy` | 7,000 × 1,315 | Same as train, no labels |
| `test_centroid.npy` | 7,000 × 2 × 1,315 | Same as train |
| `test_stars.csv` | 7,000 rows | Same columns as train, without `label` |
| `test_cadences.csv` | 1,315 rows | Cadences of the test sector |
| `sample_submission.csv` | 7,000 rows | The upload format |

### `*_stars.csv` columns

| Column | Meaning |
|---|---|
| `id` | Star ID (for example `TIC482910377`). Row *i* of the CSV is row *i* of the `.npy` files |
| `tmag` | Brightness in the TESS band (smaller = brighter). Fainter stars are noisier |
| `teff` | Surface temperature in kelvin |
| `logg` | Surface gravity (log₁₀ cm/s²). Low values = giant stars |
| `radius` | Star radius in solar radii |
| `contamination` | Fraction of the light in the aperture that comes from **other** stars nearby (0–1) |
| `camera`, `ccd` | Which camera (1–4) and detector (1–4) observed the star. **Stars on the same CCD share the same instrument trends** |
| `label` | Train only: the answer |

### `*_cadences.csv` columns

| Column | Meaning |
|---|---|
| `cadence` | Cadence number |
| `time_btjd` | Time in days (TESS Barycentric Julian Date) |
| `quality` | Bit flags: `1` = momentum dump (the telescope fired its thrusters), `2` = scattered light from the Earth or Moon, `4` = data gap |

### Loading on a laptop with 8 GB RAM

```python
import numpy as np, pandas as pd
flux = np.load('train_flux.npy', mmap_mode='r')          # 18,000 × 1,315 — reads from disk as needed
cen  = np.load('train_centroid.npy', mmap_mode='r')      # 18,000 × 2 × 1,315
stars = pd.read_csv('train_stars.csv')
cad = pd.read_csv('train_cadences.csv')

i = 0
f = flux[i] / np.nanmedian(flux[i]) - 1                  # relative brightness
print(stars.loc[i, 'label'])
```

The whole training set is about 190 MB in memory (flux + centroids), so it also loads fully on any laptop with 8 GB RAM.

---

## 4. What makes it hard

- **Tiny signals in noise.** Many planets are close to the noise level, and faint stars are very noisy.
- **Look-alikes on purpose.** Background binaries are made to look like planets; momentum-dump blips and pointing glitches can repeat on a regular period, like a transit.
- **Everything at once.** Stars are spotted while they host planets, flare while they rotate, and every star carries the camera's common trends, momentum dumps and the odd cosmic ray.
- **Faster than the camera.** Some pulsations are faster than the 30-minute sampling and show up at the wrong (aliased) frequency.
- **A new sector in the test set.** Different pointing, different thruster rhythm, stronger scattered light.
- **Imperfect labels.** About 2% of training labels are vetting mistakes between look-alike classes, as in real catalogues. The test labels are clean.

---

## 5. Evaluation

**Macro-averaged F1 score** over the 8 labels: the F1 score is computed for each label and the 8 values are averaged with equal weight. Rare and hard labels count as much as common ones, so predicting `quiet_star` for everything scores close to zero.

- The **live leaderboard** uses a public **30%** of the test stars and updates **every 4 hours**.
- The **final Round 1 ranking** uses the other **70%** (private), based on each team's best public upload.
- The final ranking decides the **top teams that go through to Round 2**.

---

## 6. Submission format

A CSV with exactly two columns and one row for each of the 7,000 test stars:

```csv
id,label
TIC948201733,planet
TIC912004561,quiet_star
TIC955310872,background_binary
...
```

- Labels must be spelled exactly as in the table above (lowercase, underscores).
- Every test `id` must appear once. Files that fail these checks are rejected straight away and **do not use up an upload**.
- Upload from the **Leaderboard** tab of your team workspace, **up to 5 times a day** (resets at midnight IST).

---

## 7. Rules

- Train only on the data provided. Open-source libraries (numpy, scikit-learn, LightGBM, PyTorch, astropy, lightkurve…) and publicly available pretrained weights are allowed.
- Do not use outside labelled data for this task, and do not hand-label the test set.
- Sharing code, predictions or labels between teams is not allowed and leads to disqualification.
- Keep your code: teams that go through may be asked to show how they made their predictions.

---

## 8. Where to start

1. Run the **starter notebook**. It loads the data, builds simple statistics, trains a model and writes a valid submission in about 2 minutes on a laptop.
2. Look at the light curves. Plot a few stars of each label — you will learn more in 20 minutes of plotting than in hours of tuning.
3. Then think like an astronomer:
   - **Remove the trends.** Divide by the median, subtract the common trend of each camera/CCD, and filter out slow variations before you look for dips.
   - **Search for periodic dips.** A Box Least Squares (BLS) period search finds transits buried in noise. Then fold the light curve on the period you found.
   - **Vet the dips.** Is the dip U- or V-shaped? Are odd and even dips the same depth? Is there a secondary eclipse at phase 0.5? **Does the centroid move during the dip?** Does the dip line up with a momentum dump (`quality & 1`)?
   - **Look at frequencies.** A periodogram separates spots (days) from pulsations (hours) and aliased fast pulsations.
   - **Find flares.** Count bright outliers that have a decaying tail, and tell them apart from single-point cosmic rays.
4. Or go deep: 1D CNNs on the raw, folded and centroid curves work well if you detrend first.

A good feature pipeline over all 25,000 stars takes **15–30 minutes** to run on a laptop. Plan your 24 hours.

**Good luck, and happy planet hunting.** 🪐
