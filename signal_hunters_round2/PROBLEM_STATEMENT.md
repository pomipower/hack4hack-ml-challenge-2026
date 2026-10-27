# Signal Hunters: is anyone out there? 🛸

**ML Challenge Hack4Hack 2026 · Round 2 (finals, one day)**
**27 Oct 2026, 10:00 AM IST → 27 Oct 2026, 11:59 PM IST**

---

## 1. The story

In Round 1 you found planets. Now we listen to them.

Radio telescopes such as the ones used by Breakthrough Listen point at nearby stars and record the radio sky in millions of very narrow frequency channels. Nature does not make radio signals only a few hertz wide. Technology does. A narrow tone that slowly **drifts** in frequency, because the transmitter sits on a rotating, orbiting planet, is exactly what a search for extraterrestrial intelligence (SETI) looks for.

The problem: **Earth is loud.** Phones, Wi-Fi, satellites, aircraft radar and the telescope's own electronics produce thousands of narrow, drifting signals every second. Nearly every "alien" signal ever found turned out to be human interference.

So astronomers use a trick: the **ON/OFF cadence**. The telescope looks at the target star (**ON**), then at a nearby empty patch of sky (**OFF**), and repeats: **ON, OFF, ON, OFF, ON, OFF**. A signal from the star should appear **only when the telescope points at it**. Interference from Earth comes in through the side of the dish and shows up whether the telescope points at the star or not.

Your job: **look at each cadence and decide what you are hearing.**

---

## 2. The task

Each sample is one cadence: **6 spectrograms** (frequency × time) in the fixed order **ON, OFF, ON, OFF, ON, OFF**. Predict **one of 9 labels**:

| Label | What it is | What it looks like |
|---|---|---|
| `eti_candidate` 👽 | A narrowband signal that comes from the target star | A thin line that **drifts** (its frequency changes steadily with time), present in **all three ON panels and none of the OFF panels**. Its position in each ON panel lies **on the same straight track through time**, including the time spent slewing to and from the OFF fields. Rare (about 3%) |
| `rfi_persistent` | Terrestrial interference that never goes away | A narrow line (usually not drifting) in **every** panel, ON and OFF. It may fade in one panel |
| `rfi_intermittent` | Terrestrial interference that comes and goes | Lines in **some** panels. The traps: lines in ON panels **and** an OFF panel; lines only in ON panels whose positions **do not connect** into one track (even when every line has the **same drift rate**); or lines only in ON panels that **do not drift at all** (a signal from a planet always drifts) |
| `satellite_pass` | A satellite crossing the beam | A bright, **steeply** drifting (sometimes curved) line in one panel or two consecutive panels |
| `pulsar` | A spinning neutron star | **Broadband** pulses repeating on a regular period, each one **dispersed** (it arrives later at lower frequencies, so it looks like a curved sweep). ON panels only |
| `frb` | A fast radio burst from another galaxy | **One** dispersed broadband sweep in **one** ON panel, sometimes with a scattering tail |
| `radar_chirp` | Radar | Steep frequency sweeps that **repeat** in a sawtooth pattern, in (almost) every panel |
| `scintillating_star` | A natural radio source twinkling through interstellar gas | Patchy, blotchy brightening across many channels in the ON panels only |
| `noise` | Nothing | Noise, the bandpass and instrument effects only |

**If a cadence shows more than one thing, the label is the most important one**, in this order:

`eti_candidate` › `frb` › `pulsar` › `scintillating_star` › `satellite_pass` › `radar_chirp` › `rfi_intermittent` › `rfi_persistent` › `noise`

For example, a cadence with a genuine ETI candidate and an unrelated persistent RFI line elsewhere in the band is an `eti_candidate`. **Many cadences contain more than one source** (a satellite crossing a pulsar observation, radar on top of persistent RFI, and so on), so the order matters.

---

## 3. The data

**Training and test cadences come from different telescopes.** Train is from "GBT", test from "PKS": a noisier receiver, a different bandpass shape (stronger ripple, roll-off and tilt, no centre-channel spike), more dropped rows and broadband glitches, larger gain changes between panels, different local interference frequencies, a different time resolution, and **a different mix of labels**. A model that learns the training telescope's quirks or its label frequencies will lose places on the private leaderboard.

| File | Shape | What's inside |
|---|---|---|
| `train_cadences.npy` | 18,000 × 6 × 16 × 128, uint8 | The training cadences: 6 panels × 16 time steps × 128 frequency channels |
| `train_meta.csv` | 18,000 rows | One row per cadence, **in the same order as the .npy**, with the label |
| `test_cadences.npy` | 10,000 × 6 × 16 × 128, uint8 | The test cadences |
| `test_meta.csv` | 10,000 rows | Same columns as train, without `label` |
| `sample_submission.csv` | 10,000 rows | The upload format |

### The values

Each value is the received power in a log scale, stored as a byte:

```
value = round(24 × (log2(power) + 4))      # power = 1 is the average noise level → value 96
power = 2 ** (value / 24 − 4)              # to get linear power back
```

- `0` = **no data** (a dropped time step). Treat it as missing, not as very low power.
- Within a panel, rows are time (top → bottom = earlier → later) and columns are frequency channels (left → right = lower → higher frequency).

### `*_meta.csv` columns

| Column | Meaning |
|---|---|
| `id` | Cadence ID. Row *i* of the CSV is row *i* of the `.npy` |
| `telescope` | `GBT` (train) or `PKS` (test) |
| `target` | The star observed (HIP catalogue number). Not useful for prediction |
| `center_freq_mhz` | Centre frequency of the 128-channel window |
| `channel_width_hz` | Width of one channel in hertz |
| `tsamp_s` | Seconds per time step |
| `slew_gap_rows` | Time steps lost while the telescope moves between panels (3–6). **Panel *p*, row *r* happened at global time step `p × (16 + slew_gap_rows) + r`** |
| `label` | Train only: the answer |

### Loading on a laptop with 8 GB RAM

```python
import numpy as np, pandas as pd
X = np.load('train_cadences.npy', mmap_mode='r')     # 18,000 × 6 × 16 × 128 bytes — read from disk as needed
meta = pd.read_csv('train_meta.csv')
cad = X[0].astype(np.float32)                         # one cadence: (6, 16, 128)
power = 2 ** (cad / 24 - 4)
print(meta.loc[0, 'label'])
```

The whole training set is 221 MB as bytes. Convert to float in batches rather than all at once (as float32 it would be 885 MB).

---

## 4. What makes it hard

- **Faint signals.** Many ETI candidates, pulsars, bursts and scintillating stars are barely above the noise in any single panel. You have to combine evidence across the three ON panels.
- **The cadence is the whole point.** A single panel can't tell an ETI candidate from interference. Present in ON, absent in OFF — and on one consistent drifting track — is what matters.
- **Traps on purpose.** Interference that happens to sit in the ON panels; ON-only lines that drift at the same rate but don't connect into one track; ON-only lines with zero drift; persistent lines that fade in one OFF panel.
- **Busy skies.** Satellites, radar and interference lines often share the window with the signal you care about. Label by the priority order.
- **Instrument effects.** Bandpass ripple and roll-off at the edges, a bright centre channel in GBT data, panel-to-panel gain changes, broadband impulsive rows and dropped rows.
- **A new telescope in the test set**, noisier and with a different mix of labels.
- **Macro-F1 with a rare class.** ETI candidates are about 3% of the data, but they count as much as every other class.

---

## 5. Evaluation

**Macro-averaged F1 score** over the 9 labels: F1 is computed for each label and the 9 values are averaged with equal weight.

- The **live leaderboard** uses a public **30%** of the test cadences and updates **every 15 minutes**.
- The **final ranking** uses the hidden **70%** (private), based on each team's best public upload.
- **Prizes are decided by the Round 2 final ranking.** Winners are announced on **28 Oct 2026, 5:00 PM IST**.

---

## 6. Submission format

A CSV with exactly two columns and one row for each of the 10,000 test cadences:

```csv
id,label
SX1a2b3c4d,noise
SX9f8e7d6c,eti_candidate
SX5a6b7c8d,rfi_persistent
...
```

- Labels must be spelled exactly as in the table above (lowercase, underscores).
- Every test `id` must appear once. Files that fail these checks are rejected straight away and **do not use up an upload**.
- Upload from the **Leaderboard** tab of your team workspace (Round 2), up to **5 uploads** in the round.
- **Before the round closes, submit your code** (a GitHub repository or notebook link) from the **Submission** tab. Prize-winning teams must be able to reproduce their best upload from that code.

---

## 7. Rules

- Train only on the data provided. Open-source libraries and publicly available pretrained weights are allowed.
- Do not use outside labelled data for this task, and do not hand-label the test set.
- Sharing code, predictions or labels between teams is not allowed and leads to disqualification.
- The organizers' decisions on verification and results are final.

---

## 8. Where to start

1. Run the **starter notebook**. It loads the data, builds simple per-panel statistics, trains a model and writes a valid submission in a few minutes on a laptop.
2. **Look at cadences.** Plot the 6 panels side by side for a few examples of each label.
3. Then think like a SETI astronomer:
   - **Flatten the bandpass**: divide each channel by its median over time (per panel, or per cadence).
   - **Find narrow lines**: for each panel, search for the straight line (start channel + drift) with the most power — a simple "drift search" over a grid of drift rates.
   - **Check the cadence**: is the line in ON 1, 3 and 5? Is it missing in OFF 2, 4 and 6? Do the three ON detections line up on one track using the global time (`slew_gap_rows`)?
   - **Broadband events**: sum over channels after de-dispersing (shifting each channel by a curved delay) to catch pulsars and FRBs.
   - **Patchiness**: compare the texture of ON and OFF panels for scintillation.
4. Or go deep: a 2D CNN that sees all 6 panels as channels, or a model that compares ON and OFF panels directly. Augment with frequency shifts and flips (flipping frequency flips the drift sign — that's fine), and normalise each cadence so the PKS test data looks like GBT.

**Clear skies, and good hunting.** 📡
