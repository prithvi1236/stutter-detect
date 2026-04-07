# Pause and Stutter Detection (Explainable Audio Pipeline)

This notebook implements an end-to-end, rule-based speech analysis pipeline
with two outputs:
1) pause detection, and
2) repetition-style stutter detection.

The assignment goal is not perfect accuracy. The goal is to show signal
understanding, clean detection logic, and evidence-based debugging.

## View on Kaggle

Open the live notebook here: [Stutter Detection on Kaggle](https://www.kaggle.com/code/prithvibhargav/stutter-detection).

## Approach

The design is intentionally modular and explainable:
- audio preprocessing,
- shared frame-level features,
- pause detection via soft voting,
- repetition detection via multi-gate filtering,
- diagnostic reporting to explain failures and threshold effects.

Every repetition event is traceable through named gates and thresholds.

## Features Used

### Core audio preprocessing
- Resample to `16 kHz` (WebRTC VAD compatibility).
- Noise reduction (`noisereduce`) using early-audio noise estimate.
- Peak normalization.
- Pre-emphasis (`y[t] = x[t] - 0.97*x[t-1]`) to preserve consonant detail.

### Frame-level features (shared grid)
- **RMS**: loudness proxy.
- **ZCR**: helps catch high-frequency speech where RMS alone is weak.
- **WebRTC VAD**: speech/non-speech vote per frame.
- **MFCC (13 dims)**: phonetic fingerprint for repetition matching.

All features use the same frame/hop (`30 ms` frame, `10 ms` hop) so alignment
is consistent across modules.

## Detection Logic

### A) Pause detection (soft vote)
Per frame:
- `score = 0.40 * RMS_speech + 0.35 * VAD_speech + 0.25 * ZCR_speech`
- if `score < SOFT_VOTE_THRESH (0.55)`, frame is marked silence

Then:
- consecutive silence >= `MIN_PAUSE_S (0.20s)` becomes a pause
- nearby pauses are merged (`merge_gap_s = 0.1s`)

This reduces brittleness compared to single-feature pause rules.

### B) Repetition detection (adjacent burst pipeline)

#### Burst creation
- Build speech bursts from the inverse silence mask.
- Keep only bursts with duration in `[MIN_BURST_S, MAX_BURST_S] = [0.06s, 0.60s]`.
- Apply burst energy floor: `MIN_BURST_RMS = 0.004`.

#### Pairwise gates (only adjacent bursts)
For each neighboring burst pair:

1. **Gate 1: Gap**
   - require `gap <= MAX_GAP_BETWEEN_S (0.50s)`

2. **Gate 2: Duration ratio**
   - require `max(dur1, dur2) / min(dur1, dur2) <= MAX_DUR_RATIO (3.0)`

3. **Gate 3: MFCC cosine (adaptive)**
   - cosine distance between mean MFCC vectors
   - threshold is computed per recording from candidate-pair distribution:
     `percentile(cosine_vals, COSINE_PERCENTILE)`
   - then clamped to `[COSINE_MIN_CLAMP, COSINE_MAX_CLAMP] = [0.30, 0.55]`
   - default fallback remains `COSINE_THRESH (0.40)` if adaptive threshold
     cannot be computed

4. **Gate 4: Normalized DTW**
   - DTW on MFCC frame sequences
   - require `< DTW_THRESH (1.5)`
   - if DTW is unavailable, fallback returns length-invariant Euclidean-style
     distance from mean sequence vectors

#### Tiny-gap safety rule
For nearly contiguous bursts (`gap < 0.02s`), enforce stricter similarity:
- `cosine < 0.22`
- `dtw < 1.00`

This protects against accidental over-linking of very close non-repetitions.

#### Event clustering and post-filter
- Consecutive confirmed pair indices are clustered into one event.
- `MIN_REP_IN_EVENT = 1` means one confirmed adjacent pair is enough.
- Additional guardrails for single-pair events:
  - `SINGLE_PAIR_MAX_BURST_S = 0.34s`
  - `SINGLE_PAIR_MAX_SPAN_S = 1.10s`

This reduces false positives from long, phrase-like pairs.

## Challenges Faced and Fixes

Only the challenges relevant to final behavior are listed below.

### 1) Valid two-part repetitions were dropped
- **Symptom:** adjacent matches were found, but event count stayed zero.
- **Root cause:** clustering threshold had an off-by-one condition requiring
  3 bursts.
- **Fix:** align event threshold with config intent (`MIN_REP_IN_EVENT = 1`
  means a single adjacent pair is valid).

### 2) Fast adjacent words looked like repetitions
- **Symptom:** non-stutter word pairs in quick speech were still flagged.
- **Root cause:** global MFCC similarity can be high when neighboring words
  share vowel energy.
- **Fix in current notebook:** combine multi-gate filtering with strict
  tiny-gap similarity checks and single-pair post-filters to reject many of
  these cases.

### 3) Quiet speech was removed too aggressively
- **Symptom:** too few bursts survived to form repetition pairs.
- **Root cause:** energy floor was rejecting low-volume but valid speech
  segments.
- **Fix in current notebook:** lower `MIN_BURST_RMS` to `0.004`, then verify
  survivors with later gates (gap, ratio, cosine, DTW) instead of deleting
  early.

## Diagnostics and Explainability

The notebook includes a diagnostic pass that prints:
- burst counts before/after energy filtering,
- per-gate pass/reject behavior,
- cosine and DTW statistics,
- cosine threshold actually used for the current recording,
- largest drop-off gate.

This turns threshold tuning into a measurable process rather than guesswork.

## Current Limitations

- Thresholds are rule-based and may need speaker-specific tuning.
- Coarticulation in very fast speech can still blur boundaries.
- Binary speech/silence segmentation may merge subtle intra-word dips.

## Future Improvements

- Add onset-focused feature gate (first 20-40 ms) to better separate similar
  vowels from true repeated onsets.
- Learn or auto-tune the cosine percentile (`COSINE_PERCENTILE`) from a small
  labeled validation set.
- Add labeled evaluation set and report precision/recall/F1.
- Explore lightweight learned re-ranking on top of current gates.
