# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## PROJECT STATUS

**Stages 1-4 of 9 are implemented; stages 5-9 are NOT IMPLEMENTED.** Built and verified here: the
dataset layer and frozen 6000/1000/1000 split, VGG16 fc2 extraction (8000 x 4096), TF-IDF + K-Means
pseudo-classes (k=20), and Fisher Score selection (4096 -> 512), each passing its own validator. Also
built: a Flask web UI serving exactly those stages. **Not built: both LSTM models, BLEU evaluation,
caption generation and narration.** No model has been trained, no caption generated and no BLEU score
measured in this repository.
Only functionality that is implemented, tested and verified *in this repository* may be marked complete.
Label anything unverified `NOT VERIFIED` and anything unbuilt `NOT IMPLEMENTED`. Never fabricate results
(BLEU, accuracy, dataset stats, feature/Fisher outputs, training completion, Flask behaviour, timings).

Keep a status table in HANDOFF.md up to date as stages are genuinely completed.

## PROJECT PURPOSE

RetineX: Deep Learning Based Image Caption Generation using Fisher Score Feature Optimization, LSTM and Neural
Voice Narration for Real-Time Visual Accessibility. An academic major project.

## PROJECT OBJECTIVES

1. Caption images with a VGG16 + LSTM model.
2. Test whether Fisher Score feature selection (4096 -> 512 dims) helps vs. the full-feature baseline.
3. Narrate captions via text-to-speech for visual accessibility.
4. Serve this through a Flask web app.

## CORE PIPELINE

Image -> VGG16 (ImageNet) fc2 -> 4096-d vector -> TF-IDF + K-Means semantic pseudo-classes ->
Fisher Score -> top-512 features -> LSTM caption generator -> BLEU evaluation -> TTS -> Flask app.

## TECHNOLOGY STACK

Expected (add to requirements only when actually used): Python, TensorFlow/Keras, NumPy, Pandas, scikit-learn,
NLTK, Pillow, Flask, gTTS (initial TTS), pytest. Dev machine: Apple M2, 8 GB RAM, Python 3.12, macOS.
Do not blindly install everything; requirements must list only real dependencies.

## DATASET

Flickr8K (8,000 images, 5 captions each). Downloaded by the user into `data/`; never committed, never
auto-downloaded without being asked.

## EXPECTED EXPERIMENT

- Image-level split **6000 / 1000 / 1000** (train/val/test), generated once with a fixed seed and **frozen** (stored as a file).
- Baseline: 4096-d VGG16 features + LSTM.
- RetineX: 512 Fisher-selected features + LSTM.
- Same split, vocabulary, hyperparameters and seeds for both models; only the image features differ.
- Metrics: BLEU-1/2/3/4, **unsmoothed corpus BLEU** (all references per image, not sentence-mean BLEU).

## ARCHITECTURE

Target layout (create files only when needed):
`data/` raw dataset (ignored) · `src/` core ML modules · `scripts/` CLI entry points per stage ·
`notebooks/` exploration only · `models/` checkpoints (ignored) · `artifacts/` features, masks, splits-derived
outputs (ignored, except small frozen split files if decided) · `experiments/` run configs/results ·
`app/` Flask · `tests/` pytest · `docs/`.
Core logic lives in Python modules, never only in notebooks.

Pipeline stages: dataset prep -> frozen split -> VGG16 fc2 features -> TF-IDF/K-Means pseudo-classes ->
Fisher mask -> baseline training -> RetineX training -> BLEU evaluation -> inference -> Flask + gTTS.

## DEVELOPMENT RULES

- Do not change a core methodological requirement above without documenting the reason and asking the user.
- Work in small, verified stages; don't run ahead of the user's instruction.
- Use current `tf.keras` APIs. Paths relative to repo root / configurable, never hard-coded absolute.
- Reference repo (MiteshPuthran/Image-Caption-Generator) is conceptual reference only; do not copy code blindly.

## STAGE 1 DECISIONS

- Dataset layer: `src/retinex/dataset.py`; commands `python -m retinex.dataset validate|split|verify`; split seed 42.
- `data/flickr8k/splits/` (IDs + `split_meta.json` checksums) is the frozen split. Later stages use `load_split`, never `make_split`; never use `--force` without user approval.
- Data source: Kaggle `adityajn105/flickr8k` (third-party copy; Illinois original link returned 404). Normalize CSV with `python -m retinex.dataset normalize --source <dir>`. Always label it a third-party copy; never call it official. Full details in HANDOFF.md.
- Frozen split (seed 42, 6000/1000/1000, 91 surplus images unused) is committed under `data/flickr8k/splits/`; verify with `python -m retinex.dataset verify`.
- Venv is `.venv/`; install with `pip install -e ".[dev]"`; run tests with `pytest`.

## STAGE 3 DECISIONS

Frozen with explicit user approval on 2026-10-12. Changing any of these changes the methodology and
requires the same approval again.

- Module `src/retinex/pseudoclasses.py`; commands `python -m retinex.pseudoclasses fit|validate`.
- **Document unit:** one document per *training* image, its captions joined in `#n` order (6,000 documents).
- **TF-IDF:** `lowercase=True`, `token_pattern=r"[a-z]{2,}"` (drops punctuation, digits, single
  characters), `min_df=2`, `ngram_range=(1,1)`, `norm="l2"`, `sublinear_tf=False`, no stop-word list.
  IDF already suppresses words common to nearly every caption, so a stop list would be redundant.
- **K-Means:** `n_clusters=20`, `random_state=42`, `n_init=10`, `algorithm="lloyd"`, `max_iter=300`.
  L2-normalised rows make Euclidean K-Means equivalent to spherical/cosine clustering.
- **K = 20 is frozen in advance and mechanically enforced.** `fit` raises on any other k and the CLI
  refuses `--k` != 20 before doing any work; there is no override, documented or otherwise, and the sweep
  is not a parameter. The cosine silhouette sweep over K in {5,10,20,30,50} is recorded in the metadata for
  transparency only and must never be used to select K (`k_selected_by_sweep: false`).
- **Silhouette:** `metric="cosine"` on the L2-normalised training TF-IDF matrix, every training document,
  `sample_size=None` (no sampling). Measured cost at this scale is about 3 s and under 1 GB, so sampling is
  unnecessary. Where a silhouette is mathematically undefined (fewer than 2 distinct labels, or as many
  labels as documents) the metadata records `score: null` with an `unavailable` reason -- never a number.
- **Train only.** TF-IDF and K-Means are fitted on training captions alone. **No labels are produced for
  validation or test images**, so no later stage can misuse them; `validate` fails if such a file appears.
- **Fisher precondition:** `validate` fails if any cluster holds fewer than 10 training images, because a
  near-singleton class makes the Fisher within-class variance unstable. The floor is the constant
  `MIN_CLUSTER_SIZE = 10` and is **not** a parameter of `validate_pseudoclasses`, so no caller can validate
  against a weaker floor; the metadata's `min_cluster_size_required` is separately checked to be >= 10.
- An image whose captions contain no `[a-z]{2,}` token after cleaning is an error naming the image id,
  never a silent all-zero TF-IDF row.
- Artifacts in `artifacts/pseudoclasses/` (git-ignored by the existing `artifacts/*` rule):
  `train_pseudoclasses.csv` (sorted by image id), `train_vocabulary.txt`, `pseudoclass_meta.json`.
- Stage 4 maps label rows to feature rows through `vgg16_fc2_ids.txt`, which stays the single source of
  truth for row order; the metadata records its SHA-256 so a mismatch is detectable.
- Stage 3's cleaning is independent of the Stage 5 LSTM tokenizer; they share no vocabulary.

## STAGE 4 DECISIONS

Frozen with explicit user approval on 2026-10-12. Changing any of these changes the methodology and
requires the same approval again.

- Module `src/retinex/fisher.py`; commands `python -m retinex.fisher select|validate`.
- **Formula**, all means over the 6,000 training rows only:

      F_j = SS_between,j / SS_within,j
      SS_between,j = sum_c  n_c (mu_cj - mu_j)^2
      SS_within,j  = sum_c  sum_{i in c} (x_ij - mu_cj)^2

- **Variance convention: population, `ddof = 0`.** That is what makes the count-weighted form
  `sum_c n_c sigma^2_cj` *exactly* equal the pooled within-class sum-of-squared-deviations form
  implemented above, so no division by `n_c` happens on the denominator path.
- **Two-pass float64** (class means, then squared deviations), never `sum(x^2) - sum(x)^2/n`, which
  subtracts large near-equal numbers. A sum of squared deviations cannot be negative, so **no
  clamping is applied**; a negative value is treated as a bug and raises.
- **No standardisation** — the ratio is invariant to per-feature scaling.
- **No epsilon** is ever added to the denominator. Zero denominators are handled by branch:
  `SS_within == 0 and SS_between > 0` -> `+inf`; `SS_within == 0 and SS_between == 0` -> `0.0`.
  Both counts are recorded in the metadata. No `0/0` division is executed, so no score is NaN.
- **`+inf` is reserved for a genuinely zero denominator.** A non-finite ratio arising from a
  *positive* denominator is an overflow, not a separable feature: it raises `FloatingPointError`
  naming the feature and both scatter terms, rather than being recorded as `+inf` and inflating
  `n_zero_denominator_with_signal`. `validate` enforces the resulting invariant
  `n_infinite_score == n_zero_denominator_with_signal`. The branch is defensive: overflow is
  unreachable from a float32 source of this size (worst finite ratio ~9.4e170 vs float64's
  1.8e308), which `test_overflow_is_unreachable_from_a_float32_source` asserts.
- **Selection: top 512 by descending score, ties broken by ascending original feature index**
  (stable argsort). The selected indices are then **sorted ascending** and that ascending array is
  the column mapping: output column `i` is source column `idx[i]`. `validate` re-derives the
  ranking from the saved scores independently and also checks that no unselected feature scores
  strictly higher than a selected one.
- **Boundary ties** are recorded honestly: the cut score, whether the cut falls inside a tie, and
  how many tied features were included and excluded.
- **Join by image id, never by row position.** `training_join` resolves every row index from
  `vgg16_fc2_ids.txt`, requires the labelled ids to be a one-to-one cover of the frozen training
  split, and raises on a labelled val/test image, a missing label, an unknown id or a duplicate id.
- `select` refuses to run unless `validate_features` and `validate_pseudoclasses` both come back
  clean, so Stage 4 cannot be built on invalid upstream artifacts.
- **The source `vgg16_fc2.npy` is opened read-only and must stay byte-identical**; `validate`
  re-checks its SHA-256 against the recorded value.
- Artifacts in `artifacts/fisher/` (git-ignored by the existing `artifacts/*` rule):
  `fisher_scores.npy` (4096 float64), `fisher_top512_indices.npy` (512 int32, ascending),
  `vgg16_fc2_fisher512.npy` (8000x512 float32), `fisher_meta.json`.
- Rows keep the order of `vgg16_fc2_ids.txt`; that file is referenced and checksummed, not copied.
- **Expected real-data counts are never hardcoded in validation.** Observed values (zero-score
  columns, infinite scores, boundary ties) are recorded from the actual run and checked only for
  internal consistency with the saved scores.
- No claim that Fisher selection improves caption quality may be made before the baseline and
  512-feature LSTM evaluations produce measured BLEU.

## VALIDATION-API RULES

The frozen experiment parameters are not call-time arguments. `validate_features(out_dir, data_dir)`,
`validate_pseudoclasses(out_dir, data_dir, features_dir)` and
`validate_fisher(out_dir, data_dir, features_dir, pseudo_dir)` take no `sizes`, `dim`,
`min_cluster_size` or `n_selected`: they read the frozen constants (`SIZES` 6000/1000/1000,
`FEATURE_DIM` 4096, `MIN_CLUSTER_SIZE` 10, `N_SELECTED` 512), so no caller can validate the real
artifacts against a weaker rule. `fisher.select` is frozen the same way. Tests that need a small
synthetic fixture use the private `_validate_features` / `_validate_pseudoclasses` /
`_validate_fisher` / `_select`, which the production CLI never calls. Any new validator follows the
same split: a frozen public entry point, a parameterised private implementation.

## TESTING RULES

pytest. Each stage gets a small test that fails if its logic breaks (shapes, split disjointness, mask size 512,
BLEU on a known example). Run tests before declaring a stage done; report failures with output.

## REPRODUCIBILITY RULES

Fixed seeds (python, numpy, TF); record versions; deterministic splits; all steps scriptable from a clean clone
(venv -> install -> place Flickr8K -> prepare -> split -> features -> pseudo-classes -> Fisher mask -> train
baseline -> train RetineX -> BLEU -> inference -> Flask -> upload -> caption -> narration).

## DATA-LEAKAGE RULES

- Split by **image**, never by caption.
- TF-IDF vectorizer, K-Means pseudo-classes and the Fisher mask are **fit on the training split only**; val/test are only transformed/masked.
- Tokenizer/vocabulary built from training captions only.
- Test split is touched only for final evaluation, never for tuning.
- A test asserts the fit inputs contain no validation or test IDs.

## GIT RULES

Local repo on `main`. Do not create the GitHub repo or push unless explicitly asked. Commit only when asked.
Never commit datasets, checkpoints, feature matrices, audio, `.env`, or logs (see `.gitignore`).

## DOCUMENTATION RULES

Keep README.md, HANDOFF.md and this file consistent with reality. Don't document commands that don't exist
yet; mark them `NOT IMPLEMENTED YET`.

## ACADEMIC-INTEGRITY RULES

Credit the reference repo and its source blog. Report only measured results from this repo, with the exact
config that produced them. Report negative or unexpected results honestly (including if Fisher selection
does not beat the baseline). No fabricated or cherry-picked numbers.
