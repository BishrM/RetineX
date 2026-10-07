# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## PROJECT STATUS

**0% IMPLEMENTED.** This is a fresh rebuild. Only the workspace, documentation and Git foundation exist.
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
