# RetineX

> **Status: 0% implemented.** This is a fresh rebuild; nothing below has been built or verified yet.

## Overview
RetineX: Deep Learning Based Image Caption Generation using Fisher Score Feature Optimization, LSTM and Neural
Voice Narration for Real-Time Visual Accessibility.

## Problem
Visually impaired users need images described automatically and read aloud.

## Objective
Generate image captions with an LSTM, test whether Fisher Score feature selection improves on full VGG16
features, and narrate captions through a web app.

## Proposed Method
Image -> VGG16 fc2 (4096-d) -> TF-IDF + K-Means pseudo-classes -> Fisher Score -> top 512 features -> LSTM ->
BLEU evaluation -> text-to-speech.

## System Architecture
NOT IMPLEMENTED YET (see `CLAUDE.md` and `HANDOFF.md` for the intended design).

## Technology Stack
Planned: Python, TensorFlow/Keras, NumPy, Pandas, scikit-learn, NLTK, Pillow, Flask, gTTS, pytest.

## Dataset
Flickr8K (images + 5 human captions each) is used for training and evaluating the captioner.
Expected size is ~8,000 images / ~40,000 captions; **measured numbers are not recorded yet** because the dataset has
not been placed on the development machine (see `HANDOFF.md`).


## Experimental Design
Frozen image-level 6000/1000/1000 split. Baseline (4096-d VGG16 + LSTM) vs RetineX (512 Fisher-selected + LSTM).
Metrics: unsmoothed corpus BLEU-1 to BLEU-4. Pseudo-classes and the Fisher mask use training data only.

## Project Structure
```
data/  src/  scripts/  notebooks/  models/  artifacts/  experiments/  app/  tests/  docs/
```
Folders are placeholders; contents will evolve.

## Installation
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # Stage 1 needs only Pillow (+ pytest)
```
Only Stage 1 dependencies are listed; later stages will add theirs.


## Dataset Setup
**Acquisition (manual).** Flickr8K must be obtained by you; it is not in this repository and must not be committed.
The reference project points to the University of Illinois request form <https://forms.illinois.edu/sec/1713398>
(link taken from that README; not verified by us). The download provides an images archive (`Flickr8k_Dataset`)
and a text archive containing `Flickr8k.token.txt`.

**Place files** (override the root with `--data-dir` or the `RETINEX_DATA_DIR` environment variable):
```
data/flickr8k/
├── images/                  # all *.jpg files, flat
├── captions/
│   └── Flickr8k.token.txt   # lines: <image>.jpg#<0-4><TAB><caption>
└── splits/                  # generated; see below
```
Only the `Flickr8k.token.txt` caption format is supported.

**Validate** (checks files exist, caption/image links, malformed or duplicate records, unreadable images; prints measured counts):
```bash
python -m retinex.dataset validate
```

**Generate the frozen split** (6000 / 1000 / 1000 images; refuses to overwrite an existing split unless `--force`):
```bash
python -m retinex.dataset split
```

**Verify the split** (sizes, no duplicates, zero overlap, every image has captions and a file, checksums unchanged):
```bash
python -m retinex.dataset verify
```


## Training
NOT IMPLEMENTED YET

## Evaluation
NOT IMPLEMENTED YET

## Inference
NOT IMPLEMENTED YET

## Flask Application
NOT IMPLEMENTED YET

## Testing
```bash
pytest        # synthetic fixtures only; does not need Flickr8K
```


## Reproducibility
The split is deterministic: image IDs are sorted, shuffled with `random.Random(42)`, then sliced into
train/val/test. If the validated population exceeds 8,000 images, the surplus is left unused and counted in
`splits/split_meta.json`. The split files (`train.txt`, `val.txt`, `test.txt`) and `split_meta.json` (seed, sizes,
SHA-256 of each file) are **frozen artifacts**: later stages read them with `retinex.dataset.load_split` and never regenerate them.
Full pipeline reproducibility beyond Stage 1: NOT IMPLEMENTED YET.

### Data leakage
Splitting is by image, never by caption. Flickr8K has ~5 near-paraphrase captions per image; if captions were split
randomly, the same photo would appear in train and test, inflating test scores. `verify` fails on any overlap between splits.


## Limitations
No results exist yet. Flickr8K is small; gTTS needs internet access.

## Future Work
To be determined after the baseline and RetineX experiments.

## Original Reference
Evolved from [MiteshPuthran/Image-Caption-Generator](https://github.com/MiteshPuthran/Image-Caption-Generator),
itself based on [Yumi's blog](https://fairyonice.github.io/Develop_an_image_captioning_deep_learning_model_using_Flickr_8K_data.html).
Used as a conceptual reference only.
