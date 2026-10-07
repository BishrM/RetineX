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
Measured on the Kaggle copy used here: 8,091 images, 40,455 captions, exactly 5 per image. The frozen split uses
8,000 of those images (6000/1000/1000); the other 91 are unused.


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
**Original dataset source.** Flickr8K comes from Hodosh, Young & Hockenmaier,
[Framing Image Description as a Ranking Task (JAIR 2013)](https://jair.org/index.php/jair/article/view/10833).
The University of Illinois distributes it through a request form, <https://forms.illinois.edu/sec/1713398> (form verified live;
download link is emailed, non-commercial research/education use, link must not be redistributed). In our case the emailed link returned 404.

**Practical access source used by this project.** Kaggle dataset
[`adityajn105/flickr8k`](https://www.kaggle.com/datasets/adityajn105/flickr8k) (uploader `adityajn105`, 1.04 GB, version 1).
It is a third-party repackaging, **not** the official distribution. Provenance is consistent with the original corpus
(8,091 images, 5 captions each, original filename pattern) but cannot be proven without the official files; see `HANDOFF.md`.
Kaggle lists it as "CC0: Public Domain"; this is the uploader's assertion. The Illinois terms (non-commercial research and education)
and the Flickr photographers' rights still apply in spirit, so use it for academic work only and do not redistribute it.

**Download and normalize.** Kaggle layout is `Images/` + a CSV `captions.txt`; our code expects `Flickr8k.token.txt`, so a
tested normalization step converts it (the download is left untouched, images are copied):
```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('adityajn105/flickr8k'))"   # prints the download path
python -m retinex.dataset normalize --source <printed path>
```
If you obtain the original files from Illinois instead, put them in the layout below directly and skip `normalize`.

**Resulting layout** (override the root with `--data-dir` or the `RETINEX_DATA_DIR` environment variable):
```
data/flickr8k/
├── images/                  # all *.jpg files, flat
├── captions/
│   └── Flickr8k.token.txt   # lines: <image>.jpg#<0-4><TAB><caption>
└── splits/                  # generated; see below
```
The parser only reads the `Flickr8k.token.txt` format. Official `Flickr_8k.*Images.txt` split files are not used; our own split is generated below.

**Validate** (on the Kaggle copy: 8,091 images, 40,455 captions, no anomalies; checks files exist, caption/image links, malformed or duplicate records, unreadable images; prints measured counts):
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
