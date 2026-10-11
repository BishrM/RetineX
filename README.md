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
pip install -e ".[dev]"
```
That installs everything the implemented stages and the web application need: Pillow, NumPy,
TensorFlow, Keras, scikit-learn and Flask as runtime dependencies, plus pytest and PyYAML for
development. Python 3.12 or newer is required (NumPy 2.5 needs 3.12).

**One prerequisite is not a pip package.** The first time VGG16 runs, Keras downloads its ImageNet
weights (about 553 MB) into `~/.keras/models/`, which needs network access once. Feature extraction
and the Workspace both depend on that file; everything else, including the whole test suite, works
offline.


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


## Feature Extraction (Stage 2)
Extracts a 4096-d VGG16 `fc2` vector for each of the 8,000 frozen images. VGG16 is ImageNet-pretrained and
frozen; it is never trained or fine-tuned, and captions are never read, so no validation or test information
can reach a feature.

```bash
python -m retinex.features extract      # batched, resumable, validates when finished
python -m retinex.features validate     # re-check existing artifacts against the frozen 8,000 x 4096 shape
```

Writes to `artifacts/features/` (git-ignored, about 131 MB):

| File | Contents |
| --- | --- |
| `vgg16_fc2.npy` | float32 matrix, shape (8000, 4096) |
| `vgg16_fc2_ids.txt` | 8,000 image ids; line n maps to row n |
| `vgg16_fc2_meta.json` | model, preprocessing, versions, hardware, SHA-256 checksums, and `runtime_seconds` — the wall time of the **last run only**, which is less than the total if the run was resumed (`resumed_from_row`) |

Row order is the sorted union of the frozen split, so it is deterministic across machines. Interrupting the
run is safe: progress is checkpointed per batch and re-running resumes from the last completed row.

## Semantic Pseudo-Classes (Stage 3)
Groups the 6,000 **training** images into 20 semantic pseudo-classes from their captions, to serve as the
class variable for Fisher Score feature selection. TF-IDF and K-Means are fitted on training captions only;
no labels are produced for validation or test images.

```bash
python -m retinex.pseudoclasses fit         # fits on training captions, then validates
python -m retinex.pseudoclasses validate    # re-check existing artifacts against the frozen split
```

Frozen configuration (see `CLAUDE.md`, STAGE 3 DECISIONS): one document per image from its 5 captions in
`#n` order; TF-IDF `token_pattern=r"[a-z]{2,}"`, `min_df=2`, unigrams, L2 norm, no stop list; K-Means
`k=20, random_state=42, n_init=10, algorithm="lloyd"`. **K=20 is frozen in advance and enforced** — `fit`
and the CLI both refuse any other k, with no override. The cosine silhouette sweep over K in
{5,10,20,30,50} is recorded for transparency and cannot select K.

Writes to `artifacts/pseudoclasses/` (git-ignored):

| File | Contents |
| --- | --- |
| `train_pseudoclasses.csv` | `image_id,pseudo_class` for the 6,000 training images, sorted by id |
| `train_vocabulary.txt` | training vocabulary with IDF values |
| `pseudoclass_meta.json` | config, seed, vocab size, cluster sizes, inertia, silhouette sweep (metric, seed, sampling, per-candidate score or reason), top terms per cluster, checksums |

`validate` fails if a cluster holds fewer than 10 images, or if the labels do not cover the official 6,000
training images. Both are frozen constants, not arguments — the public validators take no `sizes`,
`dim` or `min_cluster_size`, so validation of the real experiment cannot be weakened by a caller. It also
fails if any validation/test image carries a label, or
if the feature id list has changed since fitting. An image left with no tokens after cleaning is an error
naming the image id, never a silent all-zero row.

## Fisher Score Feature Selection (Stage 4)
Ranks the 4096 VGG16 `fc2` features by Fisher Score computed from the **6,000 training rows and their
training-only pseudo-class labels**, keeps the top 512, and materialises an 8000 x 512 matrix by applying
those fixed column indices to all rows. Applying training-derived indices to validation/test rows is a
transform, not a fit.

```bash
python -m retinex.fisher select      # scores, selects, materialises, then validates
python -m retinex.fisher validate    # re-check existing artifacts against the frozen 8000 x 512 shape
```

Frozen formula (see `CLAUDE.md`, STAGE 4 DECISIONS), means over training rows only:

```
F_j = sum_c n_c (mu_cj - mu_j)^2  /  sum_c sum_{i in c} (x_ij - mu_cj)^2
```

Population variance (`ddof=0`), which makes the denominator exactly the count-weighted
`sum_c n_c sigma^2_cj`; two-pass float64; no standardisation; **no epsilon**. Zero denominators branch to
`+inf` (separable) or `0.0` (constant feature), both counted in the metadata. `+inf` is reserved for a
genuinely zero denominator: an overflowing ratio from a positive denominator raises instead, and `validate`
enforces `n_infinite_score == n_zero_denominator_with_signal`. Top-512 is frozen — `select` and the CLI
refuse any other value.

Writes to `artifacts/fisher/` (git-ignored):

| File | Contents |
| --- | --- |
| `fisher_scores.npy` | float64, 4096 scores including the `inf`/`0.0` cases |
| `fisher_top512_indices.npy` | int32, 512 original indices, **ascending** |
| `vgg16_fc2_fisher512.npy` | float32, (8000, 512); row order matches `vgg16_fc2_ids.txt` |
| `fisher_meta.json` | formula, variance convention, class counts, boundary ties, edge-case counts, provenance digests, checksums |

`validate` independently re-derives the top-512 from the saved scores, checks the indices are ascending and
that no unselected feature outscores a selected one, compares every output column against its source column,
and verifies the original 4096-d matrix is unchanged.

Measured on the real features (2026-10-12): 512 of 4096 selected from the 6,000 training rows in 0.2 s;
scores min 0.0, median 0.0542, max 1.6783; 13 constant features scored 0.0; no infinite scores; no tie at
the cut (0.132982, highest unselected 0.132824); output (8000, 512) float32. See `HANDOFF.md` for the full
record. **Whether these 512 features beat the 4096-d baseline is unknown until the two LSTM BLEU
evaluations run.**

## Training
NOT IMPLEMENTED YET

## Evaluation
NOT IMPLEMENTED YET

## Inference
NOT IMPLEMENTED YET

## Web Application
A Flask application with two surfaces: the Workspace, which runs the implemented part of the
pipeline on one uploaded image, and Home, which explains the method and shows the real stage status.

```bash
source .venv/bin/activate
python -c "from app import create_app; create_app().run(port=5055)"
```

| Surface | URL | What it does |
| --- | --- | --- |
| Home | http://127.0.0.1:5055/ | The method, the experiment, and what has and has not been built |
| Workspace | http://127.0.0.1:5055/workspace | Upload an image, extract real features, see the live pipeline status |
| Pipeline API | http://127.0.0.1:5055/api/pipeline | `GET`, returns the nine stages with their real status |
| Analyze API | http://127.0.0.1:5055/api/analyze | `POST` multipart `image=@file`, returns measured feature statistics |

**What the Workspace really does:** it loads your image through the same code path Stage 2 used
(`keras.utils.load_img(target_size=(224,224))` to `img_to_array` to `vgg16.preprocess_input`),
produces the 4096-d fc2 vector, then applies the **saved** Stage 4 top-512 indices to it. Fisher
Score is not refitted and no training artifact is written. The uploaded image is held in memory for
that one request and is not stored.

**What it does not do:** generate a caption or speak one. Stages 5 to 9 do not exist, so the
description and narration panels are marked NOT IMPLEMENTED and the playback controls are disabled.
No placeholder caption is ever shown.

Design tokens are generated from `design.md` rather than transcribed:
```bash
python scripts/build_tokens.py            # regenerate app/static/css/tokens.css
python scripts/build_tokens.py --check    # fail if the committed CSS is stale
```

## Testing
```bash
pytest                          # the whole suite
pytest tests/test_app.py        # the web application only
```
The suite uses synthetic fixtures throughout and needs neither Flickr8K nor the VGG16 weights: the
UI tests monkeypatch the model, so nothing downloads and no page render loads it.


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
