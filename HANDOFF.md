# HANDOFF

## Project
**RetineX** — Deep Learning Based Image Caption Generation using Fisher Score Feature Optimization, LSTM and
Neural Voice Narration for Real-Time Visual Accessibility. Academic major project.

## Current status: 0% implemented
Nothing below "Intended methodology" exists yet. Workspace, docs and a local Git repo only.

| Stage | Status |
|---|---|
| Workspace / docs / git | done (initial) |
| Dataset layer code + tests (`src/retinex/dataset.py`) | COMPLETED (16 synthetic tests pass) |
| Real Flickr8K (Kaggle copy) validated, frozen split generated and verified | COMPLETED (source provenance NOT VERIFIED against the official files) |
| VGG16 fc2 feature extraction | NOT IMPLEMENTED |
| TF-IDF + K-Means pseudo-classes | NOT IMPLEMENTED |
| Fisher Score mask (top 512) | NOT IMPLEMENTED |
| Baseline LSTM (4096-d) | NOT IMPLEMENTED |
| RetineX LSTM (512-d) | NOT IMPLEMENTED |
| BLEU-1..4 evaluation | NOT IMPLEMENTED |
| Inference | NOT IMPLEMENTED |
| gTTS narration | NOT IMPLEMENTED |
| Flask app | NOT IMPLEMENTED |
| Tests for stages after Stage 1 | NOT IMPLEMENTED |

## Origin and why a rebuild
Evolved from https://github.com/MiteshPuthran/Image-Caption-Generator (itself an implementation of
[Yumi's blog](https://fairyonice.github.io/Develop_an_image_captioning_deep_learning_model_using_Flickr_8K_data.html)).
Treated as reference only. The earlier effort had: outdated Keras/TF imports, dataset path problems,
notebook-heavy logic, incomplete integration, and unclear which parts actually worked.

## Reference repo findings (inspected remotely; nothing copied)
- Single notebook `Image Captioning 8k.ipynb` (32 code cells), plus committed `saved_models/*.h5` (~41 MB), TensorBoard logs, images.
- Outdated: `keras.backend.tensorflow_backend`, `tf.ConfigProto`/`Session`, `set_random_seed`, `Tokenizer(nb_words=)`, `layers.pop()` on the VGG model. Will not run on TF 2.x/Keras 3.
- Data: hard-coded `Data/Flickr8k_Dataset/` and `Data/Flickr8k.token.txt`; VGG weights from a local `.h5`.
- Captions: lowercased, punctuation/single-char/numeric tokens removed, `startseq`/`endseq` added. Only caption #0 per image is used.
- Features: VGG16 with last layer popped -> 4096-d fc2, 224x224, `preprocess_input`.
- Split: **60/20/20 by slicing in file order** (not shuffled, not the 6000/1000/1000 target).
- Model: image Dense(256) + Embedding(64, mask_zero) -> 2x LSTM(256) -> merge -> softmax over vocab; 6 epochs, batch 32. One training sample per (image, prefix).
- Inference: greedy argmax decoding.
- Evaluation: mean **sentence** BLEU against a single reference (not corpus BLEU, no BLEU-1..4 breakdown).
- No Fisher Score, no pseudo-classes, no Flask, and no TTS in code (TTS appears only as a README use case).
- Reusable ideas only: caption cleaning, start/end tokens, fc2 features, merge-style captioner, greedy decoding.

## Intended methodology
See CLAUDE.md (authoritative): Flickr8K, image-level 6000/1000/1000 frozen split, VGG16 fc2 4096-d,
TF-IDF + K-Means pseudo-classes (train only), Fisher Score top-512 mask (train only), baseline vs RetineX LSTM,
unsmoothed corpus BLEU-1..4, gTTS, Flask.

## Planned stages
1. Dataset prep + frozen split 2. Feature extraction 3. Pseudo-classes 4. Fisher mask 5. Baseline training
6. RetineX training 7. BLEU evaluation 8. Inference 9. Flask + TTS 10. Docs/tests polish.

## Warnings for future developers
- Do not trust the reference repo's code or numbers.
- Never fit anything on val/test data (see leakage rules in CLAUDE.md).
- Only record numbers produced by this repo. Machine is an M2 with 8 GB RAM and ~16 GB free disk: batch feature extraction, keep memory in mind.

## Stage 1: dataset foundation and frozen split
**Objective:** reproducible Flickr8K loading, validation, and a frozen image-level 6000/1000/1000 split. Nothing else.

**Layout:** `data/flickr8k/{images,captions/Flickr8k.token.txt,splits}`; root configurable (`--data-dir`, `RETINEX_DATA_DIR`).
Images and captions are git-ignored; `data/flickr8k/splits/` is the one exception so the frozen split can be committed (IDs only).

**Code:** `src/retinex/dataset.py` (`parse_captions`, `validate`, `make_split`, `write_split`, `load_split`, `verify_split`, CLI
`python -m retinex.dataset validate|split|verify`). Tests: `tests/test_dataset.py`.

**Split method:** sorted unique valid IDs -> `random.Random(42).shuffle` -> first 6000 train, next 1000 val, next 1000 test.
Valid = image file exists, has >=1 caption, decodes with Pillow. `split_meta.json` records seed, sizes, population,
unused count and SHA-256 of each split file; `write_split` refuses to overwrite and `verify` detects edits.

**Decisions:**
- Real Flickr8K reportedly has slightly more than 8,000 images (commonly cited 8,091) and a stray malformed ID in the token file. NOT VERIFIED here. So the split takes exactly 8,000 of the valid images and leaves the surplus unused, reported not silently dropped. Fewer than 8,000 valid images is an error.
- Problem records (malformed, duplicate, missing, corrupt, uncaptioned) are printed by `validate`; `split` excludes them but still prints them.
- Only the `Flickr8k.token.txt` format is parsed (no Kaggle `captions.txt` CSV).
- Caption parsing keeps raw caption text (stripped only); no lowercasing, tokens, or tokenization.

**Status:** see "Dataset status" below. (Original Stage 1 caveat about missing real data is now resolved.)

## Dataset status (real data)
**Original dataset source:** Hodosh/Young/Hockenmaier (JAIR 2013), distributed by the University of Illinois via request form
(https://forms.illinois.edu/sec/1713398; form verified live, link emailed, non-commercial research/education, do not redistribute).
**Why not used:** the user completed the form and the emailed link returned 404.

**Practical access source used:** Kaggle `adityajn105/flickr8k` v1, owner adityajn105, title "Flickr 8k Dataset", 1.04 GB, license field "CC0: Public Domain".
Downloaded with `kagglehub` (no credentials needed), cached outside the repo. Source `captions.txt` SHA-256:
`e6cf5e8d8025d25b50a4945c38580d96a940b78644f31a90368dfcdca3c7e036`.
Not the official distribution. Kaggle description only copies the paper abstract; no provenance statement, no splits, no `Flickr8k.token.txt`.

**Source verification status:** PARTIAL. Evidence it is the original corpus: 8,091 images (the commonly cited original image count), exactly 5 captions each,
every filename matches `\d+_[0-9a-f]{10}.jpg`, first caption is the well-known original one, content matches the paper's description. NOT verified:
byte-identity of images/captions with the official release (no official copy to compare). It has 40,455 records vs the original token file's
reported 40,460 (the extra 5 are, to our recollection, a stray `.jpg.1` entry; NOT VERIFIED). Treat as "very likely the original corpus, repackaged".

**Format:** `Images/*.jpg` + CSV `captions.txt` (`image,caption`, quoted when containing commas). Converted by `python -m retinex.dataset normalize --source <dir>` to
`data/flickr8k/captions/Flickr8k.token.txt` (`<img>#<n>\t<caption>`, n = row order within image) and images copied to `data/flickr8k/images/`. Source untouched.
Kaggle's own split (none provided) is not used.

**Measured statistics (validator on normalized data):** 8,091 image files; 8,091 captioned images; 40,455 captions; captions/image {5: 8091};
missing images 0; uncaptioned images 0; malformed records 0; duplicate (image, index) records 0; unreadable images 0; non-.jpg files 0.
**Anomalies (reported, not hidden):** 10 caption rows are exact duplicates of another caption for the same image (same text written twice); kept, since they are legitimate
separate annotations. 12 captions contain `#` and 2,447 contain commas; both handled correctly (tested). Surplus: 8,091 valid - 8,000 = 91 images unused.

**Split results:** seed 42, sorted IDs, `random.Random(42).shuffle`, slice. train 6000 / val 1000 / test 1000; union 8000; all pairwise overlaps 0; 91 unused (not recorded in a file; = images not in any split).
Independent re-derivation from the image directory reproduced the frozen files exactly. `verify` passed. SHA-256:
- train.txt `c0e7c01a642c82c16ffdc42e934d9e61f77ffcfb171095cc3c76b7be8b936ddb`
- val.txt `40d386a4949c3cf0b225306c773aa0ac99340eff7eb1a939f2ccd038e6363e06`
- test.txt `6271db17191acce75fedcc06771d73c35ddb291a13244abc738d1f82c7ce6355`
Files: `data/flickr8k/splits/{train,val,test}.txt`, `split_meta.json` (committed; images and captions are not).
**Tests:** 20 pytest (synthetic) pass + real-data validate/split/verify executed.
**Manual steps required:** `pip install kagglehub`, download, `normalize`, then `validate`/`split` or just use the committed split with `verify`. **Limitation:** the committed split only matches a dataset whose filenames equal these IDs.
**License/access notes:** Kaggle says CC0 (uploader-asserted, unverified). Illinois terms: non-commercial research/education, no redistribution. Do not commit images or captions. Cite the JAIR paper.
**Effect on later stages:** the dataset is a third-party copy; state this in the final report.

**For future stages:** read the split via `retinex.dataset.load_split`; never call `make_split`. Split IDs are filenames incl. `.jpg`. Captions come from `parse_captions` (raw text, no tokenization yet).
