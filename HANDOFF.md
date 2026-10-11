# HANDOFF

## Project
**RetineX** — Deep Learning Based Image Caption Generation using Fisher Score Feature Optimization, LSTM and
Neural Voice Narration for Real-Time Visual Accessibility. Academic major project.

## Current status: Stages 1-2 complete
**Implemented and verified in this repo:** Stage 1 — the dataset layer (`src/retinex/dataset.py`), Flickr8K
(third-party Kaggle copy) validation, and the frozen image-level 6000/1000/1000 split, committed and passing
`verify`. Stage 2 — VGG16 fc2 feature extraction (`src/retinex/features.py`), **complete: all 8,000 rows of a
(8000, 4096) float32 matrix**, passing `python -m retinex.features validate`. 41 pytest tests pass
(20 dataset, 21 features), all on synthetic fixtures.

**Not implemented:** everything after Stage 2 — pseudo-classes, Fisher mask, both LSTM models, BLEU, inference,
gTTS and the Flask app. No model has been trained, no caption generated and no BLEU score measured in this
repo. The per-stage table below is authoritative; treat any stage not marked COMPLETED as nonexistent.

| Stage | Status |
|---|---|
| Workspace / docs / git | done (initial) |
| Dataset layer code + tests (`src/retinex/dataset.py`) | COMPLETED (20 synthetic tests pass) |
| Real Flickr8K (Kaggle copy) validated, frozen split generated and verified | COMPLETED (source provenance NOT VERIFIED against the official files) |
| VGG16 fc2 feature extraction | COMPLETED (8,000 / 8,000 rows; `validate` passes; see Stage 2 below) |
| TF-IDF + K-Means pseudo-classes | NOT IMPLEMENTED |
| Fisher Score mask (top 512) | NOT IMPLEMENTED |
| Baseline LSTM (4096-d) | NOT IMPLEMENTED |
| RetineX LSTM (512-d) | NOT IMPLEMENTED |
| BLEU-1..4 evaluation | NOT IMPLEMENTED |
| Inference | NOT IMPLEMENTED |
| gTTS narration | NOT IMPLEMENTED |
| Flask app | NOT IMPLEMENTED |
| Tests for stages after Stage 1 | PARTIAL — `tests/test_features.py` only (21 tests, synthetic); Stages 3+ NOT IMPLEMENTED |

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

## Stage 2: VGG16 fc2 feature extraction — COMPLETED
**Status: COMPLETED.** All 8,000 frozen images extracted into a (8000, 4096) float32 matrix.
`python -m retinex.features validate` passes: "shape, dtype, ids, split membership, finiteness and checksums
verified". The run was interrupted once at row 3,168 (critically low battery) and resumed from that
checkpoint; `vgg16_fc2_meta.json` records `resumed_from_row: 3168`.

**Method:** ImageNet-pretrained VGG16, `include_top=True`, truncated at the `fc2` layer, `trainable = False`.
Images loaded at 224x224 via `keras.utils.load_img` -> `img_to_array` -> `keras.applications.vgg16.preprocess_input`,
batches of 32, `predict_on_batch`. Row order is the sorted union of the frozen split, so it is deterministic
across machines and independent of split membership.

**Leakage position:** VGG16 is pretrained and frozen, so nothing is fitted on this data and extracting
val/test rows alongside train rows transfers no information. Captions are never opened by this module
(`uses_captions: false`, asserted structurally by `test_extraction_needs_no_captions`). The train-only
fitting rules in CLAUDE.md bind Stage 3 (TF-IDF/K-Means) and Stage 4 (Fisher), not this stage.

**Artifacts** (`artifacts/features/`, git-ignored, 131,072,128 bytes for the matrix):

| File | Contents |
| --- | --- |
| `vgg16_fc2.npy` | float32, shape (8000, 4096). SHA-256 `4f6a5aa0e43eba43ac9db8b06165568a53bd9d5781e91e6d457bfb169e47ec26` |
| `vgg16_fc2_ids.txt` | 8,000 ids, line n maps to row n. SHA-256 `622693cb7bc46ea7e49c3e88546a030ea45eb7f98795e300b6669e7b27b55eb7` |
| `vgg16_fc2_meta.json` | model, preprocessing, versions, hardware, split counts, `status: "complete"`, both checksums |
| `vgg16_fc2.progress.json` | Absent, as intended: `extract` deletes it after writing metadata. Its absence is the completion marker. |

**Measured consistency checks (2026-10-11, read-only)**
- `vgg16_fc2_ids.txt` is identical to the sorted union of the frozen split (`ids == frozen_ids`), and all
  6000/1000/1000 split members resolve to a row.
- `ids_sha256` in the metadata matches the id list recomputed from the split; `artifact_sha256` for both the
  matrix and the id file matches the files on disk.
- `data/flickr8k/splits/split_meta.json` SHA-256 recorded in the metadata as
  `efaa4042653e1fcbf0de2f0a8a5071cbea5cec212c791f6d87b6715e3586c740`; `dataset verify` still passes.

**Measured feature statistics** (whole matrix, float64 accumulation)
- mean 0.5572, min 0.0000, max 27.8475; no NaN or infinite values; **zero all-zero rows**.
- 69.9% of entries are exactly 0.0, which is expected: `fc2` is ReLU-activated, so sparsity is a property of
  the layer, not a sign of missing data.

**Timing — read this before quoting it.** `runtime_seconds: 453.7` in the metadata is the wall time of the
**resumed run only** (rows 3,168-7,999, i.e. 4,832 images => about 10.7 images/s, CPU-only). The first run's
elapsed time was never recorded, so **the total wall time for all 8,000 images is not measured and must not be
stated**. Hardware per the metadata: macOS 26.6.2 arm64 (M2), devices `["/physical_device:CPU:0"]` — no GPU or
Metal backend was used. Versions: Python 3.12.0, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.3.

**Reproducing or re-running**
- `python -m retinex.features validate` re-checks the existing artifacts without recomputing them.
- `python -m retinex.features extract` is resumable per batch: rows are written and `flush()`ed before the
  progress file advances, so an interruption costs at most one batch. On resume it accepts a checkpoint only
  when the recomputed id digest and the recorded shape both match.
- **Never pass `--no-resume`**, and never delete, move or regenerate the artifacts: a full re-extraction is
  expensive on this CPU-only machine (at the one measured rate of ~10.7 images/s it would be on the order of
  12-13 minutes, an extrapolation and not a measurement) and the checksums in the metadata would no longer
  describe the committed record.

**Known limitations of the safeguards (not defects in the current artifacts)**
- The `.npy` header shape is never compared with the expected shape on resume; `open_memmap(mode="r+")`
  ignores its `shape` argument (verified), so the resume gate trusts the progress file's own `shape` field.
  A short matrix plus a matching progress file would fail with a broadcast `ValueError` rather than being
  rejected up front.
- The progress file holds only a row count, no checksum of written rows, so it could not have detected a row
  that was flushed but corrupted. The finiteness, sparsity and no-all-zero-row checks above are the
  substitute evidence, and the final metadata now carries a whole-matrix SHA-256.
- If the frozen split ever changes, `extract` prints "existing artifacts do not match the frozen split;
  starting over" and reopens the matrix with mode `w+`, zeroing it. The frozen split must stay frozen.
- `json.loads` on the progress file is unguarded and that file is not written atomically, so a crash during
  its 119-byte write would make the next resume abort with a traceback. Fail-loud, not data-losing.

**Code** `src/retinex/features.py` (`frozen_ids`, `extract`, `validate_features`, CLI `extract|validate`) and
`tests/test_features.py` (21 synthetic tests: dtype, shape, row count, duplicate/unexpected/missing ids,
NaN/inf, tampered ids and matrix, incomplete status, malformed metadata, caption independence, resume skips
completed rows, resume rejected on a changed split). Stage 2 added `numpy`, `tensorflow>=2.16` and `keras>=3`
to `pyproject.toml`; before that only `pillow` was declared and a clean clone could not import this module.
