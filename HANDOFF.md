# HANDOFF

## Project
**RetineX** — Deep Learning Based Image Caption Generation using Fisher Score Feature Optimization, LSTM and
Neural Voice Narration for Real-Time Visual Accessibility. Academic major project.

## Current status: Stages 1-4 complete; web UI serving the implemented pipeline
**Implemented and verified in this repo:** Stage 1 — the dataset layer (`src/retinex/dataset.py`), Flickr8K
(third-party Kaggle copy) validation, and the frozen image-level 6000/1000/1000 split, committed and passing
`verify`. Stage 2 — VGG16 fc2 feature extraction (`src/retinex/features.py`), **complete: all 8,000 rows of a
(8000, 4096) float32 matrix**, passing `python -m retinex.features validate`. 291 pytest tests pass
(20 dataset, 23 features, 67 pseudo-classes, 77 Fisher, 104 app). All are synthetic except the few
that check the one committed Fisher index file against its pinned digest.

Stage 3 — TF-IDF + K-Means semantic pseudo-classes (`src/retinex/pseudoclasses.py`), **complete: 6,000
training images labelled into 20 clusters**, passing `python -m retinex.pseudoclasses validate`. Measured
results are in the Stage 3 section below.

Stage 4 — Fisher Score feature selection (`src/retinex/fisher.py`), **complete: 4096 -> 512 features
selected from the 6,000 training rows, 8000x512 matrix materialised**, passing
`python -m retinex.fisher validate`. Measured results are in the Stage 4 section below.

Web UI — a Flask application with a Workspace and a Home page (`app/`), serving the implemented
pipeline only. See the UI section below for measured timings.

**Not implemented:** both LSTM models, BLEU evaluation, caption inference and gTTS narration. No model has been trained, no caption generated and no BLEU score measured in this repo. The
per-stage table below is authoritative; treat any stage not marked COMPLETED as nonexistent.

| Stage | Status |
|---|---|
| Workspace / docs / git | done (initial) |
| Dataset layer code + tests (`src/retinex/dataset.py`) | COMPLETED (20 synthetic tests pass) |
| Real Flickr8K (Kaggle copy) validated, frozen split generated and verified | COMPLETED (source provenance NOT VERIFIED against the official files) |
| VGG16 fc2 feature extraction | COMPLETED (8,000 / 8,000 rows; `validate` passes; see Stage 2 below) |
| TF-IDF + K-Means pseudo-classes | COMPLETED (6,000 labels, k=20, smallest cluster 67; `validate` passes; see Stage 3 below) |
| Fisher Score mask (top 512) | COMPLETED (512 of 4096 selected; no boundary tie; `validate` passes; see Stage 4 below) |
| Baseline LSTM (4096-d) | NOT IMPLEMENTED |
| RetineX LSTM (512-d) | NOT IMPLEMENTED |
| BLEU-1..4 evaluation | NOT IMPLEMENTED |
| Caption inference | NOT IMPLEMENTED (single-image *feature* extraction works in the Workspace) |
| gTTS narration | NOT IMPLEMENTED (UI shows disabled controls and an explanation) |
| Flask app (Workspace + Home) | COMPLETED for the implemented stages (104 tests; no caption or narration, by design) |
| Tests for stages after Stage 1 | PARTIAL — `tests/test_features.py` (23), `tests/test_pseudoclasses.py` (67), `tests/test_fisher.py` (77); Stages 5+ NOT IMPLEMENTED |

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
- `python -m retinex.features validate` re-checks the existing artifacts without recomputing them. The
  public `validate_features(out_dir, data_dir)` takes no `sizes` or `dim`: it reads the frozen
  6000/1000/1000 split and 4096-d width from constants, so the real artifacts cannot be validated against a
  weaker shape. The parameterised `_validate_features` exists only for the synthetic tests.
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

## Stage 3: TF-IDF + K-Means semantic pseudo-classes — COMPLETED
**Status: COMPLETED.** `python -m retinex.pseudoclasses fit` ran on the real Flickr8K captions on
2026-10-12 with the frozen defaults and no flags; `python -m retinex.pseudoclasses validate` passes:
"train-only coverage, sorted ids, label range, cluster sizes, feature alignment and checksums verified".
All numbers below are measured from that run.

**Measured results (2026-10-12T00:48:24+0530)**
- **6,000 training documents** from **30,000 training captions** (5 per image, `#n` order).
- TF-IDF matrix **6000 x 4172**, 158,808 non-zeros; **vocabulary 4,172 terms** after `min_df=2`.
- K-Means k=20 converged in **29 iterations**, **inertia 4897.982982**.
- **Cluster sizes** (sum 6,000, smallest **67**, largest **908**, all >= the 10-image floor):
  `{0: 406, 1: 908, 2: 229, 3: 316, 4: 256, 5: 472, 6: 237, 7: 101, 8: 179, 9: 255, 10: 556, 11: 230,
  12: 114, 13: 312, 14: 67, 15: 307, 16: 344, 17: 117, 18: 463, 19: 131}`
- **Cosine silhouette, full training set, no sampling:** k=5 **0.047472**, k=10 **0.061836**,
  k=20 **0.063000**, k=30 **0.077254**, k=50 **0.086464**. The k=20 entry reuses the production labelling.
- **Honest observation about the sweep:** the silhouette rises monotonically with K, so **k=50 scored
  highest and k=20 did not**. K stayed at 20 because it was frozen in advance and the sweep is diagnostic
  only (`k_selected_by_sweep: false`). The final report must state this rather than implying k=20 was
  the best-scoring choice. All these silhouettes are low in absolute terms, which is normal for sparse
  short-text TF-IDF and means the clusters are weakly separated, not that they are meaningless.
- **Top terms per cluster** (from the centroids, in `pseudoclass_meta.json`) are semantically coherent:
  c2 snow/snowboarder/skier, c4 player/soccer/ball/football, c7 pool/swimming, c9 bike/bicycle/riding,
  c12 rock/climbing/cliff, c14 wave/surfer/surfboard/wetsuit, c17 beach/sand/ocean, c19 skateboard/ramp/trick,
  with the remainder split by subject (man, woman, boy, girl, dogs, groups). Function words such as "in",
  "the" and "on" still appear among top terms: a direct consequence of the approved decision to use no
  stop-word list. `min_df=2` and IDF reduce their weight but do not remove them.
- **Artifact checksums** recorded in the metadata and re-verified against the files after the run:
  `train_pseudoclasses.csv` SHA-256 `913197e79701215e0c1871932e82322564a03a64d2e9ef1dcf30f960cb76ca92`,
  `train_vocabulary.txt` SHA-256 `0f3a6010f5f5e4d80b5dad6018942a1c8cc3d517efc2281242cea05aad7eefd4`.
  The training-id digest is `c0e7c01a642c82c16ffdc42e934d9e61f77ffcfb171095cc3c76b7be8b936ddb`, identical
  to the frozen `train.txt` checksum, as expected since both hash the same sorted id list.
- Environment: Python 3.12.0, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.3.

**Post-run integrity checks (all passed)**
- 6,000 label rows, exactly the sorted training split; `train_ids_sha256` matches.
- **No validation or test image carries a label**, and no `val_*`/`test_*` label file exists.
- `feature_ids_sha256` and `split_meta_sha256` both still match the live files, so Stage 4 row alignment
  holds; every labelled id resolves to a row of `vgg16_fc2.npy`.
- `features validate` and `dataset verify` still pass; Stage 1/2 artifacts untouched.

**Purpose:** produce one semantic pseudo-class label per *training* image, from captions, to act as the class
variable for Stage 4's Fisher Score. Stage 3 never opens the 131 MB feature matrix; it reads
`vgg16_fc2_ids.txt` only to checksum it for row alignment.

**Frozen methodology** (approved 2026-10-12, recorded in CLAUDE.md under STAGE 3 DECISIONS)
- One document per training image: that image's captions joined in `#n` order. 6,000 documents.
- `TfidfVectorizer(lowercase=True, token_pattern=r"[a-z]{2,}", min_df=2, ngram_range=(1,1), norm="l2",
  sublinear_tf=False)`, no stop-word list. `sublinear_tf=False` is deliberate: a word repeated across an
  image's captions is a salience signal worth keeping linear.
- `KMeans(n_clusters=20, random_state=42, n_init=10, algorithm="lloyd", max_iter=300)`. L2-normalised rows
  make Euclidean K-Means equivalent to spherical/cosine clustering.
- **K=20 frozen in advance and mechanically enforced.** `fit` raises `ValueError` on any other k before
  touching the data, and the CLI refuses `--k` != 20 before any work. There is no override: the sweep is not
  a function parameter at all (`test_fit_exposes_no_sweep_override` asserts that), so it cannot be used as a
  back door to move K.
- **Silhouette:** `metric="cosine"` on the L2-normalised training TF-IDF matrix, `sample_size=None` (every
  training document, no sampling), `random_state` = the recorded seed, every candidate fitted on the training
  documents only. The metadata records each candidate's score, the metric, the seed and the sampling setting;
  a candidate whose silhouette is undefined gets `score: null` plus an `unavailable` reason, never a number.
  The frozen k=20 entry reuses the production labelling rather than refitting.

**Leakage position:** `train_documents` is the only document builder and receives the training ids alone;
`fit` asserts the document id set equals the training split and intersects neither val nor test before
fitting. No val/test labels are produced at all, and `validate` reports a stray `val_*`/`test_*` label file
as a leakage risk. `test_fit_uses_training_captions_only` spies on the vectorizer and asserts the exact
document list reaching `fit_transform`; `test_val_test_vocabulary_never_learned` asserts marker words that
occur only in held-out captions are absent from the fitted vocabulary.

**Artifacts** (`artifacts/pseudoclasses/`, git-ignored by the existing `artifacts/*` rule — verified with
`git check-ignore`; no .gitignore change was needed). Written 2026-10-12: `train_pseudoclasses.csv` 169,341 B,
`train_vocabulary.txt` 68,256 B, `pseudoclass_meta.json` 7,549 B.

| File | Contents |
| --- | --- |
| `train_pseudoclasses.csv` | `image_id,pseudo_class`, 6,000 rows, sorted by image id |
| `train_vocabulary.txt` | training vocabulary with IDF values, tab-separated |
| `pseudoclass_meta.json` | full config, seed, vocab size, cluster sizes, inertia, silhouette, top-15 terms per cluster, `uses_val_test_captions: false`, `split_meta_sha256`, `feature_ids_sha256`, artifact checksums, `status: "complete"` |

**Validation rules** (`python -m retinex.pseudoclasses validate`): artifacts present; metadata parses and
says `status: complete`, `fitted_on: train`, `uses_val_test_captions: false`, `k_selected_by_sweep: false`,
**`k` exactly 20** and `min_cluster_size_required` at least 10; **no cluster below 10 training images**,
checked against the label file using the constant `MIN_CLUSTER_SIZE = 10`; the sweep block present with
`candidates_expected == [5,10,20,30,50]`, exactly those candidate keys, `metric == "cosine"`,
`sample_size is None`, `fitted_on == "train"`, a `random_state` matching the seed, and no candidate carrying
a null score without a stated reason;
label checksums unchanged; exactly the 6,000 training ids, sorted, no duplicates; no val/test id labelled;
labels within `[0, k)`; no empty cluster; **no cluster below 10 images** (a near-singleton class makes the
Fisher within-class variance unstable); recorded `cluster_sizes` match the label file; `vgg16_fc2_ids.txt`
unchanged since fitting and every labelled id resolvable to a feature row.

**Frozen validation API:** `validate_pseudoclasses(out_dir, data_dir, features_dir)` accepts no `sizes` and
no `min_cluster_size`; it reads `SIZES` (6000/1000/1000) and `MIN_CLUSTER_SIZE` (10) from constants, so the
real artifacts cannot be validated against a weaker rule, and passing either argument is a `TypeError`. The
split-size-parameterised `_validate_pseudoclasses` is private and used only by the synthetic tests, which
still cannot relax the cluster floor because that stayed a constant on both paths.

**Edge cases handled:** an image with no captions, or none left after cleaning, raises naming the ids;
captions containing `#` or commas tokenize correctly (tested); `fit` refuses to overwrite existing artifacts
without `--force`.

**Measured cost of the sweep** (synthetic 6000x8000 L2-normalised sparse matrix at the real shape, not the
real captions): five K-Means fits totalling 1.3 s and five cosine silhouettes totalling 1.5 s, about 2.8 s in
all, peak RSS 0.84 GB. sklearn computes the silhouette through `pairwise_distances_chunked` under a 1024 MiB
working-memory budget, so the 6000x6000 distance matrix is never materialised in full. **No sampling is
needed**, and none is used.

**Risks and limitations (post-run)**
- Cluster sizes are **uneven as expected**: 67 to 908 against a 300 average. Nothing fell below the
  10-image floor, so Fisher within-class variance is computable for every class, but the largest cluster
  holds 15% of the training set and the smallest 1.1%, which will weight the Fisher ratio unevenly.
  This is a measured property of the data, not a defect to tune away.
- Cross-machine bitwise determinism is **not** guaranteed: identical labels are asserted in-process for a
  fixed seed, but threaded BLAS may differ across builds. The label-file SHA-256 and inertia are recorded so
  any drift is detectable.
- The whole `fit`, including the five-candidate sweep, completed in well under a minute on this CPU-only
  machine; the exact wall time was not instrumented, so no runtime figure is recorded.
- `scikit-learn` 1.9.1, `scipy` 1.18.1, `joblib` 1.6.0 and `threadpoolctl` 3.7.0 were installed for this
  stage; scipy requires Python >= 3.12, consistent with the project floor.

**Code** `src/retinex/pseudoclasses.py` (`train_documents`, `fit`, `read_labels`, `validate_pseudoclasses`,
CLI `fit|validate`) and `tests/test_pseudoclasses.py` (67 tests: documents, alignment, five leakage tests,
frozen-K enforcement in both `fit` and the CLI, absence of a sweep override, sweep metric/seed/sampling
recording, undefined-silhouette handling, determinism, frozen-config recording, label range, the real
10-image cluster floor, metadata drift and sweep tampering, malformed CSV/JSON, overwrite protection). The
train-only guard in `fit` raises `ValueError` rather than asserting, so `python -O` cannot strip the check
that enforces the methodology.

**Test fixture note:** the synthetic fixture uses **240** training images over 20 purely alphabetic themes, so
the frozen k=20 produces 12 images per cluster and legitimately clears the 10-image floor. The words are
alphabetic because `token_pattern=r"[a-z]{2,}"` strips digits — numbered fixture words like `theme1`/`theme2`
collapse to `theme`, leaving identical, mutually orthogonal documents that make K-Means degenerate. Neither K
nor the cluster floor is relaxed for the tests: production refuses any k but 20, and the floor has no
parameter. An undersized cluster is tested by rewriting the label file and validating through the normal path.

## Stage 4: Fisher Score feature selection — COMPLETED
**Status: COMPLETED.** `python -m retinex.fisher select` ran on the real feature matrix on 2026-10-12 with
the frozen defaults and no flags; `python -m retinex.fisher validate` passes: "formula metadata,
independent top-512 re-derivation, ascending column mapping, source-column equality, training join and
checksums verified". All numbers below are measured from that run.

**Measured results (2026-10-12T01:31:14+0530)**
- Joined **6,000 training ids** to feature rows **by image id**; 20 pseudo-classes, counts 67-908.
- Scoring and selection took **0.2 s** at `chunk_rows=500` (1.7 s total CLI wall time including imports
  and the upstream validation gate). Two passes over the 6,000 training rows in float64.
- **Zero-denominator counts: 13 with zero numerator (constant features), 0 with signal.** The 13 match the
  read-only variance diagnostic exactly; they are the all-zero fc2 columns.
- **Infinite scores: 0.** The invariant `n_infinite_score == n_zero_denominator_with_signal` holds (0 == 0),
  so no overflow occurred and no feature was perfectly separable.
- **Zero scores: 13** — identical to the constant-feature count, i.e. no feature had within-class spread
  yet identical class means.
- **Score distribution** (all 4,096 finite): min 0.0, p25 0.0320, median 0.0542, p75 0.0925, p90 0.1466,
  p99 0.3297, max **1.6783**; mean 0.0766.
- **Boundary: no tie.** Cut score **0.13298203700205952**, `tied_at_cut_total: 1`, included 1, excluded 0.
  Selected scores span 0.132982-1.678307; the highest unselected score is 0.132824, strictly below the cut.
- **Selected indices** span original columns 2-4095, strictly ascending, int32.
- **Output:** `vgg16_fc2_fisher512.npy`, shape **(8000, 512)**, dtype **float32**, all finite,
  16,384,128 bytes; row order identical to `vgg16_fc2_ids.txt`.

**Artifacts** (`artifacts/fisher/`, git-ignored by the existing `artifacts/*` rule)

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `fisher_scores.npy` | 32,896 | `6d9da06107ab4c4a8ff7af79ae1ebce156471eb481e83be6f63519ba7cafb78b` |
| `fisher_top512_indices.npy` | 2,176 | `5de508262443f0cf35dec1a5ee1326bce05f84413db296a85d72bb21ae21e749` |
| `vgg16_fc2_fisher512.npy` | 16,384,128 | `2ff02dc83008b6b999453c346074de851b98bf9ef778de3a1846f469e9c27468` |
| `fisher_meta.json` | 3,584 | (records the three above) |

**Provenance digests recorded in the metadata** (all re-verified against the live files after the run):
source matrix `4f6a5aa0...e47ec26`, feature ids `622693cb...7b55eb7`, pseudo-class labels
`913197e7...cb76ca92`, train ids `c0e7c01a...b936ddb`, split meta `efaa4042...586c740`.

**Baseline preserved:** `vgg16_fc2.npy` SHA-256 is `4f6a5aa0e43eba43ac9db8b06165568a53bd9d5781e91e6d457bfb169e47ec26`
before and after the run, matching both the Stage 2 metadata and the Stage 4 provenance record. The source
was opened `mmap_mode="r"` only.

**Independent post-run verification (read-only, written separately from `validate_fisher`)**
- The top-512 re-derived with a plain Python sort on `(-score, index)` **matches the saved indices exactly**.
- Minimum selected score 0.132982037 > maximum unselected 0.132824037, so the cut is unambiguous.
- **All 4,096,000 output values** compared against the source gather: **0 mismatches**.
- Scores recomputed from scratch with a different chunk size (1000 vs 500) differ by at most **1.06e-15**
  and give the **identical ranking** — the chunk-order caveat is real but far below any effect on selection.
- 6,000 of 8,000 rows were read for fitting; the 2,000 held-out rows contributed nothing to the scores.

**Frozen formula** (approved 2026-10-12, recorded in CLAUDE.md under STAGE 4 DECISIONS), all means over
the 6,000 training rows only:

    F_j = SS_between,j / SS_within,j
    SS_between,j = sum_c  n_c (mu_cj - mu_j)^2
    SS_within,j  = sum_c  sum_{i in c} (x_ij - mu_cj)^2

`SS_within` is the **pooled within-class sum of squared deviations**, which equals the count-weighted
`sum_c n_c sigma^2_cj` exactly because the variance convention is **population, ddof = 0**. Verified
numerically in `test_pooled_ssd_equals_count_weighted_variance_form`, and
`test_variance_convention_is_population_not_sample` asserts a `ddof=1` denominator gives different scores.

**Numerics:** two-pass float64 (class means, then squared deviations), chosen over
`sum(x^2) - sum(x)^2/n` because that shortcut subtracts large near-equal numbers; on a synthetic check it
drifted ~15x further from the reference. A sum of squared deviations cannot be negative, so **no clamping**
is applied and a negative value raises as a bug. No epsilon, no standardisation (the ratio is
scale-invariant). Zero denominators branch to `+inf` (separable) or `0.0` (constant); no `0/0` is executed,
so no score can be NaN, and a non-finite input raises naming the offending rows.

**Join by image id, never row position:** `training_join` resolves every row index through
`vgg16_fc2_ids.txt` and requires the labelled ids to be a **one-to-one cover of the frozen training split**,
raising on a labelled val/test image, a missing label, an unknown id, a duplicate feature id, or two ids
resolving to the same row. The real training rows are **non-contiguous** (indices 0-7998), which the fixture
mirrors by writing the feature ids in a scrambled order.

**Selection:** stable argsort on the negated scores gives descending score with ties broken by ascending
original index; the chosen indices are then sorted ascending, and that array is the column mapping
(output column `i` is source column `idx[i]`). The cut score, whether the cut falls inside a tie, and the
included/excluded tied counts are all recorded.

**Planned artifacts** (`artifacts/fisher/`, git-ignored by the existing `artifacts/*` rule, with one
documented exception described below)

| File | Contents |
| --- | --- |
| `fisher_scores.npy` | float64 (4096,) — all scores, `inf` and `0.0` included |
| `fisher_top512_indices.npy` | int32 (512,), strictly ascending |
| `vgg16_fc2_fisher512.npy` | float32 (8000, 512) ≈ 16.4 MB; rows follow `vgg16_fc2_ids.txt` unchanged |
| `fisher_meta.json` | formula, variance convention, denominator form, algorithm, class counts, boundary block, edge-case counts, finite score statistics, `chunk_rows`, provenance digests (source matrix, feature ids, label file, both upstream metas, split meta, train ids), output checksums, `status: "complete"` |

**Validation rules** (`python -m retinex.fisher validate`): artifacts present; metadata parses and records
`status: complete`, `fitted_on: train`, `uses_val_test_data: false`, `uses_val_test_labels: false`,
`standardized: false`, a null epsilon, the exact variance convention, denominator form, selection rule and
column order, `n_selected` 512 and `n_features_in` 4096; checksums unchanged; scores shape (4096,) with no
NaN and no negatives; indices shape (512,) int32, in range, **strictly ascending**; an **independent
re-derivation** of the top-512 from the saved scores matching the saved indices; the recorded boundary block
matching that re-derivation; the argsort-independent property that **no unselected feature scores strictly
higher than a selected one**; the recorded zero/infinite counts matching the saved scores; the reduced
matrix (8000, 512) float32 and finite, with **every column compared against its source column**; the source
matrix byte-identical to the recorded digest; and the training join still holding.

**Expected counts are NOT hardcoded.** Validation checks the recorded counts only against the saved scores
for internal consistency. The pre-run diagnostic predicted 13 constant columns and the run measured 13;
that agreement is reported, not asserted anywhere in code.

**Known limitations**
- Bitwise reproducibility depends on `chunk_rows`, the numpy build and BLAS, because float64 summation
  order changes with chunking. `chunk_rows` is recorded in the metadata and a determinism note with it.
- **Overflow is distinguished from genuine separability.** `_ratio` reserves `+inf` for
  `SS_within == 0`; a non-finite ratio from a positive denominator raises `FloatingPointError` naming the
  feature and both scatter terms, so it can never inflate `n_zero_denominator_with_signal`. `validate`
  enforces the invariant `n_infinite_score == n_zero_denominator_with_signal` and checks the recorded
  `overflow_policy`. The branch is defensive, not expected: with a float32 source and 8,000 rows the worst
  finite ratio is bounded by `(8000 * float32_max^2) / (2 * (float32_subnormal/2)^2)` ~ 9.4e170, far below
  float64's 1.8e308, so **overflow is mathematically unreachable here** — asserted by
  `test_overflow_is_unreachable_from_a_float32_source`. It is tested directly against `_ratio` with
  hand-built scatter vectors rather than through a contrived matrix, so nothing platform-dependent is
  assumed.
- `n_zero_score` and `n_zero_denominator_zero_numerator` are different quantities: a feature with
  within-class spread but identical class means also scores 0.0 while having a positive denominator.
- **No performance claim.** Whether 512 Fisher features beat the 4096-d baseline is answerable only by the
  two LSTM BLEU evaluations in Stages 6-7. A negative result will be reported as such.

**One Stage 4 artifact is committed, as a documented exception (2026-10-13).**
`artifacts/fisher/fisher_top512_indices.npy` is in the repository. Everything else under
`artifacts/` stays out.

*Why.* The Workspace reduces an uploaded image's 4096-d vector to 512 by applying this file. Without
it `/api/analyze` can only return a 503, so a deployment from a clean checkout would ship a Workspace
that cannot do the one thing it does. The file is **2,176 bytes**, int32, shape (512,), strictly
ascending, min 2, max 4095. It is derived, not source: `python -m retinex.fisher select` reproduces it
from the frozen inputs. It carries no image, no caption, no feature value and no dataset content, only
512 column numbers.

*What stays out, and why the exception is narrow.* `fisher_scores.npy`, `vgg16_fc2_fisher512.npy`
(16.4 MB), `fisher_meta.json`, `vgg16_fc2.npy` (125 MB), the pseudo-class outputs, model weights and
the dataset are all still ignored. The `.gitignore` block re-includes exactly one path, and
`test_only_the_index_file_escapes_the_artifacts_ignore_rule` lists everything under `artifacts/` that
git would carry and fails if anything but that file and `artifacts/.gitkeep` appears.

Two mechanical traps, both found by testing rather than reasoning, are worth knowing before editing
that block: `artifacts/*` excludes the *directory*, so the directory has to be re-included before a
file inside it can be, and the exception must sit **after** the `*.npy` rule, because the last
matching pattern wins and a negation placed above it is silently ineffective.

*Integrity.* `fisher_meta.json` records this file's SHA-256 but is not committed, so the expected
digest is pinned as `EXPECTED_INDICES_SHA256` in `src/retinex/fisher.py`:
`5de508262443f0cf35dec1a5ee1326bce05f84413db296a85d72bb21ae21e749`. `fisher.verify_indices` hashes
the raw bytes and raises before anything parses them, so a tampered or truncated file never reaches
`np.load`. `app.inference.load_selection` runs that check on the production path and caches the
verified array for the life of the process behind a lock, the same double-checked pattern the model
uses, so nothing is re-hashed per request. A mismatch produces the existing controlled **503** with a
fixed message; the two digests go to the log, never to the client. A caller-supplied directory is the
test seam and gets the structural checks only, because the pinned digest describes the committed file
and a synthetic fixture is not it.

*If that digest ever mismatches,* do not edit the constant to make the test pass. It means the
committed artifact changed, which is a methodology event: re-run the Stage 4 selection deliberately,
confirm the result, then update the constant and this section together.
`test_pinned_digest_matches_the_committed_file` and `test_pinned_digest_matches_the_recorded_metadata`
hold the constant, the file and the Stage 4 metadata in agreement wherever the full artifact set
exists; the metadata test skips where it does not.

**Code** `src/retinex/fisher.py` (`training_join`, `fisher_scores`, `select_top`, `apply_selection`,
`select`/`_select`, `validate_fisher`/`_validate_fisher`, CLI `select|validate`) and
`tests/test_fisher.py` (69 tests: formula vs an independent reference, the two denominator forms,
ddof sensitivity, separating features, constant features, `+inf` handling, non-finite input, ill-conditioned
columns, tie-breaking including multiple infinities and a cut through an infinite tie, id-based join and its
five failure modes, held-out rows provably not influencing scores, output shape/dtype/mapping, source
immutability, provenance digests, overwrite protection, upstream-validation gate, frozen public API, CLI
top-k refusal, nineteen validation-failure cases, and six focused tests of the ratio/overflow policy).

## Web UI: Workspace and Home — COMPLETED for the implemented stages
**Status: COMPLETED** for what exists. Flask + Jinja2 + hand-written CSS + vanilla JavaScript. No npm,
no React, no build step. Run it with:

```bash
python -c "from app import create_app; create_app().run(port=5055)"
```
Home at `/`, Workspace at `/workspace`, plus `GET /api/pipeline` and `POST /api/analyze`.

**What the UI genuinely does**
- Accepts one uploaded image, validates it, and extracts a real 4096-d VGG16 fc2 vector by calling
  **the same functions Stage 2 used** (`features._build_model` and `features._load_batch`), so the
  preprocessing cannot drift from the committed artifacts. `test_single_image_path_reuses_the_stage_2_loader`
  asserts the tensor is `np.array_equal` to what `_load_batch` produces for the same file.
- Applies the **saved** `fisher_top512_indices.npy` to that vector. Fisher Score is never refitted and
  the app never writes to `artifacts/`; a test snapshots every Stage 2/3/4 checksum around an analyze
  call and asserts equality.
- Reports measured per-request statistics: dimensions, mean, max, L2 norm, zero fraction, the eight
  strongest selected dimensions with their original 4096-d index, and the request's own timings.
- Derives the nine-stage pipeline status by **running the four real validators**, so a missing or
  tampered artifact is reported instead of a green badge. Two tests prove it: empty directories yield
  four `missing` stages, and a monkeypatched failing validator surfaces as `failed`. **Home and
  Workspace render without running the validators at all**; `GET /api/pipeline` runs them when the
  status is explicitly asked for, and `test_pages_render_without_running_the_validators` fails if a
  page starts calling `pipeline.stages()` again.

**What it deliberately does not do.** There is no code path in `app/` that can return a caption, not
even a placeholder, because no caption model exists in this repository. The description panel and the
narration panel each carry an **"In development"** chip and a short note saying what is missing and
that nothing will be invented in its place; neither prints a stage number. The three playback buttons
carry a real `disabled` attribute plus `aria-describedby` explaining why, and the transcript section
is present but empty. No BLEU score, accuracy figure or latency claim appears in any template.

**Measured timings** (2026-10-12, Apple M2, 8 GB, CPU only, TensorFlow 2.21.0, no GPU/Metal backend)

| Measurement | Value |
| --- | --- |
| VGG16 fc2 model load | **5.52 s**, once per process (5.47 s on the live server) |
| First analyze request after load | **307.6 ms** (275 ms on the live server) |
| Warm analyze requests | median **109.3 ms**, range 105-115 ms over 9 runs |
| Peak process RSS | **1.14 GB** after one load plus six inferences |
| Page weight | Home 6,685 bytes (~6.5 KB), Workspace 8,684 bytes (~8.5 KB) of HTML |

These are wall-clock measurements on this machine, not a throughput claim. Stage 2's batch rate
(10.7 images/s at batch 32) is not comparable: single-image requests carry per-call overhead. The
timings are the original 2026-10-12 run and the UI redesign did not touch the code they measure; only
the page weight was re-measured, on 2026-10-13, because the redesign removed markup from both pages.

**Environment caveats**
- The ImageNet weights (~553 MB) are downloaded by Keras into `~/.keras/models/` on first use and are
  **not** a pip dependency. One online run is needed; everything else, including the whole test suite,
  works offline because the UI tests monkeypatch the model.
- Loading VGG16 costs roughly 1 GB of resident memory in the web process. On an 8 GB machine, do not
  run training and the web app at once.
- The model loads lazily and **once**: `get_model` uses double-checked locking because `Flask.run`
  sets `threaded=True`, so two concurrent first requests would otherwise both load it.
  `test_lazy_model_load_happens_once_under_concurrency` runs four threads and asserts a single build.
- `app.run()` is the Werkzeug development server. No production WSGI server, HTTPS, rate limiting or
  authentication is configured, and none is claimed.

**UI redesign (2026-10-13).** The pages were reworked from a research-report layout into a product
one. What changed:

- The nine-stage pipeline dashboard was **removed from both pages**, along with the Home "What exists
  right now" metrics band. It was a repository status report, not interface content. Nothing behind it
  was removed: `app/pipeline.py`, the four validators and `GET /api/pipeline` are untouched.
- Home is now hero, premise, three plain-language steps, closing call to action. Academic positioning
  was removed from every user-facing surface, and a test fails if "academic", "major project",
  "student", "examiner" or "coursework" reappears on either page. This document and `CLAUDE.md` still
  describe the project accurately as an academic major project; they are not UI.
- The navigation carries **one shared underline** that glides to the hovered or focused link and
  returns to the current page. It is a progressive enhancement in `app/static/js/ui.js`; without
  JavaScript the current link keeps a plain static border and no second marker is ever drawn.
- **Page transitions** use the browser's cross-document View Transitions API through a single
  `@view-transition { navigation: auto; }` rule: 120ms out, 260ms in with a 10px rise. No script
  intercepts a link, so new tabs, modifier clicks, downloads, fragments, Back, Forward and scroll
  restoration keep their native behaviour, and a browser without the API simply navigates. The header
  and the underline carry their own `view-transition-name` so the bar does not travel with the page.
- The hero entrance is a plain CSS animation (three items, 70ms apart) rather than a scripted reveal,
  so a blocked script cannot leave content hidden. Below-the-fold reveals use an IntersectionObserver,
  fire once, and are gated on a `js` class that is only added when the script runs.
- Buttons moved from `md` to `lg` radius, with `:focus-visible`, `:active` and disabled-hover states.
  The change is in the one shared `.btn` rule; `design.md` records the decision and its reason.

**NOT VERIFIED: the motion has never been watched running.** Chrome 155 was used to confirm that
`@view-transition` parses with `navigation=auto`, that every `::view-transition` selector is accepted,
and that the reduced-motion override keeps `animation-name: none !important`, and the pages were
screenshotted statically at 1440, 900 and 500px. Capturing frames mid-navigation needs a browser
driver that was deliberately not added. Treat the animation as checked through code, static rendering
and browser CSS support, not through observation.

**Upload handling.** The filename and the browser-supplied MIME type are never trusted. Validation
order: a file was sent; body size (Werkzeug `MAX_CONTENT_LENGTH` = 8 MB, re-checked on the received
bytes); Pillow can identify the format from the bytes; format in {JPEG, PNG, WEBP, BMP}; both sides
>= 32 px and <= 10,000 px with <= 40M pixels; then a full decode to prove it really loads.
`DecompressionBombError` is named explicitly because it is **not** an `OSError` or `ValueError` and
would otherwise escape as a 500. A text file renamed `.png` is rejected on its bytes.

**Error handling.** Every failure returns a controlled response and never a partial or fabricated
success: 400 for a bad upload, 413 for an oversized body, 405/404 as JSON on `/api/` paths, 503 when
the model or the saved selection cannot be loaded, and a fixed 500 JSON message for anything
unhandled. Underlying exception text is logged, not returned, so local paths and tensor shapes do not
reach the client. A corrupt `fisher_top512_indices.npy` and a non-ascending index array are both
refused with an actionable message.

**Design system.** `design.md` is the source of truth. `app/static/css/tokens.css` is **generated**
from its frontmatter by `scripts/build_tokens.py`; `--check` fails if the committed file is stale and
`test_tokens_css_matches_design_md` enforces parity. Tests also forbid a hardcoded hex outside
`tokens.css`, any `@media (prefers-color-scheme)` at-rule (there is no dark mode), and inline `style`
attributes in templates. Accessibility: skip link first in tab order, one `h1` per page, header/nav/
main/footer landmarks, `aria-live` status region for state changes, the drop zone is a real
`<label for>` so keyboard and click work without JavaScript, 44 px targets, visible 2 px Fern focus
ring, and `prefers-reduced-motion` handled in both stylesheets including the progress bar.

**Fonts.** Avegas Royale is still absent and unlicensed, so display type renders in the documented
fallback (`"Iowan Old Style", Georgia, serif`) — expected, per design.md, not a bug. Atkinson
Hyperlegible Next is not vendored either; body text uses its documented fallback stack.

**Favicon.** A favicon now ships, in two variants, because a favicon sits in browser chrome rather
than on the page and no single value is visible on both light and dark chrome. Measured: cream on
Chrome's light tab bar `#DEE1E6` is 1.04:1, which is invisible, while Pine Green on the same surface
is 7.34:1; on a dark tab bar `#35363A` cream reaches 9.58:1 and Pine Green drops to 1.25:1.

- `app/static/brand/favicon.png`, 512x512, refilled with Pine Green `#324A3D`, linked with no media
  query so it is also the fallback for a browser that ignores `media` on a link element.
- `app/static/brand/favicon-dark.png`, 512x512, the cream artwork exactly as supplied, linked with
  `media="(prefers-color-scheme: dark)"`.
- The Apple touch icon is the green variant, because touch icons are composited on light
  home-screen backgrounds.

This is browser chrome only and does **not** give the site a dark theme; design.md's no-dark-mode
decision stands and a test still forbids any `@media (prefers-color-scheme)` at-rule in the
stylesheets. Both variants are generated by `scripts/build_favicon.py` from the supplied master
`app/static/brand/favicon-source.png` (1414x706 cream), which is never written to; `--check` fails if
a committed export is stale. The green variant changes the fill colour only, so every alpha value and
therefore the shape and its antialiasing come unchanged from the master, which a test asserts by
comparing the two alpha channels. The same script also writes `retinex-logo-flush.png`, the
navigation logo cropped to its own alpha bounds, because the supplied logo carries 34px of
transparent padding on its left edge that pushed the wordmark out of line with the content column.

**Running the UI tests**
```bash
pytest tests/test_app.py        # 104 tests, no network, no VGG16 weights needed
pytest                          # the whole suite, 291 tests
python scripts/build_tokens.py --check
python scripts/build_favicon.py --check
```

**Files** `app/__init__.py` (factory), `app/routes.py` (two pages, two API endpoints, error handlers),
`app/pipeline.py` (real stage status), `app/inference.py` (lazy model, Stage 2 code path, read-only
selection), `app/uploads.py` (byte-level validation), `app/templates/` (base, nav, footer, macros,
workspace, index, error), `app/static/css/` (generated tokens plus base, layout, components),
`app/static/js/workspace.js`, `app/static/js/ui.js` (section reveals and the sliding navigation
underline), `scripts/build_tokens.py`, `scripts/build_favicon.py`, `tests/test_app.py`. The logo was
moved to `app/static/brand/retinex-logo.png` by `git mv`; its bytes were later replaced by the
project owner with the green, transparent re-supply, and the navigation renders the flush-cropped
export derived from it.
