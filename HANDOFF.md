# HANDOFF

## Project
**RetineX** — Deep Learning Based Image Caption Generation using Fisher Score Feature Optimization, LSTM and
Neural Voice Narration for Real-Time Visual Accessibility. Academic major project.

## Current status: 0% implemented
Nothing below "Intended methodology" exists yet. Workspace, docs and a local Git repo only.

| Stage | Status |
|---|---|
| Workspace / docs / git | done (initial) |
| Dataset prep, frozen 6000/1000/1000 split | NOT IMPLEMENTED |
| VGG16 fc2 feature extraction | NOT IMPLEMENTED |
| TF-IDF + K-Means pseudo-classes | NOT IMPLEMENTED |
| Fisher Score mask (top 512) | NOT IMPLEMENTED |
| Baseline LSTM (4096-d) | NOT IMPLEMENTED |
| RetineX LSTM (512-d) | NOT IMPLEMENTED |
| BLEU-1..4 evaluation | NOT IMPLEMENTED |
| Inference | NOT IMPLEMENTED |
| gTTS narration | NOT IMPLEMENTED |
| Flask app | NOT IMPLEMENTED |
| Tests | NOT IMPLEMENTED |

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
