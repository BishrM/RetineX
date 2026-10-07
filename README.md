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
Flickr8K (8,000 images, 5 captions each). Not included; must be downloaded separately.

## Experimental Design
Frozen image-level 6000/1000/1000 split. Baseline (4096-d VGG16 + LSTM) vs RetineX (512 Fisher-selected + LSTM).
Metrics: unsmoothed corpus BLEU-1 to BLEU-4. Pseudo-classes and the Fisher mask use training data only.

## Project Structure
```
data/  src/  scripts/  notebooks/  models/  artifacts/  experiments/  app/  tests/  docs/
```
Folders are placeholders; contents will evolve.

## Installation
NOT IMPLEMENTED YET

## Dataset Setup
NOT IMPLEMENTED YET

## Training
NOT IMPLEMENTED YET

## Evaluation
NOT IMPLEMENTED YET

## Inference
NOT IMPLEMENTED YET

## Flask Application
NOT IMPLEMENTED YET

## Testing
NOT IMPLEMENTED YET

## Reproducibility
Goal: a clean clone can reproduce every stage from a venv. NOT IMPLEMENTED YET.

## Limitations
No results exist yet. Flickr8K is small; gTTS needs internet access.

## Future Work
To be determined after the baseline and RetineX experiments.

## Original Reference
Evolved from [MiteshPuthran/Image-Caption-Generator](https://github.com/MiteshPuthran/Image-Caption-Generator),
itself based on [Yumi's blog](https://fairyonice.github.io/Develop_an_image_captioning_deep_learning_model_using_Flickr_8K_data.html).
Used as a conceptual reference only.
