---
name: retinex-leakage-check
description: Use when adding or reviewing any RetineX code that fits something (tokenizer, TF-IDF, K-Means, Fisher mask, scalers, model) to make sure val/test data never influence it.
---

# RetineX data-leakage check

Before finishing any stage that fits or selects anything, confirm:

1. Split is by image and comes from `load_split`; nothing regenerates it.
2. Fit-on-train-only: vocabulary/tokenizer, TF-IDF vectorizer, K-Means pseudo-classes, Fisher mask, any scaler. Val/test are only transformed.
3. Test split is used only for final evaluation, never for tuning or early stopping (use val).
4. Baseline and RetineX share split, vocabulary, hyperparameters and seeds; only image features differ.
5. A test asserts the fit inputs contain no val/test IDs.

Report violations plainly; do not fix by weakening the check.
