---
name: retinex-dataset
description: Use when working with the Flickr8K data, the frozen train/val/test split, or running dataset validate/split/verify/normalize in RetineX.
---

# RetineX dataset operations

Code: `src/retinex/dataset.py`. Run from repo root with the venv (`.venv/bin/python -m retinex.dataset ...`).

- `normalize --source <kaggle dir>`: Kaggle CSV layout -> `data/flickr8k/{images,captions/Flickr8k.token.txt}`.
- `validate`: measures counts, reports missing/corrupt/malformed records.
- `verify`: checks the frozen split (sizes, overlap, coverage, SHA-256).
- `split`: only ever run on a fresh checkout without `data/flickr8k/splits/`.

Rules:
- The split in `data/flickr8k/splits/` is FROZEN (seed 42, 6000/1000/1000, 91 surplus images unused). Read it with `retinex.dataset.load_split`; never call `make_split` in later stages; never use `--force` without user approval.
- Source is a third-party Kaggle copy (`adityajn105/flickr8k`), not the official Illinois release. Never call it official.
- Never commit images, captions, or archives. Split files and `split_meta.json` are the only committed data.
- Run `pytest` and `verify` after touching dataset code.
