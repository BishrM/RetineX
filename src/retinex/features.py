"""VGG16 fc2 feature extraction for the frozen Flickr8K split.

Reads the frozen split written in Stage 1, extracts a 4096-d fc2 vector per image with
ImageNet-pretrained VGG16, and writes one float32 matrix plus its row-id list and metadata.

Captions are never read here: extraction depends on image pixels only, so no validation or
test information can influence a feature. The split is used only to decide *which* images to
process and is never regenerated.

CLI: python -m retinex.features {extract,validate} [--data-dir DIR] [--out-dir DIR]
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .dataset import SIZES, default_data_dir, load_split

MODEL_NAME = "VGG16"
WEIGHTS = "imagenet"
LAYER = "fc2"
FEATURE_DIM = 4096
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MATRIX = "vgg16_fc2.npy"
IDS = "vgg16_fc2_ids.txt"
META = "vgg16_fc2_meta.json"
PROGRESS = "vgg16_fc2.progress.json"


def default_out_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "features"


def frozen_ids(split_dir):
    """The 8000 frozen image ids in deterministic (sorted) order, plus the split that produced them."""
    split = load_split(split_dir)
    ids = sorted(set().union(*(set(v) for v in split.values())))
    if len(ids) != sum(len(v) for v in split.values()):
        raise ValueError("frozen split contains duplicate ids across splits; run dataset verify")
    return ids, split


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ids_digest(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def _build_model():
    """ImageNet-pretrained VGG16 truncated at fc2. Frozen: never trained or fine-tuned here."""
    import keras

    base = keras.applications.VGG16(weights=WEIGHTS, include_top=True)
    base.trainable = False
    model = keras.Model(inputs=base.inputs, outputs=base.get_layer(LAYER).output, name="vgg16_fc2")
    return model


def _load_batch(image_dir, names):
    import keras

    arr = np.stack([
        keras.utils.img_to_array(keras.utils.load_img(image_dir / n, target_size=IMAGE_SIZE))
        for n in names
    ])
    return keras.applications.vgg16.preprocess_input(arr)


def _say(msg):
    """Progress goes out unbuffered so a redirected log stays live during a long run."""
    print(msg, flush=True)


def extract(data_dir, out_dir, batch_size=BATCH_SIZE, resume=True, progress_cb=_say):
    """Extract fc2 features for every frozen image. Resumable, batched, constant memory."""
    import keras
    import tensorflow as tf

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    image_dir = data_dir / "images"
    ids, split = frozen_ids(data_dir / "splits")
    digest = _ids_digest(ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path, progress_path = out_dir / MATRIX, out_dir / PROGRESS

    done = 0
    if resume and matrix_path.exists() and progress_path.exists():
        state = json.loads(progress_path.read_text())
        if state.get("ids_sha256") == digest and state.get("shape") == [len(ids), FEATURE_DIM]:
            done = int(state.get("done", 0))
            progress_cb(f"resuming at row {done} of {len(ids)}")
        else:
            progress_cb("existing artifacts do not match the frozen split; starting over")

    mode = "r+" if done else "w+"
    feats = np.lib.format.open_memmap(matrix_path, mode=mode, dtype=np.float32,
                                      shape=(len(ids), FEATURE_DIM))
    (out_dir / IDS).write_text("\n".join(ids) + "\n")

    model = _build_model()
    devices = [d.name for d in tf.config.list_physical_devices()]
    progress_cb(f"devices: {devices}")
    started = time.time()
    try:
        for start in range(done, len(ids), batch_size):
            names = ids[start:start + batch_size]
            feats[start:start + len(names)] = model.predict_on_batch(_load_batch(image_dir, names))
            feats.flush()
            progress_path.write_text(json.dumps(
                {"done": start + len(names), "shape": [len(ids), FEATURE_DIM], "ids_sha256": digest}))
            if (start // batch_size) % 10 == 0 or start + len(names) == len(ids):
                pct = 100 * (start + len(names)) / len(ids)
                rate = (start + len(names) - done) / max(time.time() - started, 1e-9)
                progress_cb(f"  {start + len(names)}/{len(ids)} ({pct:.1f}%) {rate:.1f} img/s")
    finally:
        feats.flush()
        del feats

    # Timed from the resume point, so this covers only the rows extracted by *this* run.
    runtime = time.time() - started
    meta = {
        "feature_name": "vgg16_fc2",
        "model": MODEL_NAME,
        "weights": WEIGHTS,
        "source_layer": LAYER,
        "feature_dim": FEATURE_DIM,
        "dtype": "float32",
        "num_images": len(ids),
        "batch_size": batch_size,
        "image_size": list(IMAGE_SIZE),
        "preprocessing": "keras.utils.load_img(target_size=(224,224)) -> img_to_array -> "
                         "keras.applications.vgg16.preprocess_input",
        "row_mapping": f"row n corresponds to line n of {IDS}; ids are the sorted union of the "
                       "frozen split, deterministic across runs",
        "ids_sha256": digest,
        "uses_captions": False,
        "model_trainable": False,
        "split_counts": {k: len(v) for k, v in split.items()},
        "split_meta_sha256": _sha256(split_meta) if (split_meta := data_dir / "splits" / "split_meta.json").is_file() else None,
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(runtime, 1),
        "resumed_from_row": done,
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "devices": devices,
        "status": "complete",
    }
    meta["artifact_sha256"] = {MATRIX: _sha256(matrix_path), IDS: _sha256(out_dir / IDS)}
    (out_dir / META).write_text(json.dumps(meta, indent=2) + "\n")
    progress_path.unlink(missing_ok=True)
    return meta


def validate_features(out_dir, data_dir):
    """Validate the Stage 2 artifacts against the frozen experiment. Empty list means all checks passed.

    The split sizes (6000/1000/1000) and the feature width (4096) are not parameters: they are the
    frozen methodology, so no caller can validate the real artifacts against a weaker shape. Tests
    that need a tiny synthetic fixture call `_validate_features` directly, which cannot affect this path.
    """
    return _validate_features(out_dir, data_dir, dim=FEATURE_DIM, sizes=SIZES)


def _validate_features(out_dir, data_dir, dim, sizes):
    """Shape-parameterised implementation. Private: production goes through `validate_features`."""
    out_dir, data_dir = Path(out_dir), Path(data_dir)
    errs = []
    for name in (MATRIX, IDS, META):
        if not (out_dir / name).is_file():
            errs.append(f"missing artifact: {name}")
    if errs:
        return errs

    try:
        meta = json.loads((out_dir / META).read_text())
    except json.JSONDecodeError as e:
        return [f"{META} is not valid JSON: {e}"]
    if meta.get("status") != "complete":
        errs.append(f"{META} status is {meta.get('status')!r}, not 'complete'")

    ids = (out_dir / IDS).read_text().split()
    try:
        ids_frozen, split = frozen_ids(data_dir / "splits")
    except (FileNotFoundError, ValueError) as e:
        return errs + [f"cannot read frozen split: {e}"]

    expected = sum(sizes.values())
    if len(ids) != expected:
        errs.append(f"{IDS} has {len(ids)} ids, expected {expected}")
    if len(set(ids)) != len(ids):
        errs.append(f"{IDS} contains duplicate ids")
    if set(ids) != set(ids_frozen):
        extra, missing = set(ids) - set(ids_frozen), set(ids_frozen) - set(ids)
        if extra:
            errs.append(f"{len(extra)} id(s) not in the frozen split, e.g. {sorted(extra)[:3]}")
        if missing:
            errs.append(f"{len(missing)} frozen id(s) missing, e.g. {sorted(missing)[:3]}")
    for k, v in split.items():
        if len(v) != sizes[k]:
            errs.append(f"split {k} has {len(v)} images, expected {sizes[k]}")
    members = {k: set(v) for k, v in split.items()}
    for i in ids:
        hits = [k for k, v in members.items() if i in v]
        if len(hits) != 1:
            errs.append(f"{i} belongs to {len(hits)} splits ({hits}), expected exactly 1")
            break

    feats = np.load(out_dir / MATRIX, mmap_mode="r")
    if feats.dtype != np.float32:
        errs.append(f"matrix dtype is {feats.dtype}, expected float32")
    if feats.shape != (expected, dim):
        errs.append(f"matrix shape is {feats.shape}, expected {(expected, dim)}")
    if feats.shape[0] != len(ids):
        errs.append(f"matrix has {feats.shape[0]} rows but {IDS} has {len(ids)} ids")
    chunk = 1024
    nan = inf = 0
    for s in range(0, feats.shape[0], chunk):
        block = np.asarray(feats[s:s + chunk], dtype=np.float64)
        nan += int(np.isnan(block).sum())
        inf += int(np.isinf(block).sum())
    if nan:
        errs.append(f"matrix contains {nan} NaN value(s)")
    if inf:
        errs.append(f"matrix contains {inf} infinite value(s)")

    if meta.get("ids_sha256") and meta["ids_sha256"] != _ids_digest(ids):
        errs.append(f"{IDS} changed since extraction; row order is no longer guaranteed")
    recorded = meta.get("artifact_sha256", {})
    for name in (MATRIX, IDS):
        if recorded.get(name) and recorded[name] != _sha256(out_dir / name):
            errs.append(f"{name} changed since extraction (sha256 mismatch)")
    if meta.get("uses_captions") is not False:
        errs.append("metadata does not record uses_captions=false")
    return errs


def main(argv=None):
    p = argparse.ArgumentParser(prog="retinex.features")
    p.add_argument("command", choices=["extract", "validate"])
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--out-dir", type=Path, default=default_out_dir())
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args(argv)

    if a.command == "extract":
        if not (a.data_dir / "splits" / "train.txt").is_file():
            print(f"frozen split not found in {a.data_dir / 'splits'}; run retinex.dataset first")
            return 1
        meta = extract(a.data_dir, a.out_dir, a.batch_size, resume=not a.no_resume)
        rows = meta["num_images"] - meta["resumed_from_row"]
        print(f"wrote {meta['num_images']}x{meta['feature_dim']} float32 features to {a.out_dir}; "
              f"{rows} row(s) extracted this run in {meta['runtime_seconds']}s "
              f"(resumed from row {meta['resumed_from_row']})")

    errs = validate_features(a.out_dir, a.data_dir)
    print("\n".join(errs) if errs else
          "features OK: shape, dtype, ids, split membership, finiteness and checksums verified")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
