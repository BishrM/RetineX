"""Single-image feature extraction for the Workspace, reusing the Stage 2 and Stage 4 code paths.

What this can do: load one uploaded image through **the same functions Stage 2 used**
(`features._build_model` and `features._load_batch`), producing a 4096-d fc2 vector, then apply the
**saved** Stage 4 top-512 indices to it. Preprocessing therefore cannot drift from the extraction
that produced the committed artifacts, and a test asserts byte-equality against `_load_batch`.

What this cannot do: generate a caption. No LSTM exists in this repository, so there is no code path
here that returns a caption, not even a placeholder one. The Workspace renders that absence.

Nothing here writes to `artifacts/`. The Fisher indices are read with `mmap_mode="r"`, and the
Stage 2 matrix is never opened at all.
"""
import logging
import threading
import time
from pathlib import Path

import numpy as np

from retinex import features, fisher

log = logging.getLogger(__name__)

# One process-wide model, loaded on first use. Loading VGG16 costs real memory, so it must never
# happen at import time or during a request that does not need it. Flask's development server sets
# threaded=True, so two concurrent first requests would otherwise both load it: ~1 GB of duplicate
# work and a clobbered load timing. The lock makes the load happen exactly once.
_MODEL = None
_MODEL_LOAD_SECONDS = None
_MODEL_LOCK = threading.Lock()


class InferenceError(RuntimeError):
    """Raised when feature extraction cannot complete. Never paired with a partial result."""


def model_loaded():
    return _MODEL is not None


def model_load_seconds():
    """Measured seconds for the one model load, or None if the model has not been loaded."""
    return _MODEL_LOAD_SECONDS


def get_model():
    """Load VGG16 fc2 once, lazily and thread-safely.

    Double-checked locking: the fast path takes no lock once the model exists. A failed load leaves
    the cache empty so a later request can retry, and raises InferenceError rather than a bare
    keras error.
    """
    global _MODEL, _MODEL_LOAD_SECONDS
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:                      # another thread may have loaded it while we waited
            started = time.perf_counter()
            try:
                model = features._build_model()
            except Exception as e:              # keras/tf raise many types, including on no network
                log.exception("VGG16 fc2 could not be loaded")
                raise InferenceError(
                    "The VGG16 feature extractor could not be loaded. Its ImageNet weights may be "
                    "missing from the local keras cache, which needs one download while online."
                ) from e
            _MODEL_LOAD_SECONDS = round(time.perf_counter() - started, 3)
            _MODEL = model
    return _MODEL


def load_selection(fisher_dir=None):
    """The saved Stage 4 top-512 indices. Read-only: Fisher Score is never refitted here."""
    fisher_dir = Path(fisher_dir) if fisher_dir else fisher.default_out_dir()
    path = fisher_dir / fisher.INDICES
    if not path.is_file():
        raise InferenceError(
            f"The Fisher Score selection is missing ({path.name}). Run "
            "python -m retinex.fisher select to produce it."
        )
    try:
        idx = np.load(path, mmap_mode="r")
    except Exception as e:                       # a truncated or non-npy file raises ValueError
        log.exception("Fisher selection could not be read from %s", path)
        raise InferenceError(
            f"The Fisher Score selection ({path.name}) could not be read. It may be corrupted; "
            "re-run python -m retinex.fisher select."
        ) from e
    if idx.ndim != 1:
        raise InferenceError(f"{path.name} is not a 1-D index array")
    idx = np.asarray(idx)
    if not (np.diff(idx) > 0).all():
        raise InferenceError(f"{path.name} is not strictly ascending; refusing to use it")
    return idx


def extract_one(image_path, fisher_dir=None):
    """Extract fc2 features for one image and apply the saved selection.

    Returns measured values only. Raises InferenceError on any failure, never a partial dict.
    """
    image_path = Path(image_path)
    idx = load_selection(fisher_dir)
    model = get_model()

    started = time.perf_counter()
    try:
        # exactly the Stage 2 preprocessing: load_img(target_size=(224,224)) -> img_to_array ->
        # vgg16.preprocess_input, via the same function the committed artifacts were built with
        batch = features._load_batch(image_path.parent, [image_path.name])
        full = np.asarray(model.predict_on_batch(batch), dtype=np.float32)[0]
    except Exception as e:
        # The underlying message can carry tensor shapes and local paths, so it is logged rather
        # than returned to the client.
        log.exception("Feature extraction failed")
        raise InferenceError(
            "Feature extraction failed for this image. The server log has the details."
        ) from e
    elapsed = time.perf_counter() - started

    if full.shape != (features.FEATURE_DIM,):
        raise InferenceError(
            f"Expected a {features.FEATURE_DIM}-d vector, got {full.shape}. Refusing to report it.")
    if not np.isfinite(full).all():
        raise InferenceError("The feature vector contains non-finite values. Refusing to report it.")
    if idx.max() >= full.shape[0]:
        raise InferenceError("The saved selection indexes past the end of the feature vector.")

    reduced = full[idx]
    return {
        "feature_dim": int(full.shape[0]),
        "selected_dim": int(reduced.shape[0]),
        "preprocessing": (
            "keras.utils.load_img(target_size=(224,224)) -> img_to_array -> "
            "keras.applications.vgg16.preprocess_input"),
        "source": "retinex.features._build_model + _load_batch (the Stage 2 code path)",
        "selection_source": fisher.INDICES,
        "inference_seconds": round(elapsed, 3),
        "model_load_seconds": model_load_seconds(),
        "full": _stats(full),
        "reduced": _stats(reduced),
        "top_activations": _top(reduced, idx),
    }


def _stats(v):
    v = np.asarray(v, dtype=np.float64)
    nonzero = int((v != 0).sum())
    return {
        "dim": int(v.size),
        "min": round(float(v.min()), 6),
        "mean": round(float(v.mean()), 6),
        "max": round(float(v.max()), 6),
        "l2_norm": round(float(np.linalg.norm(v)), 6),
        "nonzero": nonzero,
        "sparsity": round(1.0 - nonzero / v.size, 4),
    }


def _top(reduced, idx, n=8):
    """The strongest selected dimensions, labelled with their original 4096-d column index."""
    order = np.argsort(-np.asarray(reduced, dtype=np.float64), kind="stable")[:n]
    return [{"selected_position": int(p), "source_index": int(idx[p]),
             "activation": round(float(reduced[p]), 6)} for p in order]
