"""Real pipeline status, derived from the artifacts on disk.

Stages 1 to 4 report whatever their own validators say, right now, including failures. Nothing here
is a hardcoded success badge: if an artifact is missing or has been tampered with, the UI says so.
Stages 5 to 9 have no module to validate, so they report NOT IMPLEMENTED.
"""
import json
from pathlib import Path

from retinex import dataset, features, fisher, pseudoclasses

DONE = "complete"
FAILED = "failed"
MISSING = "missing"
NOT_IMPLEMENTED = "not_implemented"

STATUS_LABEL = {
    DONE: "Complete",
    FAILED: "Validation failed",
    MISSING: "Artifacts missing",
    NOT_IMPLEMENTED: "Not implemented",
}


def _meta(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _checked(name, detail, validator, present):
    """Run a validator and turn its problem list into a stage record."""
    if not present:
        return {"name": name, "detail": detail, "status": MISSING, "problems": []}
    try:
        problems = validator()
    except Exception as e:                      # a broken artifact must not break the page
        return {"name": name, "detail": detail, "status": FAILED, "problems": [str(e)]}
    return {"name": name, "detail": detail,
            "status": DONE if not problems else FAILED,
            "problems": list(problems)[:5]}


def stages(data_dir=None, features_dir=None, pseudo_dir=None, fisher_dir=None):
    """The nine pipeline stages with their true current status."""
    data_dir = Path(data_dir) if data_dir else dataset.default_data_dir()
    features_dir = Path(features_dir) if features_dir else features.default_out_dir()
    pseudo_dir = Path(pseudo_dir) if pseudo_dir else pseudoclasses.default_out_dir()
    fisher_dir = Path(fisher_dir) if fisher_dir else fisher.default_out_dir()

    fmeta = _meta(features_dir / features.META)
    pmeta = _meta(pseudo_dir / pseudoclasses.META)
    kmeta = _meta(fisher_dir / fisher.META)

    out = [
        _checked("Dataset and frozen split",
                 "Flickr8K, image-level 6000/1000/1000 split, seed 42",
                 lambda: dataset.verify_split(data_dir),
                 (data_dir / "splits" / "split_meta.json").is_file()),
        _checked("VGG16 fc2 features",
                 f"{fmeta.get('num_images', '?')} x {fmeta.get('feature_dim', '?')} float32"
                 if fmeta else "8000 x 4096 float32",
                 lambda: features.validate_features(features_dir, data_dir),
                 (features_dir / features.META).is_file()),
        _checked("TF-IDF and K-Means pseudo-classes",
                 f"k={pmeta.get('k', '?')}, {pmeta.get('n_train_images', '?')} training labels"
                 if pmeta else "training captions only",
                 lambda: pseudoclasses.validate_pseudoclasses(pseudo_dir, data_dir, features_dir),
                 (pseudo_dir / pseudoclasses.META).is_file()),
        _checked("Fisher Score selection",
                 f"{kmeta.get('n_features_in', '?')} to {kmeta.get('n_selected', '?')} features"
                 if kmeta else "4096 to 512 features",
                 lambda: fisher.validate_fisher(fisher_dir, data_dir, features_dir, pseudo_dir),
                 (fisher_dir / fisher.META).is_file()),
    ]
    for name, detail in (
        ("Baseline LSTM (4096-d)", "No module exists yet"),
        ("RetineX LSTM (512-d)", "No module exists yet"),
        ("BLEU-1 to BLEU-4 evaluation", "No score has been measured in this repository"),
        ("Caption inference", "The Workspace cannot generate a caption yet"),
        ("Neural voice narration", "No text-to-speech is connected"),
    ):
        out.append({"name": name, "detail": detail, "status": NOT_IMPLEMENTED, "problems": []})

    for n, stage in enumerate(out, 1):
        stage["index"] = n
        stage["label"] = STATUS_LABEL[stage["status"]]
        stage["done"] = stage["status"] == DONE
    return out


def summary(stage_list=None):
    """Counts for the Workspace header. Measured, not asserted."""
    stage_list = stage_list if stage_list is not None else stages()
    return {
        "total": len(stage_list),
        "complete": sum(s["status"] == DONE for s in stage_list),
        "not_implemented": sum(s["status"] == NOT_IMPLEMENTED for s in stage_list),
        "problems": sum(s["status"] in (FAILED, MISSING) for s in stage_list),
    }
