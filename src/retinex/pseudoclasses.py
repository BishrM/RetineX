"""TF-IDF + K-Means semantic pseudo-classes for the frozen Flickr8K training split.

Builds one document per *training* image from that image's captions, fits TF-IDF and K-Means on
those documents only, and writes one pseudo-class label per training image for the Fisher Score
stage to use as its class variable.

Leakage position: validation and test captions are never read into a fit. `train_documents` is the
only document builder and it is given the training ids alone; `fit` asserts this before fitting.
No labels are produced for validation or test images, by design, so no later stage can misuse them.

K is frozen at 20 in advance. The silhouette sweep recorded in the metadata is transparency only and
does not select K.

CLI: python -m retinex.pseudoclasses {fit,validate} [--data-dir DIR] [--out-dir DIR]
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .dataset import SIZES, TOKEN_FILE, default_data_dir, load_split, parse_captions

K = 20
SEED = 42
N_INIT = 10
MAX_ITER = 300
ALGORITHM = "lloyd"
MIN_DF = 2
TOKEN_PATTERN = r"[a-z]{2,}"
NGRAM_RANGE = (1, 1)
NORM = "l2"
SUBLINEAR_TF = False
K_SWEEP = (5, 10, 20, 30, 50)
SILHOUETTE_METRIC = "cosine"
SILHOUETTE_SAMPLE_SIZE = None  # None = every training document; no sampling
MIN_CLUSTER_SIZE = 10
TOP_TERMS = 15

LABELS = "train_pseudoclasses.csv"
VOCAB = "train_vocabulary.txt"
META = "pseudoclass_meta.json"
FEATURE_IDS = "vgg16_fc2_ids.txt"


def default_out_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "pseudoclasses"


def default_features_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "features"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ids_digest(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def train_documents(captions, train_ids, token_pattern=TOKEN_PATTERN):
    """One document per training image: its captions joined in #n order, train ids only.

    Returns (ids, documents) with ids sorted, so row n of the matrix is ids[n]. Raises if an image
    has no captions, or if cleaning would leave it with no tokens at all -- an empty document would
    otherwise become an all-zero TF-IDF row and land in an arbitrary cluster.
    """
    import re

    ids = sorted(train_ids)
    missing = [i for i in ids if not captions.get(i)]
    if missing:
        raise ValueError(f"{len(missing)} training image(s) have no captions, e.g. {missing[:3]}")
    docs = [" ".join(captions[i]) for i in ids]
    pattern = re.compile(token_pattern)
    empty = [i for i, d in zip(ids, docs) if not pattern.findall(d.lower())]
    if empty:
        raise ValueError(f"{len(empty)} training image(s) have no tokens after cleaning: {empty[:5]}")
    return ids, docs


def _vectorizer(min_df=MIN_DF, token_pattern=TOKEN_PATTERN):
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(lowercase=True, token_pattern=token_pattern, min_df=min_df,
                           ngram_range=NGRAM_RANGE, norm=NORM, sublinear_tf=SUBLINEAR_TF)


def _kmeans(k, seed):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, random_state=seed, n_init=N_INIT, algorithm=ALGORITHM,
                  max_iter=MAX_ITER)


def _silhouette(X, labels, metric=SILHOUETTE_METRIC, sample_size=SILHOUETTE_SAMPLE_SIZE):
    """Mean silhouette over the given labelling, or (None, reason) when it is undefined.

    sklearn requires 2 <= distinct labels <= n_samples - 1. A singleton cluster is fine (sklearn
    defines its contribution as 0); a collapsed labelling or an all-singleton one is not. Returning
    a reason instead of a number keeps an undefined score out of the metadata rather than inventing one.
    """
    from sklearn.metrics import silhouette_score

    try:
        score = silhouette_score(X, labels, metric=metric, sample_size=sample_size)
    except ValueError as e:
        return None, f"silhouette undefined: {e}"
    return round(float(score), 6), None


def _top_terms(centers, terms, n=TOP_TERMS):
    """The n highest-weighted vocabulary terms of each cluster centroid."""
    return {str(c): [terms[j] for j in np.argsort(-row)[:n]] for c, row in enumerate(centers)}


def _say(msg):
    print(msg, flush=True)


def fit(data_dir, out_dir, features_dir=None, k=K, seed=SEED, min_df=MIN_DF, force=False,
        progress_cb=_say):
    """Fit TF-IDF and K-Means on training captions only and write the pseudo-class artifacts.

    `k` exists only so the frozen value is explicit at the call site: any other value is refused.
    The silhouette sweep is not a parameter -- it is the fixed K_SWEEP diagnostic and cannot
    influence the production k.
    """
    if k != K:
        raise ValueError(
            f"k={k} refused: K={K} is a frozen methodology decision for this project "
            f"(CLAUDE.md, STAGE 3 DECISIONS). The silhouette sweep over {list(K_SWEEP)} is a "
            f"diagnostic only and cannot change it. Changing K requires updating CLAUDE.md "
            f"with explicit approval.")

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    features_dir = Path(features_dir) if features_dir else default_features_dir()
    existing = [n for n in (LABELS, VOCAB, META) if (out_dir / n).exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite {existing} in {out_dir} (use --force)")

    split = load_split(data_dir / "splits")
    captions, problems = parse_captions(data_dir / "captions" / TOKEN_FILE)
    if problems:
        progress_cb(f"caption file problems reported by the parser: {len(problems)}")

    ids, docs = train_documents(captions, split["train"], TOKEN_PATTERN)
    # The leakage guard: only training ids may reach the fit, and all of them must. Raised, not
    # asserted, so `python -O` cannot strip the check that enforces the methodology.
    if leaked := sorted(set(ids) & (set(split["val"]) | set(split["test"]))):
        raise ValueError(f"LEAKAGE: {len(leaked)} validation/test image(s) in the fit, e.g. {leaked[:3]}")
    if set(ids) != set(split["train"]):
        raise ValueError("training documents do not cover the training split exactly")
    progress_cb(f"fitting on {len(docs)} training documents ({sum(map(len, (captions[i] for i in ids)))} captions)")

    vec = _vectorizer(min_df, TOKEN_PATTERN)
    X = vec.fit_transform(docs)
    terms = vec.get_feature_names_out()
    progress_cb(f"tf-idf matrix {X.shape[0]}x{X.shape[1]}, {X.nnz} non-zeros")

    km = _kmeans(k, seed)
    labels = km.fit_predict(X)
    sizes = {str(c): int((labels == c).sum()) for c in range(k)}
    progress_cb(f"k-means k={k} inertia={km.inertia_:.4f} smallest cluster={min(sizes.values())}")

    # Diagnostic only. Every candidate is fitted on the same training documents; none of this can
    # change the production k, which was fixed above.
    sweep_scores = {}
    for ks in K_SWEEP:
        if ks < 2 or ks >= X.shape[0]:
            sweep_scores[str(ks)] = {
                "score": None, "n_distinct_labels": None,
                "unavailable": f"k={ks} outside [2, {X.shape[0] - 1}] for {X.shape[0]} documents"}
            progress_cb(f"  sweep k={ks} unavailable: outside [2, {X.shape[0] - 1}]")
            continue
        if ks == k:
            lab, note = labels, "reused the production fit; not refitted"
        else:
            lab, note = _kmeans(ks, seed).fit_predict(X), None
        score, why = _silhouette(X, lab)
        entry = {"score": score, "n_distinct_labels": int(len(set(lab)))}
        if why:
            entry["unavailable"] = why
        if note:
            entry["note"] = note
        sweep_scores[str(ks)] = entry
        progress_cb(f"  sweep k={ks} silhouette({SILHOUETTE_METRIC})={score}"
                    f"{' (' + why + ')' if why else ''}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LABELS).write_text(
        "image_id,pseudo_class\n" + "".join(f"{i},{l}\n" for i, l in zip(ids, labels)))
    idf = dict(zip(terms, vec.idf_))
    (out_dir / VOCAB).write_text("".join(f"{t}\t{idf[t]:.6f}\n" for t in terms))

    split_meta = data_dir / "splits" / "split_meta.json"
    feature_ids = features_dir / FEATURE_IDS
    import scipy
    import sklearn
    meta = {
        "artifact_name": "tfidf_kmeans_pseudoclasses",
        "k": k,
        "k_frozen_in_advance": True,
        "k_selected_by_sweep": False,
        "silhouette_sweep": {
            "purpose": f"diagnostic only; k is frozen at {K} and the sweep cannot change it",
            "metric": SILHOUETTE_METRIC,
            "sample_size": SILHOUETTE_SAMPLE_SIZE,
            "random_state": seed,
            "fitted_on": "train",
            "candidates_expected": list(K_SWEEP),
            "candidates": sweep_scores,
        },
        "seed": seed,
        "kmeans": {"n_init": N_INIT, "max_iter": MAX_ITER, "algorithm": ALGORITHM,
                   "inertia": round(float(km.inertia_), 6), "n_iter": int(km.n_iter_)},
        "tfidf": {"lowercase": True, "token_pattern": TOKEN_PATTERN, "min_df": min_df,
                  "ngram_range": list(NGRAM_RANGE), "norm": NORM, "sublinear_tf": SUBLINEAR_TF,
                  "stop_words": None},
        "document_unit": "one document per training image: its captions joined in #n order",
        "n_train_images": len(ids),
        "n_documents": len(docs),
        "n_captions_used": sum(len(captions[i]) for i in ids),
        "vocab_size": int(len(terms)),
        "tfidf_shape": [int(X.shape[0]), int(X.shape[1])],
        "tfidf_nnz": int(X.nnz),
        "cluster_sizes": sizes,
        "min_cluster_size_required": MIN_CLUSTER_SIZE,
        "silhouette_at_k": {
            "score": sweep_scores[str(k)]["score"],
            "metric": SILHOUETTE_METRIC,
            "sample_size": SILHOUETTE_SAMPLE_SIZE,
            "unavailable": sweep_scores[str(k)].get("unavailable"),
        },
        "top_terms_per_cluster": _top_terms(km.cluster_centers_, terms),
        "fitted_on": "train",
        "uses_val_test_captions": False,
        "val_test_labels_produced": False,
        "label_row_order": f"{LABELS} is sorted by image_id; Stage 4 maps ids to feature rows via {FEATURE_IDS}",
        "train_ids_sha256": _ids_digest(ids),
        "split_meta_sha256": _sha256(split_meta) if split_meta.is_file() else None,
        "feature_ids_sha256": _sha256(feature_ids) if feature_ids.is_file() else None,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "status": "complete",
    }
    meta["artifact_sha256"] = {LABELS: _sha256(out_dir / LABELS), VOCAB: _sha256(out_dir / VOCAB)}
    (out_dir / META).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def read_labels(path):
    """Read the label CSV as (ids, labels). Raises on a malformed header or row."""
    lines = Path(path).read_text().splitlines()
    if not lines or lines[0] != "image_id,pseudo_class":
        raise ValueError(f"unexpected header in {path}: {lines[:1]}")
    ids, labels = [], []
    for no, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        img, sep, lab = line.partition(",")
        if not (sep and img.strip() and lab.strip().lstrip("-").isdigit()):
            raise ValueError(f"{path} line {no}: malformed row {line!r}")
        ids.append(img.strip())
        labels.append(int(lab))
    return ids, np.array(labels, dtype=int)


def validate_pseudoclasses(out_dir, data_dir, features_dir=None):
    """Validate the Stage 3 artifacts against the frozen experiment. Empty list means all problems clear.

    Neither the split sizes (6000/1000/1000) nor the 10-image cluster floor is a parameter, so no
    caller can validate the real artifacts against a weaker rule. Tests that need a tiny synthetic
    fixture call `_validate_pseudoclasses` directly, which cannot affect this path; the cluster floor
    is the constant MIN_CLUSTER_SIZE even there.
    """
    return _validate_pseudoclasses(out_dir, data_dir, features_dir, sizes=SIZES)


def _validate_pseudoclasses(out_dir, data_dir, features_dir=None, sizes=SIZES):
    """Split-size-parameterised implementation. Private: production uses `validate_pseudoclasses`."""
    out_dir, data_dir = Path(out_dir), Path(data_dir)
    features_dir = Path(features_dir) if features_dir else default_features_dir()
    errs = [f"missing artifact: {n}" for n in (LABELS, VOCAB, META) if not (out_dir / n).is_file()]
    if errs:
        return errs

    try:
        meta = json.loads((out_dir / META).read_text())
    except json.JSONDecodeError as e:
        return [f"{META} is not valid JSON: {e}"]
    if meta.get("status") != "complete":
        errs.append(f"{META} status is {meta.get('status')!r}, not 'complete'")
    if meta.get("uses_val_test_captions") is not False:
        errs.append("metadata does not record uses_val_test_captions=false")
    if meta.get("fitted_on") != "train":
        errs.append(f"metadata records fitted_on={meta.get('fitted_on')!r}, expected 'train'")
    for name in (LABELS, VOCAB):
        recorded = meta.get("artifact_sha256", {}).get(name)
        if recorded and recorded != _sha256(out_dir / name):
            errs.append(f"{name} changed since fitting (sha256 mismatch)")

    try:
        ids, labels = read_labels(out_dir / LABELS)
    except ValueError as e:
        return errs + [str(e)]
    try:
        split = load_split(data_dir / "splits")
    except FileNotFoundError as e:
        return errs + [f"cannot read frozen split: {e.filename}"]

    if len(ids) != sizes["train"]:
        errs.append(f"{LABELS} has {len(ids)} rows, expected {sizes['train']}")
    if len(set(ids)) != len(ids):
        errs.append(f"{LABELS} contains duplicate image ids")
    if ids != sorted(ids):
        errs.append(f"{LABELS} is not sorted by image_id; row order is not the documented one")
    extra, missing = set(ids) - set(split["train"]), set(split["train"]) - set(ids)
    if extra:
        errs.append(f"{len(extra)} labelled id(s) not in the training split, e.g. {sorted(extra)[:3]}")
    if missing:
        errs.append(f"{len(missing)} training id(s) unlabelled, e.g. {sorted(missing)[:3]}")
    held_out = (set(ids) & set(split["val"])) | (set(ids) & set(split["test"]))
    if held_out:
        errs.append(f"LEAKAGE: {len(held_out)} validation/test image(s) carry a label, "
                    f"e.g. {sorted(held_out)[:3]}")
    for name in ("val", "test"):
        stray = out_dir / f"{name}_pseudoclasses.csv"
        if stray.exists():
            errs.append(f"LEAKAGE RISK: {stray.name} exists; this stage must not produce {name} labels")
    if meta.get("train_ids_sha256") and meta["train_ids_sha256"] != _ids_digest(ids):
        errs.append(f"{LABELS} ids changed since fitting; label alignment is no longer guaranteed")

    k = meta.get("k")
    if not isinstance(k, int) or k < 2:
        errs.append(f"metadata records k={k!r}, expected an integer >= 2")
    elif k != K:
        errs.append(f"metadata records k={k}; the methodology freezes k={K} "
                    f"(CLAUDE.md, STAGE 3 DECISIONS)")
    else:
        if labels.size and (labels.min() < 0 or labels.max() >= k):
            errs.append(f"labels span [{labels.min()}, {labels.max()}], expected [0, {k - 1}]")
        counts = {c: int((labels == c).sum()) for c in range(k)}
        if empty := [c for c, n in counts.items() if n == 0]:
            errs.append(f"{len(empty)} empty cluster(s): {empty[:5]}")
        if small := {c: n for c, n in counts.items() if 0 < n < MIN_CLUSTER_SIZE}:
            errs.append(f"cluster(s) below the {MIN_CLUSTER_SIZE}-image minimum: {small}; "
                        "Fisher Score within-class variance would be unstable")
        if meta.get("cluster_sizes") and {str(c): n for c, n in counts.items()} != meta["cluster_sizes"]:
            errs.append("cluster_sizes in metadata do not match the label file")

    feature_ids = features_dir / FEATURE_IDS
    recorded = meta.get("feature_ids_sha256")
    if recorded:
        if not feature_ids.is_file():
            errs.append(f"metadata references {FEATURE_IDS} but {feature_ids} is missing")
        elif recorded != _sha256(feature_ids):
            errs.append(f"{FEATURE_IDS} changed since fitting; Stage 4 row alignment is not guaranteed")
        else:
            rows = feature_ids.read_text().split()
            if unresolved := [i for i in ids if i not in set(rows)]:
                errs.append(f"{len(unresolved)} labelled id(s) have no feature row, "
                            f"e.g. {unresolved[:3]}")
    if meta.get("k_selected_by_sweep") is not False:
        errs.append("metadata does not record k_selected_by_sweep=false")

    # The sweep is a fixed diagnostic: the candidate set, metric and no-sampling rule are all
    # checked independently of whatever produced the file.
    sweep = meta.get("silhouette_sweep")
    if not isinstance(sweep, dict):
        errs.append("metadata has no silhouette_sweep block")
    else:
        if sweep.get("candidates_expected") != list(K_SWEEP):
            errs.append(f"silhouette_sweep.candidates_expected is "
                        f"{sweep.get('candidates_expected')!r}, expected {list(K_SWEEP)}")
        got = set(sweep.get("candidates", {}))
        if got != {str(c) for c in K_SWEEP}:
            errs.append(f"silhouette_sweep candidates are {sorted(got)}, "
                        f"expected {[str(c) for c in K_SWEEP]}")
        if sweep.get("metric") != SILHOUETTE_METRIC:
            errs.append(f"silhouette metric is {sweep.get('metric')!r}, expected "
                        f"{SILHOUETTE_METRIC!r} on the L2-normalised TF-IDF matrix")
        if sweep.get("sample_size") is not None:
            errs.append(f"silhouette_sweep.sample_size is {sweep.get('sample_size')!r}; the "
                        "methodology specifies the full training set (no sampling)")
        if sweep.get("fitted_on") != "train":
            errs.append(f"silhouette_sweep.fitted_on is {sweep.get('fitted_on')!r}, expected 'train'")
        if sweep.get("random_state") != meta.get("seed"):
            errs.append("silhouette_sweep.random_state does not match the recorded seed")
        for c, entry in sorted(sweep.get("candidates", {}).items()):
            if not isinstance(entry, dict) or ("score" not in entry):
                errs.append(f"silhouette_sweep candidate {c} has no score field")
            elif entry.get("score") is None and not entry.get("unavailable"):
                errs.append(f"silhouette_sweep candidate {c} has no score and no reason given")

    floor = meta.get("min_cluster_size_required")
    if not isinstance(floor, int) or floor < MIN_CLUSTER_SIZE:
        errs.append(f"metadata records min_cluster_size_required={floor!r}; the methodology "
                    f"requires at least {MIN_CLUSTER_SIZE} training images per cluster")
    return errs


def main(argv=None):
    p = argparse.ArgumentParser(prog="retinex.pseudoclasses")
    p.add_argument("command", choices=["fit", "validate"])
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--out-dir", type=Path, default=default_out_dir())
    p.add_argument("--features-dir", type=Path, default=default_features_dir())
    p.add_argument("--k", type=int, default=K,
                   help=f"frozen at {K}; any other value is refused, not silently honoured")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--min-df", type=int, default=MIN_DF)
    p.add_argument("--force", action="store_true", help="fit: overwrite existing artifacts")
    a = p.parse_args(argv)

    if a.command == "fit":
        if a.k != K:
            print(f"--k {a.k} refused: K={K} is a frozen methodology decision "
                  f"(CLAUDE.md, STAGE 3 DECISIONS). The silhouette sweep over {list(K_SWEEP)} is "
                  f"a diagnostic and cannot change it; changing K needs explicit approval and a "
                  f"CLAUDE.md update.")
            return 1
        if not (a.data_dir / "splits" / "train.txt").is_file():
            print(f"frozen split not found in {a.data_dir / 'splits'}; run retinex.dataset first")
            return 1
        if not (a.data_dir / "captions" / TOKEN_FILE).is_file():
            print(f"captions not found: {a.data_dir / 'captions' / TOKEN_FILE}")
            return 1
        try:
            meta = fit(a.data_dir, a.out_dir, a.features_dir, a.k, a.seed, a.min_df, a.force)
        except (FileExistsError, ValueError) as e:
            print(f"fit failed: {e}")
            return 1
        print(f"wrote {meta['n_train_images']} training pseudo-classes (k={meta['k']}, "
              f"vocab {meta['vocab_size']}) to {a.out_dir}")

    errs = validate_pseudoclasses(a.out_dir, a.data_dir, a.features_dir)
    print("\n".join(errs) if errs else
          "pseudo-classes OK: train-only coverage, sorted ids, label range, cluster sizes, "
          "feature alignment and checksums verified")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
