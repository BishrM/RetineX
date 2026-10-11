"""Fisher Score feature selection: 4096 VGG16 fc2 features -> the top 512.

Scores are computed from the 6,000 *training* feature vectors and their training-only
pseudo-class labels (Stage 3). Applying the resulting fixed column indices to all 8,000 rows is a
transform, not a fit, so no validation or test information reaches the selection.

Frozen formula (CLAUDE.md, STAGE 4 DECISIONS), all means over the 6,000 training rows only:

    F_j = SS_between,j / SS_within,j

    SS_between,j = sum_c  n_c * (mu_cj - mu_j)^2
    SS_within,j  = sum_c  sum_{i in c} (x_ij - mu_cj)^2

`SS_within` is the count-weighted within-class variance sum under the **population** variance
convention (ddof = 0), because

    sum_c n_c * sigma^2_cj  ==  sum_c n_c * (1/n_c) sum_{i in c} (x_ij - mu_cj)^2
                            ==  sum_c sum_{i in c} (x_ij - mu_cj)^2

so the pooled sum-of-squared-deviations form implemented here is algebraically identical to
`sum_c n_c sigma^2_cj` and needs no division by n_c. It is accumulated in float64 over two passes
(class means first, then deviations) rather than via `sum(x^2) - sum(x)^2/n`, which subtracts two
large near-equal numbers and can return small negatives. A sum of squared deviations cannot be
negative, so no clamping is applied; a negative value would be a bug, not something to mask.

The ratio is invariant to per-feature scaling, so no standardisation is applied.

Edge cases, counted in the metadata rather than smoothed over:
  * SS_within == 0 and SS_between  > 0 -> score +inf (perfectly separable on the training rows)
  * SS_within == 0 and SS_between == 0 -> score 0.0  (feature constant across the training rows)
No epsilon is added to the denominator and no 0/0 division is ever executed, so no score is NaN.

CLI: python -m retinex.fisher {select,validate}
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .dataset import SIZES, default_data_dir, load_split
from .features import FEATURE_DIM, IDS as FEATURE_IDS, MATRIX as SOURCE_MATRIX
from .features import META as FEATURE_META, _ids_digest, _sha256
from .features import default_out_dir as default_features_dir
from .features import validate_features
from .pseudoclasses import LABELS as PSEUDO_LABELS, META as PSEUDO_META
from .pseudoclasses import default_out_dir as default_pseudo_dir
from .pseudoclasses import read_labels, validate_pseudoclasses

N_SELECTED = 512
CHUNK = 500
VARIANCE_CONVENTION = "population (ddof=0)"
DENOMINATOR_FORM = "pooled within-class sum of squared deviations"
FORMULA = ("F_j = sum_c n_c (mu_cj - mu_j)^2 / sum_c sum_{i in c} (x_ij - mu_cj)^2  "
           "(= sum_c n_c sigma^2_cj under ddof=0)")
SELECTION_RULE = "descending Fisher score, ties broken by ascending original feature index"
OVERFLOW_POLICY = ("+inf is reserved for SS_within == 0; a non-finite ratio from a positive "
                   "denominator raises FloatingPointError instead of being counted as separable")
COLUMN_ORDER = "selected original indices sorted ascending; output column i is source column idx[i]"

SCORES = "fisher_scores.npy"
INDICES = "fisher_top512_indices.npy"
MATRIX = "vgg16_fc2_fisher512.npy"
META = "fisher_meta.json"


def default_out_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "artifacts" / "fisher"


def _say(msg):
    print(msg, flush=True)


def _jsonable(x):
    """JSON cannot hold +/-inf; encode it as a string so the metadata stays strict JSON."""
    x = float(x)
    return "inf" if x == float("inf") else "-inf" if x == float("-inf") else x


def training_join(data_dir, features_dir, pseudo_dir, sizes=SIZES):
    """Join feature rows, pseudo-class labels and the frozen training split **by image id**.

    Row position never implies split membership or label alignment: every row index is looked up
    from the id. Returns (ids, rows, labels) with all three aligned element-wise and sorted by id.
    Raises unless the labelled ids are a one-to-one cover of the frozen training split.
    """
    data_dir, features_dir, pseudo_dir = Path(data_dir), Path(features_dir), Path(pseudo_dir)
    feature_ids = (features_dir / FEATURE_IDS).read_text().split()
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError(f"{FEATURE_IDS} contains duplicate ids; row mapping is ambiguous")
    row_of = {i: n for n, i in enumerate(feature_ids)}

    label_ids, labels = read_labels(pseudo_dir / PSEUDO_LABELS)
    if len(set(label_ids)) != len(label_ids):
        raise ValueError(f"{PSEUDO_LABELS} contains duplicate image ids")
    split = load_split(data_dir / "splits")
    train, held_out = set(split["train"]), set(split["val"]) | set(split["test"])

    if leaked := sorted(set(label_ids) & held_out):
        raise ValueError(f"LEAKAGE: {len(leaked)} validation/test image(s) carry a pseudo-class "
                         f"label, e.g. {leaked[:3]}")
    if missing := sorted(train - set(label_ids)):
        raise ValueError(f"{len(missing)} training image(s) have no pseudo-class label, "
                         f"e.g. {missing[:3]}")
    if extra := sorted(set(label_ids) - train):
        raise ValueError(f"{len(extra)} labelled id(s) are not in the frozen training split, "
                         f"e.g. {extra[:3]}")
    if len(label_ids) != sizes["train"]:
        raise ValueError(f"expected {sizes['train']} training labels, found {len(label_ids)}")
    if unresolved := sorted(set(label_ids) - set(row_of)):
        raise ValueError(f"{len(unresolved)} labelled id(s) have no feature row, "
                         f"e.g. {unresolved[:3]}")

    order = np.argsort(label_ids, kind="stable")
    ids = [label_ids[i] for i in order]
    rows = np.array([row_of[i] for i in ids], dtype=np.int64)
    labels = labels[order]
    if len(set(rows.tolist())) != len(rows):
        raise ValueError("two training ids resolved to the same feature row")
    return ids, rows, labels


def fisher_scores(matrix_path, rows, labels, dim=FEATURE_DIM, chunk=CHUNK, progress_cb=_say):
    """Two-pass float64 Fisher scores over the given rows only. Returns a dict of results."""
    X = np.load(matrix_path, mmap_mode="r")
    if X.shape[1] != dim:
        raise ValueError(f"source matrix has {X.shape[1]} columns, expected {dim}")
    if rows.max(initial=-1) >= X.shape[0]:
        raise ValueError(f"row index {rows.max()} outside the {X.shape[0]}-row source matrix")

    classes = np.unique(labels)
    per_class = [rows[labels == c] for c in classes]
    counts = np.array([len(r) for r in per_class], dtype=np.int64)

    def blocks(r):
        for a in range(0, len(r), chunk):
            blk = np.asarray(X[r[a:a + chunk]], dtype=np.float64)
            if not np.isfinite(blk).all():
                bad = np.asarray(r[a:a + chunk])[~np.isfinite(blk).all(axis=1)]
                raise ValueError(f"source matrix has non-finite values in row(s) {bad[:5].tolist()}")
            yield blk

    # pass 1: class means, then the global training mean from the same sums
    totals = np.zeros((len(classes), dim), dtype=np.float64)
    for ci, r in enumerate(per_class):
        for blk in blocks(r):
            totals[ci] += blk.sum(axis=0)
    means = totals / counts[:, None]
    global_mean = totals.sum(axis=0) / counts.sum()
    progress_cb(f"pass 1/2 done: {len(classes)} class means over {counts.sum()} training rows")

    # SS_between from the mean table alone: exact, no cancellation
    ss_between = (counts[:, None] * (means - global_mean) ** 2).sum(axis=0)

    # pass 2: pooled within-class sum of squared deviations
    ss_within = np.zeros(dim, dtype=np.float64)
    for ci, r in enumerate(per_class):
        for blk in blocks(r):
            ss_within += ((blk - means[ci]) ** 2).sum(axis=0)
    progress_cb("pass 2/2 done: pooled within-class sum of squared deviations")

    scores, separable, constant = _ratio(ss_between, ss_within)
    return {"scores": scores, "ss_between": ss_between, "ss_within": ss_within,
            "classes": classes, "class_counts": counts,
            "n_zero_denominator_with_signal": int(separable.sum()),
            "n_zero_denominator_zero_numerator": int(constant.sum()),
            "n_zero_score": int((scores == 0).sum())}


def _ratio(ss_between, ss_within):
    """Fisher ratios from the two scatter vectors, with +inf reserved for a zero denominator.

    Returns (scores, separable, constant) where `separable` marks SS_within == 0 < SS_between (score
    +inf) and `constant` marks SS_within == SS_between == 0 (score 0.0). A non-finite ratio arising
    from a *positive* denominator is an overflow, not a separable feature, so it raises
    FloatingPointError rather than being recorded as +inf and inflating the separable count.

    Overflow is unreachable from a float32 source matrix of this size -- the worst case is
    (8000 * float32_max^2) / (2 * (float32_subnormal/2)^2) ~ 9.4e170, far below float64's 1.8e308
    (asserted by test_overflow_is_unreachable_from_a_float32_source). The branch is defensive.
    """
    ss_between = np.asarray(ss_between, dtype=np.float64)
    ss_within = np.asarray(ss_within, dtype=np.float64)
    if not (np.isfinite(ss_between).all() and np.isfinite(ss_within).all()):
        raise FloatingPointError("non-finite scatter term; Fisher scores would be meaningless")
    if (ss_within < 0).any() or (ss_between < 0).any():
        raise ValueError("negative sum of squares; this is a bug, not a numerical artefact")

    scores = np.zeros(ss_between.shape, dtype=np.float64)
    positive = ss_within > 0
    with np.errstate(over="ignore", divide="raise", invalid="raise"):
        scores[positive] = ss_between[positive] / ss_within[positive]
    # Deterministic and platform-independent: inspect the result rather than trust the FPU flags.
    overflowed = positive & ~np.isfinite(scores)
    if overflowed.any():
        j = int(np.flatnonzero(overflowed)[0])
        raise FloatingPointError(
            f"Fisher ratio overflowed float64 for {int(overflowed.sum())} feature(s) with a "
            f"positive denominator, e.g. feature {j}: SS_between={ss_between[j]!r}, "
            f"SS_within={ss_within[j]!r}. +inf is reserved for SS_within == 0, so this is refused "
            f"rather than recorded as a perfectly separable feature.")

    separable = ~positive & (ss_between > 0)
    scores[separable] = np.inf
    constant = ~positive & (ss_between == 0)
    if np.isnan(scores).any():
        raise ValueError("NaN Fisher score; no 0/0 division should be reachable")
    return scores, separable, constant


def select_top(scores, n_selected=N_SELECTED):
    """Top-n feature indices: descending score, ties by ascending original index.

    Returns (indices_ascending, rank_order, boundary) where `indices_ascending` is what the reduced
    matrix columns map through, and `boundary` describes the score at the cut honestly.
    """
    scores = np.asarray(scores)
    if n_selected > scores.size:
        raise ValueError(f"cannot select {n_selected} of {scores.size} features")
    rank_order = np.argsort(-scores, kind="stable")   # stable => ties keep ascending index order
    chosen = rank_order[:n_selected]
    cut = scores[chosen[-1]]
    tied_total = int((scores == cut).sum())
    tied_included = int((scores[chosen] == cut).sum())
    boundary = {"cut_score": _jsonable(cut),
                "boundary_tie": bool(tied_total > tied_included),
                "tied_at_cut_total": tied_total,
                "tied_at_cut_included": tied_included,
                "tied_at_cut_excluded": tied_total - tied_included}
    return np.sort(chosen).astype(np.int32), rank_order, boundary


def apply_selection(matrix_path, indices, out_path, chunk=CHUNK, progress_cb=_say):
    """Materialise rows x len(indices) float32 by taking the chosen source columns, read-only source."""
    X = np.load(matrix_path, mmap_mode="r")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(X.shape[0], len(indices)))
    try:
        for a in range(0, X.shape[0], chunk):
            out[a:a + chunk] = np.asarray(X[a:a + chunk])[:, indices]
        out.flush()
    finally:
        del out
    progress_cb(f"wrote {X.shape[0]}x{len(indices)} float32 to {out_path.name}")


def select(data_dir, out_dir, features_dir=None, pseudo_dir=None, force=False, progress_cb=_say):
    """Run Fisher selection for the frozen experiment: 4096 -> 512, training rows and labels only.

    The production dimensions are not parameters. Tests use `_select`.
    """
    return _select(data_dir, out_dir, features_dir, pseudo_dir, force=force,
                   progress_cb=progress_cb, dim=FEATURE_DIM, sizes=SIZES,
                   n_selected=N_SELECTED, verify_upstream=True)


def _select(data_dir, out_dir, features_dir=None, pseudo_dir=None, *, force=False,
            progress_cb=_say, dim=FEATURE_DIM, sizes=SIZES, n_selected=N_SELECTED,
            verify_upstream=True, chunk=CHUNK):
    """Shape-parameterised implementation. Private: production goes through `select`."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    features_dir = Path(features_dir) if features_dir else default_features_dir()
    pseudo_dir = Path(pseudo_dir) if pseudo_dir else default_pseudo_dir()

    existing = [n for n in (SCORES, INDICES, MATRIX, META) if (out_dir / n).exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite {existing} in {out_dir} (use --force)")

    if verify_upstream:
        for name, errs in (("features", validate_features(features_dir, data_dir)),
                           ("pseudo-classes", validate_pseudoclasses(pseudo_dir, data_dir,
                                                                     features_dir))):
            if errs:
                raise ValueError(f"upstream {name} artifacts do not validate; refusing to build on "
                                 f"them: {errs[:3]}")
        progress_cb("upstream Stage 2 and Stage 3 artifacts validate")

    ids, rows, labels = training_join(data_dir, features_dir, pseudo_dir, sizes)
    progress_cb(f"joined {len(ids)} training ids to feature rows by image id "
                f"({len(np.unique(labels))} pseudo-classes)")

    started = time.time()
    res = fisher_scores(features_dir / SOURCE_MATRIX, rows, labels, dim, chunk, progress_cb)
    indices, rank_order, boundary = select_top(res["scores"], n_selected)
    finite = res["scores"][np.isfinite(res["scores"])]
    progress_cb(f"selected {len(indices)} features; cut score {boundary['cut_score']}, "
                f"boundary tie {boundary['boundary_tie']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / SCORES, res["scores"])
    np.save(out_dir / INDICES, indices)
    apply_selection(features_dir / SOURCE_MATRIX, indices, out_dir / MATRIX, chunk, progress_cb)
    runtime = time.time() - started

    feature_meta = features_dir / FEATURE_META
    pseudo_meta = pseudo_dir / PSEUDO_META
    split_meta = data_dir / "splits" / "split_meta.json"
    meta = {
        "artifact_name": "fisher_top512",
        "formula": FORMULA,
        "variance_convention": VARIANCE_CONVENTION,
        "denominator_form": DENOMINATOR_FORM,
        "algorithm": "two-pass float64 (class means, then pooled squared deviations)",
        "standardized": False,
        "standardization_note": "the Fisher ratio is invariant to per-feature scaling",
        "epsilon_added_to_denominator": None,
        "n_features_in": int(dim),
        "n_selected": int(n_selected),
        "n_classes": int(len(res["classes"])),
        "class_labels": res["classes"].tolist(),
        "class_counts": {str(c): int(n) for c, n in zip(res["classes"], res["class_counts"])},
        "n_train": int(len(ids)),
        "chunk_rows": int(chunk),
        "determinism_note": ("float64 summation order depends on chunk_rows, so bitwise-identical "
                             "scores require the same chunk_rows, numpy build and BLAS"),
        "selection_rule": SELECTION_RULE,
        "column_order": COLUMN_ORDER,
        "boundary": boundary,
        "n_zero_denominator_with_signal": res["n_zero_denominator_with_signal"],
        "n_zero_denominator_zero_numerator": res["n_zero_denominator_zero_numerator"],
        "n_zero_score": res["n_zero_score"],
        "n_infinite_score": int(np.isinf(res["scores"]).sum()),
        "overflow_policy": OVERFLOW_POLICY,
        "n_selected_infinite": int(np.isinf(res["scores"][indices]).sum()),
        "score_stats_finite": {
            "min": float(finite.min()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "count": int(finite.size),
        },
        "non_finite_encoding": "the JSON string 'inf' stands for +infinity",
        "row_mapping": f"rows follow {FEATURE_IDS} unchanged; column i is source column idx[i]",
        "fitted_on": "train",
        "uses_val_test_data": False,
        "uses_val_test_labels": False,
        "upstream_validated": bool(verify_upstream),
        "train_ids_sha256": _ids_digest(ids),
        "source_matrix_sha256": _sha256(features_dir / SOURCE_MATRIX),
        "feature_ids_sha256": _sha256(features_dir / FEATURE_IDS),
        "feature_meta_sha256": _sha256(feature_meta) if feature_meta.is_file() else None,
        "pseudoclass_labels_sha256": _sha256(pseudo_dir / PSEUDO_LABELS),
        "pseudoclass_meta_sha256": _sha256(pseudo_meta) if pseudo_meta.is_file() else None,
        "split_meta_sha256": _sha256(split_meta) if split_meta.is_file() else None,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runtime_seconds": round(runtime, 1),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "status": "complete",
    }
    meta["artifact_sha256"] = {n: _sha256(out_dir / n) for n in (SCORES, INDICES, MATRIX)}
    (out_dir / META).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def validate_fisher(out_dir, data_dir, features_dir=None, pseudo_dir=None):
    """Validate the Stage 4 artifacts against the frozen experiment. Empty list means all clear.

    The feature width (4096), the split sizes and the top-k (512) are not parameters: they are the
    frozen methodology, so no caller can validate the real artifacts against a weaker rule. Tests
    use `_validate_fisher`.
    """
    return _validate_fisher(out_dir, data_dir, features_dir, pseudo_dir,
                            dim=FEATURE_DIM, sizes=SIZES, n_selected=N_SELECTED)


def _validate_fisher(out_dir, data_dir, features_dir=None, pseudo_dir=None, dim=FEATURE_DIM,
                     sizes=SIZES, n_selected=N_SELECTED, chunk=CHUNK):
    """Shape-parameterised implementation. Private: production uses `validate_fisher`."""
    out_dir, data_dir = Path(out_dir), Path(data_dir)
    features_dir = Path(features_dir) if features_dir else default_features_dir()
    pseudo_dir = Path(pseudo_dir) if pseudo_dir else default_pseudo_dir()

    errs = [f"missing artifact: {n}" for n in (SCORES, INDICES, MATRIX, META)
            if not (out_dir / n).is_file()]
    if errs:
        return errs
    try:
        meta = json.loads((out_dir / META).read_text())
    except json.JSONDecodeError as e:
        return [f"{META} is not valid JSON: {e}"]

    for name in (SCORES, INDICES, MATRIX):
        recorded = meta.get("artifact_sha256", {}).get(name)
        if recorded and recorded != _sha256(out_dir / name):
            errs.append(f"{name} changed since selection (sha256 mismatch)")

    for key, want in (("status", "complete"), ("fitted_on", "train"),
                      ("variance_convention", VARIANCE_CONVENTION),
                      ("denominator_form", DENOMINATOR_FORM),
                      ("selection_rule", SELECTION_RULE), ("column_order", COLUMN_ORDER),
                      ("overflow_policy", OVERFLOW_POLICY)):
        if meta.get(key) != want:
            errs.append(f"{META} records {key}={meta.get(key)!r}, expected {want!r}")
    for key in ("uses_val_test_data", "uses_val_test_labels", "standardized"):
        if meta.get(key) is not False:
            errs.append(f"{META} does not record {key}=false")
    if meta.get("epsilon_added_to_denominator") is not None:
        errs.append(f"an epsilon of {meta['epsilon_added_to_denominator']!r} was added to the "
                    "denominator; the methodology forbids it")
    if meta.get("n_selected") != n_selected:
        errs.append(f"{META} records n_selected={meta.get('n_selected')!r}, expected {n_selected}")
    if meta.get("n_features_in") != dim:
        errs.append(f"{META} records n_features_in={meta.get('n_features_in')!r}, expected {dim}")

    scores = np.load(out_dir / SCORES)
    indices = np.load(out_dir / INDICES)
    if scores.shape != (dim,):
        errs.append(f"{SCORES} has shape {scores.shape}, expected {(dim,)}")
    if np.isnan(scores).any():
        errs.append(f"{SCORES} contains {int(np.isnan(scores).sum())} NaN score(s)")
    if (scores < 0).any():
        errs.append(f"{SCORES} contains negative score(s); the ratio cannot be negative")
    if indices.shape != (n_selected,):
        errs.append(f"{INDICES} has shape {indices.shape}, expected {(n_selected,)}")
    elif indices.dtype != np.int32:
        errs.append(f"{INDICES} dtype is {indices.dtype}, expected int32")

    if scores.shape == (dim,) and indices.shape == (n_selected,) and not np.isnan(scores).any():
        if indices.min() < 0 or indices.max() >= dim:
            errs.append(f"{INDICES} span [{indices.min()}, {indices.max()}], "
                        f"expected [0, {dim - 1}]")
        elif not (np.diff(indices) > 0).all():
            errs.append(f"{INDICES} is not strictly ascending; the column mapping is not the "
                        "documented one")
        # independent re-derivation of the ranking rule from the saved scores
        re_indices, _, re_boundary = select_top(scores, n_selected)
        if not np.array_equal(indices, re_indices):
            missing = sorted(set(re_indices.tolist()) - set(indices.tolist()))
            errs.append(f"{INDICES} is not the top {n_selected} under '{SELECTION_RULE}'; "
                        f"{len(missing)} index/indices differ, e.g. {missing[:3]}")
        if meta.get("boundary") != re_boundary:
            errs.append(f"{META} boundary {meta.get('boundary')!r} does not match the scores "
                        f"({re_boundary!r})")
        # property check independent of argsort: nothing unselected may beat anything selected
        chosen = np.zeros(dim, dtype=bool)
        chosen[indices] = True
        if chosen.any() and (~chosen).any() and scores[~chosen].max() > scores[chosen].min():
            errs.append("an unselected feature scores strictly higher than a selected one")
        # With the overflow policy in force, every +inf must come from a zero denominator.
        if meta.get("n_infinite_score") != meta.get("n_zero_denominator_with_signal"):
            errs.append(
                f"{META} records {meta.get('n_infinite_score')!r} infinite score(s) but "
                f"{meta.get('n_zero_denominator_with_signal')!r} zero-denominator feature(s); "
                "under the overflow policy those counts must be equal")
        for key, computed in (("n_infinite_score", int(np.isinf(scores).sum())),
                              ("n_zero_score", int((scores == 0).sum())),
                              ("n_selected_infinite", int(np.isinf(scores[indices]).sum()))):
            if meta.get(key) != computed:
                errs.append(f"{META} records {key}={meta.get(key)!r}, scores give {computed}")

    reduced = np.load(out_dir / MATRIX, mmap_mode="r")
    expected_rows = sum(sizes.values())
    if reduced.dtype != np.float32:
        errs.append(f"{MATRIX} dtype is {reduced.dtype}, expected float32")
    if reduced.shape != (expected_rows, n_selected):
        errs.append(f"{MATRIX} shape is {reduced.shape}, expected {(expected_rows, n_selected)}")

    source = features_dir / SOURCE_MATRIX
    recorded_source = meta.get("source_matrix_sha256")
    if not source.is_file():
        errs.append(f"source matrix {source} is missing; cannot verify the column mapping")
    else:
        if recorded_source and recorded_source != _sha256(source):
            errs.append(f"{SOURCE_MATRIX} changed since selection; the baseline matrix must stay "
                        "untouched and the column mapping is no longer verifiable")
        X = np.load(source, mmap_mode="r")
        if X.shape != (expected_rows, dim):
            errs.append(f"source matrix shape is {X.shape}, expected {(expected_rows, dim)}")
        elif reduced.shape == (expected_rows, n_selected) and indices.shape == (n_selected,) \
                and 0 <= indices.min() and indices.max() < dim:
            bad = 0
            for a in range(0, expected_rows, chunk):
                got = np.asarray(reduced[a:a + chunk])
                want = np.asarray(X[a:a + chunk])[:, indices]
                bad += int((got != want).any(axis=1).sum())
                if not np.isfinite(got).all():
                    errs.append(f"{MATRIX} has non-finite values near row {a}")
                    break
            if bad:
                errs.append(f"{MATRIX} differs from the source columns in {bad} row(s)")

    live_ids = features_dir / FEATURE_IDS
    if meta.get("feature_ids_sha256") and live_ids.is_file() \
            and meta["feature_ids_sha256"] != _sha256(live_ids):
        errs.append(f"{FEATURE_IDS} changed since selection; row alignment is not guaranteed")
    live_labels = pseudo_dir / PSEUDO_LABELS
    if meta.get("pseudoclass_labels_sha256") and live_labels.is_file() \
            and meta["pseudoclass_labels_sha256"] != _sha256(live_labels):
        errs.append(f"{PSEUDO_LABELS} changed since selection; the scores no longer match the "
                    "labels they were fitted on")

    try:
        ids, _, _ = training_join(data_dir, features_dir, pseudo_dir, sizes)
    except (ValueError, FileNotFoundError) as e:
        errs.append(f"training join no longer holds: {e}")
    else:
        if meta.get("n_train") != len(ids):
            errs.append(f"{META} records n_train={meta.get('n_train')!r}, join gives {len(ids)}")
        if meta.get("train_ids_sha256") and meta["train_ids_sha256"] != _ids_digest(ids):
            errs.append("training ids changed since selection; scores are no longer attributable")
    return errs


def main(argv=None):
    p = argparse.ArgumentParser(prog="retinex.fisher")
    p.add_argument("command", choices=["select", "validate"])
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--features-dir", type=Path, default=default_features_dir())
    p.add_argument("--pseudo-dir", type=Path, default=default_pseudo_dir())
    p.add_argument("--out-dir", type=Path, default=default_out_dir())
    p.add_argument("--n-selected", type=int, default=N_SELECTED,
                   help=f"frozen at {N_SELECTED}; any other value is refused, not silently honoured")
    p.add_argument("--force", action="store_true", help="select: overwrite existing artifacts")
    a = p.parse_args(argv)

    if a.command == "select":
        if a.n_selected != N_SELECTED:
            print(f"--n-selected {a.n_selected} refused: the top-k is frozen at {N_SELECTED} "
                  f"(CLAUDE.md, STAGE 4 DECISIONS). Changing it needs explicit approval and a "
                  f"CLAUDE.md update.")
            return 1
        try:
            meta = select(a.data_dir, a.out_dir, a.features_dir, a.pseudo_dir, a.force)
        except (FileExistsError, ValueError, FileNotFoundError) as e:
            print(f"select failed: {e}")
            return 1
        print(f"selected {meta['n_selected']} of {meta['n_features_in']} features from "
              f"{meta['n_train']} training rows in {meta['runtime_seconds']}s -> {a.out_dir}")

    errs = validate_fisher(a.out_dir, a.data_dir, a.features_dir, a.pseudo_dir)
    print("\n".join(errs) if errs else
          "fisher OK: formula metadata, independent top-512 re-derivation, ascending column "
          "mapping, source-column equality, training join and checksums verified")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
