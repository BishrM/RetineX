"""Stage 4 Fisher Score tests. Synthetic and tiny: never touches the real dataset or artifacts.

Production constants (4096 features, 512 selected, 6000/1000/1000) are never relaxed; the fixtures
go through the private `_select` / `_validate_fisher` seams, and separate tests assert the public
entry points expose no overrides.
"""
import inspect
import json

import numpy as np
import pytest

from retinex import features as fe
from retinex import fisher as fi
from retinex import pseudoclasses as pc

SMALL = {"train": 12, "val": 2, "test": 2}
N_ROWS = sum(SMALL.values())
DIM = 8
K = 3          # pseudo-classes in the fixture
TOP = 4        # features selected in the fixture


def names(n=N_ROWS):
    return [f"i{i:03d}.jpg" for i in range(n)]


def build(tmp_path, matrix=None, labels=None, dim=DIM, shuffle_labels=False):
    """A frozen split, a feature artifact set and a pseudo-class artifact set, all tiny.

    Row order of the feature matrix is deliberately NOT the split order: ids are shuffled before
    being written to the ids file, so any code that assumes position implies split membership fails.
    """
    ids = names()
    data = tmp_path / "data"
    (data / "splits").mkdir(parents=True)
    (data / "captions").mkdir()
    parts = {"train": ids[:SMALL["train"]],
             "val": ids[SMALL["train"]:SMALL["train"] + 2],
             "test": ids[SMALL["train"] + 2:]}
    for k, part in parts.items():
        (data / "splits" / f"{k}.txt").write_text("\n".join(sorted(part)) + "\n")
    (data / "captions" / "Flickr8k.token.txt").write_text(
        "".join(f"{i}#{j}\tcaption {j}\n" for i in ids for j in range(5)))

    # feature rows in a scrambled id order
    row_ids = list(ids)
    np.random.default_rng(7).shuffle(row_ids)
    fdir = tmp_path / "features"
    fdir.mkdir()
    (fdir / fe.IDS).write_text("\n".join(row_ids) + "\n")
    if matrix is None:
        rng = np.random.default_rng(0)
        matrix = rng.normal(3.0, 1.0, (N_ROWS, dim)).astype(np.float32)
    else:
        matrix = np.asarray(matrix, dtype=np.float32)
    np.save(fdir / fe.MATRIX, matrix)
    (fdir / fe.META).write_text(json.dumps({"status": "complete", "uses_captions": False}))

    pdir = tmp_path / "pseudo"
    pdir.mkdir()
    train_sorted = sorted(parts["train"])
    if labels is None:
        labels = [n % K for n in range(len(train_sorted))]
    rows = list(zip(train_sorted, labels))
    if shuffle_labels:        # same pairs, different file order: the join must not care
        rows = rows[::-1]
    (pdir / pc.LABELS).write_text("image_id,pseudo_class\n" +
                                  "".join(f"{i},{l}\n" for i, l in rows))
    (pdir / pc.META).write_text(json.dumps({"status": "complete", "k": K,
                                            "uses_val_test_captions": False}))
    return data, fdir, pdir, parts, matrix, row_ids


def run_select(tmp_path, out_name="fisher", top=TOP, dim=DIM, **kw):
    data, fdir, pdir, parts, matrix, row_ids = build(tmp_path, dim=dim, **kw)
    out = tmp_path / out_name
    meta = fi._select(data, out, fdir, pdir, dim=dim, sizes=SMALL, n_selected=top,
                      verify_upstream=False, chunk=5, progress_cb=lambda *_: None)
    return out, data, fdir, pdir, meta, matrix, row_ids


def check(out, data, fdir, pdir, top=TOP, dim=DIM):
    return fi._validate_fisher(out, data, fdir, pdir, dim=dim, sizes=SMALL, n_selected=top, chunk=5)


def brute_force(X, y):
    """Independent reference: count-weighted Fisher with population variance, written plainly."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    num = np.zeros(X.shape[1])
    den = np.zeros(X.shape[1])
    for c in np.unique(y):
        Xc = X[y == c]
        num += len(Xc) * (Xc.mean(axis=0) - mu) ** 2
        den += len(Xc) * Xc.var(axis=0, ddof=0)
    out = np.zeros(X.shape[1])
    pos = den > 0
    out[pos] = num[pos] / den[pos]
    out[~pos & (num > 0)] = np.inf
    return out


# --- formula correctness ----------------------------------------------------

def test_scores_match_an_independent_reference(tmp_path):
    out, data, fdir, pdir, meta, matrix, row_ids = run_select(tmp_path)
    ids, rows, labels = fi.training_join(data, fdir, pdir, SMALL)
    got = np.load(out / fi.SCORES)
    want = brute_force(matrix[rows], labels)
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12)


def test_pooled_ssd_equals_count_weighted_variance_form():
    """The two denominator forms are algebraically identical under ddof=0."""
    rng = np.random.default_rng(3)
    X = rng.normal(5, 2, (200, 6))
    y = rng.integers(0, 4, 200)
    ssd = np.zeros(6)
    weighted = np.zeros(6)
    for c in np.unique(y):
        Xc = X[y == c]
        ssd += ((Xc - Xc.mean(axis=0)) ** 2).sum(axis=0)
        weighted += len(Xc) * Xc.var(axis=0, ddof=0)
    assert np.allclose(ssd, weighted, rtol=0, atol=1e-9)


def test_variance_convention_is_population_not_sample(tmp_path):
    """ddof=1 would give measurably different scores; the frozen choice is ddof=0."""
    out, data, fdir, pdir, meta, matrix, _ = run_select(tmp_path)
    ids, rows, labels = fi.training_join(data, fdir, pdir, SMALL)
    X = matrix[rows].astype(np.float64)
    mu = X.mean(axis=0)
    num = sum(len(X[labels == c]) * (X[labels == c].mean(axis=0) - mu) ** 2 for c in np.unique(labels))
    den1 = sum(len(X[labels == c]) * X[labels == c].var(axis=0, ddof=1) for c in np.unique(labels))
    got = np.load(out / fi.SCORES)
    assert not np.allclose(got, num / den1)
    assert meta["variance_convention"] == "population (ddof=0)"


def test_perfectly_separating_feature_outranks_noise(tmp_path):
    data, fdir, pdir, parts, matrix, _ = build(tmp_path)
    ids, rows, labels = fi.training_join(data, fdir, pdir, SMALL)
    rng = np.random.default_rng(1)
    m = np.asarray(matrix, dtype=np.float64).copy()
    # column 5 separates the classes cleanly, with a small spread inside each class
    m[rows, 5] = labels * 50.0 + rng.normal(0, 0.01, len(rows))
    np.save(fdir / fe.MATRIX, m.astype(np.float32))
    out = tmp_path / "sep"
    fi._select(data, out, fdir, pdir, dim=DIM, sizes=SMALL, n_selected=TOP,
               verify_upstream=False, chunk=5, progress_cb=lambda *_: None)
    scores = np.load(out / fi.SCORES)
    assert scores[5] == scores.max() and 5 in np.load(out / fi.INDICES).tolist()


# --- edge cases -------------------------------------------------------------

def test_constant_feature_scores_zero_and_is_not_selected(tmp_path):
    rng = np.random.default_rng(2)
    m = rng.normal(3, 1, (N_ROWS, DIM)).astype(np.float32)
    m[:, 2] = 0.0          # all-zero column: SS_within == 0 and SS_between == 0
    m[:, 6] = 7.5          # constant non-zero column: same 0/0 case
    out, data, fdir, pdir, meta, _, _ = run_select(tmp_path, matrix=m)
    scores = np.load(out / fi.SCORES)
    assert scores[2] == 0.0 and scores[6] == 0.0
    assert meta["n_zero_denominator_zero_numerator"] == 2
    assert 2 not in np.load(out / fi.INDICES).tolist()
    assert 6 not in np.load(out / fi.INDICES).tolist()
    assert check(out, data, fdir, pdir) == []


def test_zero_denominator_with_signal_scores_positive_infinity(tmp_path):
    data, fdir, pdir, parts, matrix, row_ids = build(tmp_path)
    ids, rows, labels = fi.training_join(data, fdir, pdir, SMALL)
    m = np.asarray(matrix, dtype=np.float32).copy()
    m[rows, 1] = labels.astype(np.float32) * 10.0   # exactly constant inside every class
    np.save(fdir / fe.MATRIX, m)
    out = tmp_path / "inf"
    meta = fi._select(data, out, fdir, pdir, dim=DIM, sizes=SMALL, n_selected=TOP,
                      verify_upstream=False, chunk=5, progress_cb=lambda *_: None)
    scores = np.load(out / fi.SCORES)
    assert np.isinf(scores[1]) and scores[1] > 0
    assert meta["n_zero_denominator_with_signal"] == 1 and meta["n_selected_infinite"] == 1
    assert not np.isnan(scores).any()
    assert 1 in np.load(out / fi.INDICES).tolist()
    assert check(out, data, fdir, pdir) == []


def test_ratio_reserves_infinity_for_a_zero_denominator():
    scores, separable, constant = fi._ratio(np.array([5.0, 0.0, 8.0]), np.array([0.0, 0.0, 2.0]))
    assert np.isinf(scores[0]) and scores[1] == 0.0 and scores[2] == 4.0
    assert separable.tolist() == [True, False, False]
    assert constant.tolist() == [False, True, False]
    assert not np.isnan(scores).any()


def test_ratio_raises_on_overflow_with_a_positive_denominator():
    """An overflowing ratio is NOT a separable feature and must not be recorded as +inf."""
    with pytest.raises(FloatingPointError, match="overflowed float64"):
        fi._ratio(np.array([1e300]), np.array([1e-300]))
    # the message must name the offending feature and both scatter terms
    try:
        fi._ratio(np.array([1.0, 1e300]), np.array([1.0, 1e-300]))
    except FloatingPointError as e:
        assert "feature 1" in str(e) and "SS_between" in str(e) and "SS_within" in str(e)
        assert "reserved for SS_within == 0" in str(e)


def test_ratio_overflow_does_not_inflate_the_separable_count():
    """The failure mode this guards: overflow silently counted as a zero-denominator feature."""
    ss_between = np.array([1e300, 7.0])
    ss_within = np.array([1e-300, 0.0])       # one overflow, one genuinely separable
    with pytest.raises(FloatingPointError):
        fi._ratio(ss_between, ss_within)
    # with the overflow removed, exactly one separable feature remains
    scores, separable, _ = fi._ratio(np.array([1.0, 7.0]), np.array([1.0, 0.0]))
    assert int(separable.sum()) == 1 and np.isinf(scores[1])


def test_ratio_rejects_non_finite_scatter_terms():
    for between, within in (([np.inf], [1.0]), ([1.0], [np.inf]), ([np.nan], [1.0])):
        with pytest.raises(FloatingPointError, match="non-finite scatter term"):
            fi._ratio(np.array(between), np.array(within))


def test_ratio_rejects_negative_scatter_terms():
    with pytest.raises(ValueError, match="negative sum of squares"):
        fi._ratio(np.array([1.0]), np.array([-1e-9]))


def test_overflow_is_unreachable_from_a_float32_source():
    """Documents why the overflow branch is defensive rather than expected.

    SS_between <= N * float32_max^2; the smallest non-zero SS_within is two members of one class at
    0 and at the smallest float32 subnormal. Their ratio bounds every finite ratio this pipeline
    can produce.
    """
    m = float(np.finfo(np.float32).max)
    tiny = float(np.finfo(np.float32).smallest_subnormal)
    worst = (8000 * m ** 2) / (2 * (tiny / 2) ** 2)
    assert worst < np.finfo(np.float64).max
    assert worst == pytest.approx(9.43e170, rel=0.01)


def test_metadata_invariant_infinite_equals_zero_denominator(tmp_path):
    data, fdir, pdir, parts, matrix, _ = build(tmp_path)
    ids, rows, labels = fi.training_join(data, fdir, pdir, SMALL)
    m = np.asarray(matrix, dtype=np.float32).copy()
    m[rows, 1] = labels.astype(np.float32) * 10.0      # one genuinely separable feature
    np.save(fdir / fe.MATRIX, m)
    out = tmp_path / "inv"
    meta = fi._select(data, out, fdir, pdir, dim=DIM, sizes=SMALL, n_selected=TOP,
                      verify_upstream=False, chunk=5, progress_cb=lambda *_: None)
    assert meta["n_infinite_score"] == meta["n_zero_denominator_with_signal"] == 1
    assert meta["overflow_policy"] == fi.OVERFLOW_POLICY
    assert check(out, data, fdir, pdir) == []


def test_validate_detects_a_broken_infinite_count_invariant(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    meta = json.loads((out / fi.META).read_text())
    meta["n_zero_denominator_with_signal"] = 3        # claim separable features that do not exist
    (out / fi.META).write_text(json.dumps(meta))
    errs = check(out, data, fdir, pdir)
    assert any("those counts must be equal" in e for e in errs)


def test_validate_detects_a_tampered_overflow_policy(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    meta = json.loads((out / fi.META).read_text())
    meta["overflow_policy"] = "record overflow as +inf"
    (out / fi.META).write_text(json.dumps(meta))
    assert any("overflow_policy" in e for e in check(out, data, fdir, pdir))


def test_no_epsilon_is_recorded(tmp_path):
    out, _, _, _, meta, _, _ = run_select(tmp_path)
    assert meta["epsilon_added_to_denominator"] is None and meta["standardized"] is False


def test_non_finite_source_values_raise(tmp_path):
    rng = np.random.default_rng(4)
    m = rng.normal(3, 1, (N_ROWS, DIM)).astype(np.float32)
    data, fdir, pdir, parts, _, _ = build(tmp_path, matrix=m)
    ids, rows, _ = fi.training_join(data, fdir, pdir, SMALL)
    m[rows[0], 0] = np.nan
    np.save(fdir / fe.MATRIX, m)
    with pytest.raises(ValueError, match="non-finite"):
        fi._select(data, tmp_path / "bad", fdir, pdir, dim=DIM, sizes=SMALL, n_selected=TOP,
                   verify_upstream=False, chunk=5, progress_cb=lambda *_: None)


def test_large_magnitude_low_variance_stays_stable(tmp_path):
    """The two-pass form must not produce negative SS or NaN on a badly conditioned column."""
    rng = np.random.default_rng(5)
    m = rng.normal(3, 1, (N_ROWS, DIM)).astype(np.float32)
    m[:, 0] = 1e6 + rng.normal(0, 1e-3, N_ROWS)
    out, data, fdir, pdir, meta, _, _ = run_select(tmp_path, matrix=m)
    scores = np.load(out / fi.SCORES)
    assert not np.isnan(scores).any() and (scores >= 0).all()
    assert check(out, data, fdir, pdir) == []


# --- deterministic ranking and ties -----------------------------------------

def test_ties_break_by_ascending_original_index():
    scores = np.array([5.0, 1.0, 5.0, 5.0, 2.0, 5.0])
    idx, rank, boundary = fi.select_top(scores, 3)
    assert rank[:4].tolist() == [0, 2, 3, 5]      # equal scores in ascending index order
    assert idx.tolist() == [0, 2, 3]
    assert boundary["boundary_tie"] is True
    assert boundary["cut_score"] == 5.0
    assert boundary["tied_at_cut_total"] == 4
    assert boundary["tied_at_cut_included"] == 3
    assert boundary["tied_at_cut_excluded"] == 1


def test_multiple_infinite_scores_rank_first_by_index():
    scores = np.array([1.0, np.inf, 0.5, np.inf, np.inf, 2.0])
    idx, rank, boundary = fi.select_top(scores, 3)
    assert rank[:3].tolist() == [1, 3, 4]
    assert idx.tolist() == [1, 3, 4]
    assert boundary["cut_score"] == "inf"
    assert boundary["boundary_tie"] is False and boundary["tied_at_cut_total"] == 3


def test_infinite_tie_cut_through_is_recorded():
    scores = np.array([np.inf, np.inf, np.inf, 1.0])
    idx, _, boundary = fi.select_top(scores, 2)
    assert idx.tolist() == [0, 1]
    assert boundary["cut_score"] == "inf" and boundary["boundary_tie"] is True
    assert boundary["tied_at_cut_total"] == 3 and boundary["tied_at_cut_excluded"] == 1


def test_no_tie_is_recorded_as_no_tie():
    idx, _, boundary = fi.select_top(np.array([4.0, 3.0, 2.0, 1.0]), 2)
    assert idx.tolist() == [0, 1] and boundary["boundary_tie"] is False
    assert boundary["tied_at_cut_total"] == 1 and boundary["tied_at_cut_excluded"] == 0


def test_selected_indices_are_sorted_ascending(tmp_path):
    out, _, _, _, meta, _, _ = run_select(tmp_path)
    idx = np.load(out / fi.INDICES)
    assert (np.diff(idx) > 0).all() and idx.dtype == np.int32
    assert meta["column_order"].startswith("selected original indices sorted ascending")


def test_two_runs_give_identical_artifacts(tmp_path):
    out_a, data, fdir, pdir, meta_a, _, _ = run_select(tmp_path, out_name="a")
    meta_b = fi._select(data, tmp_path / "b", fdir, pdir, dim=DIM, sizes=SMALL, n_selected=TOP,
                        verify_upstream=False, chunk=5, progress_cb=lambda *_: None)
    assert meta_a["artifact_sha256"] == meta_b["artifact_sha256"]
    assert meta_a["boundary"] == meta_b["boundary"]


def test_select_top_rejects_more_than_available():
    with pytest.raises(ValueError, match="cannot select"):
        fi.select_top(np.arange(4.0), 5)


# --- the join: ids, never positions -----------------------------------------

def test_join_is_by_image_id_not_row_position(tmp_path):
    data, fdir, pdir, parts, matrix, row_ids = build(tmp_path)
    ids, rows, labels = fi.training_join(data, fdir, pdir, SMALL)
    assert ids == sorted(parts["train"])
    assert rows.tolist() == [row_ids.index(i) for i in ids]
    assert sorted(rows.tolist()) != list(range(len(ids)))   # proves rows are not a leading slice
    assert len(set(rows.tolist())) == len(rows)


def test_label_file_order_does_not_change_scores(tmp_path):
    out_a, data_a, f_a, p_a, _, _, _ = run_select(tmp_path / "x", out_name="a")
    out_b, data_b, f_b, p_b, _, _, _ = run_select(tmp_path / "y", out_name="b", shuffle_labels=True)
    assert np.array_equal(np.load(out_a / fi.SCORES), np.load(out_b / fi.SCORES))


def test_join_requires_one_to_one_cover_of_the_training_split(tmp_path):
    data, fdir, pdir, parts, _, _ = build(tmp_path)
    ids, labels = pc.read_labels(pdir / pc.LABELS)
    (pdir / pc.LABELS).write_text("image_id,pseudo_class\n" +
                                  "".join(f"{i},{l}\n" for i, l in list(zip(ids, labels))[:-1]))
    with pytest.raises(ValueError, match="have no pseudo-class label"):
        fi.training_join(data, fdir, pdir, SMALL)


def test_join_rejects_a_labelled_heldout_image(tmp_path):
    data, fdir, pdir, parts, _, _ = build(tmp_path)
    ids, labels = pc.read_labels(pdir / pc.LABELS)
    rows = list(zip(ids, labels))[:-1] + [(parts["val"][0], 0)]
    (pdir / pc.LABELS).write_text("image_id,pseudo_class\n" +
                                  "".join(f"{i},{l}\n" for i, l in sorted(rows)))
    with pytest.raises(ValueError, match="LEAKAGE"):
        fi.training_join(data, fdir, pdir, SMALL)


def test_join_rejects_an_unknown_labelled_id(tmp_path):
    """Keep all 12 real ids and add a stranger, so only the 'extra id' rule can fire."""
    data, fdir, pdir, parts, _, _ = build(tmp_path)
    ids, labels = pc.read_labels(pdir / pc.LABELS)
    rows = list(zip(ids, labels)) + [("stranger.jpg", 0)]
    (pdir / pc.LABELS).write_text("image_id,pseudo_class\n" +
                                  "".join(f"{i},{l}\n" for i, l in sorted(rows)))
    with pytest.raises(ValueError, match="not in the frozen training split"):
        fi.training_join(data, fdir, pdir, SMALL)


def test_join_rejects_duplicate_feature_ids(tmp_path):
    data, fdir, pdir, parts, _, row_ids = build(tmp_path)
    (fdir / fe.IDS).write_text("\n".join(row_ids[:-1] + [row_ids[0]]) + "\n")
    with pytest.raises(ValueError, match="duplicate ids"):
        fi.training_join(data, fdir, pdir, SMALL)


def test_heldout_rows_never_influence_the_scores(tmp_path):
    """Overwrite every val/test row with garbage: the scores must not move."""
    out_a, data, fdir, pdir, meta_a, matrix, _ = run_select(tmp_path, out_name="a")
    ids, rows, _ = fi.training_join(data, fdir, pdir, SMALL)
    held = sorted(set(range(N_ROWS)) - set(rows.tolist()))
    m = np.asarray(matrix, dtype=np.float32).copy()
    m[held] = 999.0
    np.save(fdir / fe.MATRIX, m)
    meta_b = fi._select(data, tmp_path / "b", fdir, pdir, dim=DIM, sizes=SMALL, n_selected=TOP,
                        verify_upstream=False, chunk=5, progress_cb=lambda *_: None)
    assert np.array_equal(np.load(out_a / fi.SCORES), np.load(tmp_path / "b" / fi.SCORES))
    assert meta_a["artifact_sha256"][fi.INDICES] == meta_b["artifact_sha256"][fi.INDICES]
    # but the reduced matrix does change, because held-out rows are transformed, not fitted
    assert not np.array_equal(np.load(out_a / fi.MATRIX), np.load(tmp_path / "b" / fi.MATRIX))


# --- output shape, mapping, baseline ----------------------------------------

def test_output_shape_dtype_and_column_mapping(tmp_path):
    out, data, fdir, pdir, meta, matrix, _ = run_select(tmp_path)
    reduced = np.load(out / fi.MATRIX)
    idx = np.load(out / fi.INDICES)
    assert reduced.shape == (N_ROWS, TOP) and reduced.dtype == np.float32
    assert np.isfinite(reduced).all()
    for i, src in enumerate(idx):
        assert np.array_equal(reduced[:, i], np.asarray(matrix)[:, src])


def test_source_matrix_is_untouched(tmp_path):
    data, fdir, pdir, parts, matrix, _ = build(tmp_path)
    before = fe._sha256(fdir / fe.MATRIX)
    meta = fi._select(data, tmp_path / "f", fdir, pdir, dim=DIM, sizes=SMALL, n_selected=TOP,
                      verify_upstream=False, chunk=5, progress_cb=lambda *_: None)
    assert fe._sha256(fdir / fe.MATRIX) == before == meta["source_matrix_sha256"]


def test_metadata_records_provenance_digests(tmp_path):
    out, data, fdir, pdir, meta, _, _ = run_select(tmp_path)
    assert meta["source_matrix_sha256"] == fe._sha256(fdir / fe.MATRIX)
    assert meta["feature_ids_sha256"] == fe._sha256(fdir / fe.IDS)
    assert meta["pseudoclass_labels_sha256"] == fe._sha256(pdir / pc.LABELS)
    ids, _, _ = fi.training_join(data, fdir, pdir, SMALL)
    assert meta["train_ids_sha256"] == fe._ids_digest(ids)
    for name in (fi.SCORES, fi.INDICES, fi.MATRIX):
        assert meta["artifact_sha256"][name] == fe._sha256(out / name)
    assert meta["fitted_on"] == "train" and meta["uses_val_test_labels"] is False
    assert meta["n_train"] == SMALL["train"] and meta["n_classes"] == K
    assert sum(meta["class_counts"].values()) == SMALL["train"]


def test_select_refuses_to_overwrite_without_force(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        fi._select(data, out, fdir, pdir, dim=DIM, sizes=SMALL, n_selected=TOP,
                   verify_upstream=False, chunk=5, progress_cb=lambda *_: None)
    fi._select(data, out, fdir, pdir, dim=DIM, sizes=SMALL, n_selected=TOP, force=True,
               verify_upstream=False, chunk=5, progress_cb=lambda *_: None)


def test_select_refuses_unvalidated_upstream_artifacts(tmp_path):
    """With verify_upstream, the tiny fixture cannot pass the frozen Stage 2/3 validators."""
    data, fdir, pdir, parts, _, _ = build(tmp_path)
    with pytest.raises(ValueError, match="upstream .* do not validate"):
        fi._select(data, tmp_path / "f", fdir, pdir, dim=DIM, sizes=SMALL, n_selected=TOP,
                   verify_upstream=True, chunk=5, progress_cb=lambda *_: None)


# --- frozen public API ------------------------------------------------------

def test_public_api_exposes_no_production_overrides(tmp_path):
    assert set(inspect.signature(fi.validate_fisher).parameters) == {
        "out_dir", "data_dir", "features_dir", "pseudo_dir"}
    assert set(inspect.signature(fi.select).parameters) == {
        "data_dir", "out_dir", "features_dir", "pseudo_dir", "force", "progress_cb"}
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    for bad in ({"dim": DIM}, {"sizes": SMALL}, {"n_selected": TOP}):
        with pytest.raises(TypeError):
            fi.validate_fisher(out, data, fdir, pdir, **bad)
        with pytest.raises(TypeError):
            fi.select(data, out, fdir, pdir, **bad)
    assert fi.N_SELECTED == 512 and fi.FEATURE_DIM == 4096


def test_public_validate_enforces_the_frozen_shape(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    errs = fi.validate_fisher(out, data, fdir, pdir)
    assert any("expected 512" in e for e in errs)
    assert any("expected 4096" in e for e in errs)
    assert check(out, data, fdir, pdir) == []


def test_cli_refuses_a_non_frozen_top_k(tmp_path, capsys):
    out = tmp_path / "never"
    rc = fi.main(["select", "--n-selected", "256", "--out-dir", str(out)])
    assert rc == 1 and "frozen at 512" in capsys.readouterr().out
    assert not out.exists()


# --- validate ---------------------------------------------------------------

def test_validate_clean_artifacts_pass(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    assert check(out, data, fdir, pdir) == []


@pytest.mark.parametrize("name", [fi.SCORES, fi.INDICES, fi.MATRIX, fi.META])
def test_validate_detects_missing_artifact(tmp_path, name):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    (out / name).unlink()
    assert check(out, data, fdir, pdir) == [f"missing artifact: {name}"]


def test_validate_detects_tampered_artifacts(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    scores = np.load(out / fi.SCORES)
    scores[0] = 1e9
    np.save(out / fi.SCORES, scores)
    assert any("sha256 mismatch" in e for e in check(out, data, fdir, pdir))


def test_validate_rederives_the_selection_independently(tmp_path):
    """Swap one selected index for an unselected one and refresh every checksum."""
    out, data, fdir, pdir, _, matrix, _ = run_select(tmp_path)
    idx = np.load(out / fi.INDICES)
    scores = np.load(out / fi.SCORES)
    victim = int(np.argmin(np.where(np.isin(np.arange(DIM), idx), scores, np.inf)))
    replacement = int(np.argmax(np.where(np.isin(np.arange(DIM), idx), -np.inf, scores)))
    forged = np.sort(np.array([i for i in idx if i != victim] + [replacement], dtype=np.int32))
    np.save(out / fi.INDICES, forged)
    fi.apply_selection(fdir / fe.MATRIX, forged, out / fi.MATRIX, 5, lambda *_: None)
    meta = json.loads((out / fi.META).read_text())
    meta["artifact_sha256"] = {n: fe._sha256(out / n) for n in (fi.SCORES, fi.INDICES, fi.MATRIX)}
    (out / fi.META).write_text(json.dumps(meta))
    errs = check(out, data, fdir, pdir)
    assert any("is not the top" in e for e in errs)
    assert any("scores strictly higher than a selected one" in e for e in errs)


def test_validate_detects_non_ascending_indices(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    idx = np.load(out / fi.INDICES)
    np.save(out / fi.INDICES, idx[::-1].copy())
    meta = json.loads((out / fi.META).read_text())
    meta["artifact_sha256"][fi.INDICES] = fe._sha256(out / fi.INDICES)
    (out / fi.META).write_text(json.dumps(meta))
    assert any("not strictly ascending" in e for e in check(out, data, fdir, pdir))


def test_validate_detects_wrong_index_count(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    assert any(f"expected ({TOP + 1},)" in e for e in check(out, data, fdir, pdir, top=TOP + 1))


def test_validate_detects_mismatched_reduced_matrix(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    reduced = np.load(out / fi.MATRIX)
    reduced[3, 0] += 1.0
    np.save(out / fi.MATRIX, reduced)
    meta = json.loads((out / fi.META).read_text())
    meta["artifact_sha256"][fi.MATRIX] = fe._sha256(out / fi.MATRIX)
    (out / fi.META).write_text(json.dumps(meta))
    assert any("differs from the source columns in 1 row" in e for e in check(out, data, fdir, pdir))


def test_validate_detects_changed_source_matrix(tmp_path):
    out, data, fdir, pdir, _, matrix, _ = run_select(tmp_path)
    m = np.asarray(matrix, dtype=np.float32).copy()
    m[0, 0] += 5.0
    np.save(fdir / fe.MATRIX, m)
    errs = check(out, data, fdir, pdir)
    assert any("baseline matrix must stay untouched" in e for e in errs)


def test_validate_detects_changed_labels_and_ids(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    ids, labels = pc.read_labels(pdir / pc.LABELS)
    labels[0] = (labels[0] + 1) % K
    (pdir / pc.LABELS).write_text("image_id,pseudo_class\n" +
                                  "".join(f"{i},{l}\n" for i, l in zip(ids, labels)))
    assert any("no longer match the labels" in e for e in check(out, data, fdir, pdir))


def test_validate_detects_broken_training_join(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    ids, labels = pc.read_labels(pdir / pc.LABELS)
    (pdir / pc.LABELS).write_text("image_id,pseudo_class\n" +
                                  "".join(f"{i},{l}\n" for i, l in list(zip(ids, labels))[:-1]))
    assert any("training join no longer holds" in e for e in check(out, data, fdir, pdir))


@pytest.mark.parametrize("patch,needle", [
    ({"status": "running"}, "status='running'"),
    ({"fitted_on": "all"}, "fitted_on='all'"),
    ({"uses_val_test_labels": True}, "uses_val_test_labels=false"),
    ({"uses_val_test_data": True}, "uses_val_test_data=false"),
    ({"standardized": True}, "standardized=false"),
    ({"epsilon_added_to_denominator": 1e-12}, "methodology forbids it"),
    ({"variance_convention": "sample (ddof=1)"}, "variance_convention"),
    ({"denominator_form": "raw sums"}, "denominator_form"),
    ({"selection_rule": "ascending score"}, "selection_rule"),
    ({"column_order": "by rank"}, "column_order"),
    ({"n_zero_score": 999}, "scores give"),
    ({"n_infinite_score": 7}, "scores give"),
    ({"n_train": 3}, "join gives"),
])
def test_validate_detects_metadata_claims(tmp_path, patch, needle):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    meta = json.loads((out / fi.META).read_text())
    meta.update(patch)
    (out / fi.META).write_text(json.dumps(meta))
    assert any(needle in e for e in check(out, data, fdir, pdir))


def test_validate_detects_wrong_boundary_metadata(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    meta = json.loads((out / fi.META).read_text())
    meta["boundary"] = dict(meta["boundary"], tied_at_cut_excluded=5, boundary_tie=True)
    (out / fi.META).write_text(json.dumps(meta))
    assert any("does not match the scores" in e for e in check(out, data, fdir, pdir))


def test_validate_detects_malformed_metadata(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    (out / fi.META).write_text("{not json")
    assert any("not valid JSON" in e for e in check(out, data, fdir, pdir))


def test_validate_detects_nan_scores(tmp_path):
    out, data, fdir, pdir, _, _, _ = run_select(tmp_path)
    scores = np.load(out / fi.SCORES)
    scores[0] = np.nan
    np.save(out / fi.SCORES, scores)
    meta = json.loads((out / fi.META).read_text())
    meta["artifact_sha256"][fi.SCORES] = fe._sha256(out / fi.SCORES)
    (out / fi.META).write_text(json.dumps(meta))
    assert any("NaN score" in e for e in check(out, data, fdir, pdir))
