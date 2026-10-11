"""Stage 3 pseudo-class tests. Synthetic and tiny: never touches the real dataset or artifacts.

The fixture carries 240 training images over 20 themes, so the frozen k=20 yields 12 images per
cluster and legitimately clears the frozen 10-image floor. Neither K nor the floor is reduced for the
tests: production refuses any k but 20, and the floor is a constant with no parameter to override.
"""
import inspect
import json
import string

import numpy as np
import pytest

from retinex import dataset as d
from retinex import pseudoclasses as pc

SMALL = {"train": 240, "val": 2, "test": 2}
N_TRAIN = SMALL["train"]
PER_CLUSTER = N_TRAIN // 20  # 12, comfortably above the frozen 10-image floor
TINY = 40  # a deliberately small split, used only where a candidate k must be out of range
A = string.ascii_lowercase


def themed_captions(n):
    """Five captions for training image n, themed on n % 20 with a per-image unique token.

    Words are purely alphabetic on purpose: token_pattern=r"[a-z]{2,}" strips digits, so numbered
    words like "theme1"/"theme2" would all collapse to "theme" and leave the documents identical.
    Adjacent themes share a word, which gives k=20 a real structure to find.
    """
    t = n % 20
    w, nxt = A[t] * 3, A[(t + 1) % 20] * 3
    return [f"{w} {w}", f"{w} {nxt}", "shared", "scene", f"zz{A[n // 20]}{A[n % 20]}"]
# Marker words that occur only in held-out captions and must never be learned.
HELDOUT = ["zzqq unseen marker word", "wwvv another unseen marker",
           "xxrr third unseen marker", "yytt fourth unseen marker"]


def build(tmp_path, train_caps=None, heldout_caps=None, n_caps=5, n_train=None):
    """A frozen split plus a caption file, deliberately small."""
    n_train = N_TRAIN if n_train is None else n_train
    data = tmp_path / "data"
    (data / "splits").mkdir(parents=True)
    (data / "captions").mkdir()
    names = [f"i{n:03d}.jpg" for n in range(n_train + 4)]
    parts = {"train": names[:n_train], "val": names[n_train:n_train + 2], "test": names[n_train + 2:]}
    for k, part in parts.items():
        (data / "splits" / f"{k}.txt").write_text("\n".join(sorted(part)) + "\n")

    train_caps = train_caps or {i: themed_captions(n)
                                for n, i in enumerate(sorted(parts["train"]))}
    heldout_caps = heldout_caps or {i: [HELDOUT[n]] * n_caps
                                    for n, i in enumerate(sorted(parts["val"] + parts["test"]))}
    lines = []
    for img, caps in sorted({**train_caps, **heldout_caps}.items()):
        lines += [f"{img}#{j}\t{c}" for j, c in enumerate(caps)]
    (data / "captions" / "Flickr8k.token.txt").write_text("\n".join(lines) + "\n")
    return data, parts


def features(tmp_path, ids):
    """A stand-in features dir holding only the row-id list that Stage 3 checksums."""
    fdir = tmp_path / "features"
    fdir.mkdir()
    (fdir / pc.FEATURE_IDS).write_text("\n".join(sorted(ids)) + "\n")
    return fdir


def run_fit(tmp_path, out_name="pseudo", **kw):
    data, parts = build(tmp_path, **kw)
    fdir = features(tmp_path, sum(parts.values(), []))
    out = tmp_path / out_name
    meta = pc.fit(data, out, fdir, min_df=1, progress_cb=lambda *_: None)
    return out, data, fdir, meta


def check(out, data, fdir, **kw):
    """Validate through the normal production path: the 10-image floor always applies."""
    kw.setdefault("sizes", SMALL)
    return pc._validate_pseudoclasses(out, data, fdir, **kw)


def patch_meta(out, **fields):
    """Patch metadata by dotted key, e.g. patch_meta(out, **{"silhouette_sweep.metric": "x"})."""
    meta = json.loads((out / pc.META).read_text())
    for key, value in fields.items():
        target, _, leaf = key.rpartition(".")
        node = meta
        for part in filter(None, target.split(".")):
            node = node[part]
        node[leaf] = value
    (out / pc.META).write_text(json.dumps(meta))
    return meta


def rewrite_labels(out, ids, labels, refresh_checksum=True):
    """Rewrite the label CSV, refreshing its checksum so the test targets one failure at a time."""
    (out / pc.LABELS).write_text("image_id,pseudo_class\n" +
                                 "".join(f"{i},{l}\n" for i, l in zip(ids, labels)))
    if refresh_checksum:  # the artifact name contains a dot, so patch it without patch_meta
        meta = json.loads((out / pc.META).read_text())
        meta["artifact_sha256"][pc.LABELS] = pc._sha256(out / pc.LABELS)
        (out / pc.META).write_text(json.dumps(meta))


# --- documents and alignment -------------------------------------------------

def test_documents_are_per_image_and_ordered():
    caps = {f"i{n}.jpg": [f"alpha{n} first", f"beta{n} second", f"gamma{n} third"] for n in range(6)}
    ids, docs = pc.train_documents(caps, list(caps))
    assert ids == sorted(caps) and len(docs) == 6
    assert docs[0] == "alpha0 first beta0 second gamma0 third"
    assert all(doc == " ".join(caps[i]) for i, doc in zip(ids, docs))


def test_documents_reject_image_without_captions(tmp_path):
    data, parts = build(tmp_path)
    caps, _ = d.parse_captions(data / "captions" / "Flickr8k.token.txt")
    del caps[sorted(parts["train"])[0]]
    with pytest.raises(ValueError, match="no captions"):
        pc.train_documents(caps, parts["train"])


def test_empty_document_after_cleaning_raises_with_id():
    caps = {f"i{n}.jpg": ["dog runs across grass"] for n in range(6)}
    bad = sorted(caps)[0]
    caps[bad] = ["!!! 123 ... 7"]  # nothing matches [a-z]{2,}
    with pytest.raises(ValueError, match=f"no tokens after cleaning.*{bad}"):
        pc.train_documents(caps, list(caps))


def test_labels_cover_training_split_exactly_and_sorted(tmp_path):
    out, data, _, meta = run_fit(tmp_path)
    ids, labels = pc.read_labels(out / pc.LABELS)
    split = d.load_split(data / "splits")
    assert ids == sorted(split["train"]) and len(labels) == N_TRAIN
    assert meta["n_train_images"] == N_TRAIN and meta["document_unit"].startswith("one document per")


def test_alignment_to_feature_rows(tmp_path):
    out, _, fdir, _ = run_fit(tmp_path)
    ids, _ = pc.read_labels(out / pc.LABELS)
    index = {i: n for n, i in enumerate((fdir / pc.FEATURE_IDS).read_text().split())}
    assert all(i in index for i in ids)


# --- leakage ----------------------------------------------------------------

def test_fit_uses_training_captions_only(tmp_path, monkeypatch):
    """The explicit leakage test: the documents handed to TF-IDF cover exactly the train split."""
    data, parts = build(tmp_path)
    fdir = features(tmp_path, sum(parts.values(), []))
    seen, fitted = {}, {}
    real_docs, real_vec = pc.train_documents, pc._vectorizer

    def doc_spy(captions, train_ids, *a, **kw):
        ids, docs = real_docs(captions, train_ids, *a, **kw)
        seen["ids"], seen["docs"] = ids, docs
        return ids, docs

    def vec_spy(*a, **kw):
        v = real_vec(*a, **kw)
        orig = v.fit_transform

        def fit_transform(docs, *aa, **kk):
            fitted["docs"] = list(docs)
            return orig(docs, *aa, **kk)

        v.fit_transform = fit_transform
        return v

    monkeypatch.setattr(pc, "train_documents", doc_spy)
    monkeypatch.setattr(pc, "_vectorizer", vec_spy)
    pc.fit(data, tmp_path / "pseudo", fdir, min_df=1, progress_cb=lambda *_: None)

    assert set(seen["ids"]) == set(parts["train"])
    assert not set(seen["ids"]) & (set(parts["val"]) | set(parts["test"]))
    assert fitted["docs"] == seen["docs"] and len(fitted["docs"]) == N_TRAIN


def test_val_test_vocabulary_never_learned(tmp_path):
    out, _, _, meta = run_fit(tmp_path)
    vocab = {line.split("\t")[0] for line in (out / pc.VOCAB).read_text().splitlines()}
    assert vocab and not ({"zzqq", "wwvv", "xxrr", "yytt", "unseen", "marker",
                           "another", "third", "fourth"} & vocab)
    assert meta["uses_val_test_captions"] is False and meta["fitted_on"] == "train"


def test_no_val_test_labels_produced(tmp_path):
    out, _, _, meta = run_fit(tmp_path)
    assert meta["val_test_labels_produced"] is False
    assert not list(out.glob("val_*")) and not list(out.glob("test_*"))


def test_fit_raises_if_heldout_id_reaches_documents(tmp_path, monkeypatch):
    """The guard must raise (not assert) so `python -O` cannot strip it."""
    data, parts = build(tmp_path)
    fdir = features(tmp_path, sum(parts.values(), []))
    real = pc.train_documents
    monkeypatch.setattr(pc, "train_documents", lambda caps, ids, *a, **kw: real(
        caps, list(ids) + [parts["val"][0]], *a, **kw))
    with pytest.raises(ValueError, match="LEAKAGE"):
        pc.fit(data, tmp_path / "pseudo", fdir, min_df=1, progress_cb=lambda *_: None)


def test_validate_flags_stray_heldout_label_file(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    (out / "val_pseudoclasses.csv").write_text("image_id,pseudo_class\n")
    assert any("must not produce val labels" in e for e in check(out, data, fdir))


def test_validate_flags_heldout_id_in_labels(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    ids, labels = pc.read_labels(out / pc.LABELS)
    split = d.load_split(data / "splits")
    rewrite_labels(out, sorted(ids[:-1] + [split["val"][0]]), labels)
    assert any("LEAKAGE" in e for e in check(out, data, fdir))


# --- frozen K ---------------------------------------------------------------

@pytest.mark.parametrize("bad_k", [2, 8, 19, 21, 50])
def test_fit_rejects_any_k_other_than_frozen(tmp_path, bad_k):
    data, parts = build(tmp_path)
    fdir = features(tmp_path, sum(parts.values(), []))
    out = tmp_path / "pseudo"
    with pytest.raises(ValueError, match=f"k={bad_k} refused.*frozen methodology"):
        pc.fit(data, out, fdir, k=bad_k, min_df=1, progress_cb=lambda *_: None)
    assert not out.exists()  # refused before any artifact was written


def test_fit_accepts_the_frozen_k_explicitly(tmp_path):
    data, parts = build(tmp_path)
    fdir = features(tmp_path, sum(parts.values(), []))
    meta = pc.fit(data, tmp_path / "pseudo", fdir, k=pc.K, min_df=1, progress_cb=lambda *_: None)
    assert meta["k"] == 20 and pc.K == 20


def test_cli_rejects_non_frozen_k_before_doing_work(tmp_path, capsys):
    data, parts = build(tmp_path)
    fdir = features(tmp_path, sum(parts.values(), []))
    out = tmp_path / "pseudo"
    rc = pc.main(["fit", "--k", "8", "--data-dir", str(data), "--out-dir", str(out),
                  "--features-dir", str(fdir)])
    assert rc == 1 and "frozen methodology decision" in capsys.readouterr().out
    assert not out.exists()


def test_fit_exposes_no_sweep_override():
    """The sweep must not be reachable as a parameter, so it cannot be used to move K."""
    params = inspect.signature(pc.fit).parameters
    assert "sweep" not in params and "k_sweep" not in params
    assert params["k"].default == pc.K
    assert pc.K_SWEEP == (5, 10, 20, 30, 50)


# --- silhouette sweep -------------------------------------------------------

def test_sweep_is_frozen_and_recorded_per_candidate(tmp_path):
    out, _, _, meta = run_fit(tmp_path)
    sweep = meta["silhouette_sweep"]
    assert sweep["candidates_expected"] == [5, 10, 20, 30, 50]
    assert set(sweep["candidates"]) == {"5", "10", "20", "30", "50"}
    for entry in sweep["candidates"].values():
        assert "score" in entry
        assert entry["score"] is not None or entry["unavailable"]


def test_sweep_records_metric_seed_and_no_sampling(tmp_path):
    out, _, _, meta = run_fit(tmp_path)
    sweep = meta["silhouette_sweep"]
    assert sweep["metric"] == "cosine" and sweep["sample_size"] is None
    assert sweep["random_state"] == meta["seed"] == pc.SEED
    assert sweep["fitted_on"] == "train"
    assert meta["silhouette_at_k"]["metric"] == "cosine"
    assert meta["silhouette_at_k"]["sample_size"] is None


def test_sweep_does_not_select_k(tmp_path):
    out, _, _, meta = run_fit(tmp_path)
    scored = {int(c): e["score"] for c, e in meta["silhouette_sweep"]["candidates"].items()
              if e["score"] is not None}
    assert scored  # at least one candidate scored
    assert meta["k"] == 20 and meta["k_frozen_in_advance"] and meta["k_selected_by_sweep"] is False


def test_sweep_reuses_production_fit_for_frozen_k(tmp_path):
    out, _, _, meta = run_fit(tmp_path)
    entry = meta["silhouette_sweep"]["candidates"]["20"]
    assert entry["note"] == "reused the production fit; not refitted"
    assert entry["score"] == meta["silhouette_at_k"]["score"]


def test_candidate_outside_valid_range_recorded_unavailable(tmp_path):
    """k=50 exceeds a 40-document split: an explicit reason, never an invented score."""
    data, parts = build(tmp_path, n_train=TINY)
    fdir = features(tmp_path, sum(parts.values(), []))
    meta = pc.fit(data, tmp_path / "pseudo", fdir, min_df=1, progress_cb=lambda *_: None)
    entry = meta["silhouette_sweep"]["candidates"]["50"]
    assert entry["score"] is None and f"outside [2, {TINY - 1}]" in entry["unavailable"]
    assert entry["n_distinct_labels"] is None


def test_silhouette_helper_reports_reason_instead_of_number():
    from sklearn.preprocessing import normalize
    X = normalize(np.eye(6, dtype=float))
    score, why = pc._silhouette(X, np.zeros(6, int))           # one label
    assert score is None and "undefined" in why
    score, why = pc._silhouette(X, np.arange(6))               # every point its own cluster
    assert score is None and "undefined" in why
    score, why = pc._silhouette(X, np.array([0, 0, 0, 1, 1, 1]))
    assert isinstance(score, float) and why is None


def test_singleton_cluster_still_scores():
    """sklearn defines a singleton's silhouette contribution as 0, so this is not degenerate."""
    from sklearn.preprocessing import normalize
    X = normalize(np.eye(6, dtype=float))
    score, why = pc._silhouette(X, np.array([0, 0, 0, 0, 0, 1]))
    assert why is None and isinstance(score, float)


# --- determinism and configuration ------------------------------------------

def test_deterministic_labels_same_seed(tmp_path):
    out_a, data, fdir, meta_a = run_fit(tmp_path)
    meta_b = pc.fit(data, tmp_path / "pseudo_b", fdir, min_df=1, progress_cb=lambda *_: None)
    ids_a, labels_a = pc.read_labels(out_a / pc.LABELS)
    ids_b, labels_b = pc.read_labels(tmp_path / "pseudo_b" / pc.LABELS)
    assert ids_a == ids_b and np.array_equal(labels_a, labels_b)
    assert meta_a["kmeans"]["inertia"] == meta_b["kmeans"]["inertia"]
    assert meta_a["artifact_sha256"][pc.LABELS] == meta_b["artifact_sha256"][pc.LABELS]
    assert meta_a["silhouette_at_k"]["score"] == meta_b["silhouette_at_k"]["score"]


def test_frozen_config_recorded(tmp_path):
    out, _, _, meta = run_fit(tmp_path)
    assert meta["k"] == 20 and meta["seed"] == pc.SEED
    assert meta["kmeans"]["n_init"] == 10 and meta["kmeans"]["algorithm"] == "lloyd"
    assert meta["kmeans"]["max_iter"] == 300
    t = meta["tfidf"]
    assert t["lowercase"] and t["token_pattern"] == r"[a-z]{2,}" and t["ngram_range"] == [1, 1]
    assert t["norm"] == "l2" and t["sublinear_tf"] is False and t["stop_words"] is None
    assert meta["min_cluster_size_required"] == 10


def test_labels_in_range_and_clusters_non_empty(tmp_path):
    out, _, _, meta = run_fit(tmp_path)
    _, labels = pc.read_labels(out / pc.LABELS)
    assert labels.min() >= 0 and labels.max() < meta["k"]
    assert sum(meta["cluster_sizes"].values()) == N_TRAIN and all(meta["cluster_sizes"].values())


def test_tokenizer_handles_hash_and_comma_captions():
    """The real data has 12 captions containing '#' and 2,447 containing commas."""
    caps = {f"i{n}.jpg": [f"a dog, running near #{n} the fence"] for n in range(6)}
    ids, docs = pc.train_documents(caps, list(caps))
    vec = pc._vectorizer(min_df=1)
    vec.fit(docs)
    vocab = set(vec.get_feature_names_out())
    assert {"dog", "running", "near", "the", "fence"} <= vocab
    assert not any(t in vocab for t in ("#", ",", "0", "a"))


def test_fit_refuses_to_overwrite_without_force(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        pc.fit(data, out, fdir, min_df=1, progress_cb=lambda *_: None)
    pc.fit(data, out, fdir, min_df=1, force=True, progress_cb=lambda *_: None)


# --- validate ---------------------------------------------------------------

def test_validate_clean_artifacts_pass(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    assert check(out, data, fdir) == []


def test_clean_fit_clears_the_ten_image_floor(tmp_path):
    out, _, _, meta = run_fit(tmp_path)
    assert min(meta["cluster_sizes"].values()) == PER_CLUSTER >= pc.MIN_CLUSTER_SIZE


def test_validate_rejects_undersized_cluster_through_normal_path(tmp_path):
    """Move all but one member of cluster 0 elsewhere: the frozen floor must reject it."""
    out, data, fdir, _ = run_fit(tmp_path)
    ids, labels = pc.read_labels(out / pc.LABELS)
    moved = [n for n, l in enumerate(labels) if l == 0][1:]  # leave exactly one member behind
    for n in moved:
        labels[n] = 1
    rewrite_labels(out, ids, labels)
    patch_meta(out, cluster_sizes={str(c): int((labels == c).sum()) for c in range(20)})
    errs = pc._validate_pseudoclasses(out, data, fdir, sizes=SMALL)   # no floor argument exists
    assert any("below the 10-image minimum: {0: 1}" in e for e in errs)
    assert "within-class variance" in " ".join(errs)


def test_production_validate_accepts_no_weakening_arguments(tmp_path):
    """Neither the cluster floor nor the split sizes may be supplied by a caller."""
    params = inspect.signature(pc.validate_pseudoclasses).parameters
    assert set(params) == {"out_dir", "data_dir", "features_dir"}
    assert pc.MIN_CLUSTER_SIZE == 10
    out, data, fdir, _ = run_fit(tmp_path)
    for bad in ({"sizes": SMALL}, {"min_cluster_size": 1}, {"sizes": SMALL, "min_cluster_size": 1}):
        with pytest.raises(TypeError):
            pc.validate_pseudoclasses(out, data, fdir, **bad)
    # the floor is a constant even on the private helper, so tests cannot weaken it either
    assert "min_cluster_size" not in inspect.signature(pc._validate_pseudoclasses).parameters


def test_production_validate_enforces_the_official_split_sizes(tmp_path):
    """Called on a 240-image synthetic fixture, production validate must demand the real 6,000."""
    out, data, fdir, _ = run_fit(tmp_path)
    errs = pc.validate_pseudoclasses(out, data, fdir)
    assert any(f"rows, expected {d.SIZES['train']}" in e for e in errs)
    assert d.SIZES == {"train": 6000, "val": 1000, "test": 1000}
    # the same artifacts are clean when checked against the fixture's own sizes
    assert pc._validate_pseudoclasses(out, data, fdir, sizes=SMALL) == []


def test_validate_detects_weakened_floor_in_metadata(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    patch_meta(out, min_cluster_size_required=3)
    assert any("requires at least 10 training images per cluster" in e
               for e in check(out, data, fdir))


@pytest.mark.parametrize("name", [pc.LABELS, pc.VOCAB, pc.META])
def test_validate_detects_missing_artifact(tmp_path, name):
    out, data, fdir, _ = run_fit(tmp_path)
    (out / name).unlink()
    assert check(out, data, fdir) == [f"missing artifact: {name}"]


def test_validate_detects_tampered_labels(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    ids, labels = pc.read_labels(out / pc.LABELS)
    labels[0] = (labels[0] + 1) % 20
    rewrite_labels(out, ids, labels, refresh_checksum=False)
    assert any("sha256 mismatch" in e for e in check(out, data, fdir))


def test_validate_detects_row_count_mismatch(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    errs = check(out, data, fdir, sizes={"train": N_TRAIN + 1, "val": 2, "test": 2})
    assert any(f"rows, expected {N_TRAIN + 1}" in e for e in errs)


def test_validate_detects_unlabelled_and_extra_ids(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    ids, labels = pc.read_labels(out / pc.LABELS)
    rewrite_labels(out, sorted(ids[:-1] + ["zz_not_in_split.jpg"]), labels)
    errs = check(out, data, fdir)
    assert any("not in the training split" in e for e in errs)
    assert any("unlabelled" in e for e in errs)


def test_validate_detects_unsorted_labels(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    ids, labels = pc.read_labels(out / pc.LABELS)
    rewrite_labels(out, ids[::-1], labels)
    assert any("not sorted" in e for e in check(out, data, fdir))


def test_validate_detects_label_out_of_range(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    ids, labels = pc.read_labels(out / pc.LABELS)
    labels[0] = 99
    rewrite_labels(out, ids, labels)
    assert any("expected [0, 19]" in e for e in check(out, data, fdir))


def test_validate_detects_cluster_size_metadata_drift(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    drift = {str(c): PER_CLUSTER for c in range(19)}
    drift["19"] = PER_CLUSTER + 1
    patch_meta(out, cluster_sizes=drift)
    assert any("do not match the label file" in e for e in check(out, data, fdir))


def test_validate_detects_feature_ids_change(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    (fdir / pc.FEATURE_IDS).write_text("something_else.jpg\n")
    assert any("Stage 4 row alignment is not guaranteed" in e for e in check(out, data, fdir))


def test_validate_detects_missing_feature_ids(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    (fdir / pc.FEATURE_IDS).unlink()
    assert any("is missing" in e for e in check(out, data, fdir))


@pytest.mark.parametrize("patch,needle", [
    ({"status": "running"}, "not 'complete'"),
    ({"uses_val_test_captions": True}, "uses_val_test_captions=false"),
    ({"fitted_on": "all"}, "expected 'train'"),
    ({"k_selected_by_sweep": True}, "k_selected_by_sweep=false"),
    ({"k": 1}, "expected an integer >= 2"),
    ({"k": 19}, "the methodology freezes k=20"),
    ({"k": 50}, "the methodology freezes k=20"),
])
def test_validate_detects_metadata_claims(tmp_path, patch, needle):
    out, data, fdir, _ = run_fit(tmp_path)
    patch_meta(out, **patch)
    assert any(needle in e for e in check(out, data, fdir))


@pytest.mark.parametrize("patch,needle", [
    ({"silhouette_sweep.metric": "euclidean"}, "expected 'cosine'"),
    ({"silhouette_sweep.sample_size": 500}, "no sampling"),
    ({"silhouette_sweep.fitted_on": "all"}, "fitted_on"),
    ({"silhouette_sweep.random_state": 7}, "does not match the recorded seed"),
    ({"silhouette_sweep.candidates_expected": [5, 10, 20]}, "candidates_expected"),
])
def test_validate_detects_sweep_tampering(tmp_path, patch, needle):
    out, data, fdir, _ = run_fit(tmp_path)
    patch_meta(out, **patch)
    assert any(needle in e for e in check(out, data, fdir))


def test_validate_detects_dropped_sweep_candidate(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    meta = json.loads((out / pc.META).read_text())
    del meta["silhouette_sweep"]["candidates"]["50"]
    (out / pc.META).write_text(json.dumps(meta))
    assert any("expected ['5', '10', '20', '30', '50']" in e for e in check(out, data, fdir))


def test_validate_detects_scoreless_candidate_without_reason(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    meta = json.loads((out / pc.META).read_text())
    meta["silhouette_sweep"]["candidates"]["10"] = {"score": None, "n_distinct_labels": 10}
    (out / pc.META).write_text(json.dumps(meta))
    assert any("no score and no reason given" in e for e in check(out, data, fdir))


def test_validate_detects_missing_sweep_block(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    meta = json.loads((out / pc.META).read_text())
    del meta["silhouette_sweep"]
    (out / pc.META).write_text(json.dumps(meta))
    assert any("no silhouette_sweep block" in e for e in check(out, data, fdir))


def test_validate_detects_malformed_metadata(tmp_path):
    out, data, fdir, _ = run_fit(tmp_path)
    (out / pc.META).write_text("{not json")
    assert any("not valid JSON" in e for e in check(out, data, fdir))


@pytest.mark.parametrize("text", ["wrong,header\ni0.jpg,0\n",
                                  "image_id,pseudo_class\ni0.jpg\n",
                                  "image_id,pseudo_class\ni0.jpg,abc\n"])
def test_read_labels_rejects_malformed_csv(tmp_path, text):
    p = tmp_path / "bad.csv"
    p.write_text(text)
    with pytest.raises(ValueError):
        pc.read_labels(p)
