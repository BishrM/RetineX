"""Stage 2 artifact validation. Synthetic and tiny: never loads the real 131 MB matrix."""
import json

import numpy as np
import pytest

from retinex import dataset as d
from retinex import features as f

SMALL = {"train": 6, "val": 2, "test": 2}
DIM = 4


def fixture(tmp_path, dim=DIM, dtype=np.float32, ids=None, rows=None, meta_patch=None):
    """A frozen split plus a matching feature artifact set, both deliberately tiny."""
    data = tmp_path / "data"
    (data / "splits").mkdir(parents=True)
    names = [f"i{n}.jpg" for n in range(10)]
    for k, part in (("train", names[:6]), ("val", names[6:8]), ("test", names[8:])):
        (data / "splits" / f"{k}.txt").write_text("\n".join(part) + "\n")

    out = tmp_path / "features"
    out.mkdir()
    ids = sorted(names) if ids is None else ids
    (out / f.IDS).write_text("\n".join(ids) + "\n")
    arr = np.zeros((len(ids) if rows is None else rows, dim), dtype=dtype)
    np.save(out / f.MATRIX, arr)
    meta = {"status": "complete", "uses_captions": False, "ids_sha256": f._ids_digest(ids),
            "artifact_sha256": {f.MATRIX: f._sha256(out / f.MATRIX), f.IDS: f._sha256(out / f.IDS)}}
    meta.update(meta_patch or {})
    (out / f.META).write_text(json.dumps(meta))
    return out, data


def check(tmp_path, **kw):
    out, data = fixture(tmp_path, **kw)
    return f.validate_features(out, data, dim=kw.get("dim", DIM), sizes=SMALL)


def test_clean_artifacts_pass(tmp_path):
    assert check(tmp_path) == []


def test_frozen_ids_are_sorted_union_of_split(tmp_path):
    _, data = fixture(tmp_path)
    ids, split = f.frozen_ids(data / "splits")
    assert ids == sorted(ids) and len(ids) == 10
    assert {k: len(v) for k, v in split.items()} == SMALL


def test_wrong_dtype_detected(tmp_path):
    assert any("dtype" in e for e in check(tmp_path, dtype=np.float64))


def test_wrong_dimension_detected(tmp_path):
    out, data = fixture(tmp_path, dim=DIM)
    assert any("shape" in e for e in f.validate_features(out, data, dim=DIM + 1, sizes=SMALL))


def test_row_count_mismatch_detected(tmp_path):
    assert any("rows" in e or "shape" in e for e in check(tmp_path, rows=9))


def test_duplicate_ids_detected(tmp_path):
    ids = sorted(f"i{n}.jpg" for n in range(10))
    assert any("duplicate" in e for e in check(tmp_path, ids=ids[:-1] + [ids[0]]))


def test_unexpected_id_detected(tmp_path):
    ids = sorted(f"i{n}.jpg" for n in range(9)) + ["stranger.jpg"]
    errs = check(tmp_path, ids=ids)
    assert any("not in the frozen split" in e for e in errs) and any("missing" in e for e in errs)


def test_id_count_mismatch_detected(tmp_path):
    assert any("expected 10" in e for e in check(tmp_path, ids=[f"i{n}.jpg" for n in range(9)]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nan_and_inf_detected(tmp_path, bad):
    out, data = fixture(tmp_path)
    arr = np.load(out / f.MATRIX)
    arr[3, 2] = bad
    np.save(out / f.MATRIX, arr)
    errs = f.validate_features(out, data, dim=DIM, sizes=SMALL)
    assert any(("NaN" if np.isnan(bad) else "infinite") in e for e in errs)


def test_tampered_ids_file_detected(tmp_path):
    out, data = fixture(tmp_path)
    ids = (out / f.IDS).read_text().split()
    (out / f.IDS).write_text("\n".join(ids[::-1]) + "\n")  # same ids, different row mapping
    errs = f.validate_features(out, data, dim=DIM, sizes=SMALL)
    assert any("row order is no longer guaranteed" in e for e in errs)


def test_tampered_matrix_detected(tmp_path):
    out, data = fixture(tmp_path)
    arr = np.load(out / f.MATRIX)
    arr[0, 0] = 1.0
    np.save(out / f.MATRIX, arr)
    assert any("sha256 mismatch" in e for e in f.validate_features(out, data, dim=DIM, sizes=SMALL))


def test_incomplete_run_detected(tmp_path):
    assert any("not 'complete'" in e for e in check(tmp_path, meta_patch={"status": "running"}))


def test_caption_independence_recorded(tmp_path):
    assert any("uses_captions" in e for e in check(tmp_path, meta_patch={"uses_captions": True}))


def test_malformed_meta_detected(tmp_path):
    out, data = fixture(tmp_path)
    (out / f.META).write_text("{not json")
    assert any("not valid JSON" in e for e in f.validate_features(out, data, dim=DIM, sizes=SMALL))


def test_missing_artifacts_detected(tmp_path):
    out, data = fixture(tmp_path)
    (out / f.MATRIX).unlink()
    assert f.validate_features(out, data, dim=DIM, sizes=SMALL) == [f"missing artifact: {f.MATRIX}"]


def test_wrong_split_size_detected(tmp_path):
    out, data = fixture(tmp_path)
    errs = f.validate_features(out, data, dim=DIM, sizes={"train": 7, "val": 2, "test": 2})
    assert any("expected 7" in e for e in errs)


def test_extraction_needs_no_captions(tmp_path, monkeypatch):
    """Structural guarantee: no caption or split-label information reaches a feature vector."""
    out, data = fixture(tmp_path)
    assert not (data / "captions").exists()
    calls = []

    class FakeModel:
        def predict_on_batch(self, batch):
            calls.append(batch.shape)
            return np.ones((batch.shape[0], DIM), dtype=np.float32)

    monkeypatch.setattr(f, "_build_model", lambda: FakeModel())
    monkeypatch.setattr(f, "_load_batch", lambda d, names: np.zeros((len(names), 2, 2, 3), np.float32))
    monkeypatch.setattr(f, "FEATURE_DIM", DIM)
    meta = f.extract(data, out, batch_size=4, resume=False, progress_cb=lambda *_: None)
    assert meta["uses_captions"] is False and meta["num_images"] == 10
    assert [s[0] for s in calls] == [4, 4, 2]
    assert f.validate_features(out, data, dim=DIM, sizes=SMALL) == []


def test_resume_skips_completed_rows(tmp_path, monkeypatch):
    out, data = fixture(tmp_path)
    ids, _ = f.frozen_ids(data / "splits")
    seen = []

    class FakeModel:
        def predict_on_batch(self, batch):
            seen.append(batch.shape[0])
            return np.ones((batch.shape[0], DIM), dtype=np.float32)

    monkeypatch.setattr(f, "_build_model", lambda: FakeModel())
    monkeypatch.setattr(f, "_load_batch", lambda d, names: np.zeros((len(names), 2, 2, 3), np.float32))
    monkeypatch.setattr(f, "FEATURE_DIM", DIM)
    np.lib.format.open_memmap(out / f.MATRIX, mode="w+", dtype=np.float32, shape=(10, DIM))
    (out / f.PROGRESS).write_text(json.dumps(
        {"done": 8, "shape": [10, DIM], "ids_sha256": f._ids_digest(ids)}))
    meta = f.extract(data, out, batch_size=4, resume=True, progress_cb=lambda *_: None)
    assert seen == [2] and meta["resumed_from_row"] == 8
    assert not (out / f.PROGRESS).exists()


def test_resume_ignored_when_split_changed(tmp_path, monkeypatch):
    out, data = fixture(tmp_path)
    monkeypatch.setattr(f, "_build_model", lambda: type("M", (), {
        "predict_on_batch": lambda self, b: np.ones((b.shape[0], DIM), np.float32)})())
    monkeypatch.setattr(f, "_load_batch", lambda d, names: np.zeros((len(names), 2, 2, 3), np.float32))
    monkeypatch.setattr(f, "FEATURE_DIM", DIM)
    np.lib.format.open_memmap(out / f.MATRIX, mode="w+", dtype=np.float32, shape=(10, DIM))
    (out / f.PROGRESS).write_text(json.dumps({"done": 8, "shape": [10, DIM], "ids_sha256": "stale"}))
    meta = f.extract(data, out, batch_size=4, resume=True, progress_cb=lambda *_: None)
    assert meta["resumed_from_row"] == 0
