import json

import pytest
from PIL import Image

from retinex import dataset as d

SMALL = {"train": 6, "val": 2, "test": 2}


def build(root, n=10, caps=5, bad_line=None):
    (root / "images").mkdir(parents=True)
    (root / "captions").mkdir()
    lines = []
    for i in range(n):
        Image.new("RGB", (4, 4)).save(root / "images" / f"img{i}.jpg")
        lines += [f"img{i}.jpg#{j}\tcaption {i} {j}" for j in range(caps)]
    if bad_line:
        lines.append(bad_line)
    (root / "captions" / d.TOKEN_FILE).write_text("\n".join(lines) + "\n")
    return root


def test_parse_groups_captions_by_image(tmp_path):
    caps, problems = d.parse_captions(build(tmp_path / "x", n=3) / "captions" / d.TOKEN_FILE)
    assert problems == [] and list(caps) == ["img0.jpg", "img1.jpg", "img2.jpg"]
    assert all(len(v) == 5 for v in caps.values()) and caps["img1.jpg"][2] == "caption 1 2"


@pytest.mark.parametrize("bad", ["no tab here", "img0.jpg\tno hash", "img0.jpg#x\tbad index",
                                 "img0.jpg#9\t   ", "#1\tno image", "img0.jpg#0\tduplicate"])
def test_malformed_and_duplicate_reported_not_dropped_silently(tmp_path, bad):
    root = build(tmp_path / "x", n=1, bad_line=bad)
    caps, problems = d.parse_captions(root / "captions" / d.TOKEN_FILE)
    assert len(problems) == 1 and len(caps["img0.jpg"]) == 5


def test_validate_clean(tmp_path):
    r = d.validate(build(tmp_path / "x"))
    assert r.ok and len(r.valid_ids) == 10 and r.n_captions == 50 and r.captions_per_image == {5: 10}


def test_validate_detects_missing_uncaptioned_corrupt(tmp_path):
    root = build(tmp_path / "x")
    (root / "images" / "img0.jpg").unlink()                        # captioned, missing
    Image.new("RGB", (4, 4)).save(root / "images" / "extra.jpg")   # no captions
    (root / "images" / "img1.jpg").write_bytes(b"not a jpeg")      # corrupt
    r = d.validate(root)
    assert r.missing_images == ["img0.jpg"] and r.uncaptioned_images == ["extra.jpg"]
    assert len(r.corrupt_images) == 1 and "img1.jpg" in r.corrupt_images[0]
    assert not r.ok and "img0.jpg" not in r.valid_ids and "img1.jpg" not in r.valid_ids


def test_split_sizes_disjoint_complete():
    ids = [f"i{n}" for n in range(10)]
    s = d.make_split(ids, sizes=SMALL)
    assert {k: len(v) for k, v in s.items()} == SMALL
    assert set().union(*map(set, s.values())) == set(ids)
    assert not set(s["train"]) & set(s["val"]) and not set(s["train"]) & set(s["test"]) \
        and not set(s["val"]) & set(s["test"])


def test_split_deterministic_and_input_order_independent():
    ids = [f"i{n}" for n in range(50)]
    sz = {"train": 30, "val": 10, "test": 10}
    assert d.make_split(ids, sizes=sz) == d.make_split(ids[::-1], sizes=sz)
    assert d.make_split(ids, seed=1, sizes=sz) != d.make_split(ids, seed=2, sizes=sz)


def test_split_extra_images_unused_and_too_few_rejected():
    s = d.make_split([f"i{n}" for n in range(12)], sizes=SMALL)
    assert sum(map(len, s.values())) == 10
    with pytest.raises(ValueError):
        d.make_split(["a", "b"], sizes=SMALL)


def frozen(tmp_path):
    root = build(tmp_path / "x")
    r = d.validate(root)
    d.write_split(root / "splits", d.make_split(r.valid_ids, sizes=SMALL), d.SEED, len(r.valid_ids))
    return root


def test_verify_ok_then_refuses_overwrite(tmp_path):
    root = frozen(tmp_path)
    assert d.verify_split(root, SMALL) == []
    with pytest.raises(FileExistsError):
        d.write_split(root / "splits", {"train": [], "val": [], "test": []}, 0, 0)
    assert d.load_split(root / "splits")["train"] == d.make_split(d.validate(root).valid_ids, sizes=SMALL)["train"]


def test_verify_detects_leakage_and_tampering(tmp_path):
    root = frozen(tmp_path)
    sd = root / "splits"
    train = (sd / "train.txt").read_text().split()
    (sd / "val.txt").write_text("\n".join([train[0]] + (sd / "val.txt").read_text().split()[1:]) + "\n")
    errs = d.verify_split(root, SMALL)
    assert any("LEAKAGE" in e for e in errs) and any("sha256" in e for e in errs)


def test_verify_detects_missing_image_and_artifacts(tmp_path):
    root = frozen(tmp_path)
    name = (root / "splits" / "test.txt").read_text().split()[0]
    (root / "images" / name).unlink()
    assert any("image file missing" in e for e in d.verify_split(root, SMALL))
    (root / "splits" / d.META).unlink()
    assert any("missing split artifact" in e for e in d.verify_split(root, SMALL))


def test_cli_validate_and_too_small_dataset(tmp_path):
    root = str(build(tmp_path / "x"))
    assert d.main(["validate", "--data-dir", root]) == 0
    assert d.main(["split", "--data-dir", root]) == 1   # 10 images < 8000 required
    assert d.main(["verify", "--data-dir", root]) == 1  # nothing generated
