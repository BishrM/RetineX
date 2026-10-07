"""Flickr8K dataset layer: caption parsing, validation, frozen image-level split.

CLI: python -m retinex.dataset {validate,split,verify} [--data-dir DIR]
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

SEED = 42
SIZES = {"train": 6000, "val": 1000, "test": 1000}
TOKEN_FILE = "Flickr8k.token.txt"
META = "split_meta.json"


def default_data_dir() -> Path:
    env = os.environ.get("RETINEX_DATA_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data" / "flickr8k"


def parse_captions(path):
    """Parse Flickr8k.token.txt ("<image>#<n>\\t<caption>").

    Returns ({image_id: [caption, ...]}, [problem, ...]). Malformed lines and
    duplicate (image, n) records are reported and skipped, never dropped silently.
    """
    caps, seen, problems = {}, set(), []
    with open(path, encoding="utf8") as f:
        for no, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            key, tab, cap = line.partition("\t")
            img, hash_, idx = key.partition("#")
            if not (tab and hash_ and img.strip() and idx.isdigit() and cap.strip()):
                problems.append(f"line {no}: malformed record: {line[:80]!r}")
            elif (img, idx) in seen:
                problems.append(f"line {no}: duplicate record {img}#{idx}")
            else:
                seen.add((img, idx))
                caps.setdefault(img, []).append(cap.strip())
    return caps, problems


@dataclass
class Report:
    n_image_files: int = 0
    n_captioned: int = 0
    n_captions: int = 0
    captions_per_image: dict = field(default_factory=dict)
    malformed: list = field(default_factory=list)
    missing_images: list = field(default_factory=list)      # captioned, no file
    uncaptioned_images: list = field(default_factory=list)  # file, no captions
    corrupt_images: list = field(default_factory=list)
    valid_ids: list = field(default_factory=list)           # file + captions + readable

    @property
    def ok(self):
        return not (self.malformed or self.missing_images or self.uncaptioned_images or self.corrupt_images)


def validate(data_dir, check_pixels=True) -> Report:
    data_dir = Path(data_dir)
    caps, malformed = parse_captions(data_dir / "captions" / TOKEN_FILE)
    files = {p.name for p in (data_dir / "images").iterdir() if p.suffix.lower() == ".jpg"}
    r = Report(n_image_files=len(files), n_captioned=len(caps), malformed=malformed,
               n_captions=sum(map(len, caps.values())),
               captions_per_image=dict(Counter(map(len, caps.values()))))
    r.missing_images = sorted(set(caps) - files)
    r.uncaptioned_images = sorted(files - set(caps))
    for name in sorted(set(caps) & files):
        if check_pixels:
            try:
                with Image.open(data_dir / "images" / name) as im:
                    im.load()
            except Exception as e:  # PIL raises many types for bad files
                r.corrupt_images.append(f"{name}: {e}")
                continue
        r.valid_ids.append(name)
    return r


def make_split(ids, seed=SEED, sizes=SIZES):
    """Deterministic image-level split. Extra images beyond sum(sizes) are left unused."""
    ids = sorted(set(ids))
    need = sum(sizes.values())
    if len(ids) < need:
        raise ValueError(f"need {need} images, have {len(ids)}")
    random.Random(seed).shuffle(ids)
    out, i = {}, 0
    for name, n in sizes.items():
        out[name] = sorted(ids[i:i + n])
        i += n
    return out


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_split(split_dir, split, seed, n_population, force=False):
    split_dir = Path(split_dir)
    names = [f"{k}.txt" for k in split] + [META]
    existing = [n for n in names if (split_dir / n).exists()]
    if existing and not force:
        raise FileExistsError(f"split is FROZEN, refusing to overwrite {existing} (use --force)")
    split_dir.mkdir(parents=True, exist_ok=True)
    for k, ids in split.items():
        (split_dir / f"{k}.txt").write_text("\n".join(ids) + "\n")
    meta = {"seed": seed, "population": n_population,
            "unused": n_population - sum(map(len, split.values())),
            "sizes": {k: len(v) for k, v in split.items()},
            "sha256": {k: _sha(split_dir / f"{k}.txt") for k in split}}
    (split_dir / META).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_split(split_dir):
    """Read the frozen split. Later stages must use this, never make_split."""
    split_dir = Path(split_dir)
    return {k: (split_dir / f"{k}.txt").read_text().split() for k in SIZES}


def verify_split(data_dir, sizes=SIZES):
    """Return a list of problems (empty = split is valid and unmodified)."""
    data_dir = Path(data_dir)
    sd, errs = data_dir / "splits", []
    try:
        split, meta = load_split(sd), json.loads((sd / META).read_text())
    except FileNotFoundError as e:
        return [f"missing split artifact: {e.filename}"]
    for k, ids in split.items():
        if len(ids) != sizes[k] or len(ids) != meta["sizes"][k]:
            errs.append(f"{k}: {len(ids)} images, expected {sizes[k]}")
        if len(set(ids)) != len(ids):
            errs.append(f"{k}: duplicate image ids")
        if _sha(sd / f"{k}.txt") != meta["sha256"][k]:
            errs.append(f"{k}: file changed since generation (sha256 mismatch)")
    sets = {k: set(v) for k, v in split.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if sets[a] & sets[b]:
            errs.append(f"LEAKAGE: {len(sets[a] & sets[b])} images shared by {a} and {b}")
    caps, _ = parse_captions(data_dir / "captions" / TOKEN_FILE) if (data_dir / "captions" / TOKEN_FILE).exists() else (None, [])
    if caps is None:
        errs.append("caption file not found; cannot check coverage")
    else:
        allids = set().union(*sets.values())
        errs += [f"{i}: no captions" for i in sorted(allids - set(caps))[:10]]
        errs += [f"{i}: image file missing" for i in sorted(allids) if not (data_dir / "images" / i).exists()][:10]
    return errs


def main(argv=None):
    p = argparse.ArgumentParser(prog="retinex.dataset")
    p.add_argument("command", choices=["validate", "split", "verify"])
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--force", action="store_true", help="split: overwrite frozen split")
    a = p.parse_args(argv)

    if a.command == "verify":
        errs = verify_split(a.data_dir)
        print("\n".join(errs) if errs else "split OK: sizes, disjointness, coverage, checksums verified")
        return 1 if errs else 0

    if not ((a.data_dir / "captions" / TOKEN_FILE).is_file() and (a.data_dir / "images").is_dir()):
        print(f"Flickr8K not found in {a.data_dir}: expected captions/{TOKEN_FILE} and images/ (see README, Dataset Setup)")
        return 1
    r = validate(a.data_dir)
    print(f"image files: {r.n_image_files}  captioned images: {r.n_captioned}  captions: {r.n_captions}")
    print(f"captions per image (count: images): {dict(sorted(r.captions_per_image.items()))}")
    for label, items in (("malformed/duplicate records", r.malformed), ("captioned but missing image", r.missing_images),
                         ("image without captions", r.uncaptioned_images), ("corrupt images", r.corrupt_images)):
        if items:
            print(f"{label}: {len(items)}")
            print("  " + "\n  ".join(items[:10]) + ("\n  ..." if len(items) > 10 else ""))
    print(f"valid images (file + captions + readable): {len(r.valid_ids)}")
    if a.command == "validate":
        print("dataset OK" if r.ok else "dataset has problems (see above)")
        return 0 if r.ok else 1

    try:
        meta = write_split(a.data_dir / "splits", make_split(r.valid_ids, a.seed), a.seed, len(r.valid_ids), a.force)
    except (ValueError, FileExistsError) as e:
        print(f"split not written: {e}")
        return 1
    print(f"wrote split: {meta['sizes']} seed={a.seed} unused={meta['unused']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
