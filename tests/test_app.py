"""Flask UI tests: routes, upload validation, preprocessing alignment, honest states, a11y, tokens.

These never load VGG16 unless a test says so explicitly; the model is monkeypatched by default so
the suite stays fast and does not depend on the ImageNet weights being cached.
"""
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from app import create_app, inference, pipeline, uploads
from retinex import features as fe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    return app.test_client()


def png_bytes(w=64, h=48, color=(20, 90, 50), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, fmt)
    return buf.getvalue()


# --- routes and markup ------------------------------------------------------

@pytest.mark.parametrize("path", ["/", "/workspace"])
def test_pages_render(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert r.headers["Content-Type"].startswith("text/html")


def test_unknown_route_is_a_real_404(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404
    assert b"does not exist yet" in r.data


@pytest.mark.parametrize("path", ["/", "/workspace"])
def test_accessibility_landmarks_and_skip_link(client, path):
    html = client.get(path).get_data(as_text=True)
    assert '<a class="skip-link" href="#main">Skip to content</a>' in html
    assert '<main id="main">' in html
    assert "<header" in html and "<footer" in html
    assert '<nav aria-label="Primary">' in html
    assert html.count("<h1") == 1                      # exactly one h1 per page
    assert '<html lang="en">' in html
    assert 'name="viewport"' in html


def test_workspace_has_labelled_keyboard_operable_upload(client):
    html = client.get("/workspace").get_data(as_text=True)
    assert '<label class="label" for="image-input">Image file</label>' in html
    # the dropzone is a <label for> the real input, so click and keyboard both work with no JS
    assert 'class="dropzone" id="dropzone" for="image-input"' in html
    assert 'type="file" id="image-input" name="image"' in html
    assert 'role="status" aria-live="polite"' in html   # state changes are announced


def test_workspace_narration_controls_are_really_disabled(client):
    html = client.get("/workspace").get_data(as_text=True)
    block = html[html.index('aria-label="Narration playback"'):]
    block = block[:block.index("</div>")]
    assert block.count("<button") == 3
    assert block.count("disabled") == 3                 # not merely styled as disabled
    assert "No audio is produced" in html
    assert "Transcript" in html                         # transcript is always present


@pytest.mark.parametrize("name", ["favicon.png", "favicon-dark.png"])
def test_favicon_variants_are_square_and_transparent(client, name):
    icon = ROOT / "app" / "static" / "brand" / name
    assert icon.is_file()
    im = Image.open(icon)
    assert im.size == (512, 512), f"{name} is {im.size}, must be 512x512"
    assert im.mode == "RGBA" and im.getchannel("A").getextrema()[0] == 0, "needs transparency"
    assert client.get(f"/static/brand/{name}").status_code == 200


@pytest.mark.parametrize("path", ["/", "/workspace"])
def test_both_favicon_variants_are_linked_with_the_right_media(client, path):
    html = client.get(path).get_data(as_text=True)
    # the fallback has no media query, so browsers ignoring `media` still get a visible icon
    assert 'href="/static/brand/favicon.png"' in html
    assert 'media="(prefers-color-scheme: dark)"' in html
    assert 'href="/static/brand/favicon-dark.png"' in html
    dark_link = html[html.index('media="(prefers-color-scheme: dark)"'):]
    assert "favicon-dark.png" in dark_link[:200], "the dark media query must point at the cream mark"
    # the Apple touch icon is composited on light backgrounds, so it must be the green one
    apple = html[html.index('rel="apple-touch-icon"'):]
    assert "favicon.png" in apple[:200] and "favicon-dark" not in apple[:200]


def test_the_page_itself_has_no_dark_theme(client):
    """The favicon media query is browser chrome only; the site stays light-only per design.md."""
    for path in (ROOT / "app" / "static" / "css").glob("*.css"):
        assert not re.search(r"@media[^{]*prefers-color-scheme", path.read_text()), path.name
    html = client.get("/workspace").get_data(as_text=True)
    # the only occurrence in the markup is on a link element, not a style or a theme toggle
    for idx in [i for i in range(len(html)) if html.startswith("prefers-color-scheme", i)]:
        assert "<link" in html[max(0, idx - 200):idx]


def test_favicon_is_generated_from_the_supplied_asset_not_hand_edited():
    """The square export must still match what build_favicon.py produces from favicon.png."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("bf", ROOT / "scripts" / "build_favicon.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    assert bf.main(["--check"]) == 0


def test_logo_dimension_attributes_match_the_real_file(client):
    """Wrong width/height reserve the wrong box and shift the layout when the image loads."""
    import re

    html = client.get("/workspace").get_data(as_text=True)
    tag = re.search(r'<img[^>]*retinex-logo-flush\.png"[^>]*>', html).group(0)
    w = int(re.search(r'width="(\d+)"', tag).group(1))
    h = int(re.search(r'height="(\d+)"', tag).group(1))
    real = Image.open(ROOT / "app" / "static" / "brand" / "retinex-logo-flush.png").size
    assert (w, h) == real, f"markup says {(w, h)}, file is {real}"


def test_logo_is_wide_enough_for_the_design_minimum():
    """design.md Brand Assets: minimum width 160px. It is rendered at 170px."""
    width = Image.open(ROOT / "app" / "static" / "brand" / "retinex-logo-flush.png").size[0]
    assert width >= 170, "the logo file is narrower than its rendered size, so it will look soft"


def test_the_rendered_logo_has_no_transparent_padding():
    """Why the flush export exists.

    The supplied master carries 34px of fully transparent padding on its left edge, which pushed
    the visible wordmark right of the content column. The export the navigation renders must have
    none, because that, and not a negative margin, is what makes the bar align with the page.
    """
    flush = Image.open(ROOT / "app" / "static" / "brand" / "retinex-logo-flush.png").convert("RGBA")
    assert flush.getchannel("A").getbbox() == (0, 0, flush.width, flush.height)

    master = Image.open(ROOT / "app" / "static" / "brand" / "retinex-logo.png").convert("RGBA")
    assert master.getchannel("A").getbbox() != (0, 0, master.width, master.height), (
        "the master no longer has padding to crop; has it been overwritten?")


def test_the_flush_export_removed_only_transparent_pixels():
    """Cropping must not touch a visible pixel, a proportion or a colour."""
    master = Image.open(ROOT / "app" / "static" / "brand" / "retinex-logo.png").convert("RGBA")
    flush = Image.open(ROOT / "app" / "static" / "brand" / "retinex-logo-flush.png").convert("RGBA")
    box = master.getchannel("A").getbbox()
    assert list(master.crop(box).get_flattened_data()) == list(flush.get_flattened_data())


def test_dark_variant_keeps_the_supplied_cream_colour():
    """favicon-dark.png is the master's own colour: cropped and padded, never recoloured."""
    from collections import Counter

    source = Image.open(ROOT / "app" / "static" / "brand" / "favicon-source.png").convert("RGBA")
    out = Image.open(ROOT / "app" / "static" / "brand" / "favicon-dark.png")
    src = Counter(p[:3] for p in source.get_flattened_data() if p[3] > 200).most_common(1)[0][0]
    got = Counter(p[:3] for p in out.get_flattened_data() if p[3] > 200).most_common(1)[0][0]
    # resampling shifts a channel by a point or two; a recolour would not be within tolerance
    assert all(abs(a - b) <= 3 for a, b in zip(src, got)), (src, got)


def test_light_variant_is_exactly_the_approved_pine_green():
    from collections import Counter

    out = Image.open(ROOT / "app" / "static" / "brand" / "favicon.png")
    opaque = [p for p in out.get_flattened_data() if p[3] > 250]
    got = Counter(p[:3] for p in opaque).most_common(1)[0][0]
    assert got == (0x32, 0x4A, 0x3D), f"expected #324A3D, got #{got[0]:02X}{got[1]:02X}{got[2]:02X}"


def test_recolouring_did_not_alter_the_artwork_shape():
    """Only the colour channels differ between the variants; every alpha value is identical."""
    light = Image.open(ROOT / "app" / "static" / "brand" / "favicon.png").getchannel("A")
    dark = Image.open(ROOT / "app" / "static" / "brand" / "favicon-dark.png").getchannel("A")
    assert list(light.get_flattened_data()) == list(dark.get_flattened_data())


def test_light_variant_is_visible_on_light_browser_chrome():
    """The reason two variants exist. Measured, not assumed."""
    def contrast(fg, bg):
        def lin(c):
            c /= 255
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        def lum(c):
            r, g, b = (lin(x) for x in c)
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        a, b = lum(fg), lum(bg)
        return (max(a, b) + 0.05) / (min(a, b) + 0.05)

    green, cream = (0x32, 0x4A, 0x3D), (0xE9, 0xE5, 0xD9)
    light_chrome, dark_chrome = (0xDE, 0xE1, 0xE6), (0x35, 0x36, 0x3A)
    assert contrast(green, light_chrome) > 3, "the light-chrome icon must be visible on light chrome"
    assert contrast(cream, dark_chrome) > 3, "the dark-chrome icon must be visible on dark chrome"
    # and the pairing is the right way round
    assert contrast(green, light_chrome) > contrast(cream, light_chrome)
    assert contrast(cream, dark_chrome) > contrast(green, dark_chrome)


def test_logo_is_served_from_the_moved_brand_path(client):
    html = client.get("/workspace").get_data(as_text=True)
    assert "/static/brand/retinex-logo-flush.png" in html
    for name in ("retinex-logo.png", "retinex-logo-flush.png"):
        assert (ROOT / "app" / "static" / "brand" / name).is_file()
        assert client.get(f"/static/brand/{name}").status_code == 200
    assert not (ROOT / "retinex-logo.png").exists()


# --- truthful states --------------------------------------------------------

def test_workspace_marks_caption_and_narration_not_implemented(client):
    html = client.get("/workspace").get_data(as_text=True)
    assert "Caption generation is not implemented" in html
    assert "Narration is not connected" in html
    assert html.count("In development") == 2        # one chip on each of the two panels
    # the panels are still driven by the single source of truth in routes.py
    from app.routes import CAPTION_STATE, NARRATION_STATE
    assert CAPTION_STATE["body"] in html and NARRATION_STATE["body"] in html


@pytest.mark.parametrize("path", ["/", "/workspace"])
def test_no_fabricated_numbers_anywhere(client, path):
    """No BLEU score, accuracy, latency claim or testimonial may appear in a template."""
    html = client.get(path).get_data(as_text=True).lower()
    for banned in ("bleu-4 score", "accuracy of", "% accurate", "testimonial",
                   "users love", "state of the art", "revolutioniz"):
        assert banned not in html
    # a bare "BLEU" is allowed only as a stage name that says it is not implemented
    if "bleu" in html:
        assert "not implemented" in html


def test_pipeline_stage_count_and_honest_split():
    stages = pipeline.stages()
    assert len(stages) == 9
    assert [s["status"] for s in stages[4:]] == ["not_implemented"] * 5


@pytest.mark.parametrize("path", ["/", "/workspace"])
def test_no_page_renders_the_pipeline_dashboard(client, path):
    """The nine stage rows and their repeated badges are not interface content.

    The stages are still computed and still served by /api/pipeline; what went away is the
    repository status report on the page, not the inspection behind it.
    """
    html = client.get(path).get_data(as_text=True)
    assert "Pipeline status" not in html
    assert "pipeline-stage" not in html
    for s in pipeline.stages():
        assert s["detail"] not in html
    for leaked in ("6000/1000/1000", "4096", "No module exists yet",
                   "The Workspace cannot generate a caption yet",
                   "No text-to-speech is connected", "Not yet evaluated"):
        assert leaked not in html


@pytest.mark.parametrize("path", ["/", "/workspace"])
def test_no_page_positions_retinex_as_a_student_project(client, path):
    html = client.get(path).get_data(as_text=True).lower()
    for banned in ("academic", "major project", "college", "university project",
                   "student", "examiner", "coursework"):
        assert banned not in html


def test_pages_render_without_running_the_validators(client, monkeypatch):
    """Neither page needs stage status now, so neither should pay for four validators."""
    def boom(*_a, **_k):
        raise AssertionError("a page must not call pipeline.stages()")

    monkeypatch.setattr(pipeline, "stages", boom)
    assert client.get("/").status_code == 200
    assert client.get("/workspace").status_code == 200


def test_pipeline_is_derived_from_validators_not_hardcoded(tmp_path):
    """Point the pipeline at empty directories: it must report problems, not success."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    stages = pipeline.stages(data_dir=empty, features_dir=empty, pseudo_dir=empty, fisher_dir=empty)
    assert [s["status"] for s in stages[:4]] == ["missing"] * 4
    assert pipeline.summary(stages)["complete"] == 0


def test_pipeline_reports_a_failing_validator(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.dataset, "verify_split", lambda *_a, **_k: ["train: duplicate ids"])
    stages = pipeline.stages()
    assert stages[0]["status"] == "failed"
    assert "train: duplicate ids" in stages[0]["problems"]


def test_api_pipeline_returns_the_same_truth(client):
    data = client.get("/api/pipeline").get_json()
    assert data["ok"] and len(data["stages"]) == 9
    assert data["summary"]["not_implemented"] == 5


# --- upload validation ------------------------------------------------------

def test_valid_upload_facts_come_from_the_bytes(client):
    facts = uploads.validate_bytes(png_bytes(70, 50), "whatever.jpg")
    assert facts["format"] == "PNG"            # despite the .jpg filename
    assert (facts["width"], facts["height"]) == (70, 50)


@pytest.mark.parametrize("raw,needle", [
    (b"", "empty"),
    (b"this is not an image", "could not be read as an image"),
    (b"\x89PNG\r\n\x1a\ntruncated garbage", "could not be read as an image"),
])
def test_upload_rejects_bad_bytes(raw, needle):
    with pytest.raises(uploads.UploadError, match=needle):
        uploads.validate_bytes(raw, "x.png")


def test_upload_rejects_a_renamed_non_image(client):
    """A text file called .png must be rejected on its bytes, not accepted on its name."""
    r = client.post("/api/analyze", data={"image": (io.BytesIO(b"<html>hi</html>"), "photo.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 400
    assert "could not be read as an image" in r.get_json()["error"]


def test_upload_rejects_too_small_dimensions():
    with pytest.raises(uploads.UploadError, match="at least 32 pixels"):
        uploads.validate_bytes(png_bytes(10, 10), "tiny.png")


def test_upload_rejects_oversized_dimensions(monkeypatch):
    monkeypatch.setattr(uploads, "MAX_SIDE", 100)
    monkeypatch.setattr(uploads, "MAX_PIXELS", 10_000)
    with pytest.raises(uploads.UploadError, match="too large to process"):
        uploads.validate_bytes(png_bytes(200, 200), "big.png")


def test_upload_rejects_oversized_bytes():
    with pytest.raises(uploads.UploadError, match="over the"):
        uploads.validate_bytes(png_bytes(64, 64), "x.png", max_bytes=10)


def test_request_body_limit_returns_413_json(client):
    big = b"x" * (uploads.MAX_BYTES + 1024)
    r = client.post("/api/analyze", data={"image": (io.BytesIO(big), "big.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 413
    assert "limit" in r.get_json()["error"]


def test_missing_file_field_is_a_clear_400(client):
    r = client.post("/api/analyze", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert "No image was uploaded" in r.get_json()["error"]


@pytest.mark.parametrize("name,expected", [
    ("../../etc/passwd", "passwd"),
    ("a/b/c.png", "c.png"),
    ("", "upload"),
    ("...", "upload"),
])
def test_filename_is_never_trusted_as_a_path(name, expected):
    assert uploads.safe_name(name) == expected


# --- preprocessing alignment with Stage 2 -----------------------------------

def test_single_image_path_reuses_the_stage_2_loader(tmp_path):
    """The Workspace must preprocess exactly as the committed artifacts were built."""
    img = tmp_path / "one.png"
    img.write_bytes(png_bytes(300, 200))
    direct = fe._load_batch(tmp_path, ["one.png"])
    assert direct.shape == (1, 224, 224, 3)

    seen = {}

    class FakeModel:
        def predict_on_batch(self, batch):
            seen["batch"] = np.asarray(batch)
            return np.ones((batch.shape[0], fe.FEATURE_DIM), dtype=np.float32)

    inference._MODEL = FakeModel()
    try:
        inference.extract_one(img)
    finally:
        inference._MODEL = None
    # byte-identical to what Stage 2's own loader produces for the same file
    assert np.array_equal(seen["batch"], direct)


def test_extract_one_applies_the_saved_indices_without_refitting(tmp_path):
    img = tmp_path / "one.png"
    img.write_bytes(png_bytes())
    vector = np.arange(fe.FEATURE_DIM, dtype=np.float32)

    class FakeModel:
        def predict_on_batch(self, batch):
            return vector.reshape(1, -1)

    idx = inference.load_selection()
    inference._MODEL = FakeModel()
    try:
        out = inference.extract_one(img)
    finally:
        inference._MODEL = None
    assert out["feature_dim"] == 4096 and out["selected_dim"] == 512
    assert out["selection_source"] == "fisher_top512_indices.npy"
    # the reported maximum is the value at the largest selected index, proving the mask was applied
    assert out["reduced"]["max"] == pytest.approx(float(vector[idx].max()))


def test_saved_selection_is_read_only_and_ascending():
    idx = inference.load_selection()
    assert idx.shape == (512,) and (np.diff(idx) > 0).all()


def test_the_production_selection_is_verified_against_its_pinned_digest(monkeypatch):
    """The real path checks the committed bytes before numpy ever parses them."""
    from retinex import fisher as fi

    seen = []
    real = fi.verify_indices
    monkeypatch.setattr(fi, "verify_indices", lambda p: seen.append(Path(p).name) or real(p))
    monkeypatch.setattr(inference, "_SELECTION", None)

    idx = inference.load_selection()
    assert seen == [fi.INDICES], "the production load must verify exactly once"
    assert idx.shape == (512,)


def test_a_tampered_selection_is_refused_without_leaking_the_digests(tmp_path, monkeypatch):
    """One flipped byte. The client message must not carry the checksums or a local path."""
    from retinex import fisher as fi

    raw = bytearray((fi.default_out_dir() / fi.INDICES).read_bytes())
    raw[-1] ^= 0x01
    (tmp_path / fi.INDICES).write_bytes(bytes(raw))

    monkeypatch.setattr(fi, "default_out_dir", lambda: tmp_path)
    monkeypatch.setattr(inference, "_SELECTION", None)

    with pytest.raises(inference.InferenceError) as excinfo:
        inference.load_selection()
    message = str(excinfo.value)
    assert "does not match its expected checksum" in message
    assert fi.EXPECTED_INDICES_SHA256 not in message
    assert str(tmp_path) not in message


def test_a_tampered_selection_through_the_api_is_a_503(client, tmp_path, monkeypatch):
    from retinex import fisher as fi

    raw = bytearray((fi.default_out_dir() / fi.INDICES).read_bytes())
    raw[0] ^= 0xFF
    (tmp_path / fi.INDICES).write_bytes(bytes(raw))
    monkeypatch.setattr(fi, "default_out_dir", lambda: tmp_path)
    monkeypatch.setattr(inference, "_SELECTION", None)

    r = client.post("/api/analyze", data={"image": (io.BytesIO(png_bytes()), "a.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 503
    body = r.get_json()
    assert body["ok"] is False and "checksum" in body["error"]
    assert "features" not in body                       # never a fabricated success
    assert fi.EXPECTED_INDICES_SHA256 not in r.get_data(as_text=True)


def test_the_digest_is_not_recomputed_on_every_request(monkeypatch):
    """Verification is per process, like the model, not per call."""
    from retinex import fisher as fi

    calls = []
    real = fi.verify_indices
    monkeypatch.setattr(fi, "verify_indices", lambda p: calls.append(p) or real(p))
    monkeypatch.setattr(inference, "_SELECTION", None)

    first = inference.load_selection()
    for _ in range(5):
        inference.load_selection()
    assert len(calls) == 1, f"hashed {len(calls)} times; it must be cached"
    assert inference.load_selection() is first      # the same cached array, not a re-read


def test_the_cached_selection_cannot_be_mutated(monkeypatch):
    """It is shared across requests and threads, so it must be read-only."""
    monkeypatch.setattr(inference, "_SELECTION", None)
    idx = inference.load_selection()
    assert not idx.flags.writeable
    with pytest.raises(ValueError):
        idx[0] = 999


def test_the_selection_is_verified_once_under_concurrency(monkeypatch):
    """Four threads arriving together must not all hash and load the file."""
    import threading

    from retinex import fisher as fi

    calls = []
    real = fi.verify_indices

    def slow(path):
        calls.append(path)
        time.sleep(0.05)                            # long enough for the others to queue
        return real(path)

    monkeypatch.setattr(fi, "verify_indices", slow)
    monkeypatch.setattr(inference, "_SELECTION", None)

    out = []
    threads = [threading.Thread(target=lambda: out.append(inference.load_selection()))
               for _ in range(4)]
    for t in threads:
        t.start()
        time.sleep(0.005)
    for t in threads:
        t.join()

    assert len(calls) == 1, f"verified {len(calls)} times"
    assert all(o is out[0] for o in out), "every thread must get the one cached array"


def test_the_cached_selection_is_detached_from_the_file_on_disk(tmp_path, monkeypatch):
    """Verify-then-cache must survive the file changing underneath it.

    np.asarray on a memory-mapped array returns a *view*, so a cache built that way keeps reading
    the bytes on disk: overwrite the artifact after the first request and every later request would
    quietly use unverified indices, with no second hash and no error. The copy is what closes that
    window, and this test is what proves the copy is really a copy.

    The real tracked artifact is never touched. A byte-identical duplicate is placed in tmp_path,
    which is what gets overwritten, so a failed assertion here cannot damage the repository.
    """
    from retinex import fisher as fi

    source = fi.default_out_dir() / fi.INDICES
    before_digest = fe._sha256(source)
    fixture = tmp_path / fi.INDICES
    fixture.write_bytes(source.read_bytes())            # same bytes, so verification passes
    monkeypatch.setattr(fi, "default_out_dir", lambda: tmp_path)
    monkeypatch.setattr(inference, "_SELECTION", None)

    idx = inference.load_selection()
    original = idx.copy()

    # it owns its data and shares nothing with any mapping of the file
    assert idx.flags.owndata, "the cached array must not be a view of the mapped file"
    assert not np.shares_memory(idx, np.load(fixture, mmap_mode="r"))

    # overwrite in place, exactly as a post-verification tamper would. 7 cannot be the last
    # index of an ascending selection whose maximum is 4095, so this is always a real change.
    raw = bytearray(fixture.read_bytes())
    raw[-4:] = (7).to_bytes(4, "little")
    with open(fixture, "r+b") as f:
        f.write(bytes(raw))
        f.flush()
        os.fsync(f.fileno())
    assert fe._sha256(fixture) != fi.EXPECTED_INDICES_SHA256, "the fixture really was changed"

    # the cached array still holds the values that were verified
    cached = inference.load_selection()
    assert cached is idx
    assert np.array_equal(cached, original)
    assert fe._sha256(source) == before_digest, "the tracked artifact must be untouched"


def test_a_caller_supplied_directory_is_not_checked_against_the_pinned_digest(tmp_path):
    """The test seam stays usable: the pinned digest describes the committed file, not a fixture."""
    np.save(tmp_path / "fisher_top512_indices.npy", np.arange(4, dtype=np.int32))
    idx = inference.load_selection(tmp_path)
    assert list(idx) == [0, 1, 2, 3]


def test_fisher_artifacts_are_never_written_by_the_app(tmp_path):
    """Running an analyze request must not change any Stage 2/3/4 artifact."""
    from retinex import fisher as fi
    before = {p.name: fe._sha256(p) for p in fi.default_out_dir().iterdir() if p.is_file()}
    before.update({p.name: fe._sha256(p) for p in fe.default_out_dir().iterdir() if p.is_file()})

    img = tmp_path / "one.png"
    img.write_bytes(png_bytes())

    class FakeModel:
        def predict_on_batch(self, batch):
            return np.ones((1, fe.FEATURE_DIM), dtype=np.float32)

    inference._MODEL = FakeModel()
    try:
        inference.extract_one(img)
    finally:
        inference._MODEL = None

    after = {p.name: fe._sha256(p) for p in fi.default_out_dir().iterdir() if p.is_file()}
    after.update({p.name: fe._sha256(p) for p in fe.default_out_dir().iterdir() if p.is_file()})
    assert before == after


# --- API behaviour and controlled failure -----------------------------------

def test_analyze_returns_measured_features_and_no_caption(client, monkeypatch):
    class FakeModel:
        def predict_on_batch(self, batch):
            return (np.arange(fe.FEATURE_DIM, dtype=np.float32) / 100.0).reshape(1, -1)

    monkeypatch.setattr(inference, "get_model", lambda: FakeModel())
    r = client.post("/api/analyze", data={"image": (io.BytesIO(png_bytes()), "a.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["features"]["feature_dim"] == 4096
    assert data["features"]["selected_dim"] == 512
    assert len(data["features"]["top_activations"]) == 8
    assert data["caption"]["implemented"] is False
    assert data["narration"]["implemented"] is False
    assert "caption" not in json.dumps(data["features"]).lower()   # no caption field smuggled in
    assert "text" not in data["caption"] or data["caption"].get("text") is None


def test_model_load_failure_returns_a_controlled_error(client, monkeypatch):
    def boom():
        raise inference.InferenceError("The VGG16 feature extractor could not be loaded.")

    monkeypatch.setattr(inference, "get_model", boom)
    r = client.post("/api/analyze", data={"image": (io.BytesIO(png_bytes()), "a.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 503
    body = r.get_json()
    assert body["ok"] is False and "could not be loaded" in body["error"]
    assert "features" not in body               # never a fabricated success


def test_inference_failure_returns_a_controlled_error(client, monkeypatch):
    class Broken:
        def predict_on_batch(self, batch):
            raise RuntimeError("tensor blew up")

    monkeypatch.setattr(inference, "get_model", lambda: Broken())
    r = client.post("/api/analyze", data={"image": (io.BytesIO(png_bytes()), "a.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 503
    assert "Feature extraction failed" in r.get_json()["error"]


def test_missing_selection_artifact_is_reported_not_faked(tmp_path, monkeypatch):
    with pytest.raises(inference.InferenceError, match="Fisher Score selection is missing"):
        inference.load_selection(tmp_path)


def test_non_finite_features_are_refused(tmp_path):
    img = tmp_path / "one.png"
    img.write_bytes(png_bytes())

    class NaNModel:
        def predict_on_batch(self, batch):
            v = np.ones((1, fe.FEATURE_DIM), dtype=np.float32)
            v[0, 5] = np.nan
            return v

    inference._MODEL = NaNModel()
    try:
        with pytest.raises(inference.InferenceError, match="non-finite"):
            inference.extract_one(img)
    finally:
        inference._MODEL = None


def test_model_is_not_loaded_at_import_or_page_render(client):
    assert inference._MODEL is None
    client.get("/workspace")
    client.get("/")
    client.get("/api/pipeline")
    assert inference._MODEL is None, "rendering a page must not load VGG16"


# --- hardening: concurrency, bombs, corrupt artifacts, safe errors ----------

def test_lazy_model_load_happens_once_under_concurrency(monkeypatch):
    """Flask.run sets threaded=True, so two first requests must not both load VGG16.

    The build sleeps while holding the lock, which is long enough for the other threads to pile up
    behind it. A barrier cannot be used here: it would wait inside the critical section and
    deadlock, which is exactly what happens without the double-checked lock being correct.
    """
    import threading
    import time as _t

    calls = []

    def slow_build():
        calls.append(1)
        _t.sleep(0.2)
        return object()

    monkeypatch.setattr(inference.features, "_build_model", slow_build)
    inference._MODEL = None
    inference._MODEL_LOAD_SECONDS = None
    got = []
    threads = [threading.Thread(target=lambda: got.append(inference.get_model()))
               for _ in range(4)]
    try:
        for th in threads:
            th.start()
            _t.sleep(0.01)             # stagger slightly so they contend on the lock
        for th in threads:
            th.join(timeout=10)
        assert not any(th.is_alive() for th in threads), "get_model deadlocked"
        assert len(calls) == 1, f"model was built {len(calls)} times"
        assert len(got) == 4 and len({id(m) for m in got}) == 1
    finally:
        inference._MODEL = None
        inference._MODEL_LOAD_SECONDS = None


def test_failed_model_load_leaves_the_cache_empty_and_retryable(monkeypatch):
    def boom():
        raise RuntimeError("no weights")

    monkeypatch.setattr(inference.features, "_build_model", boom)
    inference._MODEL = None
    with pytest.raises(inference.InferenceError, match="could not be loaded"):
        inference.get_model()
    assert inference._MODEL is None                # retryable, not poisoned


def test_model_load_error_does_not_leak_the_underlying_message(monkeypatch):
    monkeypatch.setattr(inference.features, "_build_model",
                        lambda: (_ for _ in ()).throw(RuntimeError("/Users/secret/path.h5")))
    inference._MODEL = None
    with pytest.raises(inference.InferenceError) as excinfo:
        inference.get_model()
    assert "/Users/secret" not in str(excinfo.value)


def test_decompression_bomb_is_an_upload_error_not_a_crash(monkeypatch):
    """DecompressionBombError is not an OSError or ValueError, so it needs naming explicitly."""
    def bomb(*_a, **_k):
        raise uploads.BOMB("too many pixels")

    monkeypatch.setattr(uploads.Image, "open", bomb)
    with pytest.raises(uploads.UploadError, match="more pixels than this application will decode"):
        uploads.validate_bytes(png_bytes(), "bomb.png")


def test_decompression_bomb_through_the_api_is_a_400(client, monkeypatch):
    def bomb(*_a, **_k):
        raise uploads.BOMB("too many pixels")

    monkeypatch.setattr(uploads.Image, "open", bomb)
    r = client.post("/api/analyze", data={"image": (io.BytesIO(png_bytes()), "bomb.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 400
    assert "pixels" in r.get_json()["error"]


def test_corrupted_selection_artifact_is_a_controlled_error(tmp_path):
    (tmp_path / "fisher_top512_indices.npy").write_bytes(b"not an npy file at all")
    with pytest.raises(inference.InferenceError, match="could not be read"):
        inference.load_selection(tmp_path)


def test_non_ascending_selection_is_refused(tmp_path):
    np.save(tmp_path / "fisher_top512_indices.npy", np.array([9, 3, 7], dtype=np.int32))
    with pytest.raises(inference.InferenceError, match="not strictly ascending"):
        inference.load_selection(tmp_path)


def test_corrupted_artifact_through_the_api_is_a_503(client, monkeypatch):
    monkeypatch.setattr(inference, "load_selection",
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            inference.InferenceError("The Fisher Score selection is missing.")))
    r = client.post("/api/analyze", data={"image": (io.BytesIO(png_bytes()), "a.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 503 and r.get_json()["ok"] is False


def test_unexpected_exception_returns_json_without_a_traceback(monkeypatch):
    """An unhandled error on an API path must be a fixed JSON message, never a stack trace."""
    app = create_app()
    app.config["PROPAGATE_EXCEPTIONS"] = False

    @app.get("/api/boom")
    def boom():
        raise RuntimeError("internal detail /Users/secret/file.py line 42")

    r = app.test_client().get("/api/boom")
    assert r.status_code == 500
    body = r.get_json()
    assert body["ok"] is False
    assert "Something went wrong" in body["error"]
    assert "Traceback" not in r.get_data(as_text=True)
    assert "/Users/secret" not in r.get_data(as_text=True)


def test_api_wrong_method_and_unknown_endpoint_return_json(client):
    assert client.get("/api/analyze").status_code == 405
    assert client.get("/api/analyze").get_json()["ok"] is False
    assert client.get("/api/nope").status_code == 404
    assert client.get("/api/nope").get_json()["ok"] is False


def test_html_404_is_still_html(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.headers["Content-Type"].startswith("text/html")


def test_upload_limit_is_enforced_twice(client):
    """Werkzeug rejects an oversized body, and validate_bytes checks the bytes it receives."""
    app = create_app({"TESTING": True})
    assert app.config["MAX_CONTENT_LENGTH"] == uploads.MAX_BYTES
    with pytest.raises(uploads.UploadError, match="over the"):
        uploads.validate_bytes(png_bytes(), "x.png", max_bytes=100)


# --- generated token parity -------------------------------------------------

def test_tokens_css_matches_design_md():
    """The committed CSS must be exactly what the generator produces from design.md."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("bt", ROOT / "scripts" / "build_tokens.py")
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    expected = bt.render(bt.frontmatter())
    assert (ROOT / "app" / "static" / "css" / "tokens.css").read_text() == expected


def test_every_design_md_token_reaches_the_css():
    fm = yaml.safe_load(
        (ROOT / "design.md").read_text().split("---")[1])
    css = (ROOT / "app" / "static" / "css" / "tokens.css").read_text()
    for name, value in fm["colors"].items():
        assert f"--color-{name}: {value};" in css
    for name, value in fm["spacing"].items():
        assert f"--space-{name}: {value};" in css
    for name, value in fm["rounded"].items():
        assert f"--radius-{name}: {value};" in css
    for token in fm["typography"]:
        assert f"--type-{token}-size:" in css


def test_no_hardcoded_hex_outside_the_generated_tokens():
    """design.md Implementation Rule 2: never hardcode a hex value in a component."""
    css_dir = ROOT / "app" / "static" / "css"
    for path in css_dir.glob("*.css"):
        if path.name == "tokens.css":
            continue
        hits = re.findall(r"#[0-9A-Fa-f]{3,8}\b", path.read_text())
        assert not hits, f"{path.name} hardcodes {hits}"


def test_no_dark_mode_overrides():
    """design.md: there is no dark mode. No prefers-color-scheme *at-rule* may exist.

    The literal words may appear in a comment explaining the rule, so match the at-rule itself.
    """
    for path in (ROOT / "app" / "static" / "css").glob("*.css"):
        assert not re.search(r"@media[^{]*prefers-color-scheme", path.read_text()), path.name


def test_templates_use_no_inline_styles():
    """design.md Implementation Rule 2: tokens live in CSS, not in style attributes."""
    for path in (ROOT / "app" / "templates").rglob("*.html"):
        assert 'style="' not in path.read_text(), path.name


# --- home -------------------------------------------------------------------

def test_home_has_a_clear_cta_to_the_workspace(client):
    html = client.get("/").get_data(as_text=True)
    assert html.count('href="/workspace"') >= 2        # hero and closing CTA
    assert "Open the Workspace" in html


def test_home_states_what_is_not_built(client):
    """Positioning RetineX as a product must not turn the unbuilt half into a claim."""
    html = client.get("/").get_data(as_text=True)
    assert "Caption generation and narration are still being built" in html
    assert html.count("In development") == 2        # the describe step and the speak step
    assert "does not show a sentence yet" in html


def test_home_uses_at_most_one_inverted_band(client):
    """design.md: use the inverted treatment for at most one band per page."""
    for path in ("/", "/workspace"):
        html = client.get(path).get_data(as_text=True)
        assert html.count('class="section-inverse"') <= 1


def test_home_dataset_provenance_is_honest(client):
    html = client.get("/").get_data(as_text=True)
    assert "third-party" in html
    assert "rather than calling it official" in html


def test_reduced_motion_is_honoured():
    base = (ROOT / "app" / "static" / "css" / "base.css").read_text()
    assert "prefers-reduced-motion: reduce" in base
    components = (ROOT / "app" / "static" / "css" / "components.css").read_text()
    assert "prefers-reduced-motion: reduce" in components


# --- motion and navigation enhancement --------------------------------------

def test_a_reveal_is_only_hidden_once_javascript_has_said_so():
    """The fallback that keeps content readable when the script never runs.

    Every rule that sets a reveal to zero opacity must be scoped to the .js class, which ui.js
    adds to <html>. An unscoped rule would leave the page blank with JavaScript disabled.
    """
    css = (ROOT / "app" / "static" / "css" / "base.css").read_text()
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)       # a comment may mention .js innocently
    # a keyframe may start at zero opacity; that hides nothing, because the animation ends
    css = re.sub(r"@keyframes\s+[\w-]+\s*\{(?:[^{}]|\{[^{}]*\})*\}", "", css)
    hiding = re.findall(r"([^{}]*)\{[^{}]*opacity:\s*0;", css)
    assert hiding, "expected at least one reveal rule to check"
    for selector in hiding:
        assert ".js " in selector, f"{selector.strip()!r} hides content without the js gate"


def test_the_reveal_script_loads_before_the_body_paints():
    """Deferring it would let hidden content flash in and then disappear."""
    base = (ROOT / "app" / "templates" / "base.html").read_text()
    head, body = base.split("<body>", 1)
    assert "js/ui.js" in head and "defer" not in head.split("js/ui.js")[1].split(">")[0]
    assert "js/ui.js" not in body


def test_reduced_motion_cancels_the_stagger_and_the_translate():
    """design.md: under reduced motion elements appear in their final state, with no wait."""
    css = (ROOT / "app" / "static" / "css" / "base.css").read_text()
    block = css.split("@media (prefers-reduced-motion: reduce)", 1)[1]
    assert "transition-delay: 0s !important" in block
    assert "transform: none !important" in block
    assert "opacity: 1 !important" in block


# --- page transitions -------------------------------------------------------

def base_css():
    return (ROOT / "app" / "static" / "css" / "base.css").read_text()


def test_navigation_is_transitioned_by_css_not_by_script():
    """The whole approach: one at-rule, and no script anywhere near a link.

    A scripted transition has to intercept clicks, and then it owns new tabs, modifier clicks,
    downloads, Back, Forward and scroll restoration, and gets one of them wrong. This owns none
    of them.
    """
    assert "@view-transition { navigation: auto; }" in base_css()

    js = (ROOT / "app" / "static" / "js" / "ui.js").read_text()
    for banned in ("preventDefault", '"click"', "startViewTransition",
                   "history.pushState", "location.href", "fetch("):
        assert banned not in js, f"ui.js must not {banned}"


def test_the_transition_duration_is_the_documented_one():
    """design.md Motion, Page transitions: 120ms out, 260ms in, so 260ms overall."""
    css = base_css()
    assert "::view-transition-old(root) { animation: page-out var(--dur-page-out)" in css
    assert "::view-transition-new(root) { animation: page-in var(--dur-page-in)" in css

    tokens = (ROOT / "app" / "static" / "css" / "tokens.css").read_text()
    out = int(re.search(r"--dur-page-out: (\d+)ms", tokens).group(1))
    into = int(re.search(r"--dur-page-in: (\d+)ms", tokens).group(1))
    assert out < into, "the outgoing page must clear before the incoming one has finished"
    assert 200 <= into <= 350, into
    assert into <= 400, "design.md Motion: nothing exceeds 400ms"


def test_the_incoming_page_rises_by_a_restrained_amount():
    css = base_css()
    lift = re.search(r"@keyframes page-in \{ from \{ opacity: 0; transform: translateY\((\d+)px\)",
                     css)
    assert lift and 8 <= int(lift.group(1)) <= 12, css


def test_the_header_does_not_travel_with_the_page():
    """Named elements are lifted out of the root snapshot, so the bar cannot jump or flicker."""
    css = base_css()
    assert ".site-header { view-transition-name: site-header; }" in css
    assert ".nav-indicator { view-transition-name: nav-indicator; }" in css
    names = re.findall(r"view-transition-name: ([\w-]+);", css)
    assert len(names) == len(set(names)), f"a view-transition-name must be unique: {names}"


def test_reduced_motion_turns_the_transition_off_without_breaking_navigation():
    block = base_css().split("@media (prefers-reduced-motion: reduce)", 1)[1]
    assert "::view-transition-group(*)" in block
    assert "animation: none !important" in block
    # nothing in the reduced-motion block may stop the navigation itself
    assert "display: none" not in block and "pointer-events" not in block


def direct_children(template, class_name):
    """Count the immediate element children of the first node carrying `class_name`."""
    from html.parser import HTMLParser

    class Counter(HTMLParser):
        depth = None
        count = 0
        done = False

        def handle_starttag(self, tag, attrs):
            if self.done:
                return
            classes = dict(attrs).get("class", "").split()
            if self.depth is None:
                if class_name in classes:
                    self.depth = 0
                return
            if self.depth == 0:
                self.count += 1
            self.depth += 1

        def handle_endtag(self, _tag):
            if self.done or self.depth is None:
                return
            self.depth -= 1
            if self.depth < 0:                  # the node itself closed; stop counting
                self.done = True

    c = Counter()
    c.feed(template.read_text())
    return c.count


def test_the_hero_stagger_stays_within_the_documented_limit():
    """design.md Motion forbids a cascade of more than three items."""
    css = (ROOT / "app" / "static" / "css" / "base.css").read_text()
    delays = re.findall(r"\.hero-copy > \*:nth-child\((\d+)\)", css)
    assert delays and max(int(d) for d in delays) <= 3, delays
    assert ".hero-copy > *:nth-child(4)" not in css
    assert direct_children(ROOT / "app" / "templates" / "index.html", "hero-copy") == 3


def test_the_hero_entrance_needs_no_javascript(client):
    """It is a CSS animation, so a blocked script cannot leave the hero blank."""
    html = client.get("/").get_data(as_text=True)
    hero = html.split('class="hero-copy', 1)[1].split("</section>", 1)[0]
    assert "reveal" not in hero and "data-enter" not in hero
    css = (ROOT / "app" / "static" / "css" / "base.css").read_text()
    rule = css.split(".hero-copy > * {", 1)[1].split("}", 1)[0]
    assert "backwards" in rule, "fill mode must not leave the hero hidden before it animates"


@pytest.mark.parametrize("path", ["/", "/workspace"])
def test_exactly_one_navigation_underline_exists(client, path):
    """A static border plus a sliding bar at once would draw two underlines."""
    html = client.get(path).get_data(as_text=True)
    assert html.count('class="nav-indicator"') == 1
    assert html.count('aria-current="page"') == 1          # the active page, server rendered

    css = (ROOT / "app" / "static" / "css" / "components.css").read_text()
    # the static underline is the no-JavaScript fallback and is stood down when the bar exists
    assert 'a[aria-current="page"] {\n  border-bottom: 2px solid var(--color-accent);' in css
    assert '.js .nav-links a[aria-current="page"] { border-bottom-color: transparent; }' in css


def test_the_navigation_indicator_is_hidden_until_it_is_positioned(client):
    """Shown before it is measured, it would sit at the left edge and then slide across."""
    html = client.get("/").get_data(as_text=True)
    indicator = html.split('class="nav-indicator"', 1)[1].split(">", 1)[0]
    assert "hidden" in indicator
    assert 'aria-hidden="true"' in indicator                # decoration, not content
    js = (ROOT / "app" / "static" / "js" / "ui.js").read_text()
    assert js.index("rest();") < js.index("bar.hidden = false;")


def test_navigation_links_are_still_real_links(client):
    """The indicator must not have turned the navigation into script-driven tabs."""
    html = client.get("/").get_data(as_text=True)
    nav = html.split('<nav aria-label="Primary">', 1)[1].split("</nav>", 1)[0]
    assert nav.count("<a href=") == 2
    assert 'href="/"' in nav and 'href="/workspace"' in nav
    assert "<button" not in nav and "onclick" not in nav


def test_buttons_use_the_documented_radius_in_one_place(client):
    """design.md Shapes: buttons are lg. One shared rule, not a per-button override."""
    css = (ROOT / "app" / "static" / "css" / "components.css").read_text()
    assert "border-radius: var(--radius-lg);" in css.split(".btn {", 1)[1].split("}", 1)[0]
    # no variant re-rounds itself, and nothing in the stylesheet is a pill but the status dot
    variants = re.findall(r"\.btn-\w+[^{]*\{([^}]*)\}", css)
    assert not [v for v in variants if "border-radius" in v]
    assert css.count("var(--radius-full)") == 1

    design = (ROOT / "design.md").read_text()
    assert "All use `lg` radius" in design


def test_every_button_still_has_a_visible_focus_ring():
    css = (ROOT / "app" / "static" / "css" / "components.css").read_text()
    assert ".btn:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; }" in css
    assert "outline: none" not in css and "outline: 0" not in css


def test_disabled_buttons_do_not_react_to_hover(client):
    """The three narration controls are disabled, so hovering one must change nothing."""
    css = (ROOT / "app" / "static" / "css" / "components.css").read_text()
    assert '.btn:disabled:hover, .btn[aria-disabled="true"]:hover {' in css
    html = client.get("/workspace").get_data(as_text=True)
    block = html.split('aria-label="Narration playback"', 1)[1].split("</div>", 1)[0]
    assert block.count("disabled") == 3
