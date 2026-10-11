"""Flask UI tests: routes, upload validation, preprocessing alignment, honest states, a11y, tokens.

These never load VGG16 unless a test says so explicitly; the model is monkeypatched by default so
the suite stays fast and does not depend on the ImageNet weights being cached.
"""
import io
import json
import re
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


def test_no_favicon_is_shipped_until_a_square_asset_exists(client):
    html = client.get("/workspace").get_data(as_text=True)
    assert 'rel="icon"' not in html
    assert not (ROOT / "app" / "static" / "favicon.png").exists()


def test_logo_is_served_from_the_moved_brand_path(client):
    html = client.get("/workspace").get_data(as_text=True)
    assert "/static/brand/retinex-logo.png" in html
    assert (ROOT / "app" / "static" / "brand" / "retinex-logo.png").is_file()
    assert not (ROOT / "retinex-logo.png").exists()
    r = client.get("/static/brand/retinex-logo.png")
    assert r.status_code == 200


# --- truthful states --------------------------------------------------------

def test_workspace_marks_caption_and_narration_not_implemented(client):
    html = client.get("/workspace").get_data(as_text=True)
    assert "Caption generation is not implemented" in html
    assert "Narration is not connected" in html
    assert "Not implemented" in html


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


def test_pipeline_stage_count_and_honest_split(client):
    html = client.get("/workspace").get_data(as_text=True)
    assert "Pipeline status" in html
    stages = pipeline.stages()
    assert len(stages) == 9
    assert [s["status"] for s in stages[4:]] == ["not_implemented"] * 5
    for s in stages[4:]:
        assert s["name"] in html


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
    html = client.get("/").get_data(as_text=True)
    assert "Not yet evaluated" in html
    assert "Not implemented" in html
    assert "negative result is a result" in html


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
