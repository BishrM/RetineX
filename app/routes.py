"""Routes: two pages and one JSON endpoint.

The endpoint returns measured feature statistics or a controlled error. There is no code path that
returns a caption, because no caption model exists in this repository.
"""
import logging
import tempfile
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, MethodNotAllowed, RequestEntityTooLarge

log = logging.getLogger(__name__)

from . import inference, pipeline, uploads

bp = Blueprint("main", __name__)

# What the Workspace says where a caption would go. One source of truth for the honest state:
# the page and the JSON API read these same two dicts, so they cannot drift apart.
CAPTION_STATE = {
    "implemented": False,
    "stage": 8,
    "heading": "Caption generation is not implemented",
    "body": ("The model that writes the sentence is still being built, so RetineX has nothing to "
             "write with yet and will not invent one. A real description appears here, in this "
             "panel, as soon as it can produce it."),
}
NARRATION_STATE = {
    "implemented": False,
    "stage": 9,
    "heading": "Narration is not connected",
    "body": ("Speaking a description needs the description first. The controls below are shown so "
             "the layout is honest about what will appear, and they stay genuinely disabled until "
             "there is something to say."),
}


@bp.app_errorhandler(RequestEntityTooLarge)
def too_large(_e):
    limit = current_app.config["MAX_CONTENT_LENGTH"] // 1_048_576
    message = f"That image is over the {limit} MB limit. Try a smaller file."
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": message}), 413
    return render_template("error.html", code=413, message=message), 413


@bp.app_errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "No such endpoint."}), 404
    return render_template("error.html", code=404,
                           message="That page does not exist yet."), 404


@bp.app_errorhandler(MethodNotAllowed)
def wrong_method(e):
    message = f"That endpoint does not accept {request.method}."
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": message}), 405
    return render_template("error.html", code=405, message=message), 405


@bp.app_errorhandler(Exception)
def unexpected(e):
    """Last resort. An unhandled exception must never leak a traceback or a local path.

    HTTPExceptions keep their own status and are re-raised to their specific handlers; anything
    else is logged in full and answered with a fixed message.
    """
    if isinstance(e, HTTPException):
        return e
    log.exception("Unhandled error on %s %s", request.method, request.path)
    message = "Something went wrong on the server. The server log has the details."
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": message}), 500
    return render_template("error.html", code=500, message=message), 500


# Neither page renders a pipeline dashboard any more, so neither runs the four validators on
# every request. They are still the only source of stage status, and /api/pipeline still reports
# exactly what they say.
@bp.get("/")
def home():
    return render_template("index.html")


@bp.get("/workspace")
def workspace():
    return render_template(
        "workspace.html",
        caption=CAPTION_STATE,
        narration=NARRATION_STATE,
        limits={"max_mb": uploads.MAX_BYTES // 1_048_576,
                "formats": "JPEG, PNG, WEBP or BMP",
                "min_side": uploads.MIN_SIDE},
    )


@bp.get("/api/pipeline")
def api_pipeline():
    stages = pipeline.stages()
    return jsonify({"ok": True, "stages": stages, "summary": pipeline.summary(stages)})


@bp.post("/api/analyze")
def api_analyze():
    """Extract real fc2 features for one uploaded image and apply the saved Stage 4 selection."""
    sent = request.files.get("image")
    if sent is None:
        return jsonify({"ok": False, "error": "No image was uploaded. Choose a file and try again."}), 400

    raw = sent.read()
    try:
        facts = uploads.validate_bytes(raw, sent.filename)
    except uploads.UploadError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    # written to a private temp file so the Stage 2 loader can open it by path, then removed
    with tempfile.TemporaryDirectory(prefix="retinex-upload-") as tmp:
        path = Path(tmp) / f"upload{uploads.suffix_for(facts['format'])}"
        path.write_bytes(raw)
        try:
            result = inference.extract_one(path)
        except inference.InferenceError as e:
            return jsonify({"ok": False, "error": str(e)}), 503

    return jsonify({
        "ok": True,
        "image": facts,
        "features": result,
        "caption": CAPTION_STATE,
        "narration": NARRATION_STATE,
    })
