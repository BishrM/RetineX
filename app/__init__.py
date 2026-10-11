"""RetineX Flask application factory.

Serves two surfaces: the Workspace and Home. The heavy model is never touched at import time or at
startup; see app.inference. Caption generation and narration do not exist in this repository and the
templates say so rather than simulating them.
"""
from flask import Flask

from .uploads import MAX_BYTES


def create_app(config=None):
    app = Flask(__name__, static_folder="static", template_folder="templates")
    # Werkzeug rejects a larger body before it reaches a view; uploads.py enforces the same limit
    # again on the bytes it actually receives.
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_BYTES,
        JSON_SORT_KEYS=False,
        TEMPLATES_AUTO_RELOAD=True,
    )
    if config:
        app.config.update(config)

    from .routes import bp

    app.register_blueprint(bp)
    return app
