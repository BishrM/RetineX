"""Upload validation. The browser-supplied filename and MIME type are treated as hostile.

What is actually checked, in order: a file was sent; its byte length is within the limit; Pillow can
identify it as one of the allowed formats; its pixel dimensions are sane; it decodes fully. The
declared content type and the filename extension are only ever used for messages, never for trust.
"""
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Pillow raises DecompressionBombError, which is NOT an OSError or ValueError, so it has to be
# named explicitly or it escapes as a 500. Its own ceiling is ~89M pixels; MAX_PIXELS below is
# stricter, so for any image Pillow will open, our dimension check fires first.
BOMB = Image.DecompressionBombError

MAX_BYTES = 8 * 1024 * 1024          # 8 MB
MIN_SIDE = 32                        # smaller than this cannot survive a 224x224 resize usefully
MAX_SIDE = 10_000                    # guards against decompression-bomb style dimensions
MAX_PIXELS = 40_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class UploadError(ValueError):
    """A rejected upload. The message tells the person what to do next."""


def safe_name(filename, fallback="upload"):
    """A filesystem-safe stem taken from the client name, used for display and the temp file only."""
    stem = Path(filename or "").name
    keep = "".join(c for c in stem if c.isalnum() or c in "-_. ")
    keep = keep.strip().strip(".")
    return (keep or fallback)[:120]


def validate_bytes(raw, declared_name=None, max_bytes=MAX_BYTES):
    """Validate raw upload bytes. Returns a dict of measured facts, or raises UploadError."""
    if not raw:
        raise UploadError("That file was empty. Choose an image file and try again.")
    if len(raw) > max_bytes:
        raise UploadError(
            f"That image is {len(raw) / 1_048_576:.1f} MB, over the "
            f"{max_bytes // 1_048_576} MB limit. Try a smaller file.")

    import io

    # identify before decoding: verify() reads structure without committing to the full bitmap
    try:
        probe = Image.open(io.BytesIO(raw))
        fmt = probe.format
        size = probe.size
        probe.verify()
    except BOMB as e:
        raise UploadError(
            "That image declares more pixels than this application will decode. "
            f"Resize it below {MAX_SIDE} pixels on each side."
        ) from e
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise UploadError(
            "That file could not be read as an image. Upload a JPEG, PNG, WEBP or BMP file."
        ) from e

    if fmt not in ALLOWED_FORMATS:
        raise UploadError(
            f"{fmt or 'That format'} is not supported. Upload a JPEG, PNG, WEBP or BMP file.")
    w, h = size
    if w < MIN_SIDE or h < MIN_SIDE:
        raise UploadError(
            f"That image is {w} by {h} pixels. Both sides need to be at least {MIN_SIDE} pixels.")
    if w > MAX_SIDE or h > MAX_SIDE or w * h > MAX_PIXELS:
        raise UploadError(
            f"That image is {w} by {h} pixels, which is too large to process. "
            f"Resize it below {MAX_SIDE} pixels on each side.")

    # verify() leaves the file unusable, so decode from a fresh buffer to prove it really loads
    try:
        full = Image.open(io.BytesIO(raw))
        full.load()
    except BOMB as e:
        raise UploadError(
            "That image expands to more pixels than this application will decode. "
            f"Resize it below {MAX_SIDE} pixels on each side."
        ) from e
    except (OSError, ValueError) as e:
        raise UploadError(
            "That image is incomplete or damaged and could not be decoded. Try re-exporting it."
        ) from e

    name = safe_name(declared_name)
    ext = Path(name).suffix.lower()
    return {
        "format": fmt,                   # from the bytes, not the filename
        "width": w,
        "height": h,
        "mode": full.mode,
        "bytes": len(raw),
        "display_name": name,
        # the extension is recorded for display and is explicitly not trusted for validation
        "declared_extension": ext if ext in ALLOWED_EXTENSIONS else None,
    }


def suffix_for(fmt):
    """The extension to give the temp file, chosen from the detected format."""
    return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}.get(fmt, ".img")
