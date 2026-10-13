#!/usr/bin/env python
"""Generate the served brand exports from the supplied artwork.

Two things are produced, and neither supplied master is ever written to: the square favicon
variants, and a flush-cropped copy of the logo for the navigation bar.

design.md, Brand Assets: the supplied favicon artwork is not square, so it is not usable as a
favicon as supplied. A square export with transparent padding is needed.

Two variants are produced from one master, because a favicon sits in browser chrome rather than on
the page, and no single value is visible on both light and dark chrome:

  favicon.png       Pine Green #324A3D, for light chrome. Also the fallback for browsers that
                    ignore `media` on a link element, and the Apple touch icon.
  favicon-dark.png  the cream artwork exactly as supplied, for dark chrome.

The green variant replaces the fill colour only: every alpha value, and therefore the shape and its
antialiasing, is taken unchanged from the master. The mark is a single flat colour, so nothing is
lost. The artwork is never redrawn, and favicon-source.png is never written to.

The logo is supplied with 34px of fully transparent padding on its left edge, which pushed the
visible wordmark about 7px right of the content column and made the navigation read as misaligned
against the page below it. The export crops away fully transparent pixels and nothing else, so no
visible pixel, proportion or colour changes; the alignment is then correct from the asset itself
rather than from a negative margin in a stylesheet.

Usage: python scripts/build_favicon.py [--check]
"""
import argparse
import io
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "app" / "static" / "brand"
# The supplied artwork, at whatever aspect ratio it was exported. Replace this file to change the
# icon, then re-run this script. The served favicons are generated from it and are never hand-edited.
SOURCE = BRAND / "favicon-source.png"

PINE_GREEN = (0x32, 0x4A, 0x3D)
# filename -> fill colour, or None to keep the master's own colour
VARIANTS = {
    "favicon.png": PINE_GREEN,
    "favicon-dark.png": None,
}

SIZE = 512          # one master size; browsers downscale it for the 16 and 32px tab slots
COVERAGE = 0.82     # share of the square the artwork fills, leaving an even transparent margin

# The supplied logo, and the flush-cropped export the navigation actually renders.
LOGO_SOURCE = BRAND / "retinex-logo.png"
LOGO_OUT = BRAND / "retinex-logo-flush.png"


def render(fill=None, source=SOURCE, size=SIZE, coverage=COVERAGE):
    """Square, transparent, optionally refilled. Only crops, pads, recolours and resizes."""
    im = Image.open(source).convert("RGBA")
    box = im.getbbox()                      # the artwork's own bounds, ignoring transparency
    if box is None:
        raise ValueError(f"{source.name} is fully transparent; nothing to export")
    art = im.crop(box)
    if fill is not None:
        # keep every alpha value, replace only the colour channels
        solid = Image.new("RGBA", art.size, (*fill, 255))
        solid.putalpha(art.getchannel("A"))
        art = solid
    side = round(max(art.size) / coverage)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(art, ((side - art.width) // 2, (side - art.height) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def render_logo(source=LOGO_SOURCE):
    """The supplied logo with its fully transparent margin removed. Nothing else changes.

    The bounds come from the alpha channel alone, so a padding pixel that carries a colour at
    zero opacity cannot keep the margin alive.
    """
    im = Image.open(source).convert("RGBA")
    box = im.getchannel("A").getbbox()
    if box is None:
        raise ValueError(f"{source.name} is fully transparent; nothing to export")
    return im.crop(box)


def encode(image):
    buf = io.BytesIO()
    image.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def main(argv=None):
    p = argparse.ArgumentParser(prog="build_favicon")
    p.add_argument("--check", action="store_true", help="fail if a committed file is stale")
    a = p.parse_args(argv)
    if not SOURCE.is_file():
        print(f"{SOURCE.relative_to(ROOT)} not found; put the supplied artwork there")
        return 1

    if not LOGO_SOURCE.is_file():
        print(f"{LOGO_SOURCE.relative_to(ROOT)} not found; put the supplied logo there")
        return 1

    jobs = [(BRAND / name, encode(render(fill)),
             f"{SIZE}x{SIZE} RGBA  " +
             (f"#{fill[0]:02X}{fill[1]:02X}{fill[2]:02X}" if fill else "as supplied"))
            for name, fill in VARIANTS.items()]
    logo = render_logo()
    jobs.append((LOGO_OUT, encode(logo), f"{logo.width}x{logo.height} RGBA  flush cropped"))

    stale = []
    for out, data, note in jobs:
        if a.check:
            if not out.is_file() or out.read_bytes() != data:
                stale.append(out.name)
            continue
        out.write_bytes(data)
        print(f"wrote {out.relative_to(ROOT)}  {note}  {len(data):,} bytes")

    if a.check:
        if stale:
            print(f"stale: {', '.join(stale)}; run python scripts/build_favicon.py")
            return 1
        print(f"{', '.join(o.name for o, _, _ in jobs)} are up to date with their masters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
