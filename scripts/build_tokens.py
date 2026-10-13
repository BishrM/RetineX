#!/usr/bin/env python
"""Generate app/static/css/tokens.css from the design.md frontmatter.

design.md Implementation Rule 2: emit the tokens once as CSS custom properties *generated* from the
frontmatter, never transcribed by hand. This script is that generator, and
tests/test_app.py::test_tokens_css_matches_design_md asserts the committed CSS still matches.

Usage: python scripts/build_tokens.py [--check]
  --check  exit non-zero if the committed file is stale, writing nothing
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "design.md"
OUT = ROOT / "app" / "static" / "css" / "tokens.css"
HEADER = """/* GENERATED FILE - do not edit by hand.
 * Source: design.md frontmatter. Regenerate with: python scripts/build_tokens.py
 * design.md is the source of truth; add a token there first, with its measured contrast.
 */
"""


def frontmatter(path=DESIGN):
    """The YAML block between the first two '---' lines of design.md."""
    text = path.read_text(encoding="utf8")
    if not text.startswith("---"):
        raise ValueError(f"{path} does not start with a YAML frontmatter fence")
    end = text.index("\n---", 3)
    return yaml.safe_load(text[3:end])


def _css_value(v):
    return str(v)


def render(fm):
    """Frontmatter -> CSS custom properties. Only primitives; components compose them in CSS."""
    lines = [HEADER, ":root {"]

    lines.append("  /* colors */")
    for name, value in fm["colors"].items():
        lines.append(f"  --color-{name}: {value};")

    lines.append("\n  /* typography */")
    props = {"fontFamily": "family", "fontSize": "size", "fontWeight": "weight",
             "lineHeight": "line-height", "letterSpacing": "letter-spacing"}
    families = {}
    for token, spec in fm["typography"].items():
        for key, suffix in props.items():
            if key in spec:
                value = spec[key]
                if key == "fontFamily":
                    families[value] = True
                    continue  # families are emitted once below as stacks
                lines.append(f"  --type-{token}-{suffix}: {_css_value(value)};")
        lines.append(f"  --type-{token}-family: var(--font-{_family_slug(spec['fontFamily'])});")

    lines.append("\n  /* font stacks: design.md Typography. Avegas Royale is NOT in the repository,")
    lines.append("     so display type renders in the documented fallback. That is expected. */")
    lines.append('  --font-display: "Avegas Royale", "Iowan Old Style", Georgia, serif;')
    lines.append('  --font-body: "Atkinson Hyperlegible Next", "Atkinson Hyperlegible", '
                 "system-ui, sans-serif;")

    lines.append("\n  /* spacing */")
    for name, value in fm["spacing"].items():
        lines.append(f"  --space-{name}: {value};")

    lines.append("\n  /* radius */")
    for name, value in fm["rounded"].items():
        lines.append(f"  --radius-{name}: {value};")

    lines.append("\n  /* motion: design.md Motion. Nothing exceeds 400ms. */")
    lines.append("  --ease: cubic-bezier(0.22, 0.61, 0.36, 1);")
    lines.append("  --dur-hover: 120ms;")
    lines.append("  --dur-disclosure: 200ms;")
    lines.append("  --dur-reveal: 320ms;")
    lines.append("  --dur-page-out: 120ms;")
    lines.append("  --dur-page-in: 260ms;")
    lines.append("\n  /* the single permitted shadow, tinted with Pine Teal */")
    lines.append("  --shadow-overlay: 0 2px 8px rgba(52, 78, 65, 0.10);")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _family_slug(family):
    return "display" if family == "Avegas Royale" else "body"


def main(argv=None):
    p = argparse.ArgumentParser(prog="build_tokens")
    p.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    a = p.parse_args(argv)
    css = render(frontmatter())
    if a.check:
        current = OUT.read_text(encoding="utf8") if OUT.is_file() else ""
        if current != css:
            print(f"{OUT.relative_to(ROOT)} is stale; run python scripts/build_tokens.py")
            return 1
        print(f"{OUT.relative_to(ROOT)} is up to date with design.md")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(css, encoding="utf8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(css.splitlines())} lines) from design.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
