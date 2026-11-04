#!/usr/bin/env python3
"""Checks the built book (book/book/) after `mdbook build`.

Fails (exit 1) on:
* a KaTeX error (`katex-error` or `ParseError`) in any page;
* a relative link or image in any page whose target file does not exist;
* a figure image that is missing or empty.

Prints the number of pages, rendered maths expressions and figures.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ATTR = re.compile(r'\b(?:href|src)="([^"]*)"')
# `site-url` in book.toml. mdBook writes 404.html with links rooted at it.
SITE_PATH = "/audio_samples/"


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "book")
    pages = sorted(out.glob("*.html"))
    if not pages:
        print(f"check_output: no HTML pages in {out}", file=sys.stderr)
        return 1
    problems: list[str] = []
    n_inline = n_display = n_fig = 0
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for bad in ("katex-error", "ParseError"):
            if bad in text:
                problems.append(f"{page.name}: contains {bad}")
        n_inline += text.count('class="katex"')
        n_display += text.count('class="katex-display"')
        if page.name not in ("print.html",):
            n_fig += text.count('<figure class="guide-figure"')
        for target in ATTR.findall(text):
            parts = urlsplit(target)
            if parts.scheme or target.startswith(("#", "//", "mailto:", "data:")):
                continue
            path = unquote(parts.path)
            if not path:
                continue
            if path.startswith(SITE_PATH):
                dest = (out / path.removeprefix(SITE_PATH)).resolve()
            else:
                dest = (page.parent / path).resolve()
            if not dest.exists():
                problems.append(f"{page.name}: broken link {target}")
            elif path.startswith("figures/") and dest.stat().st_size == 0:
                problems.append(f"{page.name}: empty figure {target}")
    for p in problems:
        print(f"check_output: {p}", file=sys.stderr)
    print(
        f"check_output: {len(pages)} pages, {n_inline} KaTeX expressions "
        f"({n_display} display), {n_fig} figures (print.html excluded)"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
