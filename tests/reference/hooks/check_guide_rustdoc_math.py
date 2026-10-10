#!/usr/bin/env python3
"""Check that the guide's maths reaches KaTeX intact in the rustdoc HTML.

The guide chapters (documentation/guide/*.md) are written with plain LaTeX
maths. build.rs escapes them for rustdoc (build_support/rustdoc_math.rs). This
script reads the HTML that `cargo doc` produced for each chapter and checks
that the maths KaTeX will see there is exactly the maths in the source file:
same spans, same order, same LaTeX (whitespace runs compared as one space,
since Markdown drops the indentation of continuation lines).

Spans are found the way KaTeX's auto-render does it on the page: `$$...$$`
first, then `$...$`, in the text outside <pre>/<code>. On the source side,
fenced code blocks, inline code spans and HTML comments are skipped first.

Usage:
  check_guide_rustdoc_math.py [--build] [repo_root]

--build runs `cargo doc --features educational,full_no_plotting --no-deps`
first. Exits 1 on any mismatch.
"""

from __future__ import annotations

import html.parser
import re
import subprocess
import sys
from pathlib import Path

FEATURES = "educational,full_no_plotting"


def squash(s: str) -> str:
    return " ".join(s.split())


def pair_spans(text: str) -> list[tuple[bool, str]]:
    """Split text into maths spans like KaTeX auto-render ($$ before $)."""
    spans = []
    i = 0
    while True:
        j = text.find("$", i)
        if j < 0:
            return spans
        if text.startswith("$$", j):
            k = text.find("$$", j + 2)
            if k < 0:
                raise ValueError(f"unclosed $$ near {text[j:j + 60]!r}")
            spans.append((True, squash(text[j + 2 : k])))
            i = k + 2
        else:
            k = text.find("$", j + 1)
            if k < 0:
                raise ValueError(f"unclosed $ near {text[j:j + 60]!r}")
            spans.append((False, squash(text[j + 1 : k])))
            i = k + 1


def source_text(md: str) -> str:
    """Markdown with fenced code, inline code and HTML comments removed."""
    out, fence = [], None
    for line in md.split("\n"):
        s = line.lstrip(" ")
        m = re.match(r"(`{3,}|~{3,})", s)
        if fence is None and m and len(line) - len(s) <= 3:
            fence = m.group(1)
            out.append("")
            continue
        if fence is not None:
            if s.startswith(fence) and not s[len(fence) :].strip(fence[0]).strip():
                fence = None
            out.append("")
            continue
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"(`+)(?!`).*?(?<!`)\1(?!`)", " ", text, flags=re.S)
    return text


class TopDoc(html.parser.HTMLParser):
    """Collect the text of the page's top-level docblock, minus code."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_top = False
        self.depth = 0  # div depth inside the top docblock
        self.skip = 0  # inside <pre>/<code>/<script>/<style>
        self.parts: list[str] = []
        self.found = False

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if tag == "details" and "top-doc" in cls.split():
            self.in_top = True
            return
        if not self.in_top:
            return
        if tag == "div":
            if self.depth == 0 and "docblock" in cls.split():
                self.found = True
                self.depth = 1
            elif self.depth:
                self.depth += 1
        elif self.depth and tag in ("pre", "code", "script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if not self.depth:
            return
        if tag == "div":
            self.depth -= 1
            if self.depth == 0:
                self.in_top = False
        elif tag in ("pre", "code", "script", "style"):
            self.skip -= 1
        elif tag in ("p", "li", "td", "th", "h1", "h2", "h3", "h4", "blockquote"):
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self.depth and not self.skip:
            self.parts.append(data)


def chapter_pages(root: Path) -> list[tuple[Path, Path]]:
    """(source .md, rustdoc page) pairs, read from src/guide.rs."""
    guide_rs = (root / "src" / "guide.rs").read_text(encoding="utf-8")
    doc = root / "target" / "doc" / "audio_samples" / "guide"
    pairs = [(root / "documentation/guide/README.md", doc / "index.html")]
    for md, module in re.findall(r'"/guide/([^"]+\.md)"\)\)\]\s*pub mod (\w+)', guide_rs):
        pairs.append((root / "documentation/guide" / md, doc / module / "index.html"))
    return pairs


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--build"]
    root = Path(args[0] if args else Path(__file__).resolve().parents[3])
    if "--build" in sys.argv:
        subprocess.run(
            ["cargo", "doc", "--quiet", "--features", FEATURES, "--no-deps"],
            cwd=root,
            check=True,
        )
    failures, total = 0, 0
    pairs = chapter_pages(root)
    if len(pairs) < 16:
        print(f"expected 16 guide pages, found {len(pairs)} in src/guide.rs")
        return 1
    for md, page in pairs:
        if not page.exists():
            print(f"{page}: missing (run with --build)")
            failures += 1
            continue
        parser = TopDoc()
        parser.feed(page.read_text(encoding="utf-8"))
        if not parser.found:
            print(f"{page}: no top-level docblock found")
            failures += 1
            continue
        try:
            want = pair_spans(source_text(md.read_text(encoding="utf-8")))
            got = pair_spans("".join(parser.parts))
        except ValueError as e:
            print(f"{md.name}: {e}")
            failures += 1
            continue
        total += len(want)
        if got == want:
            continue
        failures += 1
        for n, (w, g) in enumerate(zip(want, got)):
            if w != g:
                print(f"{md.name}: maths span {n} differs\n  source:  {w}\n  rustdoc: {g}")
                break
        else:
            print(f"{md.name}: {len(want)} maths spans in the source, {len(got)} in rustdoc")
    print(f"{len(pairs)} guide pages, {total} maths spans, {failures} page(s) with differences")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
