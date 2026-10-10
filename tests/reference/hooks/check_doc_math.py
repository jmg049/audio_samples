#!/usr/bin/env python3
"""Find LaTeX in rustdoc/markdown maths that markdown will corrupt.

Rustdoc renders doc comments and the guide chapters as CommonMark *before*
KaTeX sees the page. CommonMark treats a backslash followed by ASCII
punctuation as an escape and drops the backslash, so inside `$...$` or
`$$...$$`:

    \\{  ->  {      (\\left\\{ becomes \\left{, invalid LaTeX)
    \\,  ->  ,      (a thin space turns into a comma)
    \\;  \\!  \\|  \\_  \\%  ...

Write the backslash twice (`\\\\,`, `\\\\{`) so that one survives markdown.
`\\[` and `\\]` are NOT reported: they are used on purpose to render literal
brackets (e.g. x\\[n\\]).

Scans documentation/guide/*.md and the `//!` / `///` comments of src/**/*.rs.
Fenced code blocks and inline code spans are skipped. Prints one line per
finding as `path:line: <escape> in <maths span>` and exits 1 if any are found.

Usage: python3 tests/reference/hooks/check_doc_math.py [repo_root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ASCII_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
ALLOWED = {"[", "]"}

DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)")
INLINE_CODE = re.compile(r"(`+)(?:.+?)\1", re.S)


def blank(text: str, start: int, end: int) -> str:
    """Replace text[start:end] by spaces, keeping newlines (offsets stay valid)."""
    seg = "".join("\n" if c == "\n" else " " for c in text[start:end])
    return text[:start] + seg + text[end:]


def strip_code(text: str) -> str:
    """Blank out fenced code blocks and inline code spans."""
    lines = text.split("\n")
    out = []
    fence = None
    for line in lines:
        s = line.lstrip()
        m = re.match(r"(`{3,}|~{3,})", s)
        if fence is None and m:
            fence = m.group(1)[0] * len(m.group(1))
            out.append(" " * len(line))
            continue
        if fence is not None:
            if s.startswith(fence):
                fence = None
            out.append(" " * len(line))
            continue
        out.append(line)
    text = "\n".join(out)
    for m in reversed(list(INLINE_CODE.finditer(text))):
        text = blank(text, m.start(), m.end())
    return text


def bad_escapes(span: str):
    """Yield (offset, escape) for single (odd-run) backslashes before punctuation."""
    i = 0
    while i < len(span):
        if span[i] != "\\":
            i += 1
            continue
        j = i
        while j < len(span) and span[j] == "\\":
            j += 1
        run = j - i
        if run % 2 == 1 and j < len(span):
            c = span[j]
            if c in ASCII_PUNCT and c not in ALLOWED:
                yield j - 1, "\\" + c
        i = j


def scan_markdown(text: str):
    """Yield (offset, escape, span) for every finding in markdown `text`."""
    clean = strip_code(text)
    for m in DISPLAY.finditer(clean):
        for off, esc in bad_escapes(m.group(1)):
            yield m.start(1) + off, esc, m.group(0)
    no_display = clean
    for m in reversed(list(DISPLAY.finditer(clean))):
        no_display = blank(no_display, m.start(), m.end())
    for m in INLINE.finditer(no_display):
        for off, esc in bad_escapes(m.group(1)):
            yield m.start(1) + off, esc, m.group(0)


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def short(span: str, width: int = 90) -> str:
    span = " ".join(span.split())
    return span if len(span) <= width else span[: width - 3] + "..."


def check_md(path: Path, root: Path, out: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for off, esc, span in scan_markdown(text):
        out.append(f"{path.relative_to(root)}:{line_of(text, off)}: {esc} in {short(span)}")


DOC_LINE = re.compile(r"^\s*//(?:!|/(?!/))( ?)(.*)$")


def check_rs(path: Path, root: Path, out: list[str]) -> None:
    lines = path.read_text(encoding="utf-8").split("\n")
    block: list[tuple[int, str]] = []

    def flush():
        if not block:
            return
        text = "\n".join(t for _, t in block)
        for off, esc, span in scan_markdown(text):
            src_line = block[line_of(text, off) - 1][0]
            out.append(f"{path.relative_to(root)}:{src_line}: {esc} in {short(span)}")
        block.clear()

    for n, line in enumerate(lines, 1):
        m = DOC_LINE.match(line)
        if m:
            block.append((n, m.group(2)))
        else:
            flush()
    flush()


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[3])
    out: list[str] = []
    for p in sorted((root / "documentation" / "guide").glob("*.md")):
        check_md(p, root, out)
    for p in sorted((root / "src").rglob("*.rs")):
        check_rs(p, root, out)
    for line in out:
        print(line)
    print(f"{len(out)} markdown-eaten escape(s) in maths spans", file=sys.stderr)
    return 1 if out else 0


if __name__ == "__main__":
    sys.exit(main())
