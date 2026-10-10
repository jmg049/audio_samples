#!/usr/bin/env python3
"""mdBook preprocessor for the learning guide.

The chapters in ``documentation/guide/`` are shared with rustdoc, GitHub and
plain Markdown viewers, so they carry two things only rustdoc understands or
only the book can use:

* ``<!-- figure: <id> -->`` markers, on a line of their own. Each is replaced
  by the figure ``figures/<id>.svg`` and its caption, read from the manifests
  (``src/figures/*.json``) that the figures crate (``book/figures``) writes.
  A marker whose figure is missing is an error; figures that no marker uses
  are reported on stderr.
* rustdoc intra-doc links such as ``[`resample`](crate::resample)`` or
  ``[`MelParams`](spectrograms::MelParams)``. They become docs.rs search
  links, which resolve to the item whatever its kind (function, method,
  struct, trait...).

Nothing else in the chapters is changed. Fenced code blocks are left alone.

Protocol: ``guide.py supports <renderer>`` exits 0; otherwise read
``[context, book]`` as JSON on stdin and write the book JSON to stdout.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

MARKER = re.compile(r"^[ \t]*<!--\s*figure:\s*([A-Za-z0-9_-]+)\s*-->[ \t]*$")
FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
# A Markdown link target that is a Rust path: `crate::a::b`, `spectrograms::X`.
RUST_PATH_LINK = re.compile(r"\]\(((?:[A-Za-z_][A-Za-z0-9_]*)(?:::[A-Za-z_][A-Za-z0-9_]*)+)\)")
CHAPTER_NUMBER = re.compile(r"^(\d\d)_")

DOCS_RS = "https://docs.rs/{krate}/latest/{krate}/?search={query}"


def docs_rs_url(path: str) -> str:
    parts = path.split("::")
    if parts[0] in ("crate", "audio_samples"):
        krate, rest = "audio_samples", parts[1:]
    else:
        krate, rest = parts[0], parts[1:]
    return DOCS_RS.format(krate=krate, query="::".join(rest))


def load_manifests(figures_dir: Path) -> dict[str, dict]:
    figures: dict[str, dict] = {}
    if not figures_dir.is_dir():
        return figures
    for manifest in sorted(figures_dir.glob("*.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for fig in data.get("figures", []):
            figures[fig["id"]] = fig
    return figures


def figure_html(fig: dict, number: str) -> str:
    caption = html.escape(fig["caption"], quote=False).replace("$", "&#36;")
    alt = html.escape(fig.get("alt", fig["caption"]), quote=True).replace("$", "&#36;")
    lines = [
        f'<figure class="guide-figure" id="figure-{fig["id"]}">',
        f'<img src="figures/{fig["file"]}" alt="{alt}" loading="lazy"'
        + (f' width="{fig["width"]}" height="{fig["height"]}"' if "width" in fig else "")
        + ">",
        f'<figcaption><span class="figure-number">Figure {number}.</span> {caption}',
    ]
    if fig.get("interactive"):
        lines.append(
            f' <a class="figure-interactive" href="figures/{fig["interactive"]}">'
            "Interactive version</a>"
        )
    lines.append("</figcaption>")
    lines.append("</figure>")
    # Blank lines around the block keep it a raw HTML block for the parser.
    return "\n" + "\n".join(lines) + "\n"


def process(content: str, source: str, figures: dict, used: set, errors: list) -> str:
    m = CHAPTER_NUMBER.match(Path(source).name)
    chapter = str(int(m.group(1))) if m else None
    count = 0
    out = []
    fence: str | None = None
    for line in content.split("\n"):
        f = FENCE.match(line)
        if fence is not None:
            if f and f.group(1)[0] == fence[0] and len(f.group(1)) >= len(fence):
                fence = None
            out.append(line)
            continue
        if f:
            fence = f.group(1)
            out.append(line)
            continue
        marker = MARKER.match(line)
        if marker:
            fid = marker.group(1)
            fig = figures.get(fid)
            if fig is None:
                errors.append(f"{source}: figure marker '{fid}' has no generated figure")
                out.append(line)
                continue
            count += 1
            used.add(fid)
            number = f"{chapter}.{count}" if chapter else str(count)
            out.append(figure_html(fig, number))
            continue
        out.append(RUST_PATH_LINK.sub(lambda mm: "](" + docs_rs_url(mm.group(1)) + ")", line))
    return "\n".join(out)


def walk(items: list, fn) -> None:
    for item in items:
        if isinstance(item, dict) and "Chapter" in item:
            ch = item["Chapter"]
            fn(ch)
            walk(ch.get("sub_items", []), fn)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "supports":
        return 0
    context, book = json.load(sys.stdin)
    root = Path(context["root"])
    src = root / context["config"]["book"].get("src", "src")
    figures = load_manifests(src / "figures")
    used: set[str] = set()
    errors: list[str] = []

    def fix(ch: dict) -> None:
        source = ch.get("source_path") or ch.get("path") or ch["name"]
        ch["content"] = process(ch["content"], source, figures, used, errors)

    walk(book["items"], fix)

    for fid in sorted(set(figures) - used):
        print(f"[guide preprocessor] warning: figure '{fid}' is not used by any marker", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"[guide preprocessor] error: {e}", file=sys.stderr)
        print(
            "[guide preprocessor] generate the figures first (book/build.sh), "
            "or fix the marker id",
            file=sys.stderr,
        )
        return 1
    json.dump(book, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
