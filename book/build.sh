#!/usr/bin/env bash
# Builds the learning guide's mdBook: computes the figures with the library, runs
# the examples for the gallery chapter, builds the book and checks the output
# (KaTeX errors, broken links, figures).
#
#   book/build.sh           generate figures (when stale) and build into book/book/
#   book/build.sh clean     delete the generated figures and the built book
#   book/build.sh serve     generate figures, then `mdbook serve` (extra args passed on)
#   FORCE_FIGURES=1 book/build.sh   re-run every figure program
#
# Requires mdbook 0.5.4 and mdbook-katex 0.10.0 (see book/README.md).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
cmd="${1:-build}"
[ $# -gt 0 ] && shift

figures_out="$here/src/figures"
figures_crate="$here/figures"
bin_dir="$figures_crate/target/release"

if [ "$cmd" = clean ]; then
    rm -rf "$figures_out" "$here/book"
    echo "removed $figures_out and $here/book"
    exit 0
fi

missing=0
for tool in mdbook mdbook-katex python3 cargo; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: $tool not found on PATH" >&2
        missing=1
    fi
done
if [ "$missing" = 1 ]; then
    cat >&2 <<'EOF'
Install the book tools with:
    cargo install mdbook --version 0.5.4 --locked
    cargo install mdbook-katex --version 0.10.0
EOF
    exit 1
fi

# 1. Figures. Each chapter is a binary of the figures crate; a binary is run
#    again only when it is newer than the manifest it wrote last time (or a
#    figure it lists is missing), so an unchanged book rebuilds quickly.
echo "== figures"
cargo build --release --quiet --manifest-path "$figures_crate/Cargo.toml" --bins
mkdir -p "$figures_out"
for src in "$figures_crate"/src/chapters/ch*.rs; do
    name="$(basename "$src" .rs)"
    bin="$bin_dir/$name"
    manifest="$figures_out/$name.json"
    stale=0
    if [ -n "${FORCE_FIGURES:-}" ] || [ ! -f "$manifest" ] || [ "$bin" -nt "$manifest" ]; then
        stale=1
    else
        for f in $(python3 -c 'import json,sys; print(" ".join(x["file"] for x in json.load(open(sys.argv[1]))["figures"]))' "$manifest"); do
            [ -s "$figures_out/$f" ] || stale=1
        done
    fi
    if [ "$stale" = 1 ]; then
        GUIDE_FIGURES_DIR="$figures_out" "$bin"
    else
        echo "$name: up to date"
    fi
done

# 2. Examples for the gallery chapter: all must compile and run.
echo "== gallery examples"
python3 "$here/gallery_outputs.py"

# 3. Book.
if [ "$cmd" = serve ]; then
    exec mdbook serve "$@"
fi
echo "== mdbook build"
log="$(mktemp)"
trap 'rm -f "$log"' EXIT
if ! mdbook build >"$log" 2>&1; then
    cat "$log" >&2
    echo "error: mdbook build failed" >&2
    exit 1
fi
grep -v ' INFO ' "$log" || true

# 4. Output checks: KaTeX errors, broken relative links, empty figures.
echo "== checks"
python3 "$here/check_output.py" "$here/book"
echo "book written to $here/book/index.html"
