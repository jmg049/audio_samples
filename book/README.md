# The learning guide as an mdBook

Published at https://jmg049.github.io/audio_samples/.

This directory builds the learning guide (`documentation/guide/`) as an
illustrated [mdBook](https://rust-lang.github.io/mdBook/) with KaTeX maths and
figures computed by `audio_samples` itself.

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Rust toolchain | the crate's (edition 2024) | `rustup` |
| mdbook | 0.5.4 | `cargo install mdbook --version 0.5.4 --locked` |
| mdbook-katex | 0.10.0 | `cargo install mdbook-katex --version 0.10.0` |
| Python | 3.9 or later, standard library only | system package |

Install `mdbook-katex` without `--locked`: its lock file pins
`mdbook-preprocessor` 0.5.3, and the unlocked build picks 0.5.4 to match
`mdbook` 0.5.4 (the locked build works but warns about the mismatch on every
build). mdbook-katex 0.10.0 pre-renders the maths with KaTeX (through the
`katex` crate 0.4.6) at build time; the pages load the KaTeX stylesheet and
fonts from a CDN. The figures crate uses plotters 0.3.7 with its SVG backend
only (no system fonts, no browser).

## Commands

```sh
book/build.sh           # generate the figures (when stale), build book/book/, check the output
book/build.sh serve     # the same, then serve on http://localhost:3000 with live reload
book/build.sh clean     # delete the generated figures and the built book
FORCE_FIGURES=1 book/build.sh   # re-run every figure program
```

The book is written to `book/book/index.html`. `build.sh` fails if:

* a figure program fails (each one asserts the numbers its chapter asserts);
* `mdbook build` fails, which includes a figure marker without a figure;
* the output contains a KaTeX error (`katex-error` or `ParseError`), a
  relative link or image whose target does not exist, or an empty figure
  (`check_output.py`).

The pre-commit hook `book` (`tests/reference/hooks/check_book.sh`) runs the
same build, plus `cargo fmt --check` on the figures crate, when anything under
`book/` or `documentation/guide/` changes. It does not run on changes under
`src/`; run `book/build.sh` after changing code that the figures use.

## Layout

| Path | Contents |
|---|---|
| `book.toml` | mdBook configuration: preprocessors, KaTeX options, no playground |
| `src/SUMMARY.md`, `src/introduction.md` | table of contents and introduction (book only) |
| `src/*.md` symlinks | the chapters, `FOR_INSTRUCTORS.md` and `README.md` of `documentation/guide/`, used in place |
| `preprocessor/guide.py` | mdBook preprocessor: figure markers and intra-doc links |
| `theme/figures.css` | figure and caption style |
| `figures/` | Rust crate that computes and draws the figures |
| `check_output.py` | checks on the built HTML |
| `build.sh` | entry point |

Generated, and ignored by git (`book/.gitignore`): `book/book/` (the built
book), `book/src/figures/` (SVGs and manifests) and `book/figures/target/`.

## How the chapters become pages

The chapters are not copied. `src/` holds symlinks to the files in
`documentation/guide/`, and the preprocessor `preprocessor/guide.py` changes
only two things on the way through:

* **Intra-doc links.** A link whose target is a Rust path, such as
  ``[`resample`](crate::resample)`` or
  ``[`MelParams`](spectrograms::MelParams)``, becomes a docs.rs search link
  (`https://docs.rs/audio_samples/latest/audio_samples/?search=resample`),
  which finds the item whatever its kind. Links between chapters (`02_quantisation_dither.md`)
  are ordinary mdBook links.
* **Figure markers** (below).

The chapter maths is plain LaTeX (`$...$`, `$$...$$`). mdbook-katex renders it
before the Markdown parser sees it, so `\,`, `\{`, `_` and `*` inside maths
need no escaping. Rust code blocks hide their `# ` setup lines (mdBook's eye
button shows them) and have no run button.

The guide's own `README.md` is part of the book as "Guide index and setup".
mdBook's default `index` preprocessor is disabled in `book.toml` because it
would rename that page to `index.html` and break the chapters' links to
`README.md`.

## Figures and markers

A figure belongs to a place in a chapter, marked by an HTML comment on a line
of its own, with blank lines around it:

```markdown
The frequency axis is folded at ... which maps every frequency into $[0, f_N]$.

<!-- figure: ch01-folding -->

Aliasing occurs in three places:
```

The comment is invisible in rustdoc, on GitHub and in Markdown previews. In
the book the preprocessor replaces it with the image `figures/ch01-folding.svg`
and its numbered caption ("Figure 1.1. ..."). A marker whose figure does not
exist fails the build; a figure that no marker uses is reported as a warning.

The figures crate (`figures/`, its own workspace, depending on `audio_samples`
by path with the `educational` and `full_no_plotting` features) has one binary
per chapter, `figures/src/chapters/chNN.rs`. Each binary:

1. runs the computations of its chapter with the same library calls and
   parameters as the chapter's code;
2. asserts the numbers the chapter asserts that the figure shows (an alias
   frequency, a -3 dB point, a detected pitch, a tempo...), so a figure cannot
   drift from its text: if the library changes, the figure program fails;
3. draws each figure with plotters (`Figures::svg` in `figures/src/lib.rs`)
   and writes `book/src/figures/<id>.svg`;
4. writes `book/src/figures/chNN.json`, the manifest with each figure's id,
   file, size and caption, which the preprocessor reads.

To add a figure: draw it in the chapter's binary with an id `chNN-<name>` and a
plain-text caption (no maths, no Markdown), put `<!-- figure: chNN-<name> -->`
where it belongs in the chapter, and run `book/build.sh`.

Run one chapter's figures on their own with

```sh
cd book/figures
cargo run --release --bin ch01
```

Hand-written formulas in the figure programs are used only for reference
overlays that the chapters derive by hand (for example the continuous sine
through the samples in chapter 01); every value attributed to the library
comes from the library.
