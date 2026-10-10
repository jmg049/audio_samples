//! The learning guide is written with plain LaTeX maths, and `build.rs` escapes
//! it for rustdoc (`build_support/rustdoc_math.rs`). These tests render the
//! escaped chapters with pulldown-cmark, the CommonMark parser rustdoc uses,
//! with rustdoc's Markdown options, and check that the text KaTeX will see is
//! exactly the source LaTeX.
//!
//! `tests/reference/hooks/check_guide_rustdoc_math.py` makes the same check on
//! the HTML that `cargo doc` actually produces.

#[path = "../build_support/rustdoc_math.rs"]
#[allow(dead_code)]
mod rustdoc_math;

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use rustdoc_math::{MathSpan, escape_math, math_spans};
use std::path::PathBuf;

/// The options rustdoc renders doc comments with.
fn rustdoc_options() -> Options {
    Options::ENABLE_TABLES
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_SMART_PUNCTUATION
}

/// The visible prose text of a rendered document: code, code blocks and raw
/// HTML removed, soft breaks as newlines, `\0` between blocks so that no span
/// can be matched across a block boundary.
fn rendered_text(markdown: &str) -> String {
    let mut out = String::new();
    let mut in_code_block = false;
    for event in Parser::new_ext(markdown, rustdoc_options()) {
        match event {
            Event::Start(Tag::CodeBlock(_)) => in_code_block = true,
            Event::End(TagEnd::CodeBlock) => in_code_block = false,
            Event::Text(t) if !in_code_block => out.push_str(&t),
            Event::SoftBreak => out.push('\n'),
            Event::HardBreak => out.push_str("<br>"),
            Event::Code(_) | Event::End(_) => out.push('\0'),
            _ => {}
        }
    }
    out
}

/// Collapse each run of whitespace to one space. Markdown drops the leading
/// whitespace of continuation lines; to TeX all whitespace runs are the same.
fn squash(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Check that each source span appears, delimited and in order, in `text`,
/// and that `text` holds no other `$`. Whitespace runs are compared as one
/// space.
fn assert_round_trip(name: &str, spans: &[MathSpan], text: &str) {
    let text = &text.split('\0').map(squash).collect::<Vec<_>>().join("\0");
    let mut cursor = 0;
    for (n, span) in spans.iter().enumerate() {
        let delim = if span.display { "$$" } else { "$" };
        let lead = if span.tex.starts_with(char::is_whitespace) {
            " "
        } else {
            ""
        };
        let trail = if span.tex.ends_with(char::is_whitespace) {
            " "
        } else {
            ""
        };
        let needle = format!("{delim}{lead}{}{trail}{delim}", squash(&span.tex));
        let found = text[cursor..].find(&needle).unwrap_or_else(|| {
            panic!(
                "{name}: maths span {n} did not survive Markdown rendering\n  source: {needle}\n  rendered text from here: {:?}",
                &text[cursor..text.len().min(cursor + 200)]
            )
        });
        let between = &text[cursor..cursor + found];
        assert!(
            !between.contains('$'),
            "{name}: stray `$` before maths span {n}: {between:?}"
        );
        cursor += found + needle.len();
    }
    assert!(
        !text[cursor..].contains('$'),
        "{name}: stray `$` after the last maths span"
    );
}

fn guide_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("documentation/guide")
}

#[test]
fn guide_maths_survives_rustdoc_markdown() {
    let mut files = 0;
    let mut total = 0;
    for entry in std::fs::read_dir(guide_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let source = std::fs::read_to_string(&path).unwrap();
        let spans = math_spans(&source).unwrap_or_else(|e| panic!("{name}: {e}"));
        let escaped = escape_math(&source).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_round_trip(&name, &spans, &rendered_text(&escaped));
        files += 1;
        total += spans.len();
    }
    assert!(
        files >= 16,
        "expected the 14 chapters, README and FOR_INSTRUCTORS"
    );
    assert!(total > 500, "suspiciously few maths spans: {total}");
}

#[test]
fn awkward_constructs_round_trip() {
    let source = r#"Inline $x[n] = x_a(nT)$ and $a_1 * b_2 * c$, $\{0, 1\}$, $f'(x)$,
$a -- b ... c$, $|X[k]|$ and $\#\%\&$, $[0, 1)$ with $\mathrm{sinc}\!\left(\frac{t}{T}\right)$.

$$\sum_n x[n] \, \mathrm{sinc}\!\left(u\right) \;\; \text{(ok)} \quad <a> \& b$$

$$
\begin{aligned}
y[n] &= b_0 x[n] + b_1 x[n-1] \\
     &- a_1 y[n-1] \\[2pt]
z &= *a* + _b_
\end{aligned}
$$

| col | maths |
|-----|-------|
| a   | $\lvert H(e^{j\omega}) \rvert$ |

Code such as `code with $x$` is left alone.
<!-- figure: example $not maths$ -->
"#;
    let spans = math_spans(source).unwrap();
    assert_eq!(spans.len(), 12, "{spans:#?}");
    let escaped = escape_math(source).unwrap();
    assert_round_trip("awkward", &spans, &rendered_text(&escaped));
    // Unescaped, Markdown would have damaged it.
    let raw = rendered_text(source);
    assert!(!raw.contains(r"\{0, 1\}"));
}

#[test]
fn code_is_left_alone() {
    let source = "```rust\nlet s = \"$x_1$\";\n```\n\n~~~\n$a_b$\n~~~\n\n`$c_d$` $e_f$\n";
    let escaped = escape_math(source).unwrap();
    assert_eq!(
        escaped,
        "```rust\nlet s = \"$x_1$\";\n```\n\n~~~\n$a_b$\n~~~\n\n`$c_d$` $e\\_f$\n"
    );
}

#[test]
fn malformed_maths_is_rejected() {
    assert!(escape_math("$$x\n\ny$$").is_err());
    assert!(escape_math("a $$x").is_err());
}
