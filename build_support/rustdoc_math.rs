//! Escape the maths in a Markdown document so that rustdoc passes it through
//! to KaTeX unchanged.
//!
//! The learning guide (`documentation/guide/*.md`) is written in plain Markdown
//! with standard LaTeX maths (`$...$` inline, `$$...$$` display), which renders
//! in Markdown previews, on GitHub and in the mdBook. Rustdoc does not know about
//! maths: it parses the page as CommonMark first, and CommonMark treats a
//! backslash before ASCII punctuation as an escape, `_` and `*` as emphasis,
//! `[...]` as a link, `'` and `--` as smart punctuation, and so on. KaTeX then
//! sees the damaged text.
//!
//! [`escape_math`] backslash-escapes every ASCII punctuation character inside
//! every maths span. CommonMark turns an escaped punctuation character back into
//! that literal character and never treats it as markup, so the text that
//! reaches KaTeX in the rendered page is exactly the source LaTeX.
//!
//! Fenced code blocks, inline code spans and HTML comments are copied
//! unchanged, so doctests are unaffected. A `$` preceded by a backslash is not a
//! delimiter. The delimiter rules for inline maths follow the usual Markdown
//! maths convention (markdown-it-katex, pandoc): the opening `$` is not followed
//! by whitespace, the closing `$` is not preceded by whitespace and not followed
//! by a digit.
//!
//! This file is used by `build.rs` and by `tests/guide_math_roundtrip.rs`.

/// An error in the maths markup of a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MathError {
    /// 1-based line of the opening delimiter.
    pub line: usize,
    /// What is wrong.
    pub message: String,
}

impl std::fmt::Display for MathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

/// One maths span found in a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MathSpan {
    /// `true` for `$$...$$`, `false` for `$...$`.
    pub display: bool,
    /// The LaTeX between the delimiters.
    pub tex: String,
}

/// Escape every maths span of `markdown` for rustdoc. Returns the new document.
pub fn escape_math(markdown: &str) -> Result<String, MathError> {
    let mut out = String::with_capacity(markdown.len() + markdown.len() / 8);
    walk(markdown, &mut |piece| match piece {
        Piece::Text(t) => out.push_str(t),
        Piece::Math { display, tex } => {
            let delim = if display { "$$" } else { "$" };
            out.push_str(delim);
            for c in tex.chars() {
                if c.is_ascii_punctuation() {
                    out.push('\\');
                }
                out.push(c);
            }
            out.push_str(delim);
        }
    })?;
    Ok(out)
}

/// List the maths spans of `markdown`, in document order.
pub fn math_spans(markdown: &str) -> Result<Vec<MathSpan>, MathError> {
    let mut spans = Vec::new();
    walk(markdown, &mut |piece| {
        if let Piece::Math { display, tex } = piece {
            spans.push(MathSpan {
                display,
                tex: tex.to_string(),
            });
        }
    })?;
    Ok(spans)
}

enum Piece<'a> {
    Text(&'a str),
    Math { display: bool, tex: &'a str },
}

/// Split `src` into verbatim text and maths spans.
fn walk<'a>(src: &'a str, emit: &mut dyn FnMut(Piece<'a>)) -> Result<(), MathError> {
    // First split off fenced code blocks, line by line.
    let mut prose_start = 0;
    let mut pos = 0;
    let mut fence: Option<(u8, usize)> = None;
    for line in src.split_inclusive('\n') {
        let line_start = pos;
        pos += line.len();
        let trimmed = line.trim_start_matches(' ');
        let indent = line.len() - trimmed.len();
        let run_char = trimmed.as_bytes().first().copied();
        let run_len = match run_char {
            Some(c @ (b'`' | b'~')) => trimmed.bytes().take_while(|&b| b == c).count(),
            _ => 0,
        };
        match fence {
            None => {
                if indent <= 3 && run_len >= 3 {
                    let c = run_char.unwrap_or(b'`');
                    // A backtick fence's info string may not contain backticks.
                    if c == b'~' || !trimmed[run_len..].contains('`') {
                        prose(src, prose_start, line_start, emit)?;
                        emit(Piece::Text(line));
                        prose_start = pos;
                        fence = Some((c, run_len));
                    }
                }
            }
            Some((c, n)) => {
                emit(Piece::Text(line));
                prose_start = pos;
                let closes = indent <= 3
                    && run_char == Some(c)
                    && run_len >= n
                    && trimmed[run_len..].trim().is_empty();
                if closes {
                    fence = None;
                }
            }
        }
    }
    if fence.is_some() {
        emit(Piece::Text(&src[prose_start..]));
        return Ok(());
    }
    prose(src, prose_start, src.len(), emit)
}

/// Split the prose `src[start..end]` (no fenced code) into text and maths.
fn prose<'a>(
    src: &'a str,
    start: usize,
    end: usize,
    emit: &mut dyn FnMut(Piece<'a>),
) -> Result<(), MathError> {
    let b = src.as_bytes();
    let line_at = |i: usize| src[..i].matches('\n').count() + 1;
    let mut text_start = start;
    let mut i = start;
    while i < end {
        match b[i] {
            b'\\' => i += 2,
            b'`' => {
                let n = b[i..end].iter().take_while(|&&c| c == b'`').count();
                let ticks = &src[i..i + n];
                // A code span ends at the next run of exactly n backticks.
                let mut j = i + n;
                let mut close = None;
                while let Some(k) = src[j..end].find(ticks) {
                    let at = j + k;
                    let m = b[at..end].iter().take_while(|&&c| c == b'`').count();
                    if m == n {
                        close = Some(at + n);
                        break;
                    }
                    j = at + m;
                }
                i = close.unwrap_or(i + n);
            }
            b'<' if src[i..end].starts_with("<!--") => {
                i = match src[i + 4..end].find("-->") {
                    Some(k) => i + 4 + k + 3,
                    None => end,
                };
            }
            b'$' => {
                let display = b.get(i + 1) == Some(&b'$') && i + 1 < end;
                let open = if display { 2 } else { 1 };
                let body = i + open;
                let close = if display {
                    src[body..end].find("$$").map(|k| body + k)
                } else {
                    inline_close(src, body, end)
                };
                let Some(close) = close else {
                    if display {
                        return Err(MathError {
                            line: line_at(i),
                            message: "unclosed `$$`".into(),
                        });
                    }
                    i += 1;
                    continue;
                };
                let tex = &src[body..close];
                let lines: Vec<&str> = tex.split('\n').collect();
                let inner = lines.get(1..lines.len().saturating_sub(1)).unwrap_or(&[]);
                if inner.iter().any(|l| l.trim().is_empty()) {
                    return Err(MathError {
                        line: line_at(i),
                        message: "blank line inside a maths span (Markdown would end the paragraph there)".into(),
                    });
                }
                if !display && (tex.is_empty() || tex.starts_with(char::is_whitespace)) {
                    i += 1;
                    continue;
                }
                if text_start < i {
                    emit(Piece::Text(&src[text_start..i]));
                }
                emit(Piece::Math { display, tex });
                i = close + open;
                text_start = i;
            }
            _ => i += 1,
        }
    }
    if text_start < end {
        emit(Piece::Text(&src[text_start..end]));
    }
    Ok(())
}

/// Find the closing `$` of an inline span whose body starts at `body`.
fn inline_close(src: &str, body: usize, end: usize) -> Option<usize> {
    let b = src.as_bytes();
    let mut j = body;
    while j < end {
        match b[j] {
            b'\\' => j += 2,
            b'\n'
                if src[j + 1..end]
                    .trim_start_matches([' ', '\t'])
                    .starts_with('\n') =>
            {
                return None;
            }
            b'$' => {
                let before_ws = j > body && (b[j - 1] as char).is_whitespace();
                let after_digit = b.get(j + 1).is_some_and(u8::is_ascii_digit);
                if j > body && !before_ws && !after_digit {
                    return Some(j);
                }
                return None;
            }
            _ => j += 1,
        }
    }
    None
}
