//! The [`ExplainDoc`] document model and its three renderers.
//!
//! An [`ExplainDoc`] is plain data: a title, a one-paragraph summary, the
//! formulas the code implements, the symbols those formulas use, a list of
//! numbers derived from the caller's settings, optional tables, trade-offs,
//! warnings and cross-references. Keeping the content separate from its
//! presentation lets the same explanation be printed in a terminal
//! ([`ExplainDoc::to_text`]), shown in a Jupyter notebook
//! ([`ExplainDoc::to_markdown`]) or embedded in an HTML page
//! ([`ExplainDoc::to_html`]).

use std::fmt::{self, Display, Write as _};

use super::super::html_escape;

/// Syllabus topic an explanation belongs to.
///
/// Used to group and filter explanations (for example, "show me everything
/// about windowing"). New topics may be added in future releases, so the
/// enum is `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Topic {
    /// Sampling, the sampling theorem and the Nyquist frequency.
    Sampling,
    /// Amplitude quantisation and bit depth.
    Quantisation,
    /// The discrete Fourier transform.
    Dft,
    /// Window functions and spectral leakage.
    Windowing,
    /// The short-time Fourier transform.
    Stft,
    /// Convolution.
    Convolution,
    /// Finite impulse response filters.
    FirFilter,
    /// Infinite impulse response filters.
    IirFilter,
    /// Sample-rate conversion.
    Resampling,
    /// Linear predictive coding.
    Lpc,
    /// Fundamental-frequency (pitch) estimation.
    Pitch,
    /// Formant analysis.
    Formants,
    /// The cepstrum.
    Cepstrum,
    /// The mel scale and mel filterbanks.
    Mel,
    /// Mel-frequency cepstral coefficients.
    Mfcc,
    /// Short-time energy.
    Energy,
    /// Voice activity detection.
    Vad,
    /// Dynamic-range processing (compressors, limiters).
    Dynamics,
    /// Onset detection.
    Onsets,
    /// Tempo and beat tracking.
    Tempo,
    /// Psychoacoustics.
    Psychoacoustics,
    /// Audio coding.
    Codecs,
    /// Equalisation.
    Equalisation,
}

impl Topic {
    /// Human-readable name of the topic.
    ///
    /// # Returns
    ///
    /// A short title-case label, e.g. `"Windowing"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use audio_samples::educational::Topic;
    /// assert_eq!(Topic::Stft.label(), "Short-time Fourier transform");
    /// ```
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Sampling => "Sampling",
            Self::Quantisation => "Quantisation",
            Self::Dft => "Discrete Fourier transform",
            Self::Windowing => "Windowing",
            Self::Stft => "Short-time Fourier transform",
            Self::Convolution => "Convolution",
            Self::FirFilter => "FIR filters",
            Self::IirFilter => "IIR filters",
            Self::Resampling => "Resampling",
            Self::Lpc => "Linear prediction",
            Self::Pitch => "Pitch estimation",
            Self::Formants => "Formants",
            Self::Cepstrum => "Cepstrum",
            Self::Mel => "Mel scale",
            Self::Mfcc => "MFCC",
            Self::Energy => "Energy",
            Self::Vad => "Voice activity detection",
            Self::Dynamics => "Dynamic range",
            Self::Onsets => "Onset detection",
            Self::Tempo => "Tempo",
            Self::Psychoacoustics => "Psychoacoustics",
            Self::Codecs => "Codecs",
            Self::Equalisation => "Equalisation",
        }
    }
}

impl Display for Topic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// One formula, written in LaTeX, with a plain-language caption.
///
/// ## Purpose
/// Records an equation that the explained code evaluates.
///
/// ## Intended Usage
/// Built by explainers; read by renderers.
///
/// ## Invariants
/// `latex` is a math-mode fragment without surrounding `$` delimiters.
///
/// ## Assumptions
/// The fragment uses the LaTeX subset understood by both MathJax and the
/// `term-maths` terminal renderer where possible; the text renderer falls
/// back to the raw source otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Formula {
    /// LaTeX source, math mode, no delimiters.
    pub latex: String,
    /// What the formula computes, in words.
    pub caption: String,
}

/// A symbol used in the formulas, with its meaning and (optionally) the value
/// it takes for the explained configuration.
///
/// ## Purpose
/// The "legend" of the formulas.
///
/// ## Intended Usage
/// Built by explainers; read by renderers.
///
/// ## Invariants
/// `symbol` is LaTeX (math mode, no delimiters).
///
/// ## Assumptions
/// `value` is already formatted; it is display text, not data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    /// LaTeX for the symbol, e.g. `f_s`.
    pub symbol: String,
    /// Meaning of the symbol.
    pub meaning: String,
    /// Value for this configuration, if known.
    pub value: Option<String>,
    /// Unit of `value`, if any.
    pub unit: Option<String>,
}

/// A number derived from the configuration (and context), with the formula
/// used to obtain it.
///
/// ## Purpose
/// Turns symbols into concrete numbers: "Δf = 21.5 Hz".
///
/// ## Intended Usage
/// Read `value` programmatically (it is the exact `f64` the explainer
/// computed) or let a renderer format it.
///
/// ## Invariants
/// `value` is finite. Explainers skip a quantity rather than store NaN.
///
/// ## Assumptions
/// `unit` is a display string (may be empty for dimensionless numbers).
#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    /// Human-readable name, e.g. `"Frequency resolution"`. Also the lookup
    /// key for [`ExplainDoc::quantity`].
    pub name: String,
    /// LaTeX symbol, e.g. `\Delta f`.
    pub symbol: String,
    /// The computed value.
    pub value: f64,
    /// Unit, e.g. `"Hz"`; empty if dimensionless.
    pub unit: String,
    /// LaTeX formula (or short description) used to compute `value`.
    pub how: String,
}

/// A small table of computed values (e.g. poles per section, a gain table).
///
/// ## Purpose
/// Holds results that do not fit a single [`Quantity`].
///
/// ## Intended Usage
/// Built by explainers; read by renderers.
///
/// ## Invariants
/// Every row has `headers.len()` cells.
///
/// ## Assumptions
/// Cells are display text; cells starting and ending with `$` are LaTeX.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    /// Caption shown above the table.
    pub caption: String,
    /// Column headings.
    pub headers: Vec<String>,
    /// Rows of cells.
    pub rows: Vec<Vec<String>>,
}

/// Severity of a [`Warning`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum WarningLevel {
    /// Worth knowing; not a problem.
    Info,
    /// Likely to give a misleading or degraded result.
    Caution,
    /// The operation will fail or produce a wrong result.
    Problem,
}

impl WarningLevel {
    /// Lower-case label: `"info"`, `"caution"` or `"problem"`.
    ///
    /// # Returns
    ///
    /// The label as a static string.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Caution => "caution",
            Self::Problem => "problem",
        }
    }
}

impl Display for WarningLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// A computed observation about the settings ("your window is shorter than
/// two pitch periods").
///
/// ## Purpose
/// Flags settings that are invalid, suspicious or worth noting.
///
/// ## Intended Usage
/// Check [`ExplainDoc::has_warning`] in tests or tooling; renderers list them.
///
/// ## Invariants
/// Every warning is the result of a check on the parameters, never a
/// generic disclaimer.
///
/// ## Assumptions
/// None.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Warning {
    /// Severity.
    pub level: WarningLevel,
    /// What was found.
    pub message: String,
    /// What to do about it, if anything.
    pub hint: Option<String>,
}

/// A cross-reference: a Rust path, a Python name or a guide chapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeeAlso {
    /// Short label.
    pub label: String,
    /// Where to look (e.g. `audio_samples::AudioTransforms::stft`).
    pub target: String,
}

/// One structured explanation of a configuration or concept.
///
/// ## Purpose
/// Answers "what will this do, with which formula, at what resolution or
/// latency, and is anything about my settings suspicious?" before any audio
/// is processed.
///
/// ## Intended Usage
/// Obtain one from [`Explain::explain`](super::Explain::explain),
/// [`ExplainFor::explain`](super::ExplainFor::explain) or a function in
/// [`topics`](super::topics), then print it (`Display` is the terminal
/// rendering), call [`to_markdown`](Self::to_markdown) for notebooks, or
/// inspect the fields directly. Implementers of
/// [`Explain`](super::Explain) for their own types build one with
/// [`ExplainDoc::new`] and the `add_*` methods.
///
/// ## Invariants
/// Every number in `derived`, `symbols`, `tables` and `warnings` was
/// computed from the inputs.
///
/// ## Assumptions
/// LaTeX fragments follow the conventions documented on [`Formula`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ExplainDoc {
    /// Title, e.g. `"Short-time Fourier transform (STFT)"`.
    pub title: String,
    /// Syllabus topic.
    pub topic: Topic,
    /// One to three sentences describing what will happen.
    pub summary: String,
    /// Formulas the code evaluates.
    pub formulas: Vec<Formula>,
    /// Symbols used in the formulas.
    pub symbols: Vec<Symbol>,
    /// Numbers derived from the configuration.
    pub derived: Vec<Quantity>,
    /// Tables of computed values.
    pub tables: Vec<Table>,
    /// Trade-offs relevant to these settings.
    pub trade_offs: Vec<String>,
    /// Computed warnings.
    pub warnings: Vec<Warning>,
    /// Cross-references.
    pub see_also: Vec<SeeAlso>,
}

impl ExplainDoc {
    /// Creates an empty explanation with a topic and a title.
    ///
    /// # Arguments
    ///
    /// * `topic` - syllabus topic.
    /// * `title` - heading shown by every renderer.
    ///
    /// # Returns
    ///
    /// An [`ExplainDoc`] with every list empty and an empty summary.
    ///
    /// # Examples
    ///
    /// ```
    /// use audio_samples::educational::{ExplainDoc, Topic};
    ///
    /// let mut doc = ExplainDoc::new(Topic::Energy, "Frame energy");
    /// doc.set_summary("Sum of squared samples in each frame.");
    /// doc.add_formula(r"E = \sum_{n=0}^{N-1} x[n]^2", "Energy of one frame");
    /// assert!(doc.to_markdown().contains("$$"));
    /// ```
    #[must_use]
    pub fn new(topic: Topic, title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            topic,
            summary: String::new(),
            formulas: Vec::new(),
            symbols: Vec::new(),
            derived: Vec::new(),
            tables: Vec::new(),
            trade_offs: Vec::new(),
            warnings: Vec::new(),
            see_also: Vec::new(),
        }
    }

    /// Replaces the summary paragraph.
    ///
    /// # Arguments
    ///
    /// * `summary` - one to three sentences.
    pub fn set_summary(&mut self, summary: impl Into<String>) -> &mut Self {
        self.summary = summary.into();
        self
    }

    /// Appends a formula.
    ///
    /// # Arguments
    ///
    /// * `latex` - math-mode LaTeX without `$` delimiters.
    /// * `caption` - what the formula computes.
    pub fn add_formula(
        &mut self,
        latex: impl Into<String>,
        caption: impl Into<String>,
    ) -> &mut Self {
        self.formulas.push(Formula {
            latex: latex.into(),
            caption: caption.into(),
        });
        self
    }

    /// Appends a symbol without a value.
    ///
    /// # Arguments
    ///
    /// * `symbol` - LaTeX for the symbol.
    /// * `meaning` - what it stands for.
    pub fn add_symbol(
        &mut self,
        symbol: impl Into<String>,
        meaning: impl Into<String>,
    ) -> &mut Self {
        self.symbols.push(Symbol {
            symbol: symbol.into(),
            meaning: meaning.into(),
            value: None,
            unit: None,
        });
        self
    }

    /// Appends a symbol together with its value for this configuration.
    ///
    /// # Arguments
    ///
    /// * `symbol` - LaTeX for the symbol.
    /// * `meaning` - what it stands for.
    /// * `value` - formatted value.
    /// * `unit` - unit (empty for none).
    pub fn add_symbol_value(
        &mut self,
        symbol: impl Into<String>,
        meaning: impl Into<String>,
        value: impl Into<String>,
        unit: &str,
    ) -> &mut Self {
        self.symbols.push(Symbol {
            symbol: symbol.into(),
            meaning: meaning.into(),
            value: Some(value.into()),
            unit: (!unit.is_empty()).then(|| unit.to_string()),
        });
        self
    }

    /// Appends a derived quantity. Non-finite values are silently skipped so
    /// that a document never displays `NaN`.
    ///
    /// # Arguments
    ///
    /// * `name` - human-readable name (also the lookup key).
    /// * `symbol` - LaTeX symbol.
    /// * `value` - computed value.
    /// * `unit` - unit, empty if dimensionless.
    /// * `how` - LaTeX formula used.
    pub fn add_quantity(
        &mut self,
        name: impl Into<String>,
        symbol: impl Into<String>,
        value: f64,
        unit: &str,
        how: impl Into<String>,
    ) -> &mut Self {
        if value.is_finite() {
            self.derived.push(Quantity {
                name: name.into(),
                symbol: symbol.into(),
                value,
                unit: unit.to_string(),
                how: how.into(),
            });
        }
        self
    }

    /// Appends a table.
    ///
    /// # Arguments
    ///
    /// * `table` - the table to append.
    pub fn add_table(&mut self, table: Table) -> &mut Self {
        self.tables.push(table);
        self
    }

    /// Appends a trade-off statement.
    ///
    /// # Arguments
    ///
    /// * `text` - one sentence.
    pub fn add_trade_off(&mut self, text: impl Into<String>) -> &mut Self {
        self.trade_offs.push(text.into());
        self
    }

    /// Appends a warning.
    ///
    /// # Arguments
    ///
    /// * `level` - severity.
    /// * `message` - what was found.
    /// * `hint` - what to do about it (empty for none).
    pub fn add_warning(
        &mut self,
        level: WarningLevel,
        message: impl Into<String>,
        hint: impl Into<String>,
    ) -> &mut Self {
        let hint = hint.into();
        self.warnings.push(Warning {
            level,
            message: message.into(),
            hint: (!hint.is_empty()).then_some(hint),
        });
        self
    }

    /// Appends a cross-reference.
    ///
    /// # Arguments
    ///
    /// * `label` - short label.
    /// * `target` - Rust path, Python name or guide chapter.
    pub fn add_see_also(
        &mut self,
        label: impl Into<String>,
        target: impl Into<String>,
    ) -> &mut Self {
        self.see_also.push(SeeAlso {
            label: label.into(),
            target: target.into(),
        });
        self
    }

    /// Looks up a derived quantity by name.
    ///
    /// # Arguments
    ///
    /// * `name` - the quantity's `name`.
    ///
    /// # Returns
    ///
    /// The first quantity with that name, or `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use audio_samples::educational::topics;
    ///
    /// let doc = topics::sampling(48_000.0);
    /// assert_eq!(doc.quantity("Nyquist frequency").unwrap().value, 24_000.0);
    /// ```
    #[must_use]
    pub fn quantity(&self, name: &str) -> Option<&Quantity> {
        self.derived.iter().find(|q| q.name == name)
    }

    /// Returns `true` if at least one warning has exactly this level.
    ///
    /// # Arguments
    ///
    /// * `level` - the severity to look for.
    ///
    /// # Returns
    ///
    /// Whether such a warning exists.
    #[must_use]
    pub fn has_warning(&self, level: WarningLevel) -> bool {
        self.warnings.iter().any(|w| w.level == level)
    }

    /// Returns `true` if any warning's message contains `needle`.
    ///
    /// # Arguments
    ///
    /// * `needle` - substring to search for.
    ///
    /// # Returns
    ///
    /// Whether a matching warning exists.
    #[must_use]
    pub fn has_warning_containing(&self, needle: &str) -> bool {
        self.warnings.iter().any(|w| w.message.contains(needle))
    }

    /// Renders the explanation for a terminal.
    ///
    /// Formulas are drawn as two-dimensional Unicode art by `term-maths`;
    /// a formula that uses LaTeX the terminal renderer does not understand is
    /// printed as raw LaTeX instead. This is also the [`Display`] output.
    ///
    /// # Returns
    ///
    /// A multi-line string.
    ///
    /// # Examples
    ///
    /// ```
    /// use audio_samples::educational::topics;
    ///
    /// let text = topics::aliasing(7_000.0, 8_000.0).to_text();
    /// assert!(text.contains("1000 Hz"));
    /// ```
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "{}", self.title);
        let _ = writeln!(out, "{}", "=".repeat(self.title.chars().count()));
        if !self.summary.is_empty() {
            let _ = writeln!(out, "{}", self.summary);
        }

        if !self.formulas.is_empty() {
            text_heading(&mut out, "Formulas");
            for f in &self.formulas {
                let _ = writeln!(out, "  {}:", f.caption);
                for line in block_math(&f.latex).lines() {
                    let _ = writeln!(out, "      {line}");
                }
                out.push('\n');
            }
        }

        if !self.symbols.is_empty() {
            text_heading(&mut out, "Symbols");
            let rows: Vec<Vec<String>> = self
                .symbols
                .iter()
                .map(|s| {
                    let value = match (&s.value, &s.unit) {
                        (Some(v), Some(u)) => format!("{v} {u}"),
                        (Some(v), None) => v.clone(),
                        _ => String::new(),
                    };
                    vec![inline_math(&s.symbol), s.meaning.clone(), value]
                })
                .collect();
            text_table(&mut out, &[], &rows);
        }

        if !self.derived.is_empty() {
            text_heading(&mut out, "Derived quantities");
            let rows: Vec<Vec<String>> = self
                .derived
                .iter()
                .map(|q| {
                    vec![
                        q.name.clone(),
                        format!("{} = {}", inline_math(&q.symbol), value_with_unit(q)),
                        inline_math(&q.how),
                    ]
                })
                .collect();
            text_table(&mut out, &[], &rows);
        }

        for t in &self.tables {
            text_heading(&mut out, &t.caption);
            let rows: Vec<Vec<String>> = t
                .rows
                .iter()
                .map(|r| r.iter().map(|c| text_cell(c)).collect())
                .collect();
            text_table(&mut out, &t.headers, &rows);
        }

        if !self.trade_offs.is_empty() {
            text_heading(&mut out, "Trade-offs");
            for t in &self.trade_offs {
                let _ = writeln!(out, "  - {t}");
            }
        }

        if !self.warnings.is_empty() {
            text_heading(&mut out, "Warnings");
            for w in &self.warnings {
                let marker = match w.level {
                    WarningLevel::Problem => "!!",
                    WarningLevel::Caution => " !",
                    WarningLevel::Info => " i",
                };
                let _ = writeln!(out, "  {marker} [{}] {}", w.level, w.message);
                if let Some(h) = &w.hint {
                    let _ = writeln!(out, "       hint: {h}");
                }
            }
        }

        if !self.see_also.is_empty() {
            text_heading(&mut out, "See also");
            for s in &self.see_also {
                let _ = writeln!(out, "  - {}: {}", s.label, s.target);
            }
        }
        out
    }

    /// Renders the explanation as Markdown with `$$…$$` display maths and
    /// `$…$` inline maths.
    ///
    /// The output renders in Jupyter (MathJax), mdBook with MathJax enabled,
    /// and MyST. Vertical bars inside table cells are written as `\vert` so
    /// they do not split Markdown table columns.
    ///
    /// # Returns
    ///
    /// A Markdown string starting with a level-3 heading.
    ///
    /// # Examples
    ///
    /// ```
    /// use audio_samples::educational::topics;
    ///
    /// let md = topics::dft(1024, 48_000.0).to_markdown();
    /// assert!(md.starts_with("### "));
    /// assert!(md.contains("$$"));
    /// ```
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "### {}\n", md_escape(&self.title));
        if !self.summary.is_empty() {
            let _ = writeln!(out, "{}\n", md_escape(&self.summary));
        }

        if !self.formulas.is_empty() {
            out.push_str("**Formulas**\n\n");
            for f in &self.formulas {
                let _ = writeln!(out, "{}:\n\n$$\n{}\n$$\n", md_escape(&f.caption), f.latex);
            }
        }

        if !self.symbols.is_empty() {
            out.push_str("**Symbols**\n\n| Symbol | Meaning | Value |\n|---|---|---|\n");
            for s in &self.symbols {
                let value = match (&s.value, &s.unit) {
                    (Some(v), Some(u)) => format!("{v} {u}"),
                    (Some(v), None) => v.clone(),
                    _ => String::new(),
                };
                let _ = writeln!(
                    out,
                    "| {} | {} | {} |",
                    md_math(&s.symbol),
                    md_cell(&s.meaning),
                    md_cell(&value)
                );
            }
            out.push('\n');
        }

        if !self.derived.is_empty() {
            out.push_str(
                "**Derived quantities**\n\n| Quantity | Value | Formula |\n|---|---|---|\n",
            );
            for q in &self.derived {
                let _ = writeln!(
                    out,
                    "| {} | {} = {} | {} |",
                    md_cell(&q.name),
                    md_math(&q.symbol),
                    md_cell(&value_with_unit(q)),
                    md_math(&q.how)
                );
            }
            out.push('\n');
        }

        for t in &self.tables {
            let _ = writeln!(out, "**{}**\n", md_escape(&t.caption));
            let _ = writeln!(
                out,
                "| {} |",
                t.headers
                    .iter()
                    .map(|h| md_cell(h))
                    .collect::<Vec<_>>()
                    .join(" | ")
            );
            let _ = writeln!(out, "|{}", "---|".repeat(t.headers.len()));
            for r in &t.rows {
                let cells: Vec<String> = r.iter().map(|c| md_table_cell(c)).collect();
                let _ = writeln!(out, "| {} |", cells.join(" | "));
            }
            out.push('\n');
        }

        if !self.trade_offs.is_empty() {
            out.push_str("**Trade-offs**\n\n");
            for t in &self.trade_offs {
                let _ = writeln!(out, "- {}", md_escape(t));
            }
            out.push('\n');
        }

        if !self.warnings.is_empty() {
            out.push_str("**Warnings**\n\n");
            for w in &self.warnings {
                let label = match w.level {
                    WarningLevel::Problem => "Problem",
                    WarningLevel::Caution => "Caution",
                    WarningLevel::Info => "Info",
                };
                let _ = write!(out, "- **{label}:** {}", md_escape(&w.message));
                if let Some(h) = &w.hint {
                    let _ = write!(out, " *Hint:* {}", md_escape(h));
                }
                out.push('\n');
            }
            out.push('\n');
        }

        if !self.see_also.is_empty() {
            out.push_str("**See also**\n\n");
            for s in &self.see_also {
                let _ = writeln!(out, "- {}: `{}`", md_escape(&s.label), s.target);
            }
            out.push('\n');
        }
        out
    }

    /// Renders the explanation as a self-contained HTML fragment (no
    /// scripts, no styles).
    ///
    /// Display formulas are wrapped in `\[ … \]` and inline symbols in
    /// `\( … \)`, so a page that loads MathJax or KaTeX auto-render will
    /// typeset them; without such a script the LaTeX source is shown. All
    /// text is HTML-escaped.
    ///
    /// # Returns
    ///
    /// A `<section class="as-explain">` element.
    ///
    /// # Examples
    ///
    /// ```
    /// use audio_samples::educational::topics;
    ///
    /// let html = topics::quantisation(16).to_html();
    /// assert!(html.starts_with("<section"));
    /// assert!(!html.contains("<script"));
    /// ```
    #[must_use]
    pub fn to_html(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "<section class=\"as-explain\" data-topic=\"{}\">",
            html_escape(self.topic.label())
        );
        let _ = writeln!(out, "<h3>{}</h3>", html_escape(&self.title));
        if !self.summary.is_empty() {
            let _ = writeln!(
                out,
                "<p class=\"as-explain-summary\">{}</p>",
                html_escape(&self.summary)
            );
        }
        if !self.formulas.is_empty() {
            out.push_str("<h4>Formulas</h4>\n");
            for f in &self.formulas {
                let _ = writeln!(
                    out,
                    "<figure class=\"as-explain-formula\"><figcaption>{}</figcaption><div class=\"math\">\\[{}\\]</div></figure>",
                    html_escape(&f.caption),
                    html_escape(&f.latex)
                );
            }
        }
        if !self.symbols.is_empty() {
            out.push_str("<h4>Symbols</h4>\n<table class=\"as-explain-symbols\"><thead><tr><th>Symbol</th><th>Meaning</th><th>Value</th></tr></thead><tbody>\n");
            for s in &self.symbols {
                let value = match (&s.value, &s.unit) {
                    (Some(v), Some(u)) => format!("{v} {u}"),
                    (Some(v), None) => v.clone(),
                    _ => String::new(),
                };
                let _ = writeln!(
                    out,
                    "<tr><td>\\({}\\)</td><td>{}</td><td>{}</td></tr>",
                    html_escape(&s.symbol),
                    html_escape(&s.meaning),
                    html_escape(&value)
                );
            }
            out.push_str("</tbody></table>\n");
        }
        if !self.derived.is_empty() {
            out.push_str("<h4>Derived quantities</h4>\n<table class=\"as-explain-derived\"><thead><tr><th>Quantity</th><th>Value</th><th>Formula</th></tr></thead><tbody>\n");
            for q in &self.derived {
                let _ = writeln!(
                    out,
                    "<tr><td>{}</td><td>\\({}\\) = {}</td><td>\\({}\\)</td></tr>",
                    html_escape(&q.name),
                    html_escape(&q.symbol),
                    html_escape(&value_with_unit(q)),
                    html_escape(&q.how)
                );
            }
            out.push_str("</tbody></table>\n");
        }
        for t in &self.tables {
            let _ = writeln!(
                out,
                "<h4>{}</h4>\n<table class=\"as-explain-table\"><thead><tr>",
                html_escape(&t.caption)
            );
            for h in &t.headers {
                let _ = write!(out, "<th>{}</th>", html_escape(h));
            }
            out.push_str("</tr></thead><tbody>\n");
            for r in &t.rows {
                out.push_str("<tr>");
                for c in r {
                    let cell = match c.strip_prefix('$').and_then(|s| s.strip_suffix('$')) {
                        Some(latex) => format!("\\({}\\)", html_escape(latex)),
                        None => html_escape(c),
                    };
                    let _ = write!(out, "<td>{cell}</td>");
                }
                out.push_str("</tr>\n");
            }
            out.push_str("</tbody></table>\n");
        }
        if !self.trade_offs.is_empty() {
            out.push_str("<h4>Trade-offs</h4>\n<ul class=\"as-explain-tradeoffs\">\n");
            for t in &self.trade_offs {
                let _ = writeln!(out, "<li>{}</li>", html_escape(t));
            }
            out.push_str("</ul>\n");
        }
        if !self.warnings.is_empty() {
            out.push_str("<h4>Warnings</h4>\n<ul class=\"as-explain-warnings\">\n");
            for w in &self.warnings {
                let _ = write!(
                    out,
                    "<li class=\"{}\"><strong>{}:</strong> {}",
                    w.level,
                    w.level,
                    html_escape(&w.message)
                );
                if let Some(h) = &w.hint {
                    let _ = write!(out, " <em>Hint:</em> {}", html_escape(h));
                }
                out.push_str("</li>\n");
            }
            out.push_str("</ul>\n");
        }
        if !self.see_also.is_empty() {
            out.push_str("<h4>See also</h4>\n<ul class=\"as-explain-see-also\">\n");
            for s in &self.see_also {
                let _ = writeln!(
                    out,
                    "<li>{}: <code>{}</code></li>",
                    html_escape(&s.label),
                    html_escape(&s.target)
                );
            }
            out.push_str("</ul>\n");
        }
        out.push_str("</section>\n");
        out
    }
}

impl Display for ExplainDoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

/// Formats a number with about six significant figures, switching to
/// scientific notation for very large or very small magnitudes and trimming
/// trailing zeros.
pub(crate) fn fmt_num(v: f64) -> String {
    if !v.is_finite() {
        return format!("{v}");
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let a = v.abs();
    if !(1e-3..1e7).contains(&a) {
        return format!("{v:.4e}");
    }
    let int_digits = a.log10().floor() as i32 + 1;
    let decimals = (6 - int_digits).max(0) as usize;
    let s = format!("{v:.decimals$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

fn value_with_unit(q: &Quantity) -> String {
    if q.unit.is_empty() {
        fmt_num(q.value)
    } else {
        format!("{} {}", fmt_num(q.value), q.unit)
    }
}

// ---------------------------------------------------------------------------
// Terminal helpers
// ---------------------------------------------------------------------------

/// Renders display maths with `term-maths`. When the renderer leaves an
/// unrecognised `\command` in its output, the formula is shown as a
/// one-line Unicode approximation instead (see [`latex_to_plain`]).
pub(crate) fn block_math(latex: &str) -> String {
    let rendered = term_maths::render(latex).to_string();
    if rendered.trim().is_empty() || rendered.contains('\\') {
        latex_to_plain(latex)
    } else {
        rendered
    }
}

/// Renders maths on a single line: the `term-maths` rendering when it fits
/// on one line, otherwise a one-line Unicode approximation.
pub(crate) fn inline_math(latex: &str) -> String {
    if latex.is_empty() {
        return String::new();
    }
    let rendered = term_maths::render(latex).to_string();
    if rendered.lines().count() == 1 && !rendered.contains('\\') && !rendered.trim().is_empty() {
        rendered
    } else {
        latex_to_plain(latex)
    }
}

/// Converts a LaTeX fragment to a readable single line of Unicode text:
/// Greek letters and operators become symbols, `\frac{a}{b}` becomes
/// `a/b`, sub- and superscripts use Unicode digits/letters where available
/// and `_`/`^` otherwise, and formatting commands are dropped.
pub(crate) fn latex_to_plain(latex: &str) -> String {
    let chars: Vec<char> = latex.chars().collect();
    let mut pos = 0;
    let out = plain_seq(&chars, &mut pos, None);
    let mut text = out.split_whitespace().collect::<Vec<_>>().join(" ");
    for (from, to) in [
        ("⌊ ", "⌊"),
        (" ⌋", "⌋"),
        ("⌈ ", "⌈"),
        (" ⌉", "⌉"),
        ("( ", "("),
        (" )", ")"),
    ] {
        text = text.replace(from, to);
    }
    text
}

fn plain_symbol(cmd: &str) -> Option<&'static str> {
    Some(match cmd {
        "alpha" => "α",
        "beta" => "β",
        "gamma" => "γ",
        "delta" => "δ",
        "Delta" => "Δ",
        "varepsilon" | "epsilon" => "ε",
        "theta" => "θ",
        "lambda" => "λ",
        "mu" => "μ",
        "pi" => "π",
        "sigma" => "σ",
        "tau" => "τ",
        "xi" => "ξ",
        "omega" => "ω",
        "Omega" => "Ω",
        "cdot" => "·",
        "times" => "×",
        "pm" => "±",
        "le" | "leq" => "≤",
        "ge" | "geq" => "≥",
        "approx" => "≈",
        "ne" | "neq" => "≠",
        "infty" => "∞",
        "sum" => "Σ",
        "prod" => "∏",
        "lfloor" => "⌊",
        "rfloor" => "⌋",
        "lceil" => "⌈",
        "rceil" => "⌉",
        "vert" | "lvert" | "rvert" => "|",
        "rightarrow" | "to" => "→",
        "Rightarrow" => "⇒",
        "leftarrow" => "←",
        "in" => "∈",
        "wedge" => "∧",
        "forall" => "∀",
        "dots" | "ldots" | "cdots" => "…",
        "log" => "log",
        "ln" => "ln",
        "exp" => "exp",
        "sin" => "sin",
        "cos" => "cos",
        "tan" => "tan",
        "max" => "max",
        "min" => "min",
        "arg" => "arg",
        "quad" | "qquad" => "  ",
        "," | ";" | ":" | " " | "!" => " ",
        "#" => "#",
        "{" => "{",
        "}" => "}",
        "_" => "_",
        "%" => "%",
        _ => return None,
    })
}

/// Reads one argument: a `{…}` group or a single character / command.
fn plain_arg(chars: &[char], pos: &mut usize) -> String {
    while *pos < chars.len() && chars[*pos] == ' ' {
        *pos += 1;
    }
    match chars.get(*pos) {
        Some('{') => {
            *pos += 1;
            plain_seq(chars, pos, Some('}'))
        }
        Some('\\') => plain_command(chars, pos),
        Some(&c) => {
            *pos += 1;
            c.to_string()
        }
        None => String::new(),
    }
}

fn plain_command(chars: &[char], pos: &mut usize) -> String {
    *pos += 1; // backslash
    let start = *pos;
    while *pos < chars.len() && chars[*pos].is_ascii_alphabetic() {
        *pos += 1;
    }
    if *pos == start && *pos < chars.len() {
        *pos += 1; // single non-letter command such as \, or \{
    }
    let cmd: String = chars[start..*pos].iter().collect();
    match cmd.as_str() {
        "frac" | "tfrac" | "dfrac" => {
            let num = plain_arg(chars, pos);
            let den = plain_arg(chars, pos);
            let wrap = |s: String| {
                if s.chars().count() > 1 && s.contains([' ', '+', '-']) {
                    format!("({s})")
                } else {
                    s
                }
            };
            format!("{}/{}", wrap(num), wrap(den))
        }
        "sqrt" => format!("√({})", plain_arg(chars, pos)),
        "mathrm" | "text" | "mathbf" | "mathit" | "operatorname" | "textrm" => {
            plain_arg(chars, pos)
        }
        "mathbb" => {
            let a = plain_arg(chars, pos);
            match a.as_str() {
                "Z" => "ℤ".into(),
                "R" => "ℝ".into(),
                "N" => "ℕ".into(),
                "C" => "ℂ".into(),
                _ => a,
            }
        }
        "overline" | "bar" => format!("{}\u{304}", plain_arg(chars, pos)),
        "hat" => format!("{}\u{302}", plain_arg(chars, pos)),
        "left" | "right" | "big" | "Big" | "bigl" | "bigr" => String::new(),
        "begin" | "end" => {
            plain_arg(chars, pos);
            String::new()
        }
        other => plain_symbol(other).map_or_else(|| other.to_string(), str::to_string),
    }
}

fn to_script(s: &str, sup: bool) -> Option<String> {
    s.chars()
        .map(|c| {
            if sup {
                match c {
                    '0' => Some('⁰'),
                    '1' => Some('¹'),
                    '2' => Some('²'),
                    '3' => Some('³'),
                    '4' => Some('⁴'),
                    '5' => Some('⁵'),
                    '6' => Some('⁶'),
                    '7' => Some('⁷'),
                    '8' => Some('⁸'),
                    '9' => Some('⁹'),
                    '+' => Some('⁺'),
                    '-' | '−' => Some('⁻'),
                    'n' => Some('ⁿ'),
                    'i' => Some('ⁱ'),
                    _ => None,
                }
            } else {
                match c {
                    '0' => Some('₀'),
                    '1' => Some('₁'),
                    '2' => Some('₂'),
                    '3' => Some('₃'),
                    '4' => Some('₄'),
                    '5' => Some('₅'),
                    '6' => Some('₆'),
                    '7' => Some('₇'),
                    '8' => Some('₈'),
                    '9' => Some('₉'),
                    '+' => Some('₊'),
                    '-' => Some('₋'),
                    'a' => Some('ₐ'),
                    'e' => Some('ₑ'),
                    'h' => Some('ₕ'),
                    'i' => Some('ᵢ'),
                    'j' => Some('ⱼ'),
                    'k' => Some('ₖ'),
                    'l' => Some('ₗ'),
                    'm' => Some('ₘ'),
                    'n' => Some('ₙ'),
                    'o' => Some('ₒ'),
                    'p' => Some('ₚ'),
                    'r' => Some('ᵣ'),
                    's' => Some('ₛ'),
                    't' => Some('ₜ'),
                    'u' => Some('ᵤ'),
                    'x' => Some('ₓ'),
                    _ => None,
                }
            }
        })
        .collect()
}

fn plain_seq(chars: &[char], pos: &mut usize, until: Option<char>) -> String {
    let mut out = String::new();
    while *pos < chars.len() {
        let c = chars[*pos];
        if Some(c) == until {
            *pos += 1;
            break;
        }
        match c {
            '\\' => out.push_str(&plain_command(chars, pos)),
            '{' => {
                *pos += 1;
                out.push_str(&plain_seq(chars, pos, Some('}')));
            }
            '^' | '_' => {
                *pos += 1;
                let arg = plain_arg(chars, pos);
                match to_script(&arg, c == '^') {
                    Some(s) if !arg.is_empty() => out.push_str(&s),
                    _ if arg.chars().count() == 1 => {
                        out.push(c);
                        out.push_str(&arg);
                    }
                    _ => {
                        out.push(c);
                        out.push('(');
                        out.push_str(&arg);
                        out.push(')');
                    }
                }
            }
            '&' => {
                *pos += 1;
                out.push(' ');
            }
            '-' => {
                *pos += 1;
                out.push('-');
            }
            _ => {
                *pos += 1;
                out.push(c);
            }
        }
    }
    out
}

fn text_cell(cell: &str) -> String {
    match cell.strip_prefix('$').and_then(|s| s.strip_suffix('$')) {
        Some(latex) => inline_math(latex),
        None => cell.to_string(),
    }
}

fn text_heading(out: &mut String, heading: &str) {
    let _ = writeln!(out, "\n{heading}");
    let _ = writeln!(out, "{}", "-".repeat(heading.chars().count()));
}

fn text_table(out: &mut String, headers: &[String], rows: &[Vec<String>]) {
    let n_cols = rows
        .iter()
        .map(Vec::len)
        .chain(std::iter::once(headers.len()))
        .max()
        .unwrap_or(0);
    let mut widths = vec![0usize; n_cols];
    for r in rows.iter().chain(std::iter::once(&headers.to_vec())) {
        for (i, c) in r.iter().enumerate() {
            widths[i] = widths[i].max(c.chars().count());
        }
    }
    let line = |cells: &[String]| -> String {
        let mut s = String::from("  ");
        for (i, c) in cells.iter().enumerate() {
            let pad = widths[i].saturating_sub(c.chars().count());
            s.push_str(c);
            if i + 1 < cells.len() {
                s.push_str(&" ".repeat(pad + 2));
            }
        }
        s.trim_end().to_string()
    };
    if !headers.is_empty() {
        let _ = writeln!(out, "{}", line(headers));
        let total: usize = widths.iter().sum::<usize>() + 2 * n_cols.saturating_sub(1);
        let _ = writeln!(out, "  {}", "-".repeat(total));
    }
    for r in rows {
        let _ = writeln!(out, "{}", line(r));
    }
}

// ---------------------------------------------------------------------------
// Markdown helpers
// ---------------------------------------------------------------------------

/// Escapes the few characters that change meaning in running Markdown text.
fn md_escape(s: &str) -> String {
    s.replace('*', "\\*").replace('_', "\\_")
}

/// Escapes a plain-text table cell.
fn md_cell(s: &str) -> String {
    md_escape(s).replace('|', "\\|")
}

/// Wraps LaTeX as inline maths that is safe inside a Markdown table.
fn md_math(latex: &str) -> String {
    if latex.is_empty() {
        return String::new();
    }
    format!("${}$", latex.replace('|', r"\vert "))
}

fn md_table_cell(cell: &str) -> String {
    match cell.strip_prefix('$').and_then(|s| s.strip_suffix('$')) {
        Some(latex) => md_math(latex),
        None => md_cell(cell),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_num_is_compact() {
        assert_eq!(fmt_num(1024.0), "1024");
        assert_eq!(fmt_num(21.533203125), "21.5332");
        assert_eq!(fmt_num(0.5), "0.5");
        assert_eq!(fmt_num(-13.2614), "-13.2614");
        assert_eq!(fmt_num(1.5e-5), "1.5000e-5");
        assert_eq!(fmt_num(0.0), "0");
    }

    #[test]
    fn non_finite_quantities_are_skipped() {
        let mut d = ExplainDoc::new(Topic::Dft, "t");
        d.add_quantity("a", "a", f64::NAN, "", "");
        d.add_quantity("b", "b", 1.0, "", "");
        assert_eq!(d.derived.len(), 1);
    }

    #[test]
    fn renderers_include_all_sections() {
        let mut d = ExplainDoc::new(Topic::Dft, "Title");
        d.set_summary("Summary.");
        d.add_formula(r"\Delta f = \frac{f_s}{N}", "Bin spacing");
        d.add_symbol_value("N", "length", "8", "samples");
        d.add_quantity("Bin spacing", r"\Delta f", 6000.0, "Hz", r"f_s / N");
        d.add_table(Table {
            caption: "Bins".into(),
            headers: vec!["k".into(), "f".into()],
            rows: vec![vec!["$|X|$".into(), "0".into()]],
        });
        d.add_trade_off("Longer is sharper.");
        d.add_warning(WarningLevel::Caution, "Careful", "Do this");
        d.add_see_also("stft", "audio_samples::AudioTransforms::stft");

        let text = d.to_text();
        for needle in [
            "Title",
            "Summary.",
            "Bin spacing",
            "6000 Hz",
            "Careful",
            "hint: Do this",
        ] {
            assert!(text.contains(needle), "text missing {needle}:\n{text}");
        }
        // term-maths draws the fraction on three lines.
        assert!(text.contains('─'));

        let md = d.to_markdown();
        assert!(md.contains("$$\n\\Delta f = \\frac{f_s}{N}\n$$"));
        assert!(md.contains(r"$\vert X\vert $"));
        assert!(md.contains("**Caution:** Careful"));

        let html = d.to_html();
        assert!(html.contains("<h3>Title</h3>"));
        assert!(html.contains("class=\"caution\""));
        assert_eq!(format!("{d}"), text);
    }

    #[test]
    fn unsupported_latex_falls_back_to_plain_text() {
        // term-maths leaves a command it does not know in its output.
        assert_eq!(
            block_math(r"\notacommand{x} + 1"),
            latex_to_plain(r"\notacommand{x} + 1")
        );
        assert!(!block_math(r"\notacommand{x} + 1").contains('\\'));

        // A formula it knows is rendered by term-maths, here over several lines.
        let floor = r"M = \left\lfloor \frac{L - N}{H} \right\rfloor + 1";
        assert!(block_math(floor).contains('⌊') && !block_math(floor).contains('\\'));
        assert_eq!(latex_to_plain(floor), "M = ⌊(L - N)/H⌋ + 1");
        // Inline maths stays on one line either way.
        assert_eq!(inline_math(floor), "M = ⌊(L - N)/H⌋ + 1");

        assert_eq!(latex_to_plain(r"\tau_{min}"), "τₘᵢₙ");
        assert_eq!(latex_to_plain(r"10^{T/20}"), "10^(T/20)");
        assert_eq!(
            latex_to_plain(r"\mathrm{ENBW} \cdot f_s / N"),
            "ENBW · fₛ / N"
        );
        assert!(!inline_math(r"\unknowncommand{x}").contains('\\'));
    }
}
