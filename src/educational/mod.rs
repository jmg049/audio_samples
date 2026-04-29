//! Educational / explainable layer for `audio_samples`.
//!
//! Enabled with `--features educational`. Wraps owned [`AudioSamples`] in an
//! explaining chain that captures a before/after snapshot for each operation
//! and assembles the results into a self-contained HTML document with
//! KaTeX-rendered formulae, Plotly waveform panels, and a selectable theme.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use audio_samples::{AudioSamples, AudioProcessingExt, ExplainMode, NormalizationConfig, sample_rate};
//! use audio_samples::educational;
//! use audio_samples::utils::generation::sine_wave;
//! use std::time::Duration;
//!
//! let audio: AudioSamples<'static, f32> =
//!     sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 0.8);
//!
//! let (result, explanations) = audio
//!     .explaining(ExplainMode::Both)
//!     .normalize(NormalizationConfig::peak(1.0))
//!     .scale(0.5)
//!     .explain();
//!
//! educational::open_explanation_document(&explanations, "Processing walkthrough").unwrap();
//! ```

use explainable::{ExplainDisplay, Explainable, Explanation, RenderVisual};

use crate::repr::AudioSamples;
use crate::traits::StandardSample;

#[cfg(feature = "processing")]
pub(crate) mod processing;

// ── Visual surface ────────────────────────────────────────────────────────────

/// Opaque visual explanation for one operation step.
///
/// Holds Plotly inline HTML fragments when compiled with `plotting`; otherwise
/// a zero-size stub. The visual is not displayed immediately — collect all
/// explanations first and pass them to [`render_explanation_document`].
pub struct AudioSamplesVisual {
    #[cfg(feature = "plotting")]
    pub(crate) before_fragment: String,
    #[cfg(feature = "plotting")]
    pub(crate) after_fragment: String,
}

impl ExplainDisplay for AudioSamplesVisual {
    fn display(&self) {
        // Intentional no-op: individual display is suppressed.
        // Call render_explanation_document / open_explanation_document instead.
    }
}

// ── RenderVisual ──────────────────────────────────────────────────────────────

impl<T: StandardSample> RenderVisual for AudioSamples<'static, T> {
    fn render_visual(before: &Self, after: &Self) -> Box<dyn ExplainDisplay> {
        #[cfg(feature = "plotting")]
        {
            use crate::operations::plotting::WaveformPlotParams;
            use crate::operations::AudioPlotting;
            // Use None so Plotly generates a unique UUID per div — a fixed ID
            // causes every step to share one element and only the last renders.
            let before_frag = before
                .plot_waveform(&WaveformPlotParams::default())
                .map(|p| p.inline_html(None))
                .unwrap_or_default();
            let after_frag = after
                .plot_waveform(&WaveformPlotParams::default())
                .map(|p| p.inline_html(None))
                .unwrap_or_default();
            return Box::new(AudioSamplesVisual {
                before_fragment: before_frag,
                after_fragment: after_frag,
            });
        }
        #[cfg(not(feature = "plotting"))]
        {
            let _ = (before, after);
            Box::new(AudioSamplesVisual {})
        }
    }
}

// ── Explainable marker ────────────────────────────────────────────────────────

impl<T: StandardSample> Explainable for AudioSamples<'static, T> {}

// ── HTML document renderer ────────────────────────────────────────────────────

/// Assemble all explanations from a chain into a single polished HTML document.
///
/// The document is self-contained: CSS is inline, KaTeX and Plotly are loaded
/// from CDN. It includes the full Rust call chain that produced the explanations,
/// a theme switcher (Midnight / Slate / Amber / Light), and one card per
/// operation with a KaTeX formula, educational prose, and before/after waveforms.
///
/// # Example
/// ```rust,no_run
/// # use explainable::Explanation;
/// # let explanations: Vec<Explanation> = vec![];
/// let html = audio_samples::educational::render_explanation_document(
///     &explanations,
///     "Normalization walkthrough",
/// );
/// std::fs::write("explanation.html", &html).unwrap();
/// ```
pub fn render_explanation_document(explanations: &[Explanation], title: &str) -> String {
    let chain = render_chain_block(explanations);
    let cards: String = explanations
        .iter()
        .enumerate()
        .map(|(i, exp)| render_card(i, exp, i == 0))
        .collect();

    DOCUMENT_TEMPLATE
        .replace("{{TITLE}}", &html_escape(title))
        .replace("{{CHAIN}}", &chain)
        .replace("{{CARDS}}", &cards)
}

/// Write the explanation document to a temporary file and open it in the
/// default system browser.
///
/// # Errors
/// Returns `Err` if the file cannot be written or the browser cannot be opened.
pub fn open_explanation_document(
    explanations: &[Explanation],
    title: &str,
) -> std::io::Result<()> {
    let html = render_explanation_document(explanations, title);
    let path = std::env::temp_dir().join(format!(
        "audio_samples_explain_{}.html",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    ));
    std::fs::write(&path, &html)?;
    open::that(&path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(())
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Convert a display-friendly operation name into the Rust method name.
fn op_name_to_method(name: &str) -> &'static str {
    match name {
        "Normalize"           => "normalize",
        "Scale"               => "scale",
        "Apply Window"        => "apply_window",
        "Apply FIR Filter"    => "apply_filter",
        "\u{03bc}-law Compress" => "mu_compress",
        "\u{03bc}-law Expand"   => "mu_expand",
        "Low-Pass Filter"     => "low_pass_filter",
        "High-Pass Filter"    => "high_pass_filter",
        "Band-Pass Filter"    => "band_pass_filter",
        "Remove DC Offset"    => "remove_dc_offset",
        "Clip"                => "clip",
        "Clip (in-place)"     => "clip_in_place",
        _                     => "operation",
    }
}

/// Extract the `[operation: Name]` value from a raw explanation text string.
fn extract_op_name(text: &str) -> Option<&str> {
    let rest = text.strip_prefix("[operation: ")?;
    let end = rest.find("]\n")?;
    Some(&rest[..end])
}

/// Render the Rust call chain for all operations in the explanation list.
fn render_chain_block(explanations: &[Explanation]) -> String {
    let methods: Vec<&str> = explanations
        .iter()
        .filter_map(|e| e.text.as_deref())
        .filter_map(extract_op_name)
        .map(op_name_to_method)
        .collect();

    if methods.is_empty() {
        return String::new();
    }

    let chain_lines: String = methods
        .iter()
        .map(|m| {
            format!(
                "    <span class=\"cc-dot\">.</span><span class=\"cc-method\">{m}</span><span class=\"cc-args\">(...)</span>"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"<section class="chain-section">
  <div class="chain-label">Call chain</div>
  <pre class="chain-code"><span class="cc-kw">let</span> (<span class="cc-var">result</span>, <span class="cc-var">explanations</span>) = <span class="cc-name">audio</span>
{chain_lines}
    <span class="cc-dot">.</span><span class="cc-method">explain</span><span class="cc-args">();</span></pre>
</section>"#
    )
}

/// Parse the structured text format emitted by `AudioProcessingExplainText` impls.
///
/// Expected format (all fields optional, fallback when absent):
/// ```text
/// [operation: <Name>]
/// [formula: <raw latex>]
/// <unicode art lines from term-maths>
///
/// <prose paragraphs>
/// ```
///
/// Returns `(operation_name, formula_html, prose_html)`.
fn parse_explanation_text(text: &str) -> (Option<String>, String, String) {
    let mut remaining = text;

    // ── [operation: ...] ──
    let op_name = if let Some(rest) = remaining.strip_prefix("[operation: ") {
        if let Some(close) = rest.find("]\n") {
            let name = rest[..close].to_owned();
            remaining = &rest[close + 2..];
            Some(name)
        } else {
            None
        }
    } else {
        None
    };

    // ── [formula: ...] → KaTeX ──
    let formula_html = if let Some(rest) = remaining.strip_prefix("[formula: ") {
        if let Some(close) = rest.find("]\n") {
            let latex = html_escape(&rest[..close]);
            remaining = &rest[close + 2..];
            // Skip the unicode art block (everything up to the next blank line).
            if let Some(blank) = remaining.find("\n\n") {
                remaining = &remaining[blank + 2..];
            }
            format!(r#"<div class="formula-block">$${}$$</div>"#, latex)
        } else {
            String::new()
        }
    } else {
        // Fallback: old format — unicode art before the first blank line.
        if let Some(sep) = remaining.find("\n\n") {
            let art = html_escape(&remaining[..sep]);
            remaining = &remaining[sep + 2..];
            format!(r#"<pre class="formula-pre">{art}</pre>"#)
        } else {
            String::new()
        }
    };

    // ── prose ──
    // Split on double-newline for paragraphs; single newlines become <br>.
    let prose_inner = html_escape(remaining.trim());
    let prose_inner = prose_inner.replace("\n\n", r#"</p><p class="prose">"#);
    let prose_inner = prose_inner.replace('\n', "<br>");
    let prose_html = format!(r#"<p class="prose">{prose_inner}</p>"#);

    (op_name, formula_html, prose_html)
}

fn render_card(index: usize, exp: &Explanation, is_first: bool) -> String {
    let step = index + 1;

    let (op_name, formula_html, prose_html) = if let Some(t) = exp.text.as_deref() {
        parse_explanation_text(t)
    } else {
        (None, String::new(), String::new())
    };

    let op_label = op_name.as_deref().unwrap_or("Operation");
    let visual_block = render_visual_block(exp);
    let connector_class = if is_first { "step-connector first" } else { "step-connector" };

    format!(
        r#"
<div class="{connector_class}">
  <span class="step-dot">{step:02}</span>
  <span class="step-label-text">{op_label}</span>
</div>
<article class="card">
  <div class="card-content">
    <h2 class="op-title">{op_label}</h2>
    {formula_html}
    {prose_html}
  </div>
  {visual_block}
</article>
"#
    )
}

fn render_visual_block(exp: &Explanation) -> String {
    let _ = exp;
    #[cfg(feature = "plotting")]
    if let Some(visual) = &exp.visual {
        // SAFETY: `RenderVisual for AudioSamples` is the only concrete impl that
        // stores `Box<dyn ExplainDisplay>` here. The pointer cast is sound.
        #[allow(clippy::undocumented_unsafe_blocks)]
        let vis = unsafe {
            &*(visual.as_ref() as *const dyn ExplainDisplay as *const AudioSamplesVisual)
        };
        if !vis.before_fragment.is_empty() || !vis.after_fragment.is_empty() {
            return format!(
                r#"<div class="waveform-section">
  <div class="waveform-grid">
    <div class="waveform-panel before">
      <div class="waveform-panel-header">Before</div>
      <div class="waveform-inner">{}</div>
    </div>
    <div class="waveform-panel after">
      <div class="waveform-panel-header">After</div>
      <div class="waveform-inner">{}</div>
    </div>
  </div>
</div>"#,
                vis.before_fragment, vis.after_fragment
            );
        }
    }
    String::new()
}

// ── HTML template ─────────────────────────────────────────────────────────────

const DOCUMENT_TEMPLATE: &str = r#"<!DOCTYPE html>
<html lang="en" data-theme="midnight">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{TITLE}}</title>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true}],throwOnError:false});"></script>

<style>
/* ─── Reset ─────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ─── Themes ─────────────────────────────────────────────────────────────── */

/* Midnight (default) */
[data-theme="midnight"] {
  --bg:        #070a10;
  --surface:   #0f1520;
  --surface2:  #16202f;
  --border:    #1d2d42;
  --accent:    #637aff;
  --accent2:   #38bdf8;
  --text:      #cbd5e1;
  --text-muted:#475569;
  --green:     #34d399;
  --red:       #f87171;
  --katex-col: #38bdf8;
  --chain-kw:  #818cf8;
  --chain-name:#38bdf8;
  --chain-met: #a5b4fc;
  --chain-args:#475569;
  --chain-bg:  #0f1520;
  --chain-bd:  #1d2d42;
}

/* Slate — softer, slate-based dark */
[data-theme="slate"] {
  --bg:        #0f172a;
  --surface:   #1e293b;
  --surface2:  #263345;
  --border:    #334155;
  --accent:    #818cf8;
  --accent2:   #67e8f9;
  --text:      #e2e8f0;
  --text-muted:#64748b;
  --green:     #4ade80;
  --red:       #fb7185;
  --katex-col: #67e8f9;
  --chain-kw:  #c4b5fd;
  --chain-name:#67e8f9;
  --chain-met: #a5b4fc;
  --chain-args:#64748b;
  --chain-bg:  #1e293b;
  --chain-bd:  #334155;
}

/* Amber — warm dark, good for long reading */
[data-theme="amber"] {
  --bg:        #0d0900;
  --surface:   #1c1500;
  --surface2:  #261d00;
  --border:    #3d2f00;
  --accent:    #f59e0b;
  --accent2:   #fcd34d;
  --text:      #fef3c7;
  --text-muted:#92400e;
  --green:     #86efac;
  --red:       #fca5a5;
  --katex-col: #fcd34d;
  --chain-kw:  #fb923c;
  --chain-name:#fcd34d;
  --chain-met: #fde68a;
  --chain-args:#92400e;
  --chain-bg:  #1c1500;
  --chain-bd:  #3d2f00;
}

/* Light — clean, high-contrast */
[data-theme="light"] {
  --bg:        #f8fafc;
  --surface:   #ffffff;
  --surface2:  #f1f5f9;
  --border:    #e2e8f0;
  --accent:    #4f46e5;
  --accent2:   #0284c7;
  --text:      #0f172a;
  --text-muted:#64748b;
  --green:     #16a34a;
  --red:       #dc2626;
  --katex-col: #1e40af;
  --chain-kw:  #7c3aed;
  --chain-name:#0284c7;
  --chain-met: #4f46e5;
  --chain-args:#94a3b8;
  --chain-bg:  #f8fafc;
  --chain-bd:  #e2e8f0;
}

/* ─── Base ───────────────────────────────────────────────────────────────── */
:root {
  --radius-lg:   14px;
  --radius-md:   9px;
  --font-sans: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  --font-mono: "Cascadia Code", "Fira Code", "JetBrains Mono", ui-monospace, monospace;
  --card-px: 2.5rem;
  --content-max: 760px;
  --timeline-max: 1240px;
}

html { scroll-behavior: smooth; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 15px;
  line-height: 1.75;
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}

/* ─── Page header ────────────────────────────────────────────────────────── */
.page-header {
  max-width: 820px;
  margin: 0 auto;
  padding: 4rem 2rem 1.5rem;
  text-align: center;
}

.page-header h1 {
  font-size: clamp(1.8rem, 4.5vw, 2.75rem);
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 1.1;
  color: var(--text);
  margin-bottom: 0.75rem;
}

.page-header .subtitle {
  color: var(--text-muted);
  font-size: 0.875rem;
}

.page-header .subtitle strong {
  color: var(--accent);
  font-weight: 600;
}

/* ─── Theme switcher ─────────────────────────────────────────────────────── */
.theme-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 1.5rem;
}

.theme-label {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-muted);
  margin-right: 0.25rem;
}

.theme-btn {
  padding: 0.3rem 0.85rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text-muted);
  font-size: 0.75rem;
  font-weight: 600;
  font-family: var(--font-sans);
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s, background 0.15s;
  letter-spacing: 0.02em;
}

.theme-btn:hover {
  color: var(--text);
  border-color: var(--accent);
}

.theme-btn.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}

/* ─── Call-chain block ───────────────────────────────────────────────────── */
.chain-section {
  max-width: 820px;
  margin: 2rem auto 0;
  padding: 0 2rem;
}

.chain-label {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--text-muted);
  padding: 0.5rem 1rem;
  background: var(--surface2);
  border: 1px solid var(--chain-bd);
  border-bottom: none;
  border-radius: var(--radius-md) var(--radius-md) 0 0;
}

.chain-code {
  display: block;
  padding: 1.1rem 1.25rem;
  font-family: var(--font-mono);
  font-size: 0.875rem;
  line-height: 1.9;
  background: var(--chain-bg);
  border: 1px solid var(--chain-bd);
  border-radius: 0 0 var(--radius-md) var(--radius-md);
  overflow-x: auto;
  white-space: pre;
}

.cc-kw   { color: var(--chain-kw);   font-style: italic; }
.cc-var  { color: var(--text); }
.cc-name { color: var(--chain-name); font-weight: 600; }
.cc-dot  { color: var(--text-muted); }
.cc-method { color: var(--chain-met); }
.cc-args { color: var(--chain-args); }

/* ─── Timeline ───────────────────────────────────────────────────────────── */
.timeline {
  max-width: var(--timeline-max);
  margin: 0 auto;
  padding: 0 2rem 8rem;
}

/* ─── Step connector ─────────────────────────────────────────────────────── */
.step-connector {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem 0 0.75rem;
  position: relative;
}

/* Line running down from the dot to the card below */
.step-connector::after {
  content: "";
  position: absolute;
  left: 21px; top: 56px; bottom: -8px;
  width: 2px;
  background: linear-gradient(to bottom, var(--border), transparent);
  pointer-events: none;
}

/* Line running from above into the dot (hidden for first connector) */
.step-connector::before {
  content: "";
  position: absolute;
  left: 21px; top: 0; bottom: calc(100% - 14px);
  width: 2px;
  background: linear-gradient(to bottom, transparent, var(--border));
  pointer-events: none;
}

.step-connector.first::before { display: none; }

.step-dot {
  position: relative;
  z-index: 1;
  width: 44px; height: 44px;
  border-radius: 50%;
  background: var(--accent);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.78rem;
  font-weight: 800;
  color: #fff;
  flex-shrink: 0;
  box-shadow: 0 0 0 4px var(--bg), 0 0 0 6px var(--border);
  letter-spacing: 0.03em;
}

.step-label-text {
  font-size: 0.8rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-muted);
}

/* ─── Card ───────────────────────────────────────────────────────────────── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.card:hover {
  border-color: color-mix(in srgb, var(--accent) 40%, transparent);
  box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

/* ─── Card text content ──────────────────────────────────────────────────── */
.card-content {
  max-width: var(--content-max);
  margin: 0 auto;
  padding: 2.5rem var(--card-px) 2rem;
}

/* ─── Operation title ────────────────────────────────────────────────────── */
.op-title {
  font-size: 1.25rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text);
  margin-bottom: 1.5rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
}

/* ─── Formula block ──────────────────────────────────────────────────────── */
.formula-block {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1.75rem 1.25rem;
  text-align: center;
  margin-bottom: 1.75rem;
  overflow-x: auto;
}

.formula-block .katex-display { margin: 0; }
.formula-block .katex { font-size: 1.55em; }
.formula-block .katex .base,
.formula-block .katex .mord,
.formula-block .katex .mbin,
.formula-block .katex .mrel { color: var(--katex-col); }

/* Fallback for when KaTeX doesn't load */
.formula-pre {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1.25rem 1.5rem;
  font-family: var(--font-mono);
  font-size: 0.9rem;
  color: var(--accent2);
  white-space: pre;
  overflow-x: auto;
  text-align: center;
  line-height: 1.5;
  margin-bottom: 1.75rem;
}

/* ─── Prose ──────────────────────────────────────────────────────────────── */
p.prose {
  color: var(--text);
  font-size: 0.925rem;
  line-height: 1.85;
  margin-bottom: 0.75rem;
}

p.prose:last-of-type { margin-bottom: 0; }

/* ─── Waveform section — full card width ─────────────────────────────────── */
.waveform-section {
  border-top: 1px solid var(--border);
  background: var(--bg);
  padding: 1.5rem 1.75rem 1.75rem;
}

.waveform-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
}

.waveform-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.waveform-panel-header {
  padding: 0.45rem 1rem;
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  border-bottom: 1px solid var(--border);
}

.waveform-panel.before .waveform-panel-header {
  color: var(--red);
  background: color-mix(in srgb, var(--red) 8%, transparent);
}

.waveform-panel.after .waveform-panel-header {
  color: var(--green);
  background: color-mix(in srgb, var(--green) 8%, transparent);
}

.waveform-inner { min-height: 300px; }

.waveform-inner .plotly-graph-div,
.waveform-inner > div { width: 100% !important; }

/* ─── Responsive ─────────────────────────────────────────────────────────── */
@media (max-width: 860px) {
  :root { --card-px: 1.5rem; }
  .waveform-grid { grid-template-columns: 1fr; }
  .waveform-section { padding: 1.25rem; }
}

@media (max-width: 540px) {
  .page-header { padding: 2.5rem 1.25rem 1.25rem; }
  .timeline { padding: 0 1rem 4rem; }
  .chain-section { padding: 0 1rem; }
}
</style>
</head>
<body>

<header class="page-header">
  <h1>{{TITLE}}</h1>
  <p class="subtitle">
    Generated by <strong>audio_samples</strong> educational mode
    &mdash; formula, explanation, and waveform comparison for each step.
  </p>
  <div class="theme-row">
    <span class="theme-label">Theme</span>
    <button class="theme-btn active" data-theme="midnight">Midnight</button>
    <button class="theme-btn"        data-theme="slate">Slate</button>
    <button class="theme-btn"        data-theme="amber">Amber</button>
    <button class="theme-btn"        data-theme="light">Light</button>
  </div>
</header>

{{CHAIN}}

<main class="timeline">
{{CARDS}}
</main>

<script>
// ── Theme switcher ────────────────────────────────────────────────────────────
(function () {
  var root = document.documentElement;
  var btns = document.querySelectorAll('.theme-btn');

  function applyTheme(name) {
    root.dataset.theme = name;
    btns.forEach(function (b) {
      b.classList.toggle('active', b.dataset.theme === name);
    });
    try { localStorage.setItem('edu-theme', name); } catch (_) {}
  }

  btns.forEach(function (btn) {
    btn.addEventListener('click', function () { applyTheme(btn.dataset.theme); });
  });

  try {
    var saved = localStorage.getItem('edu-theme');
    if (saved) applyTheme(saved);
  } catch (_) {}
}());

// ── Plotly post-render ────────────────────────────────────────────────────────
// Relayout all Plotly divs to a consistent height and theme-neutral background.
// Runs after window load to ensure Plotly has finished drawing.
window.addEventListener('load', function () {
  function getComputedVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function relayoutAll() {
    var layout = {
      height: 300,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor:  'rgba(0,0,0,0)',
      font:  { color: getComputedVar('--text-muted'), size: 11, family: 'Inter, system-ui, sans-serif' },
      xaxis: { gridcolor: getComputedVar('--border'), linecolor: getComputedVar('--border'),
               tickcolor: getComputedVar('--text-muted'), zerolinecolor: getComputedVar('--border') },
      yaxis: { gridcolor: getComputedVar('--border'), linecolor: getComputedVar('--border'),
               tickcolor: getComputedVar('--text-muted'), zerolinecolor: getComputedVar('--border') },
      margin: { t: 16, r: 16, b: 44, l: 56 }
    };
    document.querySelectorAll('.waveform-inner .plotly-graph-div').forEach(function (div) {
      try { Plotly.relayout(div, layout); } catch (_) {}
    });
  }

  relayoutAll();

  // Re-apply theme colours when the user switches theme
  document.querySelectorAll('.theme-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      // Small delay so CSS variables are updated before we read them
      setTimeout(relayoutAll, 50);
    });
  });
});
</script>

</body>
</html>
"#;
