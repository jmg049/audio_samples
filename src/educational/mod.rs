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
//! use audio_samples::{AudioSamples, AudioProcessingExt, Explainable, ExplainMode, NormalizationConfig, sample_rate};
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

pub mod explain;

pub use explain::{
    Explain, ExplainContext, ExplainDoc, ExplainFor, Formula, Quantity, SeeAlso, Symbol, Table,
    Topic, Warning, WarningLevel, topics,
};

/// Opaque visual explanation for one operation step.
///
/// Holds Plotly inline HTML fragments when compiled with `plotting`; otherwise
/// a zero-size stub. The visual is not displayed immediately - collect all
/// explanations first and pass them to [`render_explanation_document`].
pub struct AudioSamplesVisual {
    #[cfg(feature = "plotting")]
    pub(crate) before_fragment: String,
    #[cfg(feature = "plotting")]
    pub(crate) after_fragment: String,
}

// ─── Recovering the plot fragments from a `Box<dyn ExplainDisplay>` ─────────
//
// `explainable::ExplainDisplay` has no `Any` supertrait, so a
// `Box<dyn ExplainDisplay>` cannot be downcast, and `Explanation::visual` is a
// public field that may hold a visual of any type. Casting it to
// `AudioSamplesVisual` with a pointer cast would be undefined behaviour for a
// foreign visual. Instead, the fragments are extracted with safe code only:
//
// 1. `LIVE_VISUALS` holds the heap addresses of the `AudioSamplesVisual`s
//    created by `render_visual`; `Drop` removes an address before the value is
//    freed, so a listed address always belongs to a live `AudioSamplesVisual`.
//    The registry is only used as a filter (we never dereference an address
//    from it), so it avoids calling `display()` on foreign visuals, which may
//    have side effects.
// 2. `render_visual_block` arms a thread-local capture slot and calls the
//    visual's own `display()`. `AudioSamplesVisual::display` copies its
//    fragments into the slot when it is armed and is a no-op otherwise, so
//    `Explanation::surface()` stays silent.
//
// No `unsafe` is involved: the worst a mismatch could do is call some other
// type's `display()`, never misinterpret memory.

#[cfg(feature = "plotting")]
static LIVE_VISUALS: std::sync::Mutex<Option<std::collections::HashSet<usize>>> =
    std::sync::Mutex::new(None);

#[cfg(feature = "plotting")]
thread_local! {
    static CAPTURE: std::cell::RefCell<Option<Option<(String, String)>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(feature = "plotting")]
fn live_visuals() -> std::sync::MutexGuard<'static, Option<std::collections::HashSet<usize>>> {
    LIVE_VISUALS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(feature = "plotting")]
impl Drop for AudioSamplesVisual {
    fn drop(&mut self) {
        let addr = std::ptr::from_ref::<Self>(self) as usize;
        if let Some(set) = live_visuals().as_mut() {
            set.remove(&addr);
        }
    }
}

impl ExplainDisplay for AudioSamplesVisual {
    fn display(&self) {
        // Individual display is suppressed: call render_explanation_document /
        // open_explanation_document instead. When the document renderer has
        // armed the capture slot, hand it the fragments.
        #[cfg(feature = "plotting")]
        CAPTURE.with(|slot| {
            if let Some(captured) = slot.borrow_mut().as_mut() {
                *captured = Some((self.before_fragment.clone(), self.after_fragment.clone()));
            }
        });
    }
}

impl<T: StandardSample> RenderVisual for AudioSamples<'static, T> {
    fn render_visual(before: &Self, after: &Self) -> Box<dyn ExplainDisplay> {
        #[cfg(feature = "plotting")]
        {
            use crate::operations::AudioPlotting;
            use crate::operations::plotting::WaveformPlotParams;
            // Use None so Plotly generates a unique UUID per div - a fixed ID
            // causes every step to share one element and only the last renders.
            let before_frag = before
                .plot_waveform(&WaveformPlotParams::default())
                .map(|p| p.inline_html(None))
                .unwrap_or_default();
            let after_frag = after
                .plot_waveform(&WaveformPlotParams::default())
                .map(|p| p.inline_html(None))
                .unwrap_or_default();
            let visual = Box::new(AudioSamplesVisual {
                before_fragment: before_frag,
                after_fragment: after_frag,
            });
            let addr = std::ptr::from_ref::<AudioSamplesVisual>(&visual) as usize;
            live_visuals()
                .get_or_insert_with(Default::default)
                .insert(addr);
            visual
        }
        #[cfg(not(feature = "plotting"))]
        {
            let _ = (before, after);
            Box::new(AudioSamplesVisual {})
        }
    }
}

impl<T: StandardSample> Explainable for AudioSamples<'static, T> {}

/// Assemble all explanations from a chain into a single HTML document.
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
/// Returns [`crate::AudioSampleError::Io`] if the file cannot be written or the
/// browser cannot be opened.
pub fn open_explanation_document(
    explanations: &[Explanation],
    title: &str,
) -> crate::AudioSampleResult<()> {
    let html = render_explanation_document(explanations, title);
    let path = std::env::temp_dir().join(format!(
        "audio_samples_explain_{}.html",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    ));
    std::fs::write(&path, &html)
        .map_err(|e| crate::AudioSampleError::io("write explanation document", &e))?;
    open::that(&path).map_err(|e| {
        crate::AudioSampleError::io("open explanation document", &std::io::Error::other(e))
    })?;
    Ok(())
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Convert a display-friendly operation name into the Rust method name.
///
/// A trailing `" (in-place)"` (added by the `_in_place` explanations) maps to
/// the `_in_place` variant of the method. Unknown names map to `"operation"`.
fn op_name_to_method(name: &str) -> String {
    let (base, suffix) = match name.strip_suffix(" (in-place)") {
        Some(base) => (base, "_in_place"),
        None => (name, ""),
    };
    let method = match base {
        "Normalize" => "normalize",
        "Scale" => "scale",
        "Apply Window" => "apply_window",
        "Apply FIR Filter" => "apply_filter",
        "Apply Window (f64)" => "apply_window_f64",
        "Apply FIR Filter (f64)" => "apply_filter_f64",
        "\u{03bc}-law Compress" => "mu_compress",
        "\u{03bc}-law Expand" => "mu_expand",
        "Low-Pass Filter" => "low_pass_filter",
        "High-Pass Filter" => "high_pass_filter",
        "Band-Pass Filter" => "band_pass_filter",
        "Remove DC Offset" => "remove_dc_offset",
        "Clip" => "clip",
        "Compressor" => "apply_compressor",
        "Limiter" => "apply_limiter",
        "EQ Band" => "apply_eq_band",
        "Resample" => "resample",
        "Resample by Ratio" => "resample_by_ratio",
        _ => return "operation".to_string(),
    };
    format!("{method}{suffix}")
}

/// Produce the educational explanation text for an operation that lives outside
/// the `AudioProcessing` trait (and so has no auto-generated `explain_text_*`
/// method on the explaining chain).
///
/// Dynamics (`"Compressor"`, `"Limiter"`), parametric EQ (`"EQ Band"`), and
/// `"Resample"` are dispatched here by their display name to the matching
/// explanation text. The returned string uses the same
/// `[operation: …]` / `[formula: …]` layout as the chain explanations, so it can be
/// fed straight into [`render_explanation_document`] via an [`Explanation`].
///
/// Returns `None` for an unrecognised name.
#[cfg(feature = "processing")]
pub fn explain_text_for_op<T: StandardSample>(
    op_name: &str,
    before: &AudioSamples<'static, T>,
    after: &AudioSamples<'static, T>,
) -> Option<String> {
    use processing::{
        explain_text_compressor, explain_text_eq_band, explain_text_limiter, explain_text_resample,
    };
    Some(match op_name {
        "Compressor" => explain_text_compressor(before, after),
        "Limiter" => explain_text_limiter(before, after),
        "EQ Band" => explain_text_eq_band(before, after),
        "Resample" => explain_text_resample(before, after),
        _ => return None,
    })
}

/// Extract the `[operation: Name]` value from a raw explanation text string.
fn extract_op_name(text: &str) -> Option<&str> {
    let rest = text.strip_prefix("[operation: ")?;
    let end = rest.find("]\n")?;
    Some(&rest[..end])
}

/// Render the Rust call chain for all operations in the explanation list.
fn render_chain_block(explanations: &[Explanation]) -> String {
    let methods: Vec<String> = explanations
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
///
/// ```text
/// [operation: Name]
/// [formula: raw LaTeX]
/// unicode art lines from term-maths
///
/// prose paragraphs
/// ```
///
/// Returns `(operation_name, formula_html, prose_html)`.
fn parse_explanation_text(text: &str) -> (Option<String>, String, String) {
    let mut remaining = text;

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
        // Fallback: a formula given as plain text art before the first blank line.
        if let Some(sep) = remaining.find("\n\n") {
            let art = html_escape(&remaining[..sep]);
            remaining = &remaining[sep + 2..];
            format!(r#"<pre class="formula-pre">{art}</pre>"#)
        } else {
            String::new()
        }
    };

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
    let connector_class = if is_first {
        "step-connector first"
    } else {
        "step-connector"
    };

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

/// Extracts the before/after Plotly fragments of `visual` if (and only if) it
/// is a live [`AudioSamplesVisual`] created by this crate. See the comment on
/// `LIVE_VISUALS` for why this is sound without a downcast.
#[cfg(feature = "plotting")]
fn captured_fragments(visual: &dyn ExplainDisplay) -> Option<(String, String)> {
    let addr = std::ptr::from_ref(visual).cast::<()>() as usize;
    let ours = live_visuals()
        .as_ref()
        .is_some_and(|set| set.contains(&addr));
    if !ours {
        return None;
    }
    CAPTURE.with(|slot| *slot.borrow_mut() = Some(None));
    visual.display();
    CAPTURE.with(|slot| slot.borrow_mut().take().flatten())
}

fn render_visual_block(exp: &Explanation) -> String {
    let _ = exp;
    #[cfg(feature = "plotting")]
    if let Some((before, after)) = exp.visual.as_deref().and_then(captured_fragments) {
        if !before.is_empty() || !after.is_empty() {
            return format!(
                r#"<div class="waveform-section">
  <div class="waveform-grid">
    <div class="waveform-panel before">
      <div class="waveform-panel-header">Before</div>
      <div class="waveform-inner">{before}</div>
    </div>
    <div class="waveform-panel after">
      <div class="waveform-panel-header">After</div>
      <div class="waveform-inner">{after}</div>
    </div>
  </div>
</div>"#
            );
        }
    }
    String::new()
}

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

/* Slate - softer, slate-based dark */
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

/* Amber - warm dark, good for long reading */
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

/* Light - clean, high-contrast */
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

/* ─── Waveform section - full card width ─────────────────────────────────── */
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

#[cfg(all(test, feature = "processing"))]
mod tests {
    use super::*;
    use crate::sample_rate;
    use ndarray::array;

    fn sample() -> AudioSamples<'static, f32> {
        AudioSamples::new_mono(
            array![0.1f32, 0.8, -0.2, 0.9, -0.7, 0.3, 0.5, -0.4],
            sample_rate!(44100),
        )
        .expect("valid mono buffer")
    }

    // ── Each dispatched explanation is non-empty and mentions its key concept ──

    #[test]
    fn compressor_explanation_mentions_threshold_and_ratio() {
        let a = sample();
        let text = explain_text_for_op("Compressor", &a, &a).expect("compressor is wired");
        assert!(!text.is_empty());
        assert!(text.contains("[operation: Compressor]"));
        assert!(text.contains("threshold"));
        assert!(text.contains("ratio"));
        assert!(text.contains("attack") && text.contains("release"));
    }

    #[test]
    fn limiter_explanation_mentions_ceiling() {
        let a = sample();
        let text = explain_text_for_op("Limiter", &a, &a).expect("limiter is wired");
        assert!(!text.is_empty());
        assert!(text.contains("[operation: Limiter]"));
        assert!(text.contains("ceiling"));
    }

    #[test]
    fn eq_band_explanation_mentions_frequency_gain_q() {
        let a = sample();
        let text = explain_text_for_op("EQ Band", &a, &a).expect("eq band is wired");
        assert!(!text.is_empty());
        assert!(text.contains("[operation: EQ Band]"));
        assert!(text.contains("frequency"));
        assert!(text.contains("gain"));
        assert!(text.contains("q_factor") || text.contains("Q "));
        assert!(text.contains("biquad"));
    }

    #[test]
    fn resample_explanation_mentions_sample_rate_and_aliasing() {
        let a = sample();
        let text = explain_text_for_op("Resample", &a, &a).expect("resample is wired");
        assert!(!text.is_empty());
        assert!(text.contains("[operation: Resample]"));
        assert!(text.contains("sample rate"));
        assert!(text.contains("alias"));
    }

    #[test]
    fn unrecognised_op_returns_none() {
        let a = sample();
        assert!(explain_text_for_op("Nonexistent", &a, &a).is_none());
    }

    // ── Dispatched ops resolve to their specific method, not the fallback ──────

    #[test]
    fn dispatched_ops_do_not_fall_back_to_generic_operation() {
        for (name, method) in [
            ("Compressor", "apply_compressor"),
            ("Limiter", "apply_limiter"),
            ("EQ Band", "apply_eq_band"),
            ("Resample", "resample"),
        ] {
            let resolved = op_name_to_method(name);
            assert_eq!(resolved, method, "{name} should resolve to {method}");
            assert_ne!(resolved, "operation", "{name} must not hit the fallback");
        }
    }

    #[test]
    fn existing_ops_still_resolve_and_unknown_falls_back() {
        assert_eq!(op_name_to_method("Normalize"), "normalize");
        assert_eq!(op_name_to_method("Clip"), "clip");
        assert_eq!(op_name_to_method("Totally Unknown Op"), "operation");
    }

    #[test]
    fn in_place_names_resolve_to_in_place_methods() {
        assert_eq!(op_name_to_method("Clip (in-place)"), "clip_in_place");
        assert_eq!(
            op_name_to_method("Normalize (in-place)"),
            "normalize_in_place"
        );
        assert_eq!(
            op_name_to_method("Resample (in-place)"),
            "resample_in_place"
        );
        assert_eq!(
            op_name_to_method("Resample by Ratio (in-place)"),
            "resample_by_ratio_in_place"
        );
        assert_eq!(op_name_to_method("Bogus (in-place)"), "operation");
    }

    // ── Explanatory text states the correct formulas ────────────────────────

    use crate::NormalizationConfig;
    use crate::operations::traits::{AudioProcessing, AudioProcessingExplainText};

    type Edu = AudioSamples<'static, f32>;

    fn sine(n: usize) -> Edu {
        let data: ndarray::Array1<f32> = (0..n)
            .map(|i| 0.6 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 8000.0).sin() + 0.1)
            .collect();
        AudioSamples::new_mono(data, sample_rate!(8000)).expect("valid")
    }

    /// Every chain text and dispatcher text, for blanket checks.
    fn all_texts() -> Vec<String> {
        let a = sample();
        let b = a.clone();
        let mut v = vec![
            Edu::explain_text_normalize(&a, &b),
            Edu::explain_text_scale(&a, &b),
            Edu::explain_text_apply_window(&a, &b),
            Edu::explain_text_apply_filter(&a, &b),
            Edu::explain_text_apply_window_f64(&a, &b),
            Edu::explain_text_apply_filter_f64(&a, &b),
            Edu::explain_text_mu_compress(&a, &b),
            Edu::explain_text_mu_expand(&a, &b),
            Edu::explain_text_low_pass_filter(&a, &b),
            Edu::explain_text_high_pass_filter(&a, &b),
            Edu::explain_text_band_pass_filter(&a, &b),
            Edu::explain_text_remove_dc_offset(&a, &b),
            Edu::explain_text_clip(&a, &b),
            Edu::explain_text_clip_in_place(&a, &b),
            Edu::explain_text_scale_in_place(&a, &b),
        ];
        for op in ["Compressor", "Limiter", "EQ Band", "Resample"] {
            v.push(explain_text_for_op(op, &a, &b).expect("wired"));
        }
        v
    }

    #[test]
    fn terminal_rendering_leaves_no_raw_latex_commands() {
        for text in all_texts() {
            let (_, rest) = text.split_once("]\n[formula: ").expect("has formula");
            let (_, after_formula) = rest.split_once("]\n").expect("closed formula tag");
            let (art, _) = after_formula.split_once("\n\n").expect("art then prose");
            assert!(!art.trim().is_empty(), "empty rendering in {text}");
            assert!(
                !art.contains('\\'),
                "raw LaTeX left in terminal rendering:\n{art}"
            );
            assert!(!text.contains("operatorname"), "{text}");
        }
    }

    #[test]
    fn filter_texts_describe_first_order_butterworth_not_brick_wall() {
        let a = sample();
        for text in [
            Edu::explain_text_low_pass_filter(&a, &a),
            Edu::explain_text_high_pass_filter(&a, &a),
            Edu::explain_text_band_pass_filter(&a, &a),
        ] {
            assert!(text.contains("first-order Butterworth"), "{text}");
            assert!(text.contains("\u{2212}3 dB"), "{text}");
            assert!(text.contains("6 dB per octave"), "{text}");
            assert!(!text.contains("begin{cases}"), "brick-wall formula: {text}");
            assert!(text.contains(r"\tan\frac{\pi f}{f_s}"), "{text}");
        }
        let lpf = Edu::explain_text_low_pass_filter(&a, &a);
        assert!(!lpf.contains("pitch"), "LPF does not extract pitch");
        let bpf = Edu::explain_text_band_pass_filter(&a, &a);
        assert!(!bpf.contains("retains only"));
        assert!(bpf.contains("not identical"));
    }

    /// The difference equation given in the low-/high-pass texts matches the
    /// impulse response of the implementation.
    #[test]
    fn first_order_difference_equations_in_text_match_code() {
        let fs = 8000.0f64;
        let fc = 1000.0f64;
        let wc = (std::f64::consts::PI * fc / fs).tan();
        let p = (1.0 - wc) / (1.0 + wc);
        let mut imp = ndarray::Array1::<f64>::zeros(8);
        imp[0] = 1.0;
        let x = AudioSamples::new_mono(imp, sample_rate!(8000)).unwrap();
        let lp = x.low_pass_filter(fc).unwrap();
        assert!((lp[0] - (1.0 - p) / 2.0).abs() < 1e-12);
        assert!((lp[1] - (1.0 - p) * (1.0 + p) / 2.0).abs() < 1e-12);
        let hp = x.high_pass_filter(fc).unwrap();
        assert!((hp[0] - (1.0 + p) / 2.0).abs() < 1e-12);
        assert!((hp[1] - ((1.0 + p) / 2.0 * (p - 1.0))).abs() < 1e-12);
    }

    #[test]
    fn fir_text_limits_linear_phase_to_symmetric_taps_and_valid_length() {
        let a = sine(64);
        let taps = [0.25f32, 0.5, 0.25];
        let b = a
            .apply_filter(non_empty_slice::NonEmptySlice::new(&taps).unwrap())
            .unwrap();
        let text = Edu::explain_text_apply_filter(&a, &b);
        assert!(text.contains("only when the"), "{text}");
        assert!(text.contains("symmetric"), "{text}");
        assert!(text.contains("not linear-phase"), "{text}");
        assert!(!text.contains("have a linear phase response"));
        assert!(text.contains("Filter length M:  3"), "{text}");
        assert!(text.contains("64  \u{2192}  62"), "{text}");
    }

    #[test]
    fn mu_law_text_says_14_bit_and_flags_implementation_mismatch() {
        let a = sine(32);
        let b = a.mu_compress(255.0).unwrap();
        let text = Edu::explain_text_mu_compress(&a, &b);
        assert!(text.contains("14-bit"), "{text}");
        assert!(text.contains("A-law variant codes 13-bit"), "{text}");
        assert!(!text.contains("13-bit quality signal"));
        assert!(text.contains(r"\ln(1 + \mu |x|)"), "{text}");
        // The caveat appears exactly when the implementation deviates from
        // the standard curve (compress(0) must be 0).
        let zero = AudioSamples::new_mono(ndarray::array![0.0f64], sample_rate!(8000)).unwrap();
        let buggy = zero.mu_compress(255.0).unwrap()[0].abs() > 1e-12;
        assert_eq!(text.contains("known bug"), buggy, "{text}");
        let expand = Edu::explain_text_mu_expand(&b, &a);
        assert!(!expand.contains("exact inverse of \u{03bc}-law compression"));
        assert_eq!(expand.contains("does not give back"), buggy, "{expand}");
    }

    #[test]
    fn normalize_text_follows_the_method_actually_used() {
        let a = sine(256);
        let cases = [
            (NormalizationConfig::peak(0.5), "NormalizationMethod::Peak"),
            (
                NormalizationConfig::min_max(-1.0, 1.0),
                "NormalizationMethod::MinMax",
            ),
            (NormalizationConfig::mean(), "NormalizationMethod::Mean"),
            (NormalizationConfig::zscore(), "NormalizationMethod::ZScore"),
        ];
        for (cfg, expect) in cases {
            let b = a.normalize(cfg).unwrap();
            let text = Edu::explain_text_normalize(&a, &b);
            assert!(text.contains(expect), "expected {expect}: {text}");
        }
        let peak =
            Edu::explain_text_normalize(&a, &a.normalize(NormalizationConfig::peak(0.5)).unwrap());
        assert!(peak.contains(r"A \cdot \frac{x[n]}"), "{peak}");
        let mean =
            Edu::explain_text_normalize(&a, &a.normalize(NormalizationConfig::mean()).unwrap());
        assert!(
            !mean.contains(r"\max_m"),
            "peak formula shown for mean: {mean}"
        );
    }

    #[test]
    fn f64_variant_texts_name_their_methods() {
        let a = sample();
        let w = Edu::explain_text_apply_window_f64_in_place(&a, &a);
        assert!(
            w.starts_with("[operation: Apply Window (f64) (in-place)]"),
            "{w}"
        );
        assert_eq!(
            op_name_to_method("Apply Window (f64) (in-place)"),
            "apply_window_f64_in_place"
        );
        let f = Edu::explain_text_apply_filter_f64(&a, &a);
        assert!(f.starts_with("[operation: Apply FIR Filter (f64)]"), "{f}");
        assert_eq!(
            op_name_to_method("Apply FIR Filter (f64)"),
            "apply_filter_f64"
        );
    }

    #[test]
    fn window_text_has_no_flat_top_and_uses_hann() {
        let a = sample();
        let text = Edu::explain_text_apply_window(&a, &a);
        assert!(!text.to_lowercase().contains("flat-top"));
        assert!(!text.contains("Hanning"));
        assert!(text.contains("Hann"));
    }

    #[test]
    fn dynamics_texts_match_implementation() {
        let a = sample();
        let comp = explain_text_for_op("Compressor", &a, &a).unwrap();
        assert!(comp.contains("63 %") && comp.contains("2.2"), "{comp}");
        assert!(comp.contains("Giannoulis"), "{comp}");
        assert!(comp.contains("[T \u{2212} W/2, T + W/2]"), "{comp}");
        assert!(comp.contains("max-hold"), "{comp}");
        assert!(comp.contains("time-aligned"), "{comp}");
        let lim = explain_text_for_op("Limiter", &a, &a).unwrap();
        assert!(lim.contains(r"\min\left(1, \frac{c}{e[n]}\right)"), "{lim}");
        assert!(lim.contains("not a brick-wall"), "{lim}");
        assert!(lim.contains("not guaranteed"), "{lim}");
        assert!(!lim.contains("guarding against") && !lim.contains("guards against"));
        assert!(!lim.contains(r"|y[n]| \le"));
    }

    #[test]
    fn resample_and_clip_texts_match_implementation() {
        let a = sample();
        let rs = explain_text_for_op("Resample", &a, &a).unwrap();
        assert!(rs.contains("70 dB"), "{rs}");
        assert!(rs.contains("no delay"), "{rs}");
        assert!(rs.contains("time-aligned"), "{rs}");
        assert!(!rs.contains("no anti-aliasing"), "{rs}");
        assert!(!rs.contains("linear interpolation) is fast but rolls off"));
        let b = a.clip(-0.5, 0.5).unwrap();
        let clip = Edu::explain_text_clip(&a, &b);
        assert!(clip.contains("Samples clamped:  3 of 8"), "{clip}");
        assert!(!clip.contains("proportional to the fraction"));
        assert!(clip.contains("odd harmonics"));
    }

    #[test]
    fn in_place_texts_carry_in_place_names() {
        let a = sample();
        let t = Edu::explain_text_normalize_in_place(&a, &a);
        assert!(t.starts_with("[operation: Normalize (in-place)]\n"), "{t}");
        let t = Edu::explain_text_clip_in_place(&a, &a);
        assert!(t.starts_with("[operation: Clip (in-place)]\n"), "{t}");
    }

    /// Renders a whole document to a file (instead of opening a browser) and
    /// checks that every card and the call chain made it in.
    #[test]
    fn document_renders_to_file() {
        let a = sample();
        let b = a.normalize(NormalizationConfig::peak(1.0)).unwrap();
        let exps = vec![
            explainable::Explanation::new(
                explainable::ExplainMode::Text,
                Some(Edu::explain_text_normalize_in_place(&a, &b)),
                None,
            ),
            explainable::Explanation::new(
                explainable::ExplainMode::Text,
                Some(explain_text_for_op("Limiter", &b, &b).unwrap()),
                None,
            ),
        ];
        let html = render_explanation_document(&exps, "Test <doc>");
        let path = std::env::temp_dir().join(format!(
            "audio_samples_edu_texts_test_{}.html",
            std::process::id()
        ));
        std::fs::write(&path, &html).unwrap();
        let back = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(back.contains("Test &lt;doc&gt;"));
        assert!(back.contains("normalize_in_place"));
        assert!(back.contains("apply_limiter"));
        assert!(back.contains("Normalize (in-place)"));
        assert!(back.contains(r"$$g[n] = \min"));
    }

    /// A foreign `ExplainDisplay` in `Explanation::visual` must be ignored,
    /// not reinterpreted as an `AudioSamplesVisual`.
    #[test]
    fn foreign_visuals_are_ignored_safely() {
        use std::cell::Cell;
        struct Foreign(Cell<u32>);
        impl ExplainDisplay for Foreign {
            fn display(&self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let exp = explainable::Explanation::new(
            explainable::ExplainMode::Visual,
            None,
            Some(Box::new(Foreign(Cell::new(0)))),
        );
        assert_eq!(render_visual_block(&exp), "");
    }

    #[cfg(feature = "plotting")]
    #[test]
    fn own_visuals_are_rendered_and_unregistered_on_drop() {
        let a = sample();
        let b = a.scale(0.5);
        let visual = <Edu as RenderVisual>::render_visual(&a, &b);
        let addr = std::ptr::from_ref(visual.as_ref()).cast::<()>() as usize;
        assert!(live_visuals().as_ref().unwrap().contains(&addr));
        let exp =
            explainable::Explanation::new(explainable::ExplainMode::Visual, None, Some(visual));
        let block = render_visual_block(&exp);
        assert!(block.contains("waveform-panel before"), "{block}");
        // `surface()`-style display outside the renderer stays silent.
        exp.visual.as_ref().unwrap().display();
        drop(exp);
        assert!(!live_visuals().as_ref().unwrap().contains(&addr));
    }
}
