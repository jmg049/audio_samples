use plotly::Plot;
use std::path::Path;

use super::PlotUtils;
use crate::AudioSampleResult;

/// Layout configuration for composite plots
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeLayout {
    /// Stack plots vertically (most common for waveform + spectrogram)
    Vertical,
    /// Arrange plots side-by-side
    Horizontal,
    /// Custom grid layout
    Grid { rows: usize, cols: usize },
}

/// Trait that all plot types must implement to be composable
pub trait PlotComponent {
    /// Get the underlying Plotly plot
    fn get_plot(&self) -> &Plot;

    /// Get a mutable reference to the underlying plot
    fn get_plot_mut(&mut self) -> &mut Plot;

    /// Whether this plot should share X axis with other time-based plots
    fn requires_shared_x_axis(&self) -> bool;
}

/// A composite plot that combines multiple HTML plots
///
/// Note: Due to limitations in plotly.rs, this currently generates
/// separate HTML plots and combines them in an HTML document.
/// Future versions may support proper subplot composition.
pub struct CompositePlot {
    html_plots: Vec<String>,
    layout: CompositeLayout,
}

impl CompositePlot {
    /// Create a new empty composite plot
    pub fn new() -> Self {
        Self {
            html_plots: Vec::new(),
            layout: CompositeLayout::Vertical,
        }
    }

    /// Add a plot to the composition
    pub fn add_plot<P: PlotComponent + 'static>(mut self, plot: P) -> Self {
        // Convert plot to HTML and store it
        self.html_plots.push(plot.get_plot().to_html());
        self
    }

    /// Set the layout for the composite plot
    pub fn layout(mut self, layout: CompositeLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Build the final composite plot
    pub fn build(self) -> AudioSampleResult<Self> {
        if self.html_plots.is_empty() {
            return Err(crate::AudioSampleError::Parameter(
                crate::ParameterError::InvalidValue {
                    parameter: "plots".to_string(),
                    reason: "Cannot build composite plot with no plots".to_string(),
                },
            ));
        }

        Ok(self)
    }

    /// Generate combined HTML with multiple plots arranged according to layout
    fn generate_composite_html(&self) -> String {
        let container_style = match self.layout {
            CompositeLayout::Vertical => "flex-direction: column;",
            CompositeLayout::Horizontal => "flex-direction: row;",
            CompositeLayout::Grid { rows: _, cols: _ } => "flex-direction: row; flex-wrap: wrap;",
        };

        let mut html = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Composite Plot</title>
    <style>
        body {{ margin: 0; padding: 0; }}
        .composite-container {{
            display: flex;
            {}
            width: 100%;
            height: 100vh;
        }}
        .plot-item {{
            flex: 1;
            min-height: 0;
            min-width: 0;
        }}
        .plot-item iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
    </style>
</head>
<body>
    <div class="composite-container">
"#,
            container_style
        );

        for plot_html in self.html_plots.iter() {
            // Encode the HTML as base64 data URI for iframe src
            let encoded = base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                plot_html.as_bytes(),
            );
            html.push_str(&format!(
                r#"        <div class="plot-item">
            <iframe src="data:text/html;base64,{}"></iframe>
        </div>
"#,
                encoded
            ));
        }

        html.push_str(
            r#"    </div>
</body>
</html>"#,
        );

        html
    }
}

impl Default for CompositePlot {
    fn default() -> Self {
        Self::new()
    }
}

impl PlotUtils for CompositePlot {
    fn html(&self) -> AudioSampleResult<String> {
        Ok(self.generate_composite_html())
    }

    #[cfg(feature = "html_view")]
    fn show(&self) -> AudioSampleResult<()> {
        let html = self.html()?;
        html_view::show(html).map_err(|e| {
            crate::AudioSampleError::unsupported(format!("Failed to show plot: {}", e))
        })?;
        Ok(())
    }

    fn save<P: AsRef<Path>>(&self, path: P) -> AudioSampleResult<()> {
        let path = path.as_ref();
        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("html");

        match extension.to_lowercase().as_str() {
            "html" => {
                let html = self.html()?;
                std::fs::write(path, html).map_err(|e| {
                    crate::AudioSampleError::unsupported(format!(
                        "Failed to write HTML file: {}",
                        e
                    ))
                })?;
                Ok(())
            }
            _ => Err(crate::AudioSampleError::Parameter(
                crate::ParameterError::InvalidValue {
                    parameter: "file_extension".to_string(),
                    reason: format!(
                        "Composite plots only support HTML output. Got: {}",
                        extension
                    ),
                },
            )),
        }
    }
}
