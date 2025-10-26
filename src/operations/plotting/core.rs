//! Core types and traits for the composable plotting system.
//!
//! This module defines the foundational architecture for building composable,
//! reusable plot elements that can be combined in flexible layouts.
//!
//! Now powered by Plotly.rs for professional-quality, anti-aliased plots.

use plotly::common::{
    ColorScale, ColorScalePalette, DashType, Font, Line, Marker, MarkerSymbol, Title,
};
use plotly::layout::{Axis, Layout};
use plotly::{HeatMap, Plot, Scatter};

use crate::operations::types::WindowType;

/// Result type for plotting operations - use AudioSampleResult consistently
pub use crate::AudioSampleResult as PlotResult;

/// Bounds for plot data in 2D space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlotBounds {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

impl PlotBounds {
    pub fn new(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    pub fn expand_to_include(&mut self, other: &PlotBounds) {
        self.x_min = self.x_min.min(other.x_min);
        self.x_max = self.x_max.max(other.x_max);
        self.y_min = self.y_min.min(other.y_min);
        self.y_max = self.y_max.max(other.y_max);
    }

    pub fn with_margin(&self, margin_percent: f64) -> Self {
        let x_margin = (self.x_max - self.x_min) * margin_percent;
        let y_margin = (self.y_max - self.y_min) * margin_percent;
        Self {
            x_min: self.x_min - x_margin,
            x_max: self.x_max + x_margin,
            y_min: self.y_min - y_margin,
            y_max: self.y_max + y_margin,
        }
    }
}

/// Metadata about a plot element
#[derive(Debug, Clone)]
pub struct PlotMetadata {
    pub title: Option<String>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub legend_label: Option<String>,
    pub z_order: i32, // For layering elements
}

impl Default for PlotMetadata {
    fn default() -> Self {
        Self {
            title: None,
            x_label: None,
            y_label: None,
            legend_label: None,
            z_order: 0,
        }
    }
}

/// Enum for different types of plotly traces
#[derive(Debug)]
pub enum PlotTrace {
    Scatter(Scatter<f64, f64>),
    HeatMap(HeatMap<f64, f64, f64>),
}

impl PlotTrace {
    /// Add this trace to a plotly Plot
    pub fn add_to_plot(self, plot: &mut Plot) {
        match self {
            PlotTrace::Scatter(trace) => plot.add_trace(Box::new(trace)),
            PlotTrace::HeatMap(trace) => plot.add_trace(Box::new(trace)),
        }
    }
}

/// Core trait for all plot elements
/// This trait is designed to work with trait objects and supports multiple trace types
pub trait PlotElement: Send + Sync {
    /// Get the data bounds of this element
    fn data_bounds(&self) -> PlotBounds;

    /// Get metadata about this element
    fn metadata(&self) -> &PlotMetadata;

    /// Generate Plotly traces for this element (supports multiple trace types)
    fn to_plotly_traces(&self) -> Vec<PlotTrace>;

    /// Check if this element should be included in legend
    fn has_legend(&self) -> bool {
        self.metadata().legend_label.is_some()
    }

    /// Get the z-order for layering (higher values drawn on top)
    fn z_order(&self) -> i32 {
        self.metadata().z_order
    }
}

/// Layout configuration for composing multiple plot elements
#[derive(Debug, Clone)]
pub enum LayoutConfig {
    /// Single panel with all elements overlaid
    Single,
    /// Stack elements vertically
    VerticalStack,
    /// Stack elements horizontally
    HorizontalStack,
    /// Grid layout with specified rows and columns
    Grid { rows: usize, cols: usize },
    /// Custom layout function
    Custom(fn(usize) -> Vec<SubAreaSpec>),
}

/// Specification for a sub-area in a layout
#[derive(Debug, Clone)]
pub struct SubAreaSpec {
    pub row: usize,
    pub col: usize,
    pub row_span: usize,
    pub col_span: usize,
}

/// Color palette definitions for Plotly
#[derive(Debug, Clone)]
pub enum ColorPalette {
    Default,
    Viridis,
    Plasma,
    Magma,
    Inferno,
    Turbo,
    Scientific,
    Custom(Vec<String>),
}

impl ColorPalette {
    pub fn get_color(&self, index: usize) -> String {
        match self {
            ColorPalette::Default => {
                let colors = [
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                ];
                colors[index % colors.len()].to_string()
            }
            ColorPalette::Scientific => {
                let colors = [
                    "#1f77b4", // blue
                    "#ff7f0e", // orange
                    "#2ca02c", // green
                    "#d62728", // red
                    "#9467bd", // purple
                    "#8c564b", // brown
                    "#e377c2", // pink
                    "#7f7f7f", // gray
                ];
                colors[index % colors.len()].to_string()
            }
            ColorPalette::Viridis => {
                // Use Plotly's built-in viridis scale
                let colors = [
                    "#440154", "#482777", "#3f4a8a", "#31678e", "#26838f", "#1f9d8a", "#6cce5a",
                    "#b6de2b", "#fee825",
                ];
                colors[index % colors.len()].to_string()
            }
            ColorPalette::Custom(colors) => colors[index % colors.len()].clone(),
            _ => "#1f77b4".to_string(), // Default blue
        }
    }

    pub fn to_plotly_colorscale(&self) -> ColorScale {
        match self {
            ColorPalette::Viridis => ColorScale::Palette(ColorScalePalette::Viridis),
            // Use Viridis for all others since they're not available
            _ => ColorScale::Palette(ColorScalePalette::Viridis),
        }
    }
}

/// Style configuration for line elements
#[derive(Debug, Clone)]
pub struct LineStyle {
    pub color: String, // Hex color or CSS color name
    pub width: f64,    // Plotly uses f64 for line width
    pub style: LineStyleType,
}

impl Default for LineStyle {
    fn default() -> Self {
        Self {
            color: "#1f77b4".to_string(), // Professional blue
            width: 3.0,
            style: LineStyleType::Solid,
        }
    }
}

impl LineStyle {
    pub fn to_plotly_line(&self) -> Line {
        Line::new()
            .color(self.color.clone())
            .width(self.width)
            .dash(match self.style {
                LineStyleType::Solid => DashType::Solid,
                LineStyleType::Dashed => DashType::Dash,
                LineStyleType::Dotted => DashType::Dot,
                LineStyleType::DashDot => DashType::DashDot,
            })
    }
}

/// Line style types
#[derive(Debug, Clone)]
pub enum LineStyleType {
    Solid,
    Dashed,
    Dotted,
    DashDot,
}

/// Style configuration for marker elements
#[derive(Debug, Clone)]
pub struct MarkerStyle {
    pub color: String,
    pub size: f64,
    pub shape: MarkerShape,
    pub fill: bool,
}

impl Default for MarkerStyle {
    fn default() -> Self {
        Self {
            color: "#d62728".to_string(), // Professional red
            size: 8.0,
            shape: MarkerShape::Circle,
            fill: true,
        }
    }
}

impl MarkerStyle {
    pub fn to_plotly_marker(&self) -> Marker {
        let symbol = match self.shape {
            MarkerShape::Circle => MarkerSymbol::Circle,
            MarkerShape::Square => MarkerSymbol::Square,
            MarkerShape::Triangle => MarkerSymbol::TriangleUp,
            MarkerShape::Diamond => MarkerSymbol::Diamond,
            MarkerShape::Cross => MarkerSymbol::Cross,
            MarkerShape::Plus => MarkerSymbol::X,
        };

        Marker::new()
            .color(self.color.clone())
            .size(self.size as usize)
            .symbol(symbol)
    }
}

/// Marker shape types
#[derive(Debug, Clone)]
pub enum MarkerShape {
    Circle,
    Square,
    Triangle,
    Diamond,
    Cross,
    Plus,
}

/// Configuration for axis formatting
#[derive(Debug, Clone)]
pub struct AxisConfig {
    pub label: Option<String>,
    pub tick_count: Option<usize>,
    pub log_scale: bool,
    pub grid: bool,
    pub grid_color: String,
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self {
            label: None,
            tick_count: None,
            log_scale: false,
            grid: true,
            grid_color: "#dcdcdc".to_string(),
        }
    }
}

/// Configuration for spectrograms and heatmaps
#[derive(Debug, Clone)]
pub struct SpectrogramConfig {
    pub n_fft: usize,
    pub window_size: Option<usize>,
    pub hop_length: Option<usize>,
    pub window: WindowType,
    pub colormap: ColorPalette,
    pub db_range: (f64, f64),
    pub log_freq: bool,
    pub mel_scale: bool,
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            window_size: None,
            hop_length: None,
            window: WindowType::Hanning,
            colormap: ColorPalette::Viridis,
            db_range: (-80.0, 0.0),
            log_freq: false,
            mel_scale: false,
        }
    }
}

/// Configuration for onset detection overlays
#[derive(Debug, Clone)]
pub struct OnsetConfig {
    pub marker_style: MarkerStyle,
    pub line_style: Option<LineStyle>,
    pub show_strength: bool,
    pub threshold: f64,
}

impl Default for OnsetConfig {
    fn default() -> Self {
        Self {
            marker_style: MarkerStyle {
                color: "#d62728".to_string(), // Red
                size: 8.0,
                shape: MarkerShape::Triangle,
                fill: true,
            },
            line_style: Some(LineStyle {
                color: "#d62728".to_string(), // Red
                width: 1.0,
                style: LineStyleType::Dashed,
            }),
            show_strength: true,
            threshold: 0.1,
        }
    }
}

/// Configuration for beat tracking overlays
#[derive(Debug, Clone)]
pub struct BeatConfig {
    pub marker_style: MarkerStyle,
    pub line_style: Option<LineStyle>,
    pub show_tempo: bool,
}

impl Default for BeatConfig {
    fn default() -> Self {
        Self {
            marker_style: MarkerStyle {
                color: "#2ca02c".to_string(), // Green
                size: 10.0,
                shape: MarkerShape::Diamond,
                fill: true,
            },
            line_style: Some(LineStyle {
                color: "#2ca02c".to_string(), // Green
                width: 2.0,
                style: LineStyleType::Solid,
            }),
            show_tempo: true,
        }
    }
}

/// Theme configuration for plots
#[derive(Debug, Clone)]
pub struct PlotTheme {
    pub background_color: String,
    pub grid_color: String,
    pub text_color: String,
    pub font_family: String,
    pub font_size: f64,
    pub title_font_size: f64,
    pub label_font_size: f64,
    pub tick_font_size: f64,
    pub line_width: f64,
    pub grid_line_width: f64,
    pub color_palette: ColorPalette,
}

impl PlotTheme {
    pub fn to_plotly_layout(&self, title: Option<&str>) -> Layout {
        let mut layout = Layout::new()
            .font(
                Font::new()
                    .family(&self.font_family)
                    .size(self.font_size as usize)
                    .color(self.text_color.clone()),
            )
            .paper_background_color(self.background_color.clone())
            .plot_background_color(self.background_color.clone());

        if let Some(title_text) = title {
            layout = layout.title(
                Title::with_text(title_text).font(
                    Font::new()
                        .family(&self.font_family)
                        .size(self.title_font_size as usize)
                        .color(self.text_color.clone()),
                ),
            );
        }

        layout
    }

    pub fn create_axis(&self, title: &str) -> Axis {
        Axis::new()
            .title(
                Title::with_text(title).font(
                    Font::new()
                        .family(&self.font_family)
                        .size(self.label_font_size as usize)
                        .color(self.text_color.clone()),
                ),
            )
            .tick_font(
                Font::new()
                    .family(&self.font_family)
                    .size(self.tick_font_size as usize)
                    .color(self.text_color.clone()),
            )
            .grid_color(self.grid_color.clone())
            .grid_width(self.grid_line_width as usize)
            .show_grid(true)
    }
}

impl Default for PlotTheme {
    fn default() -> Self {
        Self {
            background_color: "#ffffff".to_string(),
            grid_color: "#f0f0f0".to_string(),
            text_color: "#000000".to_string(),
            font_family: "Arial, sans-serif".to_string(),
            font_size: 14.0,
            title_font_size: 18.0,
            label_font_size: 16.0,
            tick_font_size: 12.0,
            line_width: 3.0,
            grid_line_width: 1.0,
            color_palette: ColorPalette::Default,
        }
    }
}

/// Theme variants
impl PlotTheme {
    pub fn dark() -> Self {
        Self {
            background_color: "#191919".to_string(),
            grid_color: "#3c3c3c".to_string(),
            text_color: "#f0f0f0".to_string(),
            font_family: "Arial, sans-serif".to_string(),
            font_size: 14.0,
            title_font_size: 20.0,
            label_font_size: 16.0,
            tick_font_size: 12.0,
            line_width: 3.0,
            grid_line_width: 1.0,
            color_palette: ColorPalette::Scientific,
        }
    }

    pub fn scientific() -> Self {
        Self {
            background_color: "#ffffff".to_string(),
            grid_color: "#ebebeb".to_string(),
            text_color: "#000000".to_string(),
            font_family: "Arial, sans-serif".to_string(),
            font_size: 14.0,
            title_font_size: 18.0,
            label_font_size: 16.0,
            tick_font_size: 12.0,
            line_width: 2.5,
            grid_line_width: 1.0,
            color_palette: ColorPalette::Scientific,
        }
    }

    /// High-quality theme optimized for presentations and publications
    pub fn high_quality() -> Self {
        Self {
            background_color: "#ffffff".to_string(),
            grid_color: "#f5f5f5".to_string(),
            text_color: "#282828".to_string(),
            font_family: "Arial, sans-serif".to_string(),
            font_size: 16.0,
            title_font_size: 22.0,
            label_font_size: 18.0,
            tick_font_size: 14.0,
            line_width: 4.0,
            grid_line_width: 1.0,
            color_palette: ColorPalette::Scientific,
        }
    }
}
