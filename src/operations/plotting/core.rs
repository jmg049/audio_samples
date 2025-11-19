//! Core types and traits for the composable plotting system.
//!
//! This module defines the foundational architecture for building composable,
//! reusable plot elements that can be combined in flexible layouts.
//!
//! Now powered by Plotly.rs for professional-quality, anti-aliased plots.

#[cfg(feature = "plotting")]
use plotly::common::{
    ColorScale, ColorScalePalette, DashType, Font, Line, Marker, MarkerSymbol, Title,
};
#[cfg(feature = "plotting")]
use plotly::layout::{Axis, Layout};
#[cfg(feature = "plotting")]
use plotly::{HeatMap, Plot, Scatter};

use crate::operations::types::WindowType;
use crate::{RealFloat, to_precision};

/// Result type for plotting operations - use AudioSampleResult consistently
pub use crate::AudioSampleResult as PlotResult;

/// Default line width for plot elements in pixels
pub const DEFAULT_LINE_WIDTH: f64 = 3.0;

/// Bounds for plot data in 2D space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlotBounds<F: RealFloat> {
    /// Minimum x-axis value
    pub x_min: F,
    /// Maximum x-axis value
    pub x_max: F,
    /// Minimum y-axis value
    pub y_min: F,
    /// Maximum y-axis value
    pub y_max: F,
}

impl<F: RealFloat> PlotBounds<F> {
    /// Creates new plot bounds with the specified coordinates.
    ///
    /// # Arguments
    /// * `x_min` - Minimum x-coordinate
    /// * `x_max` - Maximum x-coordinate
    /// * `y_min` - Minimum y-coordinate
    /// * `y_max` - Maximum y-coordinate
    pub const fn new(x_min: F, x_max: F, y_min: F, y_max: F) -> Self {
        Self {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    /// Expands these bounds to include another set of bounds.
    ///
    /// # Arguments
    /// * `other` - The other bounds to include
    pub fn expand_to_include(&mut self, other: &PlotBounds<F>) {
        self.x_min = self.x_min.min(other.x_min);
        self.x_max = self.x_max.max(other.x_max);
        self.y_min = self.y_min.min(other.y_min);
        self.y_max = self.y_max.max(other.y_max);
    }

    /// Creates new bounds with added margin as a percentage of the range.
    ///
    /// # Arguments
    /// * `margin_percent` - Margin to add as a fraction (e.g., 0.1 for 10%)
    ///
    /// # Returns
    /// New bounds with the specified margin added
    pub fn with_margin(&self, margin_percent: F) -> Self {
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
#[derive(Debug, Clone, Default)]
pub struct PlotMetadata {
    /// Optional title for the plot
    pub title: Option<String>,
    /// Optional label for the x-axis
    pub x_label: Option<String>,
    /// Optional label for the y-axis
    pub y_label: Option<String>,
    /// Optional label for the legend entry
    pub legend_label: Option<String>,
    /// Z-order for layering elements (higher values appear on top)
    pub z_order: i32,
}

/// Enum for different types of plotly traces
#[derive(Debug)]
pub enum PlotTrace<F: RealFloat> {
    /// A scatter plot trace for line/point plots
    Scatter(Box<Scatter<F, F>>),
    /// A heatmap trace for 2D data visualization
    HeatMap(Box<HeatMap<f64, f64, f64>>),
}

impl<F: RealFloat> PlotTrace<F> {
    /// Add this trace to a plotly Plot
    pub fn add_to_plot(self, plot: &mut Plot) {
        match self {
            PlotTrace::Scatter(trace) => plot.add_trace(trace),
            PlotTrace::HeatMap(trace) => plot.add_trace(trace),
        }
    }

    /// Set the x-axis reference for this trace
    pub fn x_axis(self, axis_ref: &str) -> Self {
        match self {
            PlotTrace::Scatter(trace) => PlotTrace::Scatter(Box::new(*trace.x_axis(axis_ref))),
            PlotTrace::HeatMap(trace) => PlotTrace::HeatMap(trace), // HeatMaps don't support axis references the same way
        }
    }

    /// Set the y-axis reference for this trace
    pub fn y_axis(self, axis_ref: &str) -> Self {
        match self {
            PlotTrace::Scatter(trace) => PlotTrace::Scatter(Box::new(*trace.y_axis(axis_ref))),
            PlotTrace::HeatMap(trace) => PlotTrace::HeatMap(trace), // HeatMaps don't support axis references the same way
        }
    }
}

/// Core trait for all plot elements
/// This trait is designed to work with trait objects and supports multiple trace types
pub trait PlotElement<F: RealFloat>: Send + Sync {
    /// Get the data bounds of this element
    fn data_bounds(&self) -> PlotBounds<F>;

    /// Get metadata about this element
    fn metadata(&self) -> &PlotMetadata;

    /// Generate Plotly traces for this element (supports multiple trace types)
    fn to_plotly_traces(&self) -> Vec<PlotTrace<F>>;

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
    Grid {
        /// Number of rows in the grid
        rows: usize,
        /// Number of columns in the grid
        cols: usize,
    },
    /// Custom layout function
    Custom(fn(usize) -> Vec<SubAreaSpec>),
}

/// Specification for a sub-area in a layout
#[derive(Debug, Clone)]
pub struct SubAreaSpec {
    /// Row index (0-based)
    pub row: usize,
    /// Column index (0-based)
    pub col: usize,
    /// Number of rows to span
    pub row_span: usize,
    /// Number of columns to span
    pub col_span: usize,
}

/// Color palette definitions for Plotly
#[derive(Debug, Clone)]
pub enum ColorPalette {
    /// Default Plotly color palette
    Default,
    /// Viridis colormap (purple to yellow)
    Viridis,
    /// Plasma colormap (purple to pink to yellow)
    Plasma,
    /// Magma colormap (black to purple to white)
    Magma,
    /// Inferno colormap (black to red to yellow)
    Inferno,
    /// Turbo colormap (blue to red)
    Turbo,
    /// Scientific color palette
    Scientific,
    /// Custom color palette with specified colors
    Custom(Vec<String>),
}

impl ColorPalette {
    /// Gets a color from the palette at the specified index.
    ///
    /// # Arguments
    /// * `index` - Index of the color to retrieve (wraps around if out of bounds)
    ///
    /// # Returns
    /// A color string in hex format
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

    /// Converts this palette to a Plotly ColorScale.
    ///
    /// # Returns
    /// A Plotly ColorScale corresponding to this palette
    pub const fn to_plotly_colorscale(&self) -> ColorScale {
        ColorScale::Palette(ColorScalePalette::Viridis)
    }
}

/// Style configuration for line elements
#[derive(Debug, Clone)]
pub struct LineStyle<F: RealFloat> {
    /// Line color as hex string or CSS color name
    pub color: String,
    /// Line width in pixels
    pub width: F,
    /// Line style type (solid, dashed, etc.)
    pub style: LineStyleType,
}

impl<F: RealFloat> Default for LineStyle<F> {
    fn default() -> Self {
        Self {
            color: "#1f77b4".to_string(), // Professional blue
            width: to_precision::<F, _>(3.0),
            style: LineStyleType::Solid,
        }
    }
}

impl<F: RealFloat> LineStyle<F> {
    /// Creates a new line style with the specified parameters.
    ///
    /// # Arguments
    /// * `color` - Color as hex string or CSS color name
    /// * `width` - Line width in pixels
    /// * `style` - Line style type
    pub fn new(color: &str, width: F, style: LineStyleType) -> Self {
        Self {
            color: color.to_string(),
            width,
            style,
        }
    }

    /// Creates a solid blue line style.
    pub fn solid_blue() -> Self {
        Self::new("#1f77b4", to_precision::<F, _>(3.0), LineStyleType::Solid)
    }

    /// Creates a solid red line style.
    pub fn solid_red() -> Self {
        Self::new("#d62728", to_precision::<F, _>(3.0), LineStyleType::Solid)
    }

    /// Creates a solid green line style.
    pub fn solid_green() -> Self {
        Self::new("#2ca02c", to_precision::<F, _>(3.0), LineStyleType::Solid)
    }

    /// Converts this line style to a Plotly Line object.
    ///
    /// # Panics
    ///
    /// Panics if width cannot be converted to f64.
    pub fn to_plotly_line(&self) -> Line {
        Line::new()
            .color(self.color.clone())
            .width(self.width.to_f64().expect("Should not fail"))
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
    /// Solid line
    Solid,
    /// Dashed line
    Dashed,
    /// Dotted line
    Dotted,
    /// Dash-dot line pattern
    DashDot,
}

/// Style configuration for marker elements
#[derive(Debug, Clone)]
pub struct MarkerStyle<F: RealFloat> {
    /// Marker color as hex string or CSS color name
    pub color: String,
    /// Marker size in pixels
    pub size: F,
    /// Shape of the marker
    pub shape: MarkerShape,
    /// Whether the marker should be filled
    pub fill: bool,
}

impl<F: RealFloat> Default for MarkerStyle<F> {
    fn default() -> Self {
        Self {
            color: "#d62728".to_string(), // Professional red
            size: to_precision::<F, _>(8.0),
            shape: MarkerShape::Circle,
            fill: true,
        }
    }
}

impl<F: RealFloat> MarkerStyle<F> {
    /// Converts this marker style to a Plotly Marker object.
    ///
    /// # Panics
    ///
    /// Panics if size cannot be converted to usize.
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
            .size(self.size.to_usize().expect("Should not fail"))
            .symbol(symbol)
    }
}

/// Marker shape types
#[derive(Debug, Clone)]
pub enum MarkerShape {
    /// Circular marker
    Circle,
    /// Square marker
    Square,
    /// Triangular marker
    Triangle,
    /// Diamond-shaped marker
    Diamond,
    /// Cross-shaped marker
    Cross,
    /// Plus-shaped marker
    Plus,
}

/// Configuration for axis formatting
#[derive(Debug, Clone)]
pub struct AxisConfig {
    /// Optional axis label
    pub label: Option<String>,
    /// Optional number of tick marks to display
    pub tick_count: Option<usize>,
    /// Whether to use logarithmic scale
    pub log_scale: bool,
    /// Whether to show grid lines
    pub grid: bool,
    /// Color of grid lines
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
pub struct SpectrogramConfig<F: RealFloat> {
    /// FFT size for frequency analysis
    pub n_fft: usize,
    /// Window size for analysis (defaults to n_fft if None)
    pub window_size: Option<usize>,
    /// Hop length between frames (defaults to window_size/4 if None)
    pub hop_length: Option<usize>,
    /// Window function type
    pub window: WindowType<F>,
    /// Color palette for the spectrogram
    pub colormap: ColorPalette,
    /// Dynamic range in dB (min, max)
    pub db_range: (F, F),
    /// Whether to use logarithmic frequency scale
    pub log_freq: bool,
    /// Whether to use mel frequency scale
    pub mel_scale: bool,
}

impl<F: RealFloat> Default for SpectrogramConfig<F> {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            window_size: None,
            hop_length: None,
            window: WindowType::Hanning,
            colormap: ColorPalette::Viridis,
            db_range: (to_precision::<F, _>(-80.0), F::zero()),
            log_freq: false,
            mel_scale: false,
        }
    }
}

/// Configuration for onset detection overlays
#[derive(Debug, Clone)]
pub struct OnsetConfig<F: RealFloat> {
    /// Style for onset markers
    pub marker_style: MarkerStyle<F>,
    /// Optional line style for connecting onset markers
    pub line_style: Option<LineStyle<F>>,
    /// Whether to show onset strength information
    pub show_strength: bool,
    /// Detection threshold for onsets
    pub threshold: F,
}

impl<F: RealFloat> Default for OnsetConfig<F> {
    fn default() -> Self {
        Self {
            marker_style: MarkerStyle {
                color: "#d62728".to_string(), // Red
                size: to_precision::<F, _>(8.0),
                shape: MarkerShape::Triangle,
                fill: true,
            },
            line_style: Some(LineStyle {
                color: "#d62728".to_string(), // Red
                width: F::one(),
                style: LineStyleType::Dashed,
            }),
            show_strength: true,
            threshold: to_precision::<F, _>(0.1),
        }
    }
}

/// Configuration for beat tracking overlays
#[derive(Debug, Clone)]
pub struct BeatConfig<F: RealFloat> {
    /// Style for beat markers
    pub marker_style: MarkerStyle<F>,
    /// Optional line style for connecting beat markers
    pub line_style: Option<LineStyle<F>>,
    /// Whether to display tempo information
    pub show_tempo: bool,
}

impl<F: RealFloat> Default for BeatConfig<F> {
    fn default() -> Self {
        Self {
            marker_style: MarkerStyle {
                color: "#2ca02c".to_string(), // Green
                size: to_precision::<F, _>(10.0),
                shape: MarkerShape::Diamond,
                fill: true,
            },
            line_style: Some(LineStyle {
                color: "#2ca02c".to_string(), // Green
                width: to_precision::<F, _>(2.0),
                style: LineStyleType::Solid,
            }),
            show_tempo: true,
        }
    }
}

/// Theme configuration for plots
#[derive(Debug, Clone)]
pub struct PlotTheme<F: RealFloat> {
    /// Background color of the plot
    pub background_color: String,
    /// Color of grid lines
    pub grid_color: String,
    /// Color of text elements
    pub text_color: String,
    /// Font family for text elements
    pub font_family: String,
    /// Default font size for text
    pub font_size: F,
    /// Font size for plot titles
    pub title_font_size: F,
    /// Font size for axis labels
    pub label_font_size: F,
    /// Font size for tick labels
    pub tick_font_size: F,
    /// Default line width for plot elements
    pub line_width: F,
    /// Width of grid lines
    pub grid_line_width: F,
    /// Color palette for data visualization
    pub color_palette: ColorPalette,
}

impl<F: RealFloat> PlotTheme<F> {
    /// Converts this theme to a Plotly Layout object.
    ///
    /// # Arguments
    /// * `title` - Optional title for the plot
    ///
    /// # Returns
    /// A Plotly Layout configured with this theme
    ///
    /// # Panics
    ///
    /// Panics if font size cannot be converted to usize.
    pub fn to_plotly_layout(&self, title: Option<&str>) -> Layout {
        let mut layout = Layout::new()
            .font(
                Font::new()
                    .family(&self.font_family)
                    .size(self.font_size.to_usize().expect("Should not fail"))
                    .color(self.text_color.clone()),
            )
            .paper_background_color(self.background_color.clone())
            .plot_background_color(self.background_color.clone());

        if let Some(title_text) = title {
            layout = layout.title(
                Title::with_text(title_text).font(
                    Font::new()
                        .family(&self.font_family)
                        .size(self.title_font_size.to_usize().expect("Should not fail"))
                        .color(self.text_color.clone()),
                ),
            );
        }

        layout
    }

    /// Creates a Plotly Axis object with theme styling.
    ///
    /// # Arguments
    /// * `title` - Title for the axis
    ///
    /// # Returns
    /// A Plotly Axis configured with this theme
    ///
    /// # Panics
    ///
    /// Panics if label font size cannot be converted to usize.
    pub fn create_axis(&self, title: &str) -> Axis {
        Axis::new()
            .title(
                Title::with_text(title).font(
                    Font::new()
                        .family(&self.font_family)
                        .size(self.label_font_size.to_usize().expect("Should not fail"))
                        .color(self.text_color.clone()),
                ),
            )
            .tick_font(
                Font::new()
                    .family(&self.font_family)
                    .size(self.tick_font_size.to_usize().expect("Should not fail"))
                    .color(self.text_color.clone()),
            )
            .grid_color(self.grid_color.clone())
            .grid_width(self.grid_line_width.to_usize().expect("Should not fail"))
            .show_grid(true)
    }
}

impl<F: RealFloat> Default for PlotTheme<F> {
    fn default() -> Self {
        Self {
            background_color: "#ffffff".to_string(),
            grid_color: "#f0f0f0".to_string(),
            text_color: "#000000".to_string(),
            font_family: "Arial, sans-serif".to_string(),
            font_size: to_precision::<F, _>(14.0),
            title_font_size: to_precision::<F, _>(18.0),
            label_font_size: to_precision::<F, _>(16.0),
            tick_font_size: to_precision::<F, _>(12.0),
            line_width: to_precision::<F, _>(3.0),
            grid_line_width: F::one(),
            color_palette: ColorPalette::Default,
        }
    }
}

/// Theme variants
impl<F: RealFloat> PlotTheme<F> {
    /// Creates a dark theme suitable for dark backgrounds.
    pub fn dark() -> Self {
        Self {
            background_color: "#191919".to_string(),
            grid_color: "#3c3c3c".to_string(),
            text_color: "#f0f0f0".to_string(),
            font_family: "Arial, sans-serif".to_string(),
            font_size: to_precision::<F, _>(14.0),
            title_font_size: to_precision::<F, _>(20.0),
            label_font_size: to_precision::<F, _>(16.0),
            tick_font_size: to_precision::<F, _>(12.0),
            line_width: to_precision::<F, _>(3.0),
            grid_line_width: F::one(),
            color_palette: ColorPalette::Scientific,
        }
    }

    /// Creates a scientific theme with high contrast and precision.
    pub fn scientific() -> Self {
        Self {
            background_color: "#ffffff".to_string(),
            grid_color: "#ebebeb".to_string(),
            text_color: "#000000".to_string(),
            font_family: "Arial, sans-serif".to_string(),
            font_size: to_precision::<F, _>(14.0),
            title_font_size: to_precision::<F, _>(20.0),
            label_font_size: to_precision::<F, _>(16.0),
            tick_font_size: to_precision::<F, _>(12.0),
            line_width: to_precision::<F, _>(2.5),
            grid_line_width: F::one(),
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
            font_size: to_precision::<F, _>(16.0),
            title_font_size: to_precision::<F, _>(22.0),
            label_font_size: to_precision::<F, _>(18.0),
            tick_font_size: to_precision::<F, _>(14.0),
            line_width: to_precision::<F, _>(4.0),
            grid_line_width: F::one(),
            color_palette: ColorPalette::Scientific,
        }
    }
}

/// Phase display mode for complex spectrum analysis
#[derive(Debug, Clone, Copy, Default)]
pub enum PhaseDisplayMode {
    /// Wrapped phase (-π to π)
    #[default]
    Wrapped,
    /// Unwrapped phase (continuous)
    Unwrapped,
    /// Phase in degrees instead of radians
    Degrees,
    /// Unwrapped phase in degrees
    UnwrappedDegrees,
}

/// Frequency axis scaling options
#[derive(Debug, Clone, Copy, Default)]
pub enum FrequencyAxisScale {
    /// Linear frequency scale
    #[default]
    Linear,
    /// Logarithmic frequency scale
    Logarithmic,
    /// Mel frequency scale
    Mel,
}

/// Configuration for complex spectrum plots
#[derive(Debug, Clone)]
pub struct ComplexSpectrumConfig<F: RealFloat> {
    /// How to display phase information
    pub phase_mode: PhaseDisplayMode,
    /// Frequency range to display (Hz)
    pub frequency_range: Option<(F, F)>,
    /// Frequency axis scaling
    pub frequency_scale: FrequencyAxisScale,
    /// Whether to show magnitude and phase in separate subplots
    pub split_view: bool,
    /// Magnitude scale in dB
    pub db_scale: bool,
    /// Line styles for magnitude and phase
    pub magnitude_style: LineStyle<F>,
    /// Line style for phase plot
    pub phase_style: LineStyle<F>,
}

impl<F: RealFloat> Default for ComplexSpectrumConfig<F> {
    fn default() -> Self {
        Self {
            phase_mode: PhaseDisplayMode::Wrapped,
            frequency_range: None,
            frequency_scale: FrequencyAxisScale::Linear,
            split_view: false,
            db_scale: true,
            magnitude_style: LineStyle {
                color: "#1f77b4".to_string(),
                width: to_precision::<F, _>(2.0),
                style: LineStyleType::Solid,
            },
            phase_style: LineStyle {
                color: "#ff7f0e".to_string(),
                width: to_precision::<F, _>(2.0),
                style: LineStyleType::Solid,
            },
        }
    }
}

/// Configuration for peak detection in frequency domain
#[derive(Debug, Clone)]
pub struct PeakDetectionConfig<F: RealFloat> {
    /// Minimum peak height (in dB if using dB scale)
    pub min_height: Option<F>,
    /// Minimum peak prominence
    pub min_prominence: Option<F>,
    /// Minimum distance between peaks (in Hz)
    pub min_distance: Option<F>,
    /// Maximum number of peaks to detect
    pub max_peaks: Option<usize>,
    /// Frequency range to search for peaks
    pub frequency_range: Option<(F, F)>,
}

impl<F: RealFloat> Default for PeakDetectionConfig<F> {
    fn default() -> Self {
        Self {
            min_height: Some(to_precision::<F, _>(-60.0)), // -60 dB threshold
            min_prominence: Some(to_precision::<F, _>(6.0)), // 6 dB prominence
            min_distance: Some(to_precision::<F, _>(10.0)), // 10 Hz minimum distance
            max_peaks: Some(10),                           // Limit to 10 peaks
            frequency_range: None,
        }
    }
}

/// Configuration for waterfall (3D time-frequency) plots
#[derive(Debug, Clone)]
pub struct WaterfallConfig<F: RealFloat> {
    /// Window size for each FFT slice
    pub window_size: usize,
    /// Hop length between windows
    pub hop_length: usize,
    /// Time range to display (seconds)
    pub time_range: Option<(F, F)>,
    /// Frequency range to display (Hz)
    pub frequency_range: Option<(F, F)>,
    /// Magnitude range for color mapping (dB)
    pub magnitude_range: (F, F),
    /// Colormap for the visualization
    pub colormap: ColorPalette,
    /// Whether to use logarithmic frequency scale
    pub log_frequency: bool,
}

impl<F: RealFloat> Default for WaterfallConfig<F> {
    fn default() -> Self {
        Self {
            window_size: 2048,
            hop_length: 512,
            time_range: None,
            frequency_range: None,
            magnitude_range: (to_precision::<F, _>(-80.0), F::zero()),
            colormap: ColorPalette::Viridis,
            log_frequency: false,
        }
    }
}

/// Configuration for window function comparison plots
#[derive(Debug, Clone)]
pub struct WindowComparisonConfig<F: RealFloat> {
    /// Window types to compare
    pub windows: Vec<WindowType<F>>,
    /// Whether to overlay or use subplots
    pub overlay_mode: bool,
    /// FFT size for comparison
    pub fft_size: usize,
    /// Frequency range to display
    pub frequency_range: Option<(F, F)>,
    /// Whether to normalize window responses
    pub normalize: bool,
}

impl<F: RealFloat> Default for WindowComparisonConfig<F> {
    fn default() -> Self {
        Self {
            windows: vec![
                WindowType::Hanning,
                WindowType::Hamming,
                WindowType::Blackman,
                WindowType::Kaiser {
                    beta: to_precision::<F, _>(8.6),
                },
            ],
            overlay_mode: true,
            fft_size: 2048,
            frequency_range: None,
            normalize: true,
        }
    }
}

/// Configuration for frequency bin tracking over time
#[derive(Debug, Clone)]
pub struct FrequencyBinConfig<F: RealFloat> {
    /// Specific frequency bins to track (in Hz)
    pub target_frequencies: Vec<F>,
    /// Window size for STFT analysis
    pub window_size: usize,
    /// Hop length between windows
    pub hop_length: usize,
    /// Time range to display
    pub time_range: Option<(F, F)>,
    /// Whether to show magnitude or phase
    pub show_magnitude: bool,
    /// Whether to show phase information
    pub show_phase: bool,
    /// Line styles for each frequency bin
    pub line_styles: Vec<LineStyle<F>>,
}

impl<F: RealFloat> Default for FrequencyBinConfig<F> {
    fn default() -> Self {
        Self {
            target_frequencies: Vec::new(),
            window_size: 2048,
            hop_length: 512,
            time_range: None,
            show_magnitude: true,
            show_phase: false,
            line_styles: Vec::new(),
        }
    }
}
