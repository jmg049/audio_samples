//! Shared code for the learning guide's figures.
//!
//! Each chapter is a binary (`src/chapters/chNN.rs`) that computes its figures
//! with `audio_samples`, asserts the numbers the chapter asserts, draws SVGs
//! with plotters and records them in a manifest (`chNN.json`). The book's
//! preprocessor (`book/preprocessor/guide.py`) reads the manifests and replaces
//! each `<!-- figure: <id> -->` marker in the chapters with the figure and its
//! caption.
//!
//! The output directory is `$GUIDE_FIGURES_DIR`, or `book/src/figures/` by
//! default.

use std::error::Error;
use std::fmt::Display;
use std::path::PathBuf;

use plotters::coord::Shift;
use plotters::coord::ranged1d::{AsRangedCoord, ValueFormatter};
use plotters::element::BitMapElement;
use plotters::prelude::*;

pub use plotters;

/// Result type of the figure programs.
pub type Result<T = ()> = std::result::Result<T, Box<dyn Error>>;

/// The SVG drawing area a figure is drawn on.
pub type Area<'a> = DrawingArea<SVGBackend<'a>, Shift>;

/// Default figure size in pixels.
pub const SIZE: (u32, u32) = (720, 400);

// Okabe-Ito colour-blind-safe palette, used in this order for series.
pub const BLUE: RGBColor = RGBColor(0, 114, 178);
pub const ORANGE: RGBColor = RGBColor(230, 159, 0);
pub const GREEN: RGBColor = RGBColor(0, 158, 115);
pub const VERMILION: RGBColor = RGBColor(213, 94, 0);
pub const PURPLE: RGBColor = RGBColor(204, 121, 167);
pub const SKY: RGBColor = RGBColor(86, 180, 233);
pub const GREY: RGBColor = RGBColor(130, 130, 130);
pub const INK: RGBColor = RGBColor(30, 30, 30);
/// Series colours in order of use.
pub const PALETTE: [RGBColor; 6] = [BLUE, VERMILION, GREEN, ORANGE, PURPLE, SKY];

const FONT: &str = "sans-serif";

/// Converts a plotters error (whose type borrows the backend) to an owned one.
fn err(e: impl Display) -> Box<dyn Error> {
    e.to_string().into()
}

// ---------------------------------------------------------------------------
// Checks

/// Panics with a descriptive message unless `|got - want| <= tol`.
///
/// Use it for every number a figure shows that the chapter also asserts, so
/// the figure cannot drift from the text.
pub fn assert_close(what: &str, got: f64, want: f64, tol: f64) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, the chapter asserts {want} (tolerance {tol})"
    );
}

/// Panics with `what` unless `cond` holds.
pub fn check(what: impl Display, cond: bool) {
    assert!(cond, "check failed: {what}");
}

// ---------------------------------------------------------------------------
// Output and manifest

struct Entry {
    id: String,
    file: String,
    caption: String,
    width: u32,
    height: u32,
}

/// Collects one chapter's figures and writes its manifest.
pub struct Figures {
    chapter: String,
    out: PathBuf,
    entries: Vec<Entry>,
}

impl Figures {
    /// Starts the figures of `chapter` (`"ch01"`, ...).
    pub fn new(chapter: &str) -> Result<Self> {
        let out = match std::env::var_os("GUIDE_FIGURES_DIR") {
            Some(dir) => PathBuf::from(dir),
            None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../src/figures"),
        };
        std::fs::create_dir_all(&out)?;
        Ok(Self {
            chapter: chapter.to_owned(),
            out,
            entries: Vec::new(),
        })
    }

    /// Draws `<id>.svg` of the given size with `draw` and records `caption`.
    ///
    /// `id` must start with the chapter name (`ch01-...`). The caption is
    /// plain text (no Markdown, no maths).
    pub fn svg(
        &mut self,
        id: &str,
        caption: &str,
        size: (u32, u32),
        draw: impl FnOnce(&Area) -> Result,
    ) -> Result {
        assert!(
            id.starts_with(&format!("{}-", self.chapter)),
            "figure id {id} must start with {}-",
            self.chapter
        );
        assert!(
            !self.entries.iter().any(|e| e.id == id),
            "duplicate figure id {id}"
        );
        assert!(
            !caption.contains('$'),
            "caption of {id} must not contain maths"
        );
        let file = format!("{id}.svg");
        let path = self.out.join(&file);
        {
            let root = SVGBackend::new(&path, size).into_drawing_area();
            root.fill(&WHITE)?;
            draw(&root)?;
            root.present()?;
        }
        self.entries.push(Entry {
            id: id.to_owned(),
            file,
            caption: caption.to_owned(),
            width: size.0,
            height: size.1,
        });
        Ok(())
    }

    /// Writes `<chapter>.json` listing the figures drawn.
    pub fn finish(self) -> Result {
        let figures: Vec<serde_json::Value> = self
            .entries
            .iter()
            .map(|e| {
                serde_json::json!({
                    "id": e.id,
                    "file": e.file,
                    "caption": e.caption,
                    "width": e.width,
                    "height": e.height,
                })
            })
            .collect();
        let manifest = serde_json::json!({ "chapter": self.chapter, "figures": figures });
        let path = self.out.join(format!("{}.json", self.chapter));
        std::fs::write(&path, serde_json::to_string_pretty(&manifest)?)?;
        println!(
            "{}: {} figure(s) written to {}",
            self.chapter,
            self.entries.len(),
            self.out.display()
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Line charts

/// How a series is drawn.
#[derive(Clone, Copy, Debug)]
pub enum Style {
    /// Solid line of the given width.
    Line(u32),
    /// Dashed line of the given width.
    Dashed(u32),
    /// Filled circles of the given radius.
    Dots(u32),
    /// Hollow circles of the given radius.
    Rings(u32),
    /// Crosses of the given half-size.
    Crosses(u32),
    /// Vertical stems from y = 0 with a dot on top (impulse responses).
    Stems,
    /// Staircase: each value held until the next x (sample-and-hold).
    Steps(u32),
}

/// One data series.
#[derive(Clone, Debug)]
pub struct Series {
    pub name: Option<String>,
    pub points: Vec<(f64, f64)>,
    pub color: RGBColor,
    pub style: Style,
}

impl Series {
    /// A solid line, width 2.
    pub fn line(name: &str, points: Vec<(f64, f64)>, color: RGBColor) -> Self {
        Self {
            name: label(name),
            points,
            color,
            style: Style::Line(2),
        }
    }

    /// A series with an explicit style.
    pub fn new(name: &str, points: Vec<(f64, f64)>, color: RGBColor, style: Style) -> Self {
        Self {
            name: label(name),
            points,
            color,
            style,
        }
    }
}

fn label(name: &str) -> Option<String> {
    (!name.is_empty()).then(|| name.to_owned())
}

/// A vertical or horizontal reference line with an optional label.
#[derive(Clone, Debug)]
pub struct Rule {
    pub at: f64,
    pub label: String,
    pub color: RGBColor,
    pub dashed: bool,
}

/// A text label at a data position.
#[derive(Clone, Debug)]
pub struct Note {
    pub x: f64,
    pub y: f64,
    pub text: String,
    pub color: RGBColor,
}

/// Where the legend goes.
pub use plotters::chart::SeriesLabelPosition as LegendPos;

/// A 2-D line/point chart on linear or logarithmic axes.
#[derive(Clone, Debug)]
pub struct Chart {
    pub title: Option<String>,
    pub x_label: String,
    pub y_label: String,
    pub x: (f64, f64),
    pub y: (f64, f64),
    pub x_log: bool,
    pub y_log: bool,
    pub series: Vec<Series>,
    pub vrules: Vec<Rule>,
    pub hrules: Vec<Rule>,
    pub notes: Vec<Note>,
    pub legend: LegendPos,
    /// Number of tick labels on the x axis (linear axes).
    pub x_ticks: usize,
    pub y_ticks: usize,
    /// Tick label formatters; default `{:.?}`-like short numbers.
    pub x_fmt: Option<fn(f64) -> String>,
    pub y_fmt: Option<fn(f64) -> String>,
}

impl Chart {
    /// A chart with labelled axes and the given ranges.
    pub fn new(x_label: &str, y_label: &str, x: (f64, f64), y: (f64, f64)) -> Self {
        Self {
            title: None,
            x_label: x_label.to_owned(),
            y_label: y_label.to_owned(),
            x,
            y,
            x_log: false,
            y_log: false,
            series: Vec::new(),
            vrules: Vec::new(),
            hrules: Vec::new(),
            notes: Vec::new(),
            legend: LegendPos::UpperRight,
            x_ticks: 10,
            y_ticks: 8,
            x_fmt: None,
            y_fmt: None,
        }
    }

    pub fn title(mut self, t: &str) -> Self {
        self.title = Some(t.to_owned());
        self
    }
    pub fn log_x(mut self) -> Self {
        self.x_log = true;
        self
    }
    pub fn log_y(mut self) -> Self {
        self.y_log = true;
        self
    }
    pub fn legend(mut self, pos: LegendPos) -> Self {
        self.legend = pos;
        self
    }
    pub fn ticks(mut self, x: usize, y: usize) -> Self {
        self.x_ticks = x;
        self.y_ticks = y;
        self
    }
    pub fn x_format(mut self, f: fn(f64) -> String) -> Self {
        self.x_fmt = Some(f);
        self
    }
    pub fn y_format(mut self, f: fn(f64) -> String) -> Self {
        self.y_fmt = Some(f);
        self
    }
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, s: Series) -> Self {
        self.series.push(s);
        self
    }
    /// A dashed vertical line at x = `at`, labelled near the top.
    pub fn vrule(mut self, at: f64, label: &str, color: RGBColor) -> Self {
        self.vrules.push(Rule {
            at,
            label: label.to_owned(),
            color,
            dashed: true,
        });
        self
    }
    /// A dashed horizontal line at y = `at`, labelled at the right.
    pub fn hrule(mut self, at: f64, label: &str, color: RGBColor) -> Self {
        self.hrules.push(Rule {
            at,
            label: label.to_owned(),
            color,
            dashed: true,
        });
        self
    }
    /// A text label with its top-left corner at (x, y) in data units.
    pub fn note(mut self, x: f64, y: f64, text: &str, color: RGBColor) -> Self {
        self.notes.push(Note {
            x,
            y,
            text: text.to_owned(),
            color,
        });
        self
    }

    /// Draws the chart on `area`.
    pub fn draw(&self, area: &Area) -> Result {
        let (x0, x1) = self.x;
        let (y0, y1) = self.y;
        match (self.x_log, self.y_log) {
            (false, false) => self.draw_on(area, x0..x1, y0..y1),
            (true, false) => self.draw_on(area, (x0..x1).log_scale(), y0..y1),
            (false, true) => self.draw_on(area, x0..x1, (y0..y1).log_scale()),
            (true, true) => self.draw_on(area, (x0..x1).log_scale(), (y0..y1).log_scale()),
        }
    }

    fn draw_on<'b, X, Y>(&self, area: &Area<'b>, xs: X, ys: Y) -> Result
    where
        X: AsRangedCoord<Value = f64>,
        Y: AsRangedCoord<Value = f64>,
        X::CoordDescType: ValueFormatter<f64>,
        Y::CoordDescType: ValueFormatter<f64>,
    {
        let mut builder = ChartBuilder::on(area);
        builder
            .margin_top(if self.title.is_some() { 6 } else { 14 })
            .margin_left(10)
            .margin_right(22)
            .margin_bottom(8)
            .x_label_area_size(44)
            .y_label_area_size(64);
        if let Some(t) = &self.title {
            builder.caption(t, (FONT, 17).into_font().color(&INK));
        }
        let mut chart = builder.build_cartesian_2d(xs, ys).map_err(err)?;

        let x_fmt = self.x_fmt.unwrap_or(short);
        let y_fmt = self.y_fmt.unwrap_or(short);
        let xf = |v: &f64| x_fmt(*v);
        let yf = |v: &f64| y_fmt(*v);
        chart
            .configure_mesh()
            .x_desc(self.x_label.as_str())
            .y_desc(self.y_label.as_str())
            .x_labels(self.x_ticks)
            .y_labels(self.y_ticks)
            .x_label_formatter(&xf)
            .y_label_formatter(&yf)
            .axis_desc_style((FONT, 15).into_font().color(&INK))
            .label_style((FONT, 13).into_font().color(&INK))
            .max_light_lines(1)
            .light_line_style(RGBColor(244, 244, 244))
            .bold_line_style(RGBColor(215, 215, 215))
            .axis_style(RGBColor(90, 90, 90))
            .draw()
            .map_err(err)?;

        let (xa, xb) = self.x;
        let (ya, yb) = self.y;
        for r in &self.hrules {
            let style = ShapeStyle::from(&r.color).stroke_width(1);
            chart
                .draw_series(DashedLineSeries::new([(xa, r.at), (xb, r.at)], 6, 4, style))
                .map_err(err)?;
            if !r.label.is_empty() {
                // Label just above the line, at the left of the plot.
                let pos = chart.backend_coord(&(xa, r.at));
                let area_px = chart.plotting_area().get_pixel_range();
                chart
                    .plotting_area()
                    .strip_coord_spec()
                    .draw(&Text::new(
                        r.label.clone(),
                        (pos.0 - area_px.0.start + 6, pos.1 - area_px.1.start - 17),
                        (FONT, 13).into_font().color(&r.color),
                    ))
                    .map_err(err)?;
            }
        }
        for (i, r) in self.vrules.iter().enumerate() {
            let style = ShapeStyle::from(&r.color).stroke_width(1);
            chart
                .draw_series(DashedLineSeries::new([(r.at, ya), (r.at, yb)], 6, 4, style))
                .map_err(err)?;
            if !r.label.is_empty() {
                let pos = chart.backend_coord(&(r.at, yb));
                let area_px = chart.plotting_area().get_pixel_range();
                let w = text_width(&r.label, 13);
                // Labels go to the right of the line unless they would leave
                // the plot; successive rules are staggered vertically.
                let x = if pos.0 + 5 + w < area_px.0.end {
                    pos.0 + 5
                } else {
                    pos.0 - 5 - w
                };
                let y = pos.1 + 4 + 16 * (i as i32 % 3);
                chart
                    .plotting_area()
                    .strip_coord_spec()
                    .draw(&Text::new(
                        r.label.clone(),
                        (x - area_px.0.start, y - area_px.1.start),
                        (FONT, 13).into_font().color(&r.color),
                    ))
                    .map_err(err)?;
            }
        }

        for s in &self.series {
            let color = s.color;
            let anno = match s.style {
                Style::Line(w) => chart
                    .draw_series(LineSeries::new(
                        s.points.iter().copied(),
                        ShapeStyle::from(&color).stroke_width(w),
                    ))
                    .map_err(err)?,
                Style::Dashed(w) => chart
                    .draw_series(DashedLineSeries::new(
                        s.points.iter().copied(),
                        8,
                        5,
                        ShapeStyle::from(&color).stroke_width(w),
                    ))
                    .map_err(err)?,
                Style::Steps(w) => {
                    let mut pts = Vec::with_capacity(2 * s.points.len());
                    for (i, &(x, y)) in s.points.iter().enumerate() {
                        if i > 0 {
                            pts.push((x, s.points[i - 1].1));
                        }
                        pts.push((x, y));
                    }
                    chart
                        .draw_series(LineSeries::new(
                            pts,
                            ShapeStyle::from(&color).stroke_width(w),
                        ))
                        .map_err(err)?
                }
                Style::Dots(r) => chart
                    .draw_series(
                        s.points
                            .iter()
                            .map(|&p| Circle::new(p, r, ShapeStyle::from(&color).filled())),
                    )
                    .map_err(err)?,
                Style::Rings(r) => chart
                    .draw_series(
                        s.points
                            .iter()
                            .map(|&p| Circle::new(p, r, ShapeStyle::from(&color).stroke_width(2))),
                    )
                    .map_err(err)?,
                Style::Crosses(r) => chart
                    .draw_series(
                        s.points
                            .iter()
                            .map(|&p| Cross::new(p, r, ShapeStyle::from(&color).stroke_width(2))),
                    )
                    .map_err(err)?,
                Style::Stems => {
                    let base = if self.y_log { ya } else { 0.0 };
                    chart
                        .draw_series(s.points.iter().map(|&(x, y)| {
                            PathElement::new(
                                vec![(x, base), (x, y)],
                                ShapeStyle::from(&color).stroke_width(1),
                            )
                        }))
                        .map_err(err)?;
                    chart
                        .draw_series(
                            s.points
                                .iter()
                                .map(|&p| Circle::new(p, 3, ShapeStyle::from(&color).filled())),
                        )
                        .map_err(err)?
                }
            };
            if let Some(name) = &s.name {
                let style = s.style;
                anno.label(name.as_str()).legend(move |(x, y)| {
                    // One composite for every style; the parts a style
                    // does not use are transparent.
                    let g = glyph_parts(style);
                    let on = |b: bool, w: u32| {
                        if b {
                            ShapeStyle::from(&color).stroke_width(w)
                        } else {
                            ShapeStyle::from(&TRANSPARENT)
                        }
                    };
                    let dot = if g.filled {
                        ShapeStyle::from(&color).filled()
                    } else {
                        on(g.ring, 2)
                    };
                    EmptyElement::at((x, y))
                        + PathElement::new(vec![(0, 0), (g.seg1, 0)], on(g.seg1 > 0, g.width))
                        + PathElement::new(vec![(13, 0), (20, 0)], on(g.dashed, g.width))
                        + Circle::new((10, 0), 4, dot)
                        + Cross::new((10, 0), 4, on(g.cross, 2))
                });
            }
        }

        for n in &self.notes {
            chart
                .draw_series(std::iter::once(Text::new(
                    n.text.clone(),
                    (n.x, n.y),
                    (FONT, 13).into_font().color(&n.color),
                )))
                .map_err(err)?;
        }

        if self.series.iter().any(|s| s.name.is_some()) {
            chart
                .configure_series_labels()
                .position(self.legend.clone())
                .background_style(WHITE.mix(0.9))
                .border_style(RGBColor(190, 190, 190))
                .label_font((FONT, 13).into_font().color(&INK))
                .margin(8)
                .draw()
                .map_err(err)?;
        }
        Ok(())
    }
}

struct Glyph {
    seg1: i32,
    width: u32,
    dashed: bool,
    filled: bool,
    ring: bool,
    cross: bool,
}

fn glyph_parts(style: Style) -> Glyph {
    let mut g = Glyph {
        seg1: 0,
        width: 2,
        dashed: false,
        filled: false,
        ring: false,
        cross: false,
    };
    match style {
        Style::Line(w) | Style::Steps(w) => {
            g.seg1 = 20;
            g.width = w;
        }
        Style::Dashed(w) => {
            g.seg1 = 7;
            g.width = w;
            g.dashed = true;
        }
        Style::Dots(_) | Style::Stems => g.filled = true,
        Style::Rings(_) => g.ring = true,
        Style::Crosses(_) => g.cross = true,
    }
    g
}

/// Rough text width in pixels (plotters' font-less layout is approximate too).
fn text_width(s: &str, size: i32) -> i32 {
    (s.chars().count() as f64 * f64::from(size) * 0.55) as i32
}

/// Short tick label: integers without a decimal point, otherwise up to three
/// significant decimals.
pub fn short(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    if (v - v.round()).abs() < 1e-9 * v.abs().max(1.0) && v.abs() < 1e7 {
        return format!("{}", v.round() as i64);
    }
    let s = format!("{v:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.').to_owned();
    if s == "-0" { "0".into() } else { s }
}

/// Tick label in kHz for an axis in Hz ("2k", "500").
pub fn khz(v: f64) -> String {
    if v.abs() >= 1000.0 {
        format!("{}k", short(v / 1000.0))
    } else {
        short(v)
    }
}

// ---------------------------------------------------------------------------
// Layout helpers

/// Splits `area` into `n` rows of equal height.
pub fn rows<'a>(area: &Area<'a>, n: usize) -> Vec<Area<'a>> {
    area.split_evenly((n, 1))
}

/// Splits `area` into `n` columns of equal width.
pub fn cols<'a>(area: &Area<'a>, n: usize) -> Vec<Area<'a>> {
    area.split_evenly((1, n))
}

// ---------------------------------------------------------------------------
// Heat maps (spectrograms)

/// A heat map: `values[row][col]` with rows along y (bottom to top) and
/// columns along x, drawn as a raster embedded in the SVG, with a colour bar.
pub struct Heatmap<'v> {
    pub values: &'v [Vec<f64>],
    pub x: (f64, f64),
    pub y: (f64, f64),
    /// Values are clipped to this range before colouring.
    pub z: (f64, f64),
    pub x_label: String,
    pub y_label: String,
    pub z_label: String,
    pub title: Option<String>,
    pub y_fmt: Option<fn(f64) -> String>,
    /// Lines drawn over the raster, in data coordinates.
    pub overlays: Vec<Series>,
}

impl Heatmap<'_> {
    /// Draws the heat map and its colour bar on `area`.
    pub fn draw(&self, area: &Area) -> Result {
        let (main, bar) = area.split_horizontally(area.dim_in_pixel().0 as i32 - 78);
        let mut builder = ChartBuilder::on(&main);
        builder
            .margin_top(if self.title.is_some() { 6 } else { 14 })
            .margin_left(10)
            .margin_right(10)
            .margin_bottom(8)
            .x_label_area_size(44)
            .y_label_area_size(64);
        if let Some(t) = &self.title {
            builder.caption(t, (FONT, 17).into_font().color(&INK));
        }
        let mut chart = builder.build_cartesian_2d(self.x.0..self.x.1, self.y.0..self.y.1)?;
        let y_fmt = self.y_fmt.unwrap_or(short);
        let yf = |v: &f64| y_fmt(*v);
        let xf = |v: &f64| short(*v);
        chart
            .configure_mesh()
            .disable_mesh()
            .x_desc(self.x_label.as_str())
            .y_desc(self.y_label.as_str())
            .x_label_formatter(&xf)
            .y_label_formatter(&yf)
            .axis_desc_style((FONT, 15).into_font().color(&INK))
            .label_style((FONT, 13).into_font().color(&INK))
            .axis_style(RGBColor(90, 90, 90))
            .draw()?;

        let (xr, yr) = chart.plotting_area().get_pixel_range();
        let (w, h) = ((xr.end - xr.start) as usize, (yr.end - yr.start) as usize);
        let n_rows = self.values.len();
        let n_cols = self.values.first().map_or(0, Vec::len);
        let mut buf = vec![0u8; w * h * 3];
        for py in 0..h {
            // Pixel row 0 is the top: the highest y.
            let r = ((h - 1 - py) * n_rows / h).min(n_rows - 1);
            for px in 0..w {
                let c = (px * n_cols / w).min(n_cols - 1);
                let t = (self.values[r][c] - self.z.0) / (self.z.1 - self.z.0);
                let RGBColor(cr, cg, cb) = colormap(t);
                let i = 3 * (py * w + px);
                buf[i] = cr;
                buf[i + 1] = cg;
                buf[i + 2] = cb;
            }
        }
        let img: BitMapElement<(i32, i32)> =
            BitMapElement::with_owned_buffer((0, 0), (w as u32, h as u32), buf)
                .ok_or("heat map buffer size")?;
        chart.plotting_area().strip_coord_spec().draw(&img)?;

        for s in &self.overlays {
            let anno = chart.draw_series(LineSeries::new(
                s.points.iter().copied(),
                ShapeStyle::from(&s.color).stroke_width(2),
            ))?;
            if let Some(name) = &s.name {
                let color = s.color;
                anno.label(name.as_str()).legend(move |(x, y)| {
                    PathElement::new(
                        vec![(x, y), (x + 20, y)],
                        ShapeStyle::from(&color).stroke_width(2),
                    )
                });
            }
        }
        if self.overlays.iter().any(|s| s.name.is_some()) {
            chart
                .configure_series_labels()
                .position(LegendPos::UpperRight)
                .background_style(WHITE.mix(0.9))
                .border_style(RGBColor(190, 190, 190))
                .label_font((FONT, 13).into_font().color(&INK))
                .draw()?;
        }

        // Colour bar.
        let mut cb = ChartBuilder::on(&bar);
        cb.margin_top(if self.title.is_some() { 30 } else { 14 })
            .margin_bottom(52)
            .margin_right(4)
            .y_label_area_size(40);
        let mut cbar = cb.build_cartesian_2d(0.0..1.0, self.z.0..self.z.1)?;
        cbar.configure_mesh()
            .disable_mesh()
            .disable_x_axis()
            .y_labels(6)
            .y_label_formatter(&|v| short(*v))
            .y_desc(self.z_label.as_str())
            .axis_desc_style((FONT, 13).into_font().color(&INK))
            .label_style((FONT, 12).into_font().color(&INK))
            .draw()?;
        let steps = 120;
        let dz = (self.z.1 - self.z.0) / steps as f64;
        cbar.draw_series((0..steps).map(|i| {
            let z = self.z.0 + i as f64 * dz;
            Rectangle::new(
                [(0.0, z), (1.0, z + dz)],
                colormap(i as f64 / steps as f64).filled(),
            )
        }))?;
        Ok(())
    }
}

/// Perceptually uniform colour map (viridis, interpolated from 9 anchors),
/// `t` in [0, 1] (clipped).
pub fn colormap(t: f64) -> RGBColor {
    const V: [(f64, f64, f64); 9] = [
        (68.0, 1.0, 84.0),
        (71.0, 44.0, 122.0),
        (59.0, 81.0, 139.0),
        (44.0, 113.0, 142.0),
        (33.0, 144.0, 141.0),
        (39.0, 173.0, 129.0),
        (92.0, 200.0, 99.0),
        (170.0, 220.0, 50.0),
        (253.0, 231.0, 37.0),
    ];
    let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    let x = t * (V.len() - 1) as f64;
    let i = (x.floor() as usize).min(V.len() - 2);
    let f = x - i as f64;
    let mix = |a: f64, b: f64| (a + (b - a) * f).round() as u8;
    RGBColor(
        mix(V[i].0, V[i + 1].0),
        mix(V[i].1, V[i + 1].1),
        mix(V[i].2, V[i + 1].2),
    )
}

// ---------------------------------------------------------------------------
// Small numeric helpers for plotting (not for results).

/// `20 log10(v)`, floored at `floor` dB so that zeros stay on the chart.
pub fn db(v: f64, floor: f64) -> f64 {
    if v > 0.0 {
        (20.0 * v.log10()).max(floor)
    } else {
        floor
    }
}

/// Pairs `xs[i]` with `ys[i]`.
pub fn zip(
    xs: impl IntoIterator<Item = f64>,
    ys: impl IntoIterator<Item = f64>,
) -> Vec<(f64, f64)> {
    xs.into_iter().zip(ys).collect()
}

/// `(i * step, ys[i])` for every value.
pub fn indexed(ys: impl IntoIterator<Item = f64>, step: f64) -> Vec<(f64, f64)> {
    ys.into_iter()
        .enumerate()
        .map(|(i, y)| (i as f64 * step, y))
        .collect()
}
