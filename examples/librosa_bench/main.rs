//! Native half of the librosa comparison benchmark (`benches/librosa_compare/`).
//!
//! Reads `case config` pairs (one per line) from `--pairs FILE` (written by
//! `bench_py.py --list --tsv`), and for each pair times one call of the
//! audio_samples equivalent of the librosa function on the same input the
//! Python half uses (the signal files in `--signals DIR`). Prints one JSON
//! object per pair.
//!
//! Timing rule (the same as the Python half): two warm-up calls (one if a call
//! takes over a second), then repeat until at least `--min-time` seconds
//! (default 0.3) and 5 repetitions have elapsed, at most 50.
//!
//! A counting global allocator then measures one more call: the number of
//! allocations (including reallocations), the bytes requested and the peak of
//! live heap bytes above the level at the start of the call. The warm-up calls
//! have already filled the thread-local caches (FFT plans, filterbanks,
//! windows, resampler kernels), so these numbers are per call in steady state.
//!
//! ```sh
//! cargo run --release --example librosa_bench --features full_no_plotting -- \
//!     --signals target/librosa_bench/signals --pairs pairs.tsv
//! ```

mod array_ops;
mod constant_q;
mod convert;
mod ctx;
mod effects;
mod features;
mod inverse;
mod notation;
mod pitch;
mod pitch_filterbank;
mod rhythm;
mod segment;
mod sequence;
mod spectrum;
mod tonal;

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use audio_samples::AudioFloat;

pub use ctx::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

// ---------------------------------------------------------------------------
// Counting allocator

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

fn grow(n: u64) {
    let live = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's arguments and
// only updates counters, so `System`'s guarantees carry over.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            ALLOCS.fetch_add(1, Relaxed);
            BYTES.fetch_add(layout.size() as u64, Relaxed);
            grow(layout.size() as u64);
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged; the caller upholds `alloc_zeroed`'s contract.
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            ALLOCS.fetch_add(1, Relaxed);
            BYTES.fetch_add(layout.size() as u64, Relaxed);
            grow(layout.size() as u64);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` through this allocator with `layout`.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size() as u64, Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr` was allocated by `System` through this allocator with `layout`.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            ALLOCS.fetch_add(1, Relaxed);
            let old = layout.size() as u64;
            let new = new_size as u64;
            if new > old {
                BYTES.fetch_add(new - old, Relaxed);
                grow(new - old);
            } else {
                LIVE.fetch_sub(old - new, Relaxed);
            }
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

struct AllocStats {
    allocs: u64,
    bytes: u64,
    peak: u64,
}

fn measure_allocs(f: &mut Thunk<'_>) -> AllocStats {
    let a0 = ALLOCS.load(Relaxed);
    let b0 = BYTES.load(Relaxed);
    let live0 = LIVE.load(Relaxed);
    PEAK.store(live0, Relaxed);
    f();
    AllocStats {
        allocs: ALLOCS.load(Relaxed) - a0,
        bytes: BYTES.load(Relaxed) - b0,
        peak: PEAK.load(Relaxed).saturating_sub(live0),
    }
}

// ---------------------------------------------------------------------------
// Timing

struct Opts {
    signals: String,
    pairs: String,
    min_time: f64,
    no_alloc: bool,
}

fn parse_args() -> Opts {
    let mut o = Opts {
        signals: "target/librosa_bench/signals".into(),
        pairs: String::new(),
        min_time: 0.3,
        no_alloc: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("missing value for argument");
        match a.as_str() {
            "--signals" => o.signals = val(),
            "--pairs" => o.pairs = val(),
            "--min-time" => o.min_time = val().parse().expect("--min-time"),
            "--no-alloc" => o.no_alloc = true,
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(!o.pairs.is_empty(), "--pairs FILE is required");
    o
}

fn time_call(f: &mut Thunk<'_>, min_time: f64) -> Vec<f64> {
    let t0 = Instant::now();
    f();
    if t0.elapsed().as_secs_f64() < 1.0 {
        f();
    }
    let mut times = Vec::new();
    let start = Instant::now();
    while times.len() < 50 && (times.len() < 5 || start.elapsed().as_secs_f64() < min_time) {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times
}

/// Keeps the core busy for half a second before the first measurement.
///
/// After an idle spell, the first 0.2-0.3 s of work on this machine (i7-13700KF,
/// pinned P-core) run about 10 % slower: 16 kHz to 22.05 kHz `f64` resampling
/// took 1.15 ms per call in 7 of 7 fresh processes, against 1.05 ms after this
/// warm-up (7 of 7), at the same reported clock. The Python runs start after
/// more than a second of interpreter start-up and imports, so they are not
/// affected.
fn warm_up_core() {
    let t = Instant::now();
    let mut x = 1u64;
    while t.elapsed().as_secs_f64() < 0.5 {
        for _ in 0..1000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        }
    }
    std::hint::black_box(x);
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    // numpy's default (linear) percentile
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Builds the thunk for `case` on the given configuration, or `None` when no
/// group knows the case.
fn build<'a, F: AudioFloat>(case: &str, ctx: &'a Ctxs<F>) -> Option<Thunk<'a>> {
    macro_rules! chain {
        ($kind:ident, $x:expr) => {
            spectrum::$kind(case, $x)
                .or_else(|| features::$kind(case, $x))
                .or_else(|| tonal::$kind(case, $x))
                .or_else(|| constant_q::$kind(case, $x))
                .or_else(|| pitch::$kind(case, $x))
                .or_else(|| effects::$kind(case, $x))
                .or_else(|| rhythm::$kind(case, $x))
                .or_else(|| sequence::$kind(case, $x))
                .or_else(|| convert::$kind(case, $x))
                .or_else(|| inverse::$kind(case, $x))
                .or_else(|| pitch_filterbank::$kind(case, $x))
                .or_else(|| segment::$kind(case, $x))
                .or_else(|| array_ops::$kind(case, $x))
                .or_else(|| notation::$kind(case, $x))
        };
    }
    match ctx {
        Ctxs::Audio(x) => chain!(audio, x),
        Ctxs::Const(x) => chain!(constant, x),
        Ctxs::Vec(x) => chain!(vec, x),
        Ctxs::Seq(x) => chain!(seq, x),
    }
}

enum Ctxs<F: AudioFloat> {
    Audio(Box<AudioCtx<F>>),
    Const(ConstCtx<F>),
    Vec(VecCtx<F>),
    Seq(SeqCtx<F>),
}

fn make_ctx<F: AudioFloat>(config: &str, signals: &str) -> Ctxs<F> {
    match config.split('_').next().unwrap() {
        "const" => Ctxs::Const(ConstCtx::new()),
        "vec" => Ctxs::Vec(VecCtx::new()),
        "seq" => Ctxs::Seq(SeqCtx::new()),
        _ => Ctxs::Audio(Box::new(AudioCtx::load(config, signals))),
    }
}

fn run<F: AudioFloat>(pairs: &[(String, String)], opts: &Opts, out: &mut impl Write) {
    let mut current: Option<(String, Ctxs<F>)> = None;
    for (case, config) in pairs {
        if current.as_ref().is_none_or(|(c, _)| c != config) {
            drop(current.take()); // free the previous configuration's inputs first
            current = Some((config.clone(), make_ctx::<F>(config, &opts.signals)));
        }
        let ctx = &current.as_ref().unwrap().1;
        let Some(mut f) = build(case, ctx) else {
            eprintln!("rust     {case:34} {config:22} (no native case)");
            continue;
        };
        let mut times = time_call(&mut f, opts.min_time);
        times.sort_by(f64::total_cmp);
        let med = percentile(&times, 0.5);
        let mut rec = format!(
            "{{\"lib\": \"rust\", \"case\": \"{case}\", \"config\": \"{config}\", \"mode\": \"time\", \
             \"median_ms\": {med:.6}, \"p25_ms\": {:.6}, \"p75_ms\": {:.6}, \"min_ms\": {:.6}, \
             \"max_ms\": {:.6}, \"reps\": {}",
            percentile(&times, 0.25),
            percentile(&times, 0.75),
            times[0],
            times[times.len() - 1],
            times.len()
        );
        if !opts.no_alloc {
            let s = measure_allocs(&mut f);
            rec.push_str(&format!(
                ", \"allocs\": {}, \"alloc_bytes\": {}, \"peak_bytes\": {}",
                s.allocs, s.bytes, s.peak
            ));
        }
        rec.push('}');
        eprintln!("rust     {case:34} {config:22} {med:10.3} ms");
        writeln!(out, "{rec}").unwrap();
        out.flush().unwrap();
        drop(f);
    }
}

/// A `(case, config)` pair to run.
type Pair = (String, String);

fn main() {
    warm_up_core();
    let opts = parse_args();
    let text = std::fs::read_to_string(&opts.pairs).expect("read --pairs");
    let pairs = text.lines().filter(|l| !l.trim().is_empty()).map(|l| {
        let mut it = l.split_whitespace();
        (it.next().unwrap().to_owned(), it.next().unwrap().to_owned())
    });
    let (mut f32s, mut f64s): (Vec<Pair>, Vec<Pair>) =
        pairs.partition(|(_, c)| c.ends_with("_f32"));
    // group by configuration so each input is loaded once
    f32s.sort_by(|a, b| a.1.cmp(&b.1));
    f64s.sort_by(|a, b| a.1.cmp(&b.1));
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    run::<f32>(&f32s, &opts, &mut out);
    run::<f64>(&f64s, &opts, &mut out);
}
