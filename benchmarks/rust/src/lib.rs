//! Shared timing and CSV utilities for all benchmark binaries.

use std::time::Instant;

// ---------------------------------------------------------------------------
// Process initialisation
// ---------------------------------------------------------------------------

/// Call once at the start of every benchmark binary's `main()`.
///
/// On Linux, disables glibc's mmap threshold and heap trimming. Without
/// this, freeing a buffer larger than the default 128 KB threshold returns
/// its physical pages to the OS, and the next allocation of the same buffer
/// page-faults on every page. Whether a buffer is served this way depends on
/// its size and on what the allocator did before, so one signal length can
/// time several times slower than its neighbours for reasons unrelated to
/// the code under test. Setting both thresholds to `i32::MAX` keeps the heap
/// expanded for the life of the process.
pub fn bench_init() {
    #[cfg(target_os = "linux")]
    // SAFETY: mallopt is a safe libc function; no invariants to uphold.
    unsafe {
        unsafe extern "C" { fn mallopt(param: i32, value: i32) -> i32; }
        // Disable mmap for individual allocations (M_MMAP_THRESHOLD = -3).
        // By default glibc uses mmap for allocations ≥ 128 KB.  mmap'd blocks
        // are returned to the OS on free() via munmap, so every re-allocation of
        // the same buffer incurs page faults.  Setting the threshold to MAX forces
        // all allocations through the brk heap regardless of size.
        mallopt(-3 /* M_MMAP_THRESHOLD */, i32::MAX);
        // Disable heap trimming (M_TRIM_THRESHOLD = -1).
        // Without this, glibc calls brk() to return heap pages to the OS after
        // freeing blocks above 128 KB, causing the same page-fault storm for
        // brk-based allocations.
        mallopt(-1 /* M_TRIM_THRESHOLD */, i32::MAX);
    }
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

pub struct Stats {
    pub min_us:    f64,
    pub mean_us:   f64,
    pub median_us: f64,
    pub max_us:    f64,
    pub stddev_us: f64,
}

/// Run `op` for `warmup` iterations (not timed), then `iterations` timed
/// iterations.  Returns timing statistics in microseconds.
pub fn run_timed(iterations: usize, warmup: usize, mut op: impl FnMut()) -> Stats {
    for _ in 0..warmup {
        op();
    }

    let mut times_ns: Vec<u128> = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t0 = Instant::now();
        op();
        times_ns.push(t0.elapsed().as_nanos());
    }

    times_ns.sort_unstable();

    let n = iterations as f64;
    let mn  = times_ns[0] as f64;
    let mx  = times_ns[iterations - 1] as f64;
    let med = times_ns[iterations / 2] as f64;
    let mean: f64 = times_ns.iter().map(|&t| t as f64).sum::<f64>() / n;
    let var: f64 = times_ns.iter()
        .map(|&t| { let d = t as f64 - mean; d * d })
        .sum::<f64>() / n;
    let stddev = var.sqrt();

    Stats {
        min_us:    mn     / 1_000.0,
        mean_us:   mean   / 1_000.0,
        median_us: med    / 1_000.0,
        max_us:    mx     / 1_000.0,
        stddev_us: stddev / 1_000.0,
    }
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

pub fn print_csv_header() {
    println!("operation,implementation,duration_s,n_samples,iterations,warmup,\
              min_us,mean_us,median_us,max_us,stddev_us");
}

pub fn print_csv_row(
    op: &str, impl_name: &str,
    duration_s: u64, n_samples: usize,
    iterations: usize, warmup: usize,
    s: &Stats,
) {
    println!("{op},{impl_name},{duration_s},{n_samples},{iterations},{warmup},\
              {:.2},{:.2},{:.2},{:.2},{:.2}",
             s.min_us, s.mean_us, s.median_us, s.max_us, s.stddev_us);
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

pub struct Args {
    pub duration_s:  u64,
    pub iterations:  usize,
    pub warmup:      usize,
}

impl Args {
    pub fn parse_with_defaults(default_iters: usize, default_warmup: usize) -> Self {
        let args: Vec<String> = std::env::args().collect();
        let get = |flag: &str, def: usize| -> usize {
            args.windows(2)
                .find(|w| w[0] == flag)
                .and_then(|w| w[1].parse().ok())
                .unwrap_or(def)
        };
        let get_u64 = |flag: &str, def: u64| -> u64 {
            args.windows(2)
                .find(|w| w[0] == flag)
                .and_then(|w| w[1].parse().ok())
                .unwrap_or(def)
        };
        Args {
            duration_s: get_u64("--duration",   1),
            iterations: get("--iterations", default_iters),
            warmup:     get("--warmup",     default_warmup),
        }
    }
}
