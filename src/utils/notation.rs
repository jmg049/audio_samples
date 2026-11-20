//! Tuning systems and musical notation: interval generators, Functional Just System,
//! svara, mela and thaat names.
//!
//! - Tuning: [`pythagorean_intervals`], [`plimit_intervals`], [`interval_frequencies`].
//! - Functional Just System (FJS): [`fifths_to_note`], [`interval_to_fjs`], [`hz_to_fjs`].
//! - Hindustani notation: [`list_thaat`], [`thaat_to_degrees`], [`midi_to_svara_h`],
//!   [`hz_to_svara_h`], [`note_to_svara_h`].
//! - Carnatic notation: [`list_mela`], [`mela_to_degrees`], [`mela_to_svara`],
//!   [`midi_to_svara_c`], [`hz_to_svara_c`], [`note_to_svara_c`].
//!
//! Analysis that labels pitches needs more than equal temperament and Western note
//! names. Just intonation gives other frequency grids for a constant-Q transform, and
//! Indian classical music names pitches relative to a chosen tonic.
//!
//! The functions work on scalars and strings; mapping over arrays is left to the caller.
//! Names use Unicode symbols or ASCII as asked. Unknown names and out-of-range values give
//! a [`ParameterError`].
//!
//! ```
//! use audio_samples::utils::notation::{interval_to_fjs, mela_to_svara, midi_to_svara_h};
//!
//! assert_eq!(interval_to_fjs(5.0 / 4.0, "C", 65.0 / 63.0, true).unwrap(), "E⁵");
//! assert_eq!(midi_to_svara_h(67.0, 60.0, true, true, true), "P");
//! assert_eq!(mela_to_svara(1, true, false).unwrap()[1], "R1");
//! ```
//!
//! ## References
//!
//! W. Reinhard, *Functional Just System*, 2011 onward; J. Tenney, *John Cage
//! and the theory of harmony*, 1983 (harmonic distance); N. Sambamoorthy, *South Indian
//! Music*, 1958 (melakarta); V. N. Bhatkhande, *Hindustani Sangeet Paddhati*, 1934 (thaat).

use std::sync::OnceLock;

use crate::utils::audio_math::{hz_to_midi, key_to_notes, note_to_midi_fractional};
use crate::{AudioSampleResult, ParameterError};

fn invalid<T>(parameter: &str, reason: impl Into<String>) -> AudioSampleResult<T> {
    Err(ParameterError::invalid_value(parameter, reason.into()).into())
}

fn round_half_even(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - v.signum()
    } else {
        r
    }
}

/// A prime factorisation: `(prime, exponent)` pairs, including a power of 2 where it is
/// not zero.
pub type Factors = Vec<(u32, i32)>;

/// Reduces log-ratios to one octave and sorts them or keeps their order.
fn fractional_parts(values: &[f64]) -> Vec<(f64, i64)> {
    values
        .iter()
        .map(|&v| {
            let (mut frac, mut whole) = (v.fract(), v.trunc() as i64);
            if frac < 0.0 {
                frac += 1.0;
                whole -= 1;
            }
            (frac, whole)
        })
        .collect()
}

fn sort_order(values: &[(f64, i64)], sort: bool) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    if sort {
        order.sort_by(|&a, &b| values[a].0.total_cmp(&values[b].0));
    }
    order
}

/// One octave of the Pythagorean tuning: powers of 3 reduced to `[1, 2)`.
///
/// # Arguments
/// - `bins_per_octave`: the number of intervals.
/// - `sort`: in increasing order; otherwise in order of the power of 3.
///
/// # Returns
/// The ratios.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::pythagorean_intervals;
///
/// let r = pythagorean_intervals(3, true);
/// assert!((r[0] - 1.0).abs() < 1e-12);
/// assert!((r[1] - 1.125).abs() < 1e-12);
/// assert!((r[2] - 1.5).abs() < 1e-12);
/// ```
#[must_use]
pub fn pythagorean_intervals(bins_per_octave: usize, sort: bool) -> Vec<f64> {
    let logs: Vec<f64> = (0..bins_per_octave)
        .map(|k| k as f64 * 3f64.log2())
        .collect();
    let parts = fractional_parts(&logs);
    sort_order(&parts, sort)
        .into_iter()
        .map(|i| 2f64.powf(parts[i].0))
        .collect()
}

/// The prime factors of [`pythagorean_intervals`].
///
/// # Arguments
/// - `bins_per_octave`: the number of intervals.
/// - `sort`: as for [`pythagorean_intervals`].
///
/// # Returns
/// `[(2, a), (3, b)]` per interval.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::pythagorean_factors;
///
/// assert_eq!(pythagorean_factors(2, false)[1], vec![(2, -1), (3, 1)]);
/// ```
#[must_use]
pub fn pythagorean_factors(bins_per_octave: usize, sort: bool) -> Vec<Factors> {
    let logs: Vec<f64> = (0..bins_per_octave)
        .map(|k| k as f64 * 3f64.log2())
        .collect();
    let parts = fractional_parts(&logs);
    sort_order(&parts, sort)
        .into_iter()
        .map(|i| vec![(2, -parts[i].1 as i32), (3, i as i32)])
        .collect()
}

/// The closed set of intervals grown from the prime-limit lattice, with each step adding
/// the point nearest in total harmonic distance to the points already chosen.
fn plimit_lattice(primes: &[u32], bins_per_octave: usize) -> Vec<Vec<i32>> {
    let logs: Vec<f64> = primes.iter().map(|&p| f64::from(p).log2()).collect();
    let harmonic = |a: &[i32], b: &[i32]| -> f64 {
        // log2(a b / gcd(a, b)^2) over the prime exponents.
        let mut sum = 0.0;
        for ((&x, &y), l) in a.iter().zip(b).zip(&logs) {
            let (xn, xd) = (x.max(0), (-x).max(0));
            let (yn, yd) = (y.max(0), (-y).max(0));
            let gcd = xn.min(yn) - xd.max(yd);
            sum += f64::from(x + y - 2 * gcd) * l;
        }
        (sum * 1e6).round_ties_even() / 1e6
    };
    let n = primes.len();
    let mut seeds: Vec<Vec<i32>> = Vec::with_capacity(2 * n);
    for i in 0..n {
        for s in [1, -1] {
            let mut seed = vec![0; n];
            seed[i] = s;
            seeds.push(seed);
        }
    }
    let mut frontier = seeds.clone();
    let mut intervals: Vec<Vec<i32>> = vec![vec![0; n]];
    // The total distance of each frontier point to the chosen set.
    let mut scores: Vec<f64> = frontier
        .iter()
        .map(|p| harmonic(p, &intervals[0]))
        .collect();
    while intervals.len() < bins_per_octave && !frontier.is_empty() {
        let abs_dot = |p: &[i32]| -> f64 {
            p.iter()
                .zip(&logs)
                .map(|(&x, l)| f64::from(x.abs()) * l)
                .sum()
        };
        let mut best = 0;
        let mut score = f64::INFINITY;
        for (f, &hd) in scores.iter().enumerate() {
            let close = (hd - score).abs() <= 1e-8 + 1e-5 * score.abs();
            if hd < score || (close && abs_dot(&frontier[f]) < abs_dot(&frontier[best])) {
                score = hd;
                best = f;
            }
        }
        let chosen = frontier.remove(best);
        scores.remove(best);
        for (p, s) in frontier.iter().zip(scores.iter_mut()) {
            *s += harmonic(p, &chosen);
        }
        for seed in &seeds {
            let candidate: Vec<i32> = chosen.iter().zip(seed).map(|(a, b)| a + b).collect();
            if !intervals.contains(&candidate)
                && candidate != chosen
                && !frontier.contains(&candidate)
            {
                let total = intervals
                    .iter()
                    .chain(std::iter::once(&chosen))
                    .map(|s| harmonic(&candidate, s))
                    .sum();
                frontier.push(candidate);
                scores.push(total);
            }
        }
        intervals.push(chosen);
    }
    intervals
}

fn check_primes(primes: &[u32]) -> AudioSampleResult<()> {
    if primes.is_empty() {
        return invalid("primes", "must not be empty");
    }
    if primes.iter().any(|&p| p < 2) {
        return invalid("primes", "must be at least 2");
    }
    Ok(())
}

/// One octave of a just-intonation system over the given primes.
///
/// # Arguments
/// - `primes`: the primes, such as `[3, 5]` for 5-limit.
/// - `bins_per_octave`: the number of intervals.
/// - `sort`: in increasing order; otherwise in the order they were chosen.
///
/// # Returns
/// The ratios in `[1, 2)`.
///
/// # Errors
/// [`ParameterError`] if `primes` is empty or holds a value below 2.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::plimit_intervals;
///
/// let r = plimit_intervals(&[3, 5], 3, true).unwrap();
/// assert!((r[1] - 1.25).abs() < 1e-9 || (r[1] - 1.2).abs() < 1e-9 || (r[1] - 4.0 / 3.0).abs() < 1e-9);
/// assert_eq!(r[0], 1.0);
/// ```
pub fn plimit_intervals(
    primes: &[u32],
    bins_per_octave: usize,
    sort: bool,
) -> AudioSampleResult<Vec<f64>> {
    check_primes(primes)?;
    let points = plimit_lattice(primes, bins_per_octave);
    let logs: Vec<f64> = primes.iter().map(|&p| f64::from(p).log2()).collect();
    let values: Vec<f64> = points
        .iter()
        .map(|p| p.iter().zip(&logs).map(|(&x, l)| f64::from(x) * l).sum())
        .collect();
    let parts = fractional_parts(&values);
    Ok(sort_order(&parts, sort)
        .into_iter()
        .map(|i| 2f64.powf(parts[i].0))
        .collect())
}

/// The prime factors of [`plimit_intervals`].
///
/// # Arguments
/// - `primes`, `bins_per_octave`, `sort`: as for [`plimit_intervals`].
///
/// # Returns
/// `(prime, exponent)` pairs per interval, without zero exponents.
///
/// # Errors
/// As for [`plimit_intervals`].
pub fn plimit_factors(
    primes: &[u32],
    bins_per_octave: usize,
    sort: bool,
) -> AudioSampleResult<Vec<Factors>> {
    check_primes(primes)?;
    let points = plimit_lattice(primes, bins_per_octave);
    let logs: Vec<f64> = primes.iter().map(|&p| f64::from(p).log2()).collect();
    let values: Vec<f64> = points
        .iter()
        .map(|p| p.iter().zip(&logs).map(|(&x, l)| f64::from(x) * l).sum())
        .collect();
    let parts = fractional_parts(&values);
    Ok(sort_order(&parts, sort)
        .into_iter()
        .map(|i| {
            let mut f = Vec::new();
            if parts[i].1 != 0 {
                f.push((2, -parts[i].1 as i32));
            }
            for (&p, &e) in primes.iter().zip(&points[i]) {
                if e != 0 {
                    f.push((p, e));
                }
            }
            f
        })
        .collect())
}

/// A tuning system for [`interval_frequencies`].
#[derive(Debug, Clone, PartialEq)]
pub enum TuningSystem {
    /// Equal temperament, with a tuning offset.
    Equal,
    /// Pythagorean tuning: 3-limit by powers of 3.
    Pythagorean,
    /// 3-limit just intonation.
    Ji3,
    /// 5-limit just intonation.
    Ji5,
    /// 7-limit just intonation.
    Ji7,
    /// Explicit ratios for one octave.
    Custom(Vec<f64>),
}

impl TuningSystem {
    /// The ratios of one octave.
    ///
    /// # Arguments
    /// - `bins_per_octave`: intervals per octave; ignored by [`TuningSystem::Custom`].
    /// - `tuning`: offset of [`TuningSystem::Equal`] in fractions of a bin.
    /// - `sort`: as for [`pythagorean_intervals`].
    ///
    /// # Errors
    /// [`ParameterError`] if a custom ratio is not positive and finite, or the list is
    /// empty.
    pub fn ratios(
        &self,
        bins_per_octave: usize,
        tuning: f64,
        sort: bool,
    ) -> AudioSampleResult<Vec<f64>> {
        Ok(match self {
            Self::Equal => (0..bins_per_octave)
                .map(|i| 2f64.powf((tuning + i as f64) / bins_per_octave as f64))
                .collect(),
            Self::Pythagorean => pythagorean_intervals(bins_per_octave, sort),
            Self::Ji3 => plimit_intervals(&[3], bins_per_octave, sort)?,
            Self::Ji5 => plimit_intervals(&[3, 5], bins_per_octave, sort)?,
            Self::Ji7 => plimit_intervals(&[3, 5, 7], bins_per_octave, sort)?,
            Self::Custom(v) => {
                if v.is_empty() || v.iter().any(|r| !(r.is_finite() && *r > 0.0)) {
                    return invalid("intervals", "must be positive, finite and not empty");
                }
                v.clone()
            }
        })
    }

    /// Finds a system by name: `equal`, `pythagorean`, `ji3`, `ji5` or `ji7`.
    ///
    /// # Returns
    /// The system, or `None` for another name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "equal" => Self::Equal,
            "pythagorean" => Self::Pythagorean,
            "ji3" => Self::Ji3,
            "ji5" => Self::Ji5,
            "ji7" => Self::Ji7,
            _ => return None,
        })
    }
}

/// Frequencies of a tuning system over several octaves.
///
/// # Arguments
/// - `n_bins`: the number of frequencies.
/// - `fmin`: the lowest frequency, in Hz.
/// - `intervals`: the tuning system.
/// - `bins_per_octave`: intervals per octave.
/// - `tuning`: offset of equal temperament in fractions of a bin.
/// - `sort`: in increasing order.
///
/// # Returns
/// The frequencies.
///
/// # Errors
/// [`ParameterError`] if `fmin` is not positive and finite, or the system is invalid.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::{TuningSystem, interval_frequencies};
///
/// let f = interval_frequencies(2, 100.0, &TuningSystem::Pythagorean, 2, 0.0, true).unwrap();
/// assert_eq!(f[0], 100.0);
/// assert!((f[1] - 150.0).abs() < 1e-9);
/// ```
pub fn interval_frequencies(
    n_bins: usize,
    fmin: f64,
    intervals: &TuningSystem,
    bins_per_octave: usize,
    tuning: f64,
    sort: bool,
) -> AudioSampleResult<Vec<f64>> {
    if !(fmin.is_finite() && fmin > 0.0) {
        return invalid("fmin", "must be positive and finite");
    }
    let ratios = intervals.ratios(bins_per_octave, tuning, sort)?;
    if ratios.is_empty() {
        return invalid("bins_per_octave", "must be positive");
    }
    let per_octave = ratios.len();
    let mut out: Vec<f64> = (0..n_bins)
        .map(|i| 2f64.powi((i / per_octave) as i32) * ratios[i % per_octave])
        .collect();
    if sort {
        out.sort_by(f64::total_cmp);
    }
    for v in &mut out {
        *v *= fmin;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Functional Just System
// ---------------------------------------------------------------------------

/// The letter and accidental count of a note name.
fn parse_note(name: &str) -> AudioSampleResult<(char, i64)> {
    let mut chars = name.chars();
    let letter = chars.next().map(|c| c.to_ascii_uppercase());
    let Some(letter @ 'A'..='G') = letter else {
        return invalid("note", format!("improper note format: {name:?}"));
    };
    let mut offset = 0;
    for c in chars {
        offset += match c {
            '#' | '♯' => 1,
            'b' | '!' | '♭' => -1,
            '𝄪' => 2,
            '𝄫' => -2,
            'n' | '♮' => 0,
            _ => break,
        };
    }
    Ok((letter, offset))
}

/// The note a number of perfect fifths from a unison.
///
/// # Arguments
/// - `unison`: the starting note, such as `"C"` or `"F#"`.
/// - `fifths`: signed number of fifths.
/// - `unicode`: use Unicode accidentals; otherwise ASCII.
///
/// # Returns
/// The note's name, without an octave.
///
/// # Errors
/// [`ParameterError`] if `unison` is not a note name.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::fifths_to_note;
///
/// assert_eq!(fifths_to_note("C", 2, true).unwrap(), "D");
/// assert_eq!(fifths_to_note("C", -2, false).unwrap(), "Bb");
/// ```
pub fn fifths_to_note(unison: &str, fifths: i64, unicode: bool) -> AudioSampleResult<String> {
    const CIRCLE: [char; 7] = ['F', 'C', 'G', 'D', 'A', 'E', 'B'];
    let (letter, offset) = parse_note(unison)?;
    let start = CIRCLE.iter().position(|&c| c == letter).unwrap_or_default() as i64;
    let position = start + fifths;
    let mut out = String::with_capacity(8);
    out.push(CIRCLE[position.rem_euclid(7) as usize]);
    let accidentals = offset + position.div_euclid(7);
    let (single, double) = match (accidentals >= 0, unicode) {
        (true, true) => ("♯", "𝄪"),
        (true, false) => ("#", "##"),
        (false, true) => ("♭", "𝄫"),
        (false, false) => ("b", "bb"),
    };
    let count = accidentals.unsigned_abs();
    for _ in 0..count / 2 {
        out.push_str(double);
    }
    if count % 2 == 1 {
        out.push_str(single);
    }
    Ok(out)
}

fn octave_fold(x: f64) -> f64 {
    x * 2f64.powf(-x.log2().floor())
}

fn balanced_fold(x: f64) -> f64 {
    x * 2f64.powf(-x.log2().round_ties_even())
}

/// The smallest number of fifths that comes within `tolerance` of an interval, at most 31
/// each way, positive first.
fn fifth_search(interval: f64, tolerance: f64) -> Option<i64> {
    let limit = tolerance.log2().abs();
    for power in 0..32_i32 {
        for sign in [1, -1] {
            let value = interval / 3f64.powi(power * sign);
            if balanced_fold(value).log2().abs() <= limit {
                return Some(i64::from(power * sign));
            }
        }
    }
    None
}

/// Candidates for the primes above 3 of an octave-reduced interval, by increasing height:
/// one of 11 to 23 at most once, 7 at most squared and 5 at most to the fourth power.
fn high_prime_candidates() -> &'static [(f64, [i32; 7])] {
    static TABLE: OnceLock<Vec<(f64, [i32; 7])>> = OnceLock::new();
    TABLE.get_or_init(|| {
        const PRIMES: [f64; 7] = [5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0];
        let mut out = Vec::new();
        for e5 in -4_i32..=4 {
            for e7 in -2_i32..=2 {
                for which in 0..=5 {
                    let signs: &[i32] = if which == 0 { &[0] } else { &[1, -1] };
                    for &s in signs {
                        let mut powers = [e5, e7, 0, 0, 0, 0, 0];
                        if which > 0 {
                            powers[which + 1] = s;
                        }
                        let log: f64 = powers
                            .iter()
                            .zip(PRIMES)
                            .map(|(&e, p)| f64::from(e.abs()) * p.log2())
                            .sum();
                        out.push((log, powers));
                    }
                }
            }
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out
    })
}

/// The exponents of 5, 7, 11, 13, 17, 19 and 23 of an interval, up to a power of 3.
fn high_prime_powers(interval: f64) -> Option<[i32; 7]> {
    const PRIMES: [f64; 7] = [5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0];
    const TOLERANCE: f64 = 1.2e-6;
    let target = octave_fold(interval);
    let three = 3f64.log2();
    // Fractional parts of the powers of 3, up to 71 each way.
    for (_, powers) in high_prime_candidates() {
        let mut part = 1.0;
        for (&e, p) in powers.iter().zip(PRIMES) {
            part *= p.powi(e);
        }
        let need = (target / part).log2();
        for k in -71_i32..=71 {
            let log = f64::from(k) * three;
            let frac = (log - log.floor()) - (need - need.floor());
            let d = frac - frac.round();
            if (d * std::f64::consts::LN_2 * target).abs() <= TOLERANCE {
                return Some(*powers);
            }
        }
    }
    None
}

/// The Functional Just System spelling of an interval.
///
/// The letter comes from the nearest stack of fifths. Primes above 3 appear as a
/// superscript (otonal) or subscript (utonal). Ratios built from the primes up to 23 are
/// recognised, to six decimal places.
///
/// # Arguments
/// - `interval`: the ratio, positive.
/// - `unison`: the note of ratio 1.
/// - `tolerance`: how close in ratio the stack of fifths must come. The usual value is
///   `65 / 63`.
/// - `unicode`: use Unicode accidentals and scripts; otherwise `^n` and `_n`.
///
/// # Returns
/// The spelling.
///
/// # Errors
/// [`ParameterError`] if the interval is not positive, no stack of up to 31 fifths comes
/// within the tolerance, or the ratio is not recognised.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::interval_to_fjs;
///
/// assert_eq!(interval_to_fjs(3.0 / 2.0, "C", 65.0 / 63.0, true).unwrap(), "G");
/// assert_eq!(interval_to_fjs(7.0 / 4.0, "C", 65.0 / 63.0, false).unwrap(), "Bb^7");
/// ```
pub fn interval_to_fjs(
    interval: f64,
    unison: &str,
    tolerance: f64,
    unicode: bool,
) -> AudioSampleResult<String> {
    if !(interval.is_finite() && interval > 0.0) {
        return invalid("interval", format!("{interval} must be strictly positive"));
    }
    let Some(fifths) = fifth_search(interval, tolerance) else {
        return invalid(
            "interval",
            format!("no stack of fifths is within tolerance of {interval}"),
        );
    };
    let mut out = fifths_to_note(unison, fifths, unicode)?;
    let Some(powers) = high_prime_powers(interval) else {
        return invalid("interval", format!("unknown interval {interval}"));
    };
    const PRIMES: [u128; 7] = [5, 7, 11, 13, 17, 19, 23];
    let (mut otonal, mut utonal) = (1_u128, 1_u128);
    for (&e, p) in powers.iter().zip(PRIMES) {
        if e > 0 {
            otonal *= p.pow(e as u32);
        } else if e < 0 {
            utonal *= p.pow((-e) as u32);
        }
    }
    let script = |value: u128, digits: &[char; 10], out: &mut String| {
        for d in value.to_string().bytes() {
            out.push(digits[usize::from(d - b'0')]);
        }
    };
    const SUPER: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    const SUB: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
    if otonal > 1 {
        if unicode {
            script(otonal, &SUPER, &mut out);
        } else {
            out.push('^');
            out.push_str(&otonal.to_string());
        }
    }
    if utonal > 1 {
        if unicode {
            script(utonal, &SUB, &mut out);
        } else {
            out.push('_');
            out.push_str(&utonal.to_string());
        }
    }
    Ok(out)
}

/// The Functional Just System spelling of a frequency relative to a reference.
///
/// # Arguments
/// - `frequency`: in Hz.
/// - `fmin`: the frequency of ratio 1, in Hz.
/// - `unison`: the note of ratio 1; `None` names the nearest equal-tempered note to `fmin`.
/// - `unicode`: as for [`interval_to_fjs`].
///
/// # Returns
/// The spelling.
///
/// # Errors
/// As for [`interval_to_fjs`].
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::hz_to_fjs;
///
/// assert_eq!(hz_to_fjs(330.0, 220.0, None, false).unwrap(), "E");
/// ```
pub fn hz_to_fjs(
    frequency: f64,
    fmin: f64,
    unison: Option<&str>,
    unicode: bool,
) -> AudioSampleResult<String> {
    if !(fmin.is_finite() && fmin > 0.0) {
        return invalid("fmin", "must be positive and finite");
    }
    let unison = match unison {
        Some(u) => u.to_owned(),
        None => {
            let notes = key_to_notes("C:maj", false, false)?;
            let number = round_half_even(hz_to_midi(fmin)) as i64;
            notes[number.rem_euclid(12) as usize].clone()
        }
    };
    interval_to_fjs(frequency / fmin, &unison, 65.0 / 63.0, unicode)
}

// ---------------------------------------------------------------------------
// Hindustani and Carnatic names
// ---------------------------------------------------------------------------

const THAATS: [(&str, [u8; 7]); 10] = [
    ("bilaval", [0, 2, 4, 5, 7, 9, 11]),
    ("khamaj", [0, 2, 4, 5, 7, 9, 10]),
    ("kafi", [0, 2, 3, 5, 7, 9, 10]),
    ("asavari", [0, 2, 3, 5, 7, 8, 10]),
    ("bhairavi", [0, 1, 3, 5, 7, 8, 10]),
    ("kalyan", [0, 2, 4, 6, 7, 9, 11]),
    ("marva", [0, 1, 4, 6, 7, 9, 11]),
    ("poorvi", [0, 1, 4, 6, 7, 8, 11]),
    ("todi", [0, 1, 3, 6, 7, 8, 11]),
    ("bhairav", [0, 1, 4, 5, 7, 8, 11]),
];

const MELAS: [&str; 72] = [
    "kanakangi",
    "ratnangi",
    "ganamurthi",
    "vanaspathi",
    "manavathi",
    "tanarupi",
    "senavathi",
    "hanumathodi",
    "dhenuka",
    "natakapriya",
    "kokilapriya",
    "rupavathi",
    "gayakapriya",
    "vakulabharanam",
    "mayamalavagaula",
    "chakravakom",
    "suryakantham",
    "hatakambari",
    "jhankaradhwani",
    "natabhairavi",
    "keeravani",
    "kharaharapriya",
    "gaurimanohari",
    "varunapriya",
    "mararanjini",
    "charukesi",
    "sarasangi",
    "harikambhoji",
    "dheerasankarabharanam",
    "naganandini",
    "yagapriya",
    "ragavardhini",
    "gangeyabhushani",
    "vagadheeswari",
    "sulini",
    "chalanatta",
    "salagam",
    "jalarnavam",
    "jhalavarali",
    "navaneetham",
    "pavani",
    "raghupriya",
    "gavambodhi",
    "bhavapriya",
    "subhapanthuvarali",
    "shadvidhamargini",
    "suvarnangi",
    "divyamani",
    "dhavalambari",
    "namanarayani",
    "kamavardhini",
    "ramapriya",
    "gamanasrama",
    "viswambhari",
    "syamalangi",
    "shanmukhapriya",
    "simhendramadhyamam",
    "hemavathi",
    "dharmavathi",
    "neethimathi",
    "kanthamani",
    "rishabhapriya",
    "latangi",
    "vachaspathi",
    "mechakalyani",
    "chitrambari",
    "sucharitra",
    "jyotisvarupini",
    "dhatuvardhini",
    "nasikabhushani",
    "kosalam",
    "rasikapriya",
];

/// Where a name given in any letter case stands among lower-case names. ASCII input does
/// not allocate; other input goes through Unicode lower-casing, which maps a few non-ASCII
/// letters to ASCII ones.
fn position_ignoring_case<'a>(
    mut names: impl Iterator<Item = &'a str>,
    wanted: &str,
) -> Option<usize> {
    if wanted.is_ascii() {
        names.position(|name| name.eq_ignore_ascii_case(wanted))
    } else {
        let lower = wanted.to_lowercase();
        names.position(|name| name == lower)
    }
}

/// The ten thaats, by name.
///
/// # Returns
/// The names in their usual order.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::list_thaat;
///
/// assert_eq!(list_thaat()[0], "bilaval");
/// assert_eq!(list_thaat().len(), 10);
/// ```
#[must_use]
pub fn list_thaat() -> [&'static str; 10] {
    THAATS.map(|(name, _)| name)
}

/// The scale degrees of a thaat, in semitones above Sa.
///
/// # Arguments
/// - `thaat`: its name, in any letter case.
///
/// # Returns
/// Seven degrees.
///
/// # Errors
/// [`ParameterError`] if the name is unknown.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::thaat_to_degrees;
///
/// assert_eq!(thaat_to_degrees("Kafi").unwrap(), [0, 2, 3, 5, 7, 9, 10]);
/// ```
pub fn thaat_to_degrees(thaat: &str) -> AudioSampleResult<[u8; 7]> {
    position_ignoring_case(THAATS.iter().map(|(name, _)| *name), thaat)
        .map(|i| THAATS[i].1)
        .ok_or_else(|| {
            ParameterError::invalid_value("thaat", format!("unknown thaat {thaat:?}")).into()
        })
}

/// The 72 melakarta ragas, by name, in order of their numbers.
///
/// # Returns
/// The names; the raga at index `i` has number `i + 1`.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::list_mela;
///
/// assert_eq!(list_mela()[71], "rasikapriya");
/// ```
#[must_use]
pub const fn list_mela() -> &'static [&'static str; 72] {
    &MELAS
}

/// The number of a melakarta raga from its name.
///
/// # Arguments
/// - `name`: its name, in any letter case.
///
/// # Returns
/// The number, from 1 to 72.
///
/// # Errors
/// [`ParameterError`] if the name is unknown.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::mela_number;
///
/// assert_eq!(mela_number("Kanakangi").unwrap(), 1);
/// ```
pub fn mela_number(name: &str) -> AudioSampleResult<u8> {
    position_ignoring_case(MELAS.iter().copied(), name)
        .map(|i| (i + 1) as u8)
        .ok_or_else(|| {
            ParameterError::invalid_value("mela", format!("unknown mela {name:?}")).into()
        })
}

fn check_mela(mela: u8) -> AudioSampleResult<usize> {
    if !(1..=72).contains(&mela) {
        return invalid("mela", format!("{mela} must be in the range 1 to 72"));
    }
    Ok(usize::from(mela) - 1)
}

/// The scale degrees of a melakarta raga, in semitones above Sa.
///
/// Sa and Pa are fixed; the position within each group of 6 and of 36 ragas selects the
/// other svaras.
///
/// # Arguments
/// - `mela`: the raga's number, from 1 to 72.
///
/// # Returns
/// Seven degrees.
///
/// # Errors
/// [`ParameterError`] if `mela` is out of range.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::mela_to_degrees;
///
/// assert_eq!(mela_to_degrees(29).unwrap(), [0, 2, 4, 5, 7, 9, 11]);
/// ```
pub fn mela_to_degrees(mela: u8) -> AudioSampleResult<[u8; 7]> {
    let index = check_mela(mela)?;
    let (ri, ga) = match (index % 36) / 6 {
        0 => (1, 2),
        1 => (1, 3),
        2 => (1, 4),
        3 => (2, 3),
        4 => (2, 4),
        _ => (3, 4),
    };
    let ma = if index < 36 { 5 } else { 6 };
    let (dha, ni) = match index % 6 {
        0 => (8, 9),
        1 => (8, 10),
        2 => (8, 11),
        3 => (9, 10),
        4 => (9, 11),
        _ => (10, 11),
    };
    Ok([0, ri, ga, ma, 7, dha, ni])
}

/// The name of a svara, held inline so that it needs no allocation.
///
/// It reads as a `str` and compares equal to strings with the same text.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::midi_to_svara_h;
///
/// let name = midi_to_svara_h(61.0, 60.0, false, true, false);
/// assert_eq!(name, "re");
/// assert_eq!(name.len(), 2);
/// assert_eq!(name.to_string(), "re");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Svara {
    /// UTF-8; the longest name, a three-letter stem with a subscript digit and a combining
    /// dot, takes eight bytes.
    bytes: [u8; 15],
    len: u8,
}

impl Svara {
    /// The name as a string slice.
    ///
    /// # Returns
    /// The text of the name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        // Only whole characters are ever appended.
        std::str::from_utf8(&self.bytes[..usize::from(self.len)]).unwrap_or_default()
    }

    fn push(&mut self, c: char) {
        let start = usize::from(self.len);
        let added = c.encode_utf8(&mut self.bytes[start..]).len();
        self.len += added as u8;
    }

    fn push_str(&mut self, text: &str) {
        text.chars().for_each(|c| self.push(c));
    }

    /// A name from its long form: all of it, or its first letter when `abbr`.
    fn stem(long: &str, abbr: bool) -> Self {
        let mut name = Self::default();
        if abbr {
            name.push_str(&long[..1]);
        } else {
            name.push_str(long);
        }
        name
    }
}

impl std::ops::Deref for Svara {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for Svara {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for Svara {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::fmt::Debug for Svara {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.as_str(), f)
    }
}

impl From<Svara> for String {
    fn from(name: Svara) -> Self {
        name.as_str().to_owned()
    }
}

impl PartialEq<str> for Svara {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for Svara {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<String> for Svara {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

fn subscript_digit(unicode: bool, digit: char) -> char {
    if unicode {
        match digit {
            '1' => '\u{2081}',
            '2' => '\u{2082}',
            _ => '\u{2083}',
        }
    } else {
        digit
    }
}

fn svara_name(stem: &str, digit: char, abbr: bool, unicode: bool) -> Svara {
    let mut name = Svara::stem(stem, abbr);
    name.push(subscript_digit(unicode, digit));
    name
}

/// The names of the twelve svaras of a melakarta raga, from Sa.
///
/// # Arguments
/// - `mela`: the raga's number, from 1 to 72.
/// - `abbr`: short forms such as `R1` rather than `Ri1`.
/// - `unicode`: subscript digits; otherwise plain digits.
///
/// # Returns
/// Twelve names.
///
/// # Errors
/// [`ParameterError`] if `mela` is out of range.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::mela_to_svara;
///
/// let names = mela_to_svara(1, false, false).unwrap();
/// assert_eq!(names[0], "Sa");
/// assert_eq!(names[2], "Ga1");
/// ```
pub fn mela_to_svara(mela: u8, abbr: bool, unicode: bool) -> AudioSampleResult<[Svara; 12]> {
    let index = check_mela(mela)?;
    let lower = index % 36;
    let upper = index % 6;
    Ok([
        Svara::stem("Sa", abbr),
        svara_name("Ri", '1', abbr, unicode),
        if lower < 6 {
            svara_name("Ga", '1', abbr, unicode)
        } else {
            svara_name("Ri", '2', abbr, unicode)
        },
        if lower < 30 {
            svara_name("Ga", '2', abbr, unicode)
        } else {
            svara_name("Ri", '3', abbr, unicode)
        },
        svara_name("Ga", '3', abbr, unicode),
        svara_name("Ma", '1', abbr, unicode),
        svara_name("Ma", '2', abbr, unicode),
        Svara::stem("Pa", abbr),
        svara_name("Dha", '1', abbr, unicode),
        if upper == 0 {
            svara_name("Ni", '1', abbr, unicode)
        } else {
            svara_name("Dha", '2', abbr, unicode)
        },
        if upper == 5 {
            svara_name("Dha", '3', abbr, unicode)
        } else {
            svara_name("Ni", '2', abbr, unicode)
        },
        svara_name("Ni", '3', abbr, unicode),
    ])
}

/// The name of a MIDI number in a key.
///
/// # Arguments
/// - `midi`: the MIDI number. A non-finite value gives an empty string.
/// - `octave`: include the octave.
/// - `cents`: include the deviation in cents; needs `octave`.
/// - `key`: the key, such as `"C:maj"`, that decides the spelling.
/// - `unicode`: Unicode accidentals; otherwise ASCII.
///
/// # Returns
/// The name.
///
/// # Errors
/// [`ParameterError`] if `cents` is asked for without `octave`, or the key is invalid.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::midi_name;
///
/// assert_eq!(midi_name(69.25, true, true, "C:maj", false).unwrap(), "A4+25");
/// assert_eq!(midi_name(70.0, false, false, "F:maj", false).unwrap(), "Bb");
/// ```
pub fn midi_name(
    midi: f64,
    octave: bool,
    cents: bool,
    key: &str,
    unicode: bool,
) -> AudioSampleResult<String> {
    if cents && !octave {
        return invalid("cents", "needs octave information");
    }
    if !midi.is_finite() {
        return Ok(String::new());
    }
    let notes = crate::utils::audio_math::key_to_notes(key, unicode, false)?;
    midi_name_in(&notes, midi, octave, cents)
}

/// [`midi_name`] with the key's twelve note names already spelled (as
/// [`key_to_notes`] gives them), for
/// naming many numbers in one key.
///
/// # Errors
/// [`ParameterError`] if `cents` is set without `octave`, or `notes` does not
/// hold twelve names.
///
/// # Examples
/// ```
/// use audio_samples::utils::audio_math::key_to_notes;
/// use audio_samples::utils::notation::midi_name_in;
///
/// let notes = key_to_notes("F:maj", false, false).unwrap();
/// assert_eq!(midi_name_in(&notes, 70.0, true, false).unwrap(), "Bb4");
/// ```
pub fn midi_name_in(
    notes: &[String],
    midi: f64,
    octave: bool,
    cents: bool,
) -> AudioSampleResult<String> {
    if cents && !octave {
        return invalid("cents", "needs octave information");
    }
    if notes.len() != 12 {
        return invalid("notes", "needs the twelve note names of a key");
    }
    if !midi.is_finite() {
        return Ok(String::new());
    }
    let number = round_half_even(midi) as i64;
    let mut out = notes[number.rem_euclid(12) as usize].clone();
    if octave {
        out.push_str(&(number / 12 - 1).to_string());
    }
    if cents {
        let c = round_half_even((midi - number as f64) * 100.0) as i64;
        out.push_str(&format!("{c:+}"));
    }
    Ok(out)
}

/// Adds the octave mark: a dot above or `'` for the octave above Sa, a dot below or `,`
/// for the octave below.
fn decorate(svara: Svara, number: i64, octave: bool, unicode: bool) -> Svara {
    if !octave {
        return svara;
    }
    let mark = if (12..24).contains(&number) {
        Some(('\u{0307}', '\''))
    } else if (-12..0).contains(&number) {
        Some(('\u{0323}', ','))
    } else {
        None
    };
    let Some((combining, ascii)) = mark else {
        return svara;
    };
    let mut out = Svara::default();
    if unicode {
        let mut chars = svara.chars();
        if let Some(first) = chars.next() {
            out.push(first);
        }
        out.push(combining);
        out.push_str(chars.as_str());
    } else {
        out.push_str(svara.as_str());
        out.push(ascii);
    }
    out
}

/// The Hindustani svara of a MIDI number.
///
/// # Arguments
/// - `midi`: the MIDI number. A non-finite value gives an empty string.
/// - `sa`: the MIDI number of Sa.
/// - `abbr`: short forms such as `S` rather than `Sa`.
/// - `octave`: mark the octaves next to Sa's.
/// - `unicode`: dots; otherwise `'` and `,`.
///
/// # Returns
/// The svara.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::midi_to_svara_h;
///
/// assert_eq!(midi_to_svara_h(61.0, 60.0, false, true, false), "re");
/// assert_eq!(midi_to_svara_h(72.0, 60.0, true, true, false), "S'");
/// ```
#[must_use]
pub fn midi_to_svara_h(midi: f64, sa: f64, abbr: bool, octave: bool, unicode: bool) -> Svara {
    const NAMES: [&str; 12] = [
        "Sa", "re", "Re", "ga", "Ga", "ma", "Ma", "Pa", "dha", "Dha", "ni", "Ni",
    ];
    if !midi.is_finite() {
        return Svara::default();
    }
    let number = round_half_even(midi - sa) as i64;
    let svara = Svara::stem(NAMES[number.rem_euclid(12) as usize], abbr);
    decorate(svara, number, octave, unicode)
}

/// The Carnatic svara of a MIDI number within a melakarta raga.
///
/// # Arguments
/// - `midi`: the MIDI number. A non-finite value gives an empty string.
/// - `sa`: the MIDI number of Sa.
/// - `mela`: the raga's number, from 1 to 72.
/// - `abbr`, `octave`, `unicode`: as for [`midi_to_svara_h`].
///
/// # Returns
/// The svara.
///
/// # Errors
/// [`ParameterError`] if `mela` is out of range.
///
/// # Examples
/// ```
/// use audio_samples::utils::notation::midi_to_svara_c;
///
/// assert_eq!(midi_to_svara_c(62.0, 60.0, 1, true, true, false).unwrap(), "G1");
/// ```
pub fn midi_to_svara_c(
    midi: f64,
    sa: f64,
    mela: u8,
    abbr: bool,
    octave: bool,
    unicode: bool,
) -> AudioSampleResult<Svara> {
    let names = mela_to_svara(mela, abbr, unicode)?;
    if !midi.is_finite() {
        return Ok(Svara::default());
    }
    let number = round_half_even(midi - sa) as i64;
    let svara = names[number.rem_euclid(12) as usize];
    Ok(decorate(svara, number, octave, unicode))
}

/// [`midi_to_svara_h`] from frequencies in Hz.
///
/// # Arguments
/// - `frequency`: in Hz.
/// - `sa`: the frequency of Sa, in Hz.
/// - `abbr`, `octave`, `unicode`: as for [`midi_to_svara_h`].
///
/// # Returns
/// The svara.
#[must_use]
pub fn hz_to_svara_h(frequency: f64, sa: f64, abbr: bool, octave: bool, unicode: bool) -> Svara {
    midi_to_svara_h(hz_to_midi(frequency), hz_to_midi(sa), abbr, octave, unicode)
}

/// [`midi_to_svara_c`] from frequencies in Hz.
///
/// # Arguments
/// - `frequency`: in Hz.
/// - `sa`: the frequency of Sa, in Hz.
/// - `mela`, `abbr`, `octave`, `unicode`: as for [`midi_to_svara_c`].
///
/// # Returns
/// The svara.
///
/// # Errors
/// As for [`midi_to_svara_c`].
pub fn hz_to_svara_c(
    frequency: f64,
    sa: f64,
    mela: u8,
    abbr: bool,
    octave: bool,
    unicode: bool,
) -> AudioSampleResult<Svara> {
    midi_to_svara_c(
        hz_to_midi(frequency),
        hz_to_midi(sa),
        mela,
        abbr,
        octave,
        unicode,
    )
}

fn note_midi(name: &str) -> AudioSampleResult<f64> {
    note_to_midi_fractional(name)
}

/// [`midi_to_svara_h`] from note names.
///
/// # Arguments
/// - `note`: the note, such as `"C5"`.
/// - `sa`: the note of Sa.
/// - `abbr`, `octave`, `unicode`: as for [`midi_to_svara_h`].
///
/// # Returns
/// The svara.
///
/// # Errors
/// [`ParameterError`] if a note name is invalid.
pub fn note_to_svara_h(
    note: &str,
    sa: &str,
    abbr: bool,
    octave: bool,
    unicode: bool,
) -> AudioSampleResult<Svara> {
    Ok(midi_to_svara_h(
        note_midi(note)?,
        note_midi(sa)?,
        abbr,
        octave,
        unicode,
    ))
}

/// [`midi_to_svara_c`] from note names.
///
/// # Arguments
/// - `note`: the note, such as `"C5"`.
/// - `sa`: the note of Sa.
/// - `mela`, `abbr`, `octave`, `unicode`: as for [`midi_to_svara_c`].
///
/// # Returns
/// The svara.
///
/// # Errors
/// [`ParameterError`] if a note name is invalid or `mela` is out of range.
pub fn note_to_svara_c(
    note: &str,
    sa: &str,
    mela: u8,
    abbr: bool,
    octave: bool,
    unicode: bool,
) -> AudioSampleResult<Svara> {
    midi_to_svara_c(
        note_midi(note)?,
        note_midi(sa)?,
        mela,
        abbr,
        octave,
        unicode,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_arguments_are_rejected() {
        assert!(fifths_to_note("X", 1, true).is_err());
        assert!(interval_to_fjs(-1.0, "C", 1.03, true).is_err());
        assert!(thaat_to_degrees("nope").is_err());
        assert!(mela_number("nope").is_err());
        assert!(mela_to_degrees(0).is_err());
        assert!(mela_to_svara(73, true, true).is_err());
        assert!(plimit_intervals(&[], 4, true).is_err());
        assert!(interval_frequencies(4, -1.0, &TuningSystem::Equal, 4, 0.0, true).is_err());
        assert!(
            TuningSystem::Custom(vec![0.0])
                .ratios(1, 0.0, true)
                .is_err()
        );
    }

    #[test]
    fn non_finite_midi_is_empty() {
        assert_eq!(midi_to_svara_h(f64::NAN, 60.0, true, true, true), "");
        assert_eq!(
            midi_to_svara_c(f64::INFINITY, 60.0, 1, true, true, true).unwrap(),
            ""
        );
    }
}
