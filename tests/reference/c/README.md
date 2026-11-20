# filter_ref: independent IIR ground truth

`filter_ref` is a small C++17 command-line program. It designs IIR filters with
two long-standing third-party libraries and runs them through each library's own
processing code. Its output is ground truth for checking the Rust IIR designs:

| library | pinned version | precision |
|---|---|---|
| [DSPFilters](https://github.com/vinniefalco/DSPFilters) (Vinnie Falco) | commit `acc49170e79a94fcb9c04b8a2116e9f8dffd1c7d` | double |
| [liquid-dsp](https://github.com/jgaeddert/liquid-dsp) | tag `v1.8.3` (`10041f70cebbe3b97887e75bb41e48b73dda1b23`) | float32 |

All inputs use the SciPy convention. The harness converts them to each
library's own parameters (see "Parameter mapping" below).

## Build

```sh
tests/reference/c/build.sh
```

The script clones the pinned sources into `third_party/`. It builds liquid-dsp
as a static library with SIMD and FFTW turned off, compiles DSPFilters directly
from its sources with `-DNDEBUG`, and links `build/filter_ref`. Running it again
is safe: it re-uses the existing checkouts and builds.

Verification against SciPy:

```sh
cd tests/reference/c
~/.local/bin/uv venv .venv && ~/.local/bin/uv pip install --python .venv/bin/python numpy scipy
.venv/bin/python verify_harness.py          # full grid, about 14k cases, under a minute
.venv/bin/python verify_harness.py --quick  # smaller grid
```

## Protocol (stdin to stdout)

Each non-empty input line is one case, with fields separated by whitespace:

```
<id> <lib> <family> <response> <order> <fs> <f1> <f2> <rp> <rs> <gain_db> <q> <input_path>
```

* `lib`: `dspfilters` | `liquid`
* `family`: `butter` `cheby1` `cheby2` `ellip` `bessel` `rbj_lowpass` `rbj_highpass`
  `rbj_bandpass` `rbj_bandstop` `rbj_peak` `rbj_lowshelf` `rbj_highshelf`.
  The harness also accepts these Bessel variants:
  * `bessel` = `bessel_phase`, which matches SciPy's default `norm='phase'`
  * `bessel_mag` matches `norm='mag'`
  * `bessel_delay` matches `norm='delay'`
  * `bessel_native` is the library's own normalisation with no mapping, and is
    always `exact_convention: false`
* `response`: `lowpass` `highpass` `bandpass` `bandstop`. Use `-` for `rbj_*`.
* Frequencies are in Hz. Low-pass and high-pass use only `f1` (the cutoff).
  Band-pass and band-stop use `f1` = lower edge and `f2` = upper edge, as in
  `scipy.signal.butter(N, [f1, f2], btype=..., fs=fs)`. For `rbj_*`, `f1` is
  the centre or corner frequency, `q` is the Q and `gain_db` is the gain.
* `rp` is the pass-band ripple in dB (cheby1, ellip). `rs` is the stop-band
  attenuation in dB (cheby2, ellip). Give unused values as 0.
* `input_path` is a raw little-endian float64 mono file, or `-` for none.

The program writes one JSON object per case. Doubles are printed with `%.17g`,
and non-finite values are written as `null`.

```
{"id": "...", "ok": true, "exact_convention": true|false, "note": "...",
 "sos": [[b0,b1,b2,1,a1,a2], ...], "impulse": [2048 samples], "output": [len(input) samples or []]}
{"id": "...", "ok": false, "error": "..."}
```

* `sos` holds the library's own sections, normalised so that a0 = 1.
* `impulse` and `output` come from the library's own processing, starting from
  zero state:
  * DSPFilters: `SimpleFilter<..., TransposedDirectFormI>::process`
  * liquid-dsp: `iirfilt_rrrf_execute` on `iirfilt_rrrf_create_sos`, in float32
* `exact_convention: false` means the case is not a trustworthy SciPy-equivalent
  reference. `note` gives the reason:
  * no exact mapping exists (`bessel_native`), or
  * the case hits a measured precision defect in liquid-dsp:
    * its float32 analog prototype differs from the double-precision design by
      more than 2e-6 relative, or
    * float32 round-off in its processing changes the impulse response by more
      than 1e-3 relative, or
    * a designed pole has |p| >= 1.

## Parameter mapping (SciPy to library)

`W(f) = tan(pi f / fs)` is the prewarped frequency.

**DSPFilters**

* Low-pass and high-pass: `setup(order, fs, f1, ...)`. The library prewarps
  `tan(pi fc)` the same way SciPy does.
* Band-pass and band-stop: `setup(order, fs, centre=(f1+f2)/2, width=f2-f1, ...)`.
  The library places the digital band edges at `centre ± width/2` and applies
  the Constantinides transform. This is identical to SciPy's prewarp of both
  edges followed by `lp2bp`/`lp2bs` and the bilinear transform.
* `cheby1`: `rippleDb = rp`.
* `cheby2`: `stopBandDb = rs`. The stop-band edge is at the cutoff, as in SciPy.
* `ellip`: `rippleDb = rp`, `rolloff = 1 + ln((xi - 1)/5)`. The library defines
  `xi = ws/wp = 5 e^(rolloff-1) + 1`. The harness gets `xi` from SciPy's degree
  equation: `xi = 1/sqrt(ellipdeg(N, (10^(rp/10)-1)/(10^(rs/10)-1)))`.
* Bessel: the library uses raw reverse-Bessel roots, which is `norm='delay'`.
  Other norms are produced by rescaling the edges on the prewarped axis:
  * `k = theta_N(0)^(-1/N)` for `phase`, `1/w3dB` for `mag`, `1` for `delay`
  * low-pass: `W' = kW`
  * high-pass: `W' = W/k`
  * band-pass: `BW' = k·BW` with the geometric centre `W1·W2` kept
  * band-stop: `BW' = BW/k` with the geometric centre kept
* RBJ:
  * `LowPass`, `HighPass`: setup with `q` as given.
  * `BandPass2` (constant 0 dB peak) and `BandStop`: their `bandWidth`
    argument is used as Q.
  * Peaking (`BandShelf`): `BW = 2 asinh(1/(2Q)) sin(w0) / (ln2 · w0)`.
  * Shelves: `S = 1/((1/Q² - 2)/(A + 1/A) + 1)` with `A = 10^(gain/40)`.
* Odd-order high-pass sections come out with inverted polarity. The harness
  restores the sign with the library's own `Cascade::applyScale(-1)` and says
  so in `note`. The cause is listed under the bugs below.

**liquid-dsp** (`liquid_iirdes`)

* Low-pass and high-pass: `fc = f1/fs`, `f0 = 0`.
* Band-pass and band-stop: `fc = f2/fs` and
  `f0 = acos(cos(pi(f1+f2)/fs) / cos(pi(f2-f1)/fs)) / (2 pi)`. liquid-dsp uses
  the K = 1 Constantinides transform, and with this `f0` its prewarp
  `|cos 2pi fc - cos 2pi f0| / sin 2pi fc` equals `tan(pi(f2-f1)/fs)`.
* `cheby1`: `ap = rp`.
* `cheby2`: `as = 10 log10(10^(rs/10) - 1)`. liquid-dsp uses
  `epsilon = 10^(-as/20)`, whereas the textbook value is
  `1/sqrt(10^(rs/10)-1)`.
* `ellip`: `ap = rp`, `as = rs`. Its pass-band edge is at the cutoff, as in
  SciPy.
* Bessel: liquid-dsp divides the delay-normalised poles by the approximation
  `sqrtf((2N-1) ln 2)`. The harness rescales the edges as for DSPFilters, with
  `rho = k_target · sqrtf((2N-1) ln 2)`.
* RBJ biquads are not provided by liquid-dsp, so these cases return `ok: false`.

## Known third-party defects

* **DSPFilters `BiquadBase::setOnePole`.** It writes the first-order numerator
  as `-z0 + z^-1` instead of `1 - z0 z^-1`. For the zero at z = +1 of an
  odd-order high-pass this flips the polarity (H(Nyquist) = -1).
  `Cascade::setLayout` normalises only `|H|`, so the sign is never repaired.
* **DSPFilters `DirectFormI`/`DirectFormII` states.** These are the default
  `SimpleFilter` states. They add an alternating ±1e-8 "anti-denormal" signal to
  every sample, so the harness uses `TransposedDirectFormI`, which ignores it.
* **DSPFilters band transforms.** Debug builds assert exact floating-point
  equality of conjugate pairs, so the library is built with `-DNDEBUG`.
* **DSPFilters elliptic prototype.** Root polishing stops at 1e-6 (Bairstow
  tolerance in `findfact`). Agreement is about 2e-7 in pole position and
  1e-4 dB, not 1e-12.
* **liquid-dsp `cheby1_azpkf` / `cheby2_azpkf`.** They compute
  `tm = (t0 - 1/eps)^(1/n)` in float, which cancels catastrophically. The
  correct value is `1/tp`. For cheby2 this causes pass-band errors of up to
  about 3 dB at rs = 80 dB.
* **liquid-dsp `ellipkf` / `ellipdegf` / Landen recursions in float.** These
  lose precision when k1 = ep/es is small and when the selectivity k is close
  to 1. The design ends up with the wrong selectivity, with errors up to about
  1-3 dB.
* **liquid-dsp processing.** Float32 DF-II round-off can swamp the result for
  very low-frequency or wide-band band designs, even when the float sections
  are stable.
