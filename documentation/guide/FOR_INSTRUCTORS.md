# For instructors

This page is for lecturers and teaching assistants who are considering
`audio_samples` for a speech and audio processing course. It says what the
library offers a course, how the teaching material is organised and checked,
how to verify results against SciPy and librosa, and where the library's
conventions differ from those references.

## What the library gives a course

- **One object that carries its own metadata.** An `AudioSamples<T>` holds the
  samples, the sample rate and the channel layout together, so students cannot
  silently mix sample rates or forget which axis is time.
- **Rust and Python with the same core.** The Python package
  (`audio_samples` on PyPI) is a thin binding over this crate, so a result
  computed in a notebook and in a Rust lab comes from the same code.
- **Errors that teach.** Invalid parameters produce diagnostics that name the
  parameter, the value and the valid range (with the `fancy` feature, as
  coloured `miette` reports). In Python they are raised as
  `audio_samples.exceptions` classes carrying `.code` and `.help`.
- **A library that explains itself** (`educational` feature). A supported
  configuration can describe what it will do before it runs: the formula the
  code evaluates, every symbol with its unit, the derived resolution and
  latency numbers, and warnings about suspicious settings.

  ```rust
  use audio_samples::educational::{Explain, ExplainContext};
  use spectrograms::{StftParams, WindowType};
  use std::num::NonZeroUsize;

  let stft = StftParams::new(
      NonZeroUsize::new(256).unwrap(),
      NonZeroUsize::new(128).unwrap(),
      WindowType::Hanning,
      true,
  )
  .unwrap();
  let ctx = ExplainContext::new()
      .with_sample_rate(16_000.0)
      .with_f0_range(80.0, 300.0);
  let doc = stft.explain_with(&ctx);
  // 256 samples at 16 kHz last 16 ms, shorter than two periods of 80 Hz
  // (25 ms), so the explanation carries a warning.
  assert!(!doc.warnings.is_empty());
  println!("{doc}"); // terminal text; doc.to_markdown() for Jupyter or an LMS page
  ```

  The same is available in Python as `audio.explain(params)` and the
  `audio_samples.explain` module, rendered as Markdown with maths in Jupyter.

## How the material is organised

The [guide](crate::guide) has one chapter per topic of a typical module, from
sampling and quantisation to codecs. Every chapter follows the same pattern:

1. **Concept**: the theory, with every symbol and unit defined.
2. **With the library**: the API that does it.
3. **Under the hood**: the same result re-derived from first principles in a
   few lines of plain code, checked against the library.
4. **Ask the library to explain**: the explain API for that topic.
5. **Exercises**, with an answer key that is itself executable.

Each chapter has a matching Rust lab, `examples/tutorial_NN_<topic>.rs`, and a
matching Python notebook in the Python documentation
(`docs/source/tutorials/NN_<topic>.ipynb`).

### Everything is checked

Every number stated in the material is computed and asserted in code:

- every guide code block is a doctest
  (`cargo test --doc --features educational,full_no_plotting -- guide`);
- every lab `assert!`s what it prints
  (`cargo run --release --example tutorial_NN_<topic> --features educational,full_no_plotting`);
- every notebook ends its exercises with asserted solution cells and is
  executed headlessly by `pytest --nbmake docs/source/tutorials`;
- a `pre-commit` hook runs the guide doctests, the explain API tests and the
  changed labs whenever teaching material changes.

If you adapt a chapter, keep this property: change the number in the code, and
the test tells you whether the prose is now wrong.

## Suggested labs

Each lab is sized for a two-hour session. The Rust lab and the Python notebook
cover the same ground, so pick the language that suits the cohort.

| Lab | Topic | Students should leave able to |
|---|---|---|
| 01 | Sampling and aliasing | predict where a tone above Nyquist lands; explain why decimation needs a low-pass filter |
| 02 | Quantisation and dither | derive and measure SQNR ≈ 6.02 b + 1.76 dB; show what TPDF dither does to quiet signals |
| 03 | DFT, windows and leakage | map bins to hertz; choose a window from its main-lobe width and side-lobe level |
| 04 | STFT and spectrograms | trade time resolution against frequency resolution; compute frame counts; read a mel spectrogram |
| 05 | Convolution and FIR filters | relate full and valid convolution lengths; design a windowed-sinc low-pass; measure group delay |
| 06 | IIR filters | design a Butterworth filter, find its poles and zeros, judge stability, measure −3 dB at the cutoff |
| 07 | Resampling | resample without aliasing; measure the alias rejection of each quality tier |
| 08 | LPC and formants | fit an all-pole model with Levinson–Durbin and read formants from its roots |
| 09 | Pitch | estimate f0 with autocorrelation and YIN; explain octave errors |
| 10 | Cepstrum, mel and MFCC | compute a cepstrum and an MFCC pipeline step by step |
| 11 | Energy and VAD | build an energy and zero-crossing voice activity detector with hangover |
| 12 | Dynamics | draw a compressor's static curve and measure attack and release time constants |
| 13 | Onsets and tempo | compute spectral flux, pick peaks and estimate tempo |
| 14 | Psychoacoustics and codecs | convert between perceptual scales; implement μ-law companding |

Assessment ideas that use the explain API: give students a configuration and
ask them to predict the explanation's numbers (Δf, Δt, frame count, gain at the
cutoff) before running it; or give them a deliberately poor configuration and
ask them to remove every warning.

## Verifying results against SciPy and librosa

The IIR filters are already checked against SciPy, FFmpeg, DSPFilters and
liquid-dsp golden fixtures (`tests/filter_reference.rs`, generator in
`tests/reference/`). For other topics, compare with these conventions in mind:

| Topic | `audio_samples` | Reference default | To compare like with like |
|---|---|---|---|
| STFT window | symmetric Hann (`WindowType::Hanning`) | librosa and SciPy: periodic Hann | pass `scipy.signal.windows.hann(N, sym=True)` as librosa's `window=` |
| STFT padding | `centre = true` pads ⌊N/2⌋ zeros at each end | librosa ≥ 0.10: `pad_mode="constant"` | same |
| `rfft` | magnitudes \|X\[k\]\| | numpy: complex spectrum | `np.abs(np.fft.rfft(x))` |
| Mel spectrogram | Slaney mel scale, no filter normalisation, applied to power | librosa: `htk=False`, `norm="slaney"` | `librosa.filters.mel(..., norm=None)` |
| Mel helpers in `utils::audio_math` | HTK formula | librosa default: Slaney | `librosa.hz_to_mel(f, htk=True)` |
| dB | 10·log10(max(P, floor)) re 1: an absolute floor | librosa `power_to_db`: `top_db=80` relative to the maximum | `top_db=None` and the same `amin` |
| MFCC | unnormalised DCT-II, lifter 22 | librosa: DCT-II `norm="ortho"`, lifter 0 | rescale by the DCT normalisation factors; set the lifter |
| Zero-crossing rate | crossings per second over the whole signal | librosa: fraction of samples per frame | divide by the sample rate; frame the signal |
| Welch PSD | SciPy `welch(scaling="density")` scaling, symmetric Hann, no detrending | SciPy: periodic Hann, constant detrend | `window=hann(N, sym=True)`, `detrend=False` |
| Integer to float | asymmetric: −1.0 ↔ −32768, +1.0 ↔ 32767 | soundfile: divide by 32768 | differs by at most one least significant bit |
| YIN | de Cheveigné and Kawahara steps 1–5 with parabolic interpolation; `detect_pitch_yin` is clip-wide | `librosa.yin`: framewise | compare with `track_pitch`, framewise |

Known limitations worth telling students about are documented in each module.
For example, in this release `mu_compress` deviates from the standard μ-law
curve, the external `spectrograms` crate's Kaiser window is inaccurate for
β > 3.75, and the gammatone filterbank's default centre spacing is linear in Hz.

## Setting up a course environment

- Rust: `cargo add audio_samples --features educational,full_no_plotting` (add
  `plotting` for HTML plots). Reading and writing WAV or FLAC files is in the
  companion crate `audio_samples_io`.
- Python: `pip install audio_samples`, then
  `pip install matplotlib scipy librosa jupyter` for the notebooks.
