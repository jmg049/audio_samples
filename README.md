<div align="center">

# AudioSamples

## Fast, simple, and expressive audio in Rust

<img src="logo.png" title="AudioSamples Logo -- Ferrous' Mustachioed Cousin From East Berlin, Eisenhaltig" width="200"/>

[![Crates.io][crate-img]][crate] [![Docs.rs][docs-img]][docs] [![License: MIT][license-img]][license]
</div>

---

## Overview

Most audio libraries expose samples as raw numeric buffers. In Python,
audio is typically represented as a NumPy array whose `dtype` is
explicit, but whose meaning is not: sample rate
amplitude range, memory interleaving, and PCM versus floating-point
semantics are tracked externally, if at all. In Rust, the situation is
reversed but not resolved. Libraries provide fast and safe low-level
primitives, yet users are still responsible for managing raw buffers,
writing ad hoc conversion code, and manually preserving invariants
across crates.

AudioSamples is designed to close this gap by providing a strongly
typed audio representation that makes audio semantics explicit and
enforced by construction. Sample format, numeric domain, channel
structure, and layout are encoded in the type system, and all
operations preserve or explicitly update these invariants.

The result is an API that supports both exploratory workflows and
reliable system-level use, without requiring users to remember hidden
conventions or reimplement common audio logic.

AudioSamples is the core data and processing layer of the broader
audio related crates. It defines the canonical audio object and the
operations that act upon it.

Other crates that build on this foundation:

- `audio_samples_io` for decoding and encoding audio containers into typed
  audio objects
- `audio_samples_playback` for device-level output
- `audio_samples_python` for Python bindings, enabling AudioSamples to act as a
  type-safe backend for Python workflows
  
**NOTE** The crate is still a WIP so some features particularly plotting are not fully complete.

---

## Installation

```bash
cargo add audio_samples
```

See the [Features](#features) for more details.

---

## Quick Start

### Generating and mixing signals

This example generates a sine wave in a target sample format, converts
it to floating-point samples, and mixes it with a second signal.

```rust
use audio_samples::{
    sample_rate
    AudioProcessing, AudioTypeConversion, cosine_wave, operations::types::NormalizationMethod,
    sine_wave,
};
use std::time::Duration;

fn main() {
    let sample_rate = sample_rate!(44_100);
    let duration = Duration::from_secs_f64(1.0);
    let frequency = 440.0;
    let amplitude = 0.5;

    // Generate a sine wave with i16 output samples.
    let pcm_sine = sine_wave::<i16>(frequency, duration, sample_rate, amplitude);

    // Convert to floating-point representation
    let float_sine = pcm_sine.as_f32::<f32>();

    // Generate a second signal directly as floating-point samples
    let cosine = cosine_wave::<f32>(frequency / 2.0, duration, sample_rate, amplitude);

    // Mix the two signals
    let mixed = (float_sine + cosine).normalize(-1.0, 1.0, NormalizationMethod::MinMax);
}
```

---

### Spectral transforms and analysis

AudioSamples supports spectral and time–frequency transforms via the
`AudioTransforms` trait, enabled by the `spectral-analysis` feature.
These operations produce standard frequency-domain and
time–frequency representations used in audio analysis and research.

Enable the feature:

```bash
cargo add audio_samples --features spectral-analysis
```

#### Example: STFT, spectrogram, and MFCC computation

```rust
use audio_samples::{
    sample_rate,
    AudioProcessing, AudioTypeConversion, cosine_wave, operations::types::NormalizationMethod,
    sine_wave,
};
use std::time::Duration;

fn main() {
    let sample_rate = sample_rate!(44_100);
    let duration = Duration::from_secs_f64(1.0);
    let frequency = 440.0;
    let amplitude = 0.5;

    // Generate a sine wave with i16 output samples.
    let pcm_sine = sine_wave::<i16>(frequency, duration, sample_rate, amplitude);

    // Convert to floating-point representation
    let float_sine = pcm_sine.as_f32::<f32>();

    // Generate a second signal directly as floating-point samples
    let cosine = cosine_wave::<f32>(frequency / 2.0, duration, sample_rate, amplitude);

    // Mix the two signals
    let mixed = (float_sine + cosine).normalize(-1.0, 1.0, NormalizationMethod::MinMax);
}
```


## Creating AudioSamples

### Why does AudioSamples creation return a Result?

Creating ***valid*** AudioSamples instances is ``hard''. For a piece of audio data to be valid, it must satisfy a number of invariants:

- The sample buffer length must be consistent with the declared number of channels and frames.
- The sample rate must be a positive, non-zero value (ideally an integer).
- The number of channels must be a poisitive, non-zero value.
- The sample format must be supported.

Any of these can lead to failure at creation time, hence the use of `AudioSampleResult` to encapsulate potential errors.

However, the intended usage of audio does not make this any worse and is more explicit about potential failure modes. For example, when loading audio from a file ([audio_samples_io](https://github.com/jmg049/audio_samples_io)) there is still only one final failure point, either we could load the audio and represent it as valid AudioSamples, or we could not.

```rust
use audio_samples_io::{read, AudioIOResult};
use audio_samples::AudioSamples;

fn main() -> AudioIOResult<()> {
    let audio: AudioSamples<f32> = read("path/to/audio/file.wav")?;
    println!("{:#}", audio);
    Ok(())
}
```

The crate tries to provide methods to create valid AudioSamples, provided you the programmer/user can ensure the invariants are met. For example, by using a ``non_empty_slice::NonEmptyVec`` to ensure that the sample buffer is never empty and the ``sample_rate!`` macro to ensure a positive, non-zero sample rate at compile time.

```rust
use audio_samples::{AudioSamples, sample_rate};
use non_empty_slice::non_empty_vec;

let audio = AudioSamples::from_mono_vec(
    non_empty_vec![0.1, 0.2, 0.3],
    sample_rate!(44100),
);
```

but ultimately, checks must be performed to ensure validity, and so the return type is a `Result`.

## <a name="features">Features</a>

### Default features

- `statistics`
- `processing`
- `editing`
- `channels`

### Major functionality groups

- `fft`
- `resampling`
- `plotting`

### Transform and analysis features

- `spectral-analysis`
- `beat-detection` (requires `spectral-analysis`)

### Plotting sub-features

- `static-plots` (PNG/SVG output - [requires browser setup](#static-plots-setup))

### Performance features

- `simd` (nightly only)
- `fixed-size-audio`

### Utility features

- `formatting`
- `random-generation`

---

## Documentation

Full API documentation is available at
[https://docs.rs/audio_samples](https://docs.rs/audio_samples)

---

## Examples

A range of examples is included in the repository.

Additional demos include:

- [DTMF encoder and decoder](https://github.com/jmg049/dtmf-demo)
- Basic synthesis examples
- Audio inspection utilities

These additional demos are located in their own repos due to them depending on `audio_samples` and `audio_samples_io`

---

## Companion Crates

### [`audio_samples_io`](https://github.com/jmg049/audio_samples_io)

Rust crate providing audio file I/O utilities and helpers.
`audio_samples_io` is the IO extension of the [audio_samples](https://crates.io/crates/audio_samples) crate.

### [`audio_samples_playback`](https://github.com/jmg049/audio_playback)

Device-level playback built on AudioSamples.

### [`audio_samples_python`](https://github.com/jmg049/audio_python)

Python bindings exposing AudioSamples, AudioIO and AudioPlayback.

### [`html_view`](https://github.com/jmg049/HTMLView)

A lightweight, cross-platform HTML viewer for Rust.

`html_view` provides a minimal, ergonomic API for rendering HTML content in a native window, similar in spirit to `matplotlib.pyplot.show()` for visualisation rather than UI development.

### [`dtmf_tones`](https://github.com/jmg049/dtmf_tones)

A zero-heap, `no_std` friendly, **const-first** implementation of the standard DTMF (Dual-Tone Multi-Frequency) keypad used in telephony systems.  
This crate provides compile-time safe mappings between keypad keys and their canonical low/high frequencies, along with **runtime helpers** for practical audio processing.

### [`i24`](https://github.com/jmg049/i24)

i24 provides a 24-bit signed integer type for Rust, filling the gap between i16 and i32. This type is particularly useful in audio processing, certain embedded systems, and other scenarios where 24-bit precision is required but 32 bits would be excessive

---

## <a name="static-plots-setup">Static Plots Setup</a>

The `static-plots` feature enables PNG and SVG export of plots. This requires a browser and webdriver to be installed and configured via environment variables **before building**.

### Quick Setup (Copy-Paste)

Choose your platform and browser, then run **one** of these commands before building with `--features static-plots`:

#### Linux

**Chrome/Chromium (recommended):**
```bash
export BROWSER_PATH=$(command -v chromium || command -v google-chrome || command -v chromium-browser || echo "/usr/bin/chromium")
export WEBDRIVER_PATH=$(command -v chromedriver || echo "auto")
```

**Firefox:**
```bash
export BROWSER_PATH=$(command -v firefox || echo "/usr/bin/firefox")
export WEBDRIVER_PATH=$(command -v geckodriver || echo "/usr/local/bin/geckodriver")
```

#### macOS

**Chrome:**
```bash
export BROWSER_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
export WEBDRIVER_PATH=$(command -v chromedriver || echo "/usr/local/bin/chromedriver")
```

**Firefox:**
```bash
export BROWSER_PATH="/Applications/Firefox.app/Contents/MacOS/firefox"
export WEBDRIVER_PATH=$(command -v geckodriver || echo "/usr/local/bin/geckodriver")
```

#### Windows (PowerShell)

**Chrome:**
```powershell
$env:BROWSER_PATH="C:\Program Files\Google\Chrome\Application\chrome.exe"
$env:WEBDRIVER_PATH="C:\Program Files\chromedriver.exe"
```

**Firefox:**
```powershell
$env:BROWSER_PATH="C:\Program Files\Mozilla Firefox\firefox.exe"
$env:WEBDRIVER_PATH="C:\Program Files\geckodriver.exe"
```

### Build with Static Plots

After setting the environment variables:

```bash
cargo build --features "plotting static-plots transforms"
# or run examples
cargo run --example plotting_basic --features "plotting static-plots transforms"
```

### Permanent Setup

To avoid setting these every time, add to your shell configuration:

**Linux/macOS** (`~/.bashrc` or `~/.zshrc`):
```bash
# Chrome/Chromium
export BROWSER_PATH=/usr/bin/chromium
export WEBDRIVER_PATH=/usr/local/bin/chromedriver

# OR Firefox
export BROWSER_PATH=/usr/bin/firefox
export WEBDRIVER_PATH=/usr/local/bin/geckodriver
```

**Windows** (set system environment variables via System Properties → Environment Variables, or PowerShell profile):
```powershell
[Environment]::SetEnvironmentVariable("BROWSER_PATH", "C:\Program Files\Google\Chrome\Application\chrome.exe", "User")
[Environment]::SetEnvironmentVariable("WEBDRIVER_PATH", "C:\Program Files\chromedriver.exe", "User")
```

### Installation

If you don't have a browser/webdriver installed:

**Ubuntu/Debian:**
```bash
# Chrome/Chromium
sudo apt install chromium-browser chromium-chromedriver
# OR Firefox
sudo apt install firefox && cargo install geckodriver
```

**macOS:**
```bash
# Chrome
brew install --cask google-chrome && brew install chromedriver
# OR Firefox
brew install --cask firefox && brew install geckodriver
```

**Arch/Manjaro:**
```bash
# Chrome/Chromium
sudo pacman -S chromium chromedriver
# OR Firefox
sudo pacman -S firefox geckodriver
```

**Windows:**
- Download [Chrome](https://www.google.com/chrome/) or [Firefox](https://www.mozilla.org/firefox/)
- Download [chromedriver](https://chromedriver.chromium.org/downloads) or [geckodriver](https://github.com/mozilla/geckodriver/releases)
- Extract drivers to a location in your PATH or specify the full path

### Troubleshooting

**Build fails with `Failed to detect browser path`:**
1. Ensure the browser is installed and executable
2. Set the exact path: `export BROWSER_PATH=/path/to/your/browser`
3. Set the webdriver path: `export WEBDRIVER_PATH=/path/to/driver`
4. Verify: `echo $BROWSER_PATH && echo $WEBDRIVER_PATH` (Linux/macOS) or `echo $env:BROWSER_PATH; echo $env:WEBDRIVER_PATH` (Windows)

**Driver version mismatch:**
- Ensure your webdriver version matches your browser version
- Update both to the latest versions

---

## License

MIT License

## Citing

If you use AudioSamples in your research, please consider citing the crate:

```bibtex
@inproceedings{geraghty2026audio,
  author    = {Geraghty, Jack and Golpayegani, Fatemeh and Hines, Andrew},
  title     = {Audio Made Simple: A Modern Framework for Audio Processing},
  booktitle = {ACM Multimedia Systems Conference 2026 (MMSys '26)},
  year      = {2026},
  month     = apr,
  publisher = {ACM},
  address   = {Hong Kong, Hong Kong},
  doi       = {10.1145/3793853.3799811},
  note      = {Accepted for publication}
}
```

## Contributing

Contributions are welcome. Please submit a pull request and see
[CONTRIBUTING.md](CONTRIBUTING.md) for guidance.

[crate]: https://crates.io/crates/audio_samples  
[crate-img]: https://img.shields.io/crates/v/audio_samples?style=for-the-badge&color=009E73&label=crates.io

[docs]: https://docs.rs/audio_samples  
[docs-img]: https://img.shields.io/badge/docs.rs-online-009E73?style=for-the-badge&labelColor=gray

[license-img]: https://img.shields.io/crates/l/audio_samples?style=for-the-badge&label=license&labelColor=gray  
[license]: https://github.com/jmg049/audio_samples/blob/main/LICENSE
