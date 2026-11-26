# 14. Psychoacoustics and codecs

Telephone speech, MP3, AAC and Opus all discard information. Telephone speech
(G.711) does so with simple, signal-independent methods: band-limiting and
companding (coarse steps for loud samples, fine steps for quiet ones).
Perceptual codecs (MP3, AAC, Opus) discard much more, because not everything
in the signal is audible: quiet sounds near loud ones are masked, and very
quiet sounds fall below the threshold of hearing. (Perceived loudness grows
roughly as a power of sound intensity, Stevens' power law, rather than in
proportion to it.) This chapter quantifies those effects, implements
G.711 μ-law companding from the standard, and measures the library's
perceptual codec.

## Concept

**dBFS is not dB SPL.** Sound pressure level is an absolute physical
quantity,

$$L_p = 20 \log_{10}(p / p_0) \ \text{dB SPL}, \quad p_0 = 20\ \mu\text{Pa}$$

where $p$ is the RMS sound pressure in pascals; 1 Pa is 93.98 dB SPL. Digital
audio has only **dBFS**, decibels relative to digital full scale
($20\log_{10}$ of an amplitude, full scale = 1). As in chapters 02 and 11, an
RMS level of 0 dBFS here is an RMS of 1, so a full-scale sine is $-3.01$ dBFS
(AES17 calls the same sine 0 dBFS); a level read from the largest sample is
labelled "dBFS peak". The link between digital and acoustic levels is the
whole playback chain (converter, amplifier, volume knob, transducer, distance),
so a file has no absolute level. Psychoacoustic models that compare a signal
with the threshold of hearing must *assume* a calibration; the library's
masking model assumes that 0 dB re one digital unit² corresponds to 0 dB SPL.

**Perceptual frequency scales.** The ear analyses sound with a bank of
overlapping auditory filters whose bandwidth grows with frequency. Zwicker's
*critical bandwidth* (the Bark scale) is roughly constant, about 100 Hz, below
about 500 Hz and roughly proportional to frequency above; the *equivalent
rectangular bandwidth* (ERB) of Glasberg and Moore grows steadily with
frequency even at low frequencies. Three scales describe this; this library
implements them as follows.

- **Mel (HTK form)**, in
  [`utils::audio_math::hz_to_mel`](crate::utils::audio_math::hz_to_mel):
  $m = 2595 \log_{10}(1 + f/700)$, with $f$ in Hz and $m$ in mel
  (1000 Hz ≈ 1000 mel). The crate's mel *spectrogram* uses the Slaney mel scale
  instead (linear below 1 kHz), see chapter 10.
- **Bark (Traunmüller 1990)**, in
  [`codecs::perceptual::bands::hz_to_bark`](crate::codecs::perceptual::bands::hz_to_bark):
  $z = 26.81 f / (1960 + f) - 0.53$ Bark, without Traunmüller's corrections
  below 2 and above 20.1 Bark. One Bark is one critical band.
- **ERB (Glasberg and Moore 1990).** The equivalent rectangular *bandwidth*
  of the auditory filter centred at $f$ is
  $\mathrm{ERB}(f) = 24.7 (4.37 f / 1000 + 1)$ Hz, which is what
  [`hz_to_erb`](crate::codecs::perceptual::bands::hz_to_erb) returns (a
  bandwidth in Hz, despite its name). The perceptual *coordinate* is the
  ERB-number scale $E(f) = 21.4 \log_{10}(1 + 0.00437 f)$ Cam, in
  [`hz_to_erb_number`](crate::codecs::perceptual::bands::hz_to_erb_number).
  [`BandLayout::erb`](crate::BandLayout::erb) spaces its bands evenly on it.

<!-- figure: ch14-scales -->

**Absolute threshold of hearing.** Terhardt's (1979) fit to the threshold in
quiet, in dB SPL with $f$ in Hz and $k = f / 1000$:

$$T_q(f) = 3.64\, k^{-0.8} - 6.5\, e^{-0.6 (k - 3.3)^2} + 10^{-3} k^4$$

It is high at low frequencies, has a minimum of about $-5$ dB SPL near
3.3 kHz (the ear-canal resonance), and rises again above that minimum, slowly
at first and steeply above about 10 kHz.

<!-- figure: ch14-threshold -->

**Simultaneous masking.** A loud component raises the threshold for other
components nearby in frequency. Measured on the Bark scale, the masked
threshold falls off steeply *below* the masker (about 25 dB/Bark) and more
gently *above* it (about 10 dB/Bark at moderate levels, shallower for loud
maskers): masking spreads upwards in frequency. A classic analytic form is the
spreading function of Schroeder, Atal and Hall (1979):

$$S(\Delta z) = 15.81 + 7.5(\Delta z + 0.474) - 17.5\sqrt{1 + (\Delta z + 0.474)^2} \ \text{dB}$$

where $\Delta z$ is the maskee's Bark position minus the masker's. Far from the
masker its slopes approach $7.5 - 17.5 = -10$ dB/Bark above and
$7.5 + 17.5 = +25$ dB/Bark below. Noise-like maskers mask more than tones: in
Johnston's (1988) model the masked threshold lies about 5.5 dB below a noise
masker but $14.5 + z$ dB below a tonal one, where $z$ is the masker's Bark
position (so about 15.5 to 38.5 dB across the 24 critical bands). A perceptual coder may add quantisation noise up
to the masked threshold without it being audible, so it allocates bits only
where the **signal-to-mask ratio** (signal level minus masked threshold, dB)
is positive.

**What the library's masking model does.** `PsychoacousticConfig` stores a
simplified form of the Johnston offsets, a **constant** 14.5 dB for tonal
maskers (Johnston's $14.5 + z$ evaluated at $z = 0$) and 5.5 dB for noise, and
two straight-line slopes. Its presets use **25 dB/Bark upward and 6 dB/Bark
downward**, which is the reverse of the psychoacoustic asymmetry above (the
field documentation says so); we demonstrate it below rather than rely on it.
Despite its name, the `mpeg1` preset does not reproduce the ISO/IEC 11172-3
(MPEG-1) psychoacoustic model's spreading function either.

**μ-law companding (ITU-T G.711).** Speech has a wide dynamic range, and
uniform quantisation gives quiet talkers a poor SNR (6.02 dB per bit, minus
every dB the signal is below full scale; chapter 02). Companding quantises
the *logarithm* of the amplitude instead. The continuous μ-law curve is

$$F(x) = \mathrm{sgn}(x)\, \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}, \quad \mu = 255$$

for $x \in [-1, 1]$. G.711 implements a segmented (piecewise-linear)
approximation: a **14-bit** linear sample becomes an **8-bit** code made of a
sign bit, a 3-bit segment number and a 4-bit mantissa. The magnitude plus a
bias of 33 is located in one of 8 segments (each twice as wide as the last);
16 equal steps in each segment give step sizes of 2, 4, ..., 256 (in 14-bit
units; the very first code of segment 0 covers magnitude 0 alone, a step of 1). The result is a roughly constant SNR of 35-40 dB over a wide range of
levels, with 8 bits instead of 14.

The library's [`mu_compress`](crate::AudioProcessing::mu_compress) implements
this curve without quantising (it maps 0 to 0),
and [`mu_expand`](crate::AudioProcessing::mu_expand) inverts it. This chapter
also writes the curve out by hand and checks both against it.

**Perceptual transform codecs.** All three codecs below split the signal into
frequency bands with a modified discrete cosine transform (MDCT), a lapped
transform with perfect reconstruction, but they use it differently:

- **MP3** (MPEG-1 Layer III) uses a *hybrid* filterbank: a 32-band polyphase
  filterbank followed by an MDCT in each band. A psychoacoustic model computes
  the masked threshold per band, and a bit allocator assigns bits where the
  signal-to-mask ratio is high.
- **AAC** uses a plain MDCT with 50 % overlap (long and short blocks) and the
  same principle: a psychoacoustic model sets the masked threshold per band,
  and quantisation noise is shaped to stay under it.
- **Opus's CELT layer** uses an MDCT with a *low-overlap* window (a fixed
  2.5 ms overlap, much less than 50 % of a 20 ms frame) to keep the delay low.
  It runs no explicit masking model: it codes each band's energy explicitly
  and the normalised shape of the band separately, so the energy in every band
  is preserved (the "constrained energy" idea), and its bit allocation is
  largely fixed or signalled.

The library's
[`PerceptualCodec`](crate::PerceptualCodec) follows this outline (MDCT,
Bark bands, the masking model above, bit allocation, scalar quantisation),
and [`codecs::opus`](crate::codecs::opus) is an Opus-*inspired* teaching codec
with SILK-like (LPC), CELT-like (MDCT) and hybrid modes. **Neither is
bitstream compatible with MP3 or Opus**: there is no entropy or range coder
and no standard bitstream; the encoded value is an in-memory structure.

## With the library

The scales and the threshold of hearing, as implemented:

```rust
use audio_samples::absolute_threshold_of_hearing;
use audio_samples::codecs::perceptual::bands::{
    bark_to_hz, erb_number_to_hz, hz_to_bark, hz_to_erb, hz_to_erb_number,
};
use audio_samples::utils::audio_math::{hz_to_mel, mel_to_hz};

// HTK mel: 1000 Hz is (almost exactly) 1000 mel, and the inverse round-trips.
assert!((hz_to_mel(1_000.0) - 1_000.0).abs() < 0.02);
assert!((mel_to_hz(hz_to_mel(4_000.0)) - 4_000.0).abs() < 1e-9);
// Traunmueller Bark: 1 kHz is 8.53 Bark, 4 kHz is 17.46 Bark (f32 arithmetic).
assert!((hz_to_bark(1_000.0) - 8.527).abs() < 1e-3);
assert!((hz_to_bark(4_000.0) - 17.46).abs() < 0.01);
assert!((bark_to_hz(hz_to_bark(1_000.0)) - 1_000.0).abs() < 0.05);
// Glasberg-Moore ERB *bandwidth*: 132.6 Hz at 1 kHz, 456.5 Hz at 4 kHz.
assert!((hz_to_erb(1_000.0) - 132.639).abs() < 1e-3);
assert!((hz_to_erb(4_000.0) - 456.456).abs() < 1e-2);
// Johnston's tonal offset 14.5 + z dB over the 24 critical bands (z = 1..24):
// 15.5 to 38.5 dB. The library's constant 14.5 dB is its z = 0 value.
let johnston_tonal = |z: f64| 14.5 + z;
assert_eq!((johnston_tonal(1.0), johnston_tonal(24.0)), (15.5, 38.5));
// The ERB-number (perceptual coordinate): 15.62 Cam at 1 kHz, and its inverse.
let erb_number = |f: f64| 21.4 * (1.0 + 0.00437 * f).log10();
assert!((erb_number(1_000.0) - 15.62).abs() < 0.01);
assert!((f64::from(hz_to_erb_number(1_000.0)) - erb_number(1_000.0)).abs() < 1e-4);
assert!((erb_number_to_hz(hz_to_erb_number(1_000.0)) - 1_000.0).abs() < 0.05);

// Threshold in quiet (dB SPL): 22.95 at 100 Hz, 3.37 at 1 kHz, -4.98 at 3.3 kHz.
assert!((absolute_threshold_of_hearing(100.0) - 22.95).abs() < 0.01);
assert!((absolute_threshold_of_hearing(1_000.0) - 3.37).abs() < 0.01);
assert!((absolute_threshold_of_hearing(3_300.0) + 4.98).abs() < 0.01);
```

The masking model's spreading, with the `mpeg1` preset: a tonal masker's
threshold is $14.5 + 25 \cdot 2 = 64.5$ dB down two Bark *above* it, but only
$14.5 + 6 \cdot 2 = 26.5$ dB down two Bark *below* it, so this preset lets
masking spread further downwards than upwards, opposite to the
psychoacoustic data.

```rust
use audio_samples::{MaskerType, PsychoacousticConfig, spreading_attenuation};
use std::num::NonZeroUsize;

let w = PsychoacousticConfig::uniform_weights(NonZeroUsize::new(24).unwrap());
let cfg = PsychoacousticConfig::mpeg1(w.as_non_empty_slice());
let above = spreading_attenuation(10.0, 12.0, MaskerType::Tonal, &cfg);
let below = spreading_attenuation(10.0, 8.0, MaskerType::Tonal, &cfg);
assert!((above - 64.5).abs() < 1e-4);
assert!((below - 26.5).abs() < 1e-4);
// Noise maskers use the smaller 5.5 dB offset: they mask more.
assert!((spreading_attenuation(10.0, 10.0, MaskerType::Noise, &cfg) - 5.5).abs() < 1e-6);
```

<!-- figure: ch14-masking -->

In the perceptual codec, a larger bit budget gives less quantisation noise.
Encode half a second of a 1 kHz
tone (24 Bark bands, 2048-sample MDCT) with two bit budgets and measure the
SNR of the decoded signal:

```rust
use audio_samples::codecs::{decode, encode};
use audio_samples::{BandLayout, PerceptualCodec, PsychoacousticConfig, sample_rate, sine_wave};
use spectrograms::WindowType;
use std::num::NonZeroUsize;
use std::time::Duration;

let x = sine_wave::<f32>(1_000.0, Duration::from_millis(500), sample_rate!(44100), 0.5);
let snr_db = |budget: u32| {
    let n_bands = NonZeroUsize::new(24).unwrap();
    let layout = BandLayout::bark(n_bands, 44_100.0, NonZeroUsize::new(1024).unwrap());
    let w = PsychoacousticConfig::uniform_weights(n_bands);
    let codec = PerceptualCodec::with_window_size(
        layout, PsychoacousticConfig::mpeg1(w.as_non_empty_slice()),
        WindowType::Hanning, budget, 1, NonZeroUsize::new(2048).unwrap(),
    );
    let y = decode::<PerceptualCodec, f32>(encode(&x, codec).unwrap()).unwrap();
    let (a, b) = (x.as_slice().unwrap(), y.as_slice().unwrap());
    let s: f64 = a.iter().map(|&v| f64::from(v).powi(2)).sum();
    let e: f64 = a.iter().zip(b).map(|(&p, &q)| f64::from(p - q).powi(2)).sum();
    10.0 * (s / e).log10()
};
let (low, high) = (snr_db(32_000), snr_db(128_000));
// About 42 dB and 51 dB (1 dB tolerance: the figures depend on the bit allocator).
assert!((low - 42.1).abs() < 1.0);
assert!((high - 50.6).abs() < 1.0);
assert!(high > low);
```

SNR is *not* what the codec optimises: a perceptual codec aims to keep the
noise below the masked threshold, which can sound transparent at an SNR far
below what a waveform coder would need.

<!-- figure: ch14-codec-snr -->

## How it works

**The threshold of hearing and the spreading function, from the formulas.**

```rust
use audio_samples::absolute_threshold_of_hearing;

let terhardt = |f: f64| {
    let k = f / 1000.0;
    3.64 * k.powf(-0.8) - 6.5 * (-0.6 * (k - 3.3).powi(2)).exp() + 1e-3 * k.powi(4)
};
// The library evaluates the same formula in f32 (and clamps to [-20, 60] dB SPL).
for f in (20..=16_000).step_by(10) {
    let lib = f64::from(absolute_threshold_of_hearing(f as f32));
    assert!((lib - terhardt(f as f64).clamp(-20.0, 60.0)).abs() < 1e-3);
}
// Minimum: about -5 dB SPL near 3.3 kHz.
let (f_min, t_min) = (20..=16_000)
    .map(|f| (f, terhardt(f as f64)))
    .min_by(|a, b| a.1.total_cmp(&b.1))
    .unwrap();
assert!((f_min as f64 - 3_320.0).abs() <= 20.0 && (t_min + 4.98).abs() < 0.01);

// Schroeder et al. (1979): shallow above the masker, steep below.
let s = |dz: f64| 15.81 + 7.5 * (dz + 0.474) - 17.5 * (1.0 + (dz + 0.474).powi(2)).sqrt();
let upper_slope = (s(10.0) - s(8.0)) / 2.0; // dB per Bark, far above the masker
let lower_slope = (s(-8.0) - s(-10.0)) / 2.0; // dB per Bark, far below (rising towards it)
assert!((upper_slope + 10.0).abs() < 0.3);
assert!((lower_slope - 25.0).abs() < 0.3);
```

**G.711 μ-law, from the standard.** Encode a 14-bit sample to an 8-bit code
and back; check the code against the continuous curve, and compare the SNR
with 8-bit linear PCM at several levels.

```rust
use std::f64::consts::PI;

fn ulaw_encode(x: i32) -> u8 {
    let (mag, sign) = if x < 0 { (-x, 0x80u8) } else { (x, 0x00u8) };
    let biased = (mag.min(8_159) + 33).min(8_191); // clip, bias, saturate
    let seg = (0..8).find(|&s| biased < (64 << s)).unwrap(); // segment 0..7
    let mant = ((biased >> (seg + 1)) & 0x0F) as u8; // 4-bit mantissa
    !(sign | ((seg as u8) << 4) | mant) // G.711 transmits the code inverted
}
fn ulaw_decode(code: u8) -> i32 {
    let c = !code;
    let (seg, mant) = (i32::from((c >> 4) & 7), i32::from(c & 0x0F));
    let mag = (((mant << 1) + 33) << seg) - 33; // middle of the step
    if c & 0x80 != 0 { -mag } else { mag }
}
assert_eq!(ulaw_encode(0), 0xFF); // positive zero
assert_eq!(ulaw_decode(ulaw_encode(8_159)), 8_031); // largest magnitude
// The error never exceeds half a step: 2^seg in segment seg (1, 2, ..., 128).
for x in -8_159i32..=8_159 {
    let seg = (0..8).find(|&s| (x.abs() + 33).min(8_191) < (64 << s)).unwrap();
    assert!((x - ulaw_decode(ulaw_encode(x))).abs() <= 1 << seg);
}
// The segment/mantissa index (0..127) stays within 2 codes of 127 F(x), mu = 255.
let f = |x: f64| (1.0 + 255.0 * x).ln() / 256f64.ln();
let worst = (1..=8_159)
    .map(|x| (f64::from(!ulaw_encode(x) & 0x7F) - 127.0 * f(f64::from(x) / 8_159.0)).abs())
    .fold(0.0, f64::max);
assert!(worst < 2.0);

// SNR of a sine at a given peak level for a quantiser q.
let snr = |level_db: f64, q: &dyn Fn(f64) -> f64| {
    let a = 10f64.powf(level_db / 20.0);
    let (mut s, mut e) = (0.0, 0.0);
    for n in 0..8_000 {
        let x = a * (2.0 * PI * 0.1234567 * n as f64).sin();
        s += x * x;
        e += (x - q(x)).powi(2);
    }
    10.0 * (s / e).log10()
};
let ulaw = |x: f64| f64::from(ulaw_decode(ulaw_encode((x * 8_159.0).round() as i32))) / 8_159.0;
let lin8 = |x: f64| (x * 127.0).round() / 127.0;
let levels = [0.0, -10.0, -20.0, -30.0, -40.0];
let su: Vec<f64> = levels.iter().map(|&l| snr(l, &ulaw)).collect();
let sl: Vec<f64> = levels.iter().map(|&l| snr(l, &lin8)).collect();
// mu-law stays between 34 and 40.5 dB over 40 dB of level ...
assert!(su.iter().all(|s| (34.0..40.5).contains(s)));
// ... while 8-bit linear starts at 6.02 * 8 + 1.76 = 49.9 dB and loses ~40 dB.
assert!((sl[0] - 49.9).abs() < 0.5);
assert!((sl[0] - sl[4] - 40.0).abs() < 1.5);
assert!(su[4] > sl[4] + 20.0);
```

<!-- figure: ch14-mulaw -->

The library's compressor equals the curve written out by hand (so
$F(0) = 0$ and $F(\pm 1) = \pm 1$), and its expander inverts it:

```rust
use audio_samples::{AudioProcessing, AudioSamples, sample_rate};
use non_empty_slice::NonEmptyVec;

let mu: f64 = 255.0;
let f = |x: f64| x.signum() * (1.0 + mu * x.abs()).ln() / (1.0 + mu).ln();
let xs = [-0.9, -0.3, -0.01, 0.0, 0.02, 0.5, 1.0];
let audio: AudioSamples<f64> =
    AudioSamples::from_mono_vec(NonEmptyVec::new(xs.to_vec()).unwrap(), sample_rate!(8000));
let compressed = audio.mu_compress(255.0).unwrap();
for (x, y) in xs.iter().zip(compressed.as_slice().unwrap()) {
    assert!((f(*x) - y).abs() < 1e-12);
}
assert_eq!(compressed.as_slice().unwrap()[3], 0.0); // F(0) = 0
assert!((compressed.as_slice().unwrap()[6] - 1.0).abs() < 1e-12); // F(1) = 1
let back = compressed.mu_expand(255.0).unwrap();
for (x, y) in xs.iter().zip(back.as_slice().unwrap()) {
    assert!((x - y).abs() < 1e-12);
}
```

## Ask the library to explain

There is no explainer for psychoacoustics or codecs (see
[`topics`](crate::educational::topics) for what exists). The closest is the
quantisation topic, which gives the uniform-quantiser SNR that μ-law improves
on for quiet signals:

```rust
use audio_samples::educational::topics;

let doc = topics::quantisation(8);
println!("{doc}");
// 6.02 * 8 + 1.76 = 49.9 dB for a full-scale sine with 8-bit uniform quantisation.
assert!((doc.quantity("Rule of thumb").unwrap().value - 49.92).abs() < 0.01);
assert!((doc.quantity("SQNR (full-scale sine)").unwrap().value - 49.92).abs() < 0.1);
```

## Exercises

1. **Bark distance.** Predict roughly how many Bark lie between 1 kHz and
   4 kHz, then compute it with `hz_to_bark`.
2. **Threshold at 50 Hz.** Is a 50 Hz tone at 30 dB SPL audible in quiet
   according to Terhardt's formula?
3. **Smallest and largest μ-law steps.** Using the segment structure, what is
   the ratio between the largest and the smallest G.711 step, in dB? Why does
   this keep the SNR roughly constant?
4. **Calibration.** A recording's speech peaks at $-12$ dBFS peak. The playback
   chain is calibrated so that a full-scale sine (RMS $-3.01$ dBFS) is
   94 dB SPL. What SPL do the speech peaks correspond to (as a sine of that
   peak)?

## Answers

1. $17.46 - 8.53 = 8.93$ Bark: about 9 critical bands for two octaves, while
   0 to 1 kHz is 8.5 Bark.

```rust
use audio_samples::codecs::perceptual::bands::hz_to_bark;

let d = hz_to_bark(4_000.0) - hz_to_bark(1_000.0);
assert!((d - 8.93).abs() < 0.01);
```

2. $T_q(50) = 3.64 \cdot 0.05^{-0.8} - \dots = 40.0$ dB SPL, so a 30 dB SPL
   tone at 50 Hz is about 10 dB below threshold: inaudible.

```rust
use audio_samples::absolute_threshold_of_hearing;

let t = f64::from(absolute_threshold_of_hearing(50.0));
assert!((t - 40.0).abs() < 0.1);
assert!(30.0 < t);
```

3. Steps go from 2 (segment 0) to 256 (segment 7) in 14-bit units, a ratio
   of 128 or 42.1 dB. The step grows in proportion to the signal amplitude,
   so the quantisation noise grows with the signal and the SNR stays roughly
   constant (until the signal falls into segment 0, where μ-law behaves like
   uniform quantisation).

```rust
let steps: Vec<i32> = (0..8).map(|seg| 2 << seg).collect();
assert_eq!((steps[0], steps[7]), (2, 256));
assert!((20.0 * (256.0f64 / 2.0).log10() - 42.14).abs() < 0.01);
```

4. A sine with peak $-12$ dBFS has RMS $-15.01$ dBFS, which is 12 dB below the
   calibration tone: $94 - 12 = 82$ dB SPL.

```rust
let calibration_spl = 94.0; // full-scale sine: peak 0 dBFS, RMS -3.01 dBFS
let peak_dbfs = -12.0;
let spl: f64 = calibration_spl + peak_dbfs; // same waveform shape, so offsets carry over
assert!((spl - 82.0).abs() < 1e-12);
```

## Common pitfalls

- **dBFS conventions.** Some tools quote the RMS of a full-scale sine as
  0 dBFS (AES17); this library uses $20\log_{10}(\text{RMS})$, which gives
  $-3.01$ dBFS.
- **"Mel" is ambiguous.** HTK (`2595 log10(1 + f/700)`) and Slaney
  (linear below 1 kHz, logarithmic above) differ; `audio_math` uses HTK, the
  mel spectrogram uses Slaney.
- **`hz_to_erb` returns a bandwidth**, not an ERB-number; uniformly spaced
  ERB *bandwidths* are uniformly spaced in Hz. Use `hz_to_erb_number` for the
  perceptual coordinate.
- **Masking direction.** Masking spreads *upwards* in frequency. The library's
  `PsychoacousticConfig` presets invert the slopes, and its `conservative` /
  `aggressive` preset names are also the reverse of their effect; set the
  fields yourself if you need a faithful model.
- **`mu_compress` is the continuous curve, not G.711.** It applies $F(x)$
  without quantising; the 8-bit segmented codes of G.711 are a different,
  piecewise-linear approximation. For integer μ-law codes use
  [`mu_law_encode`](crate::mu_law_encode) / [`mu_law_decode`](crate::mu_law_decode).
- **Not Opus.** `codecs::opus` is a teaching codec; it cannot read or write
  `.opus` files.

## References

- E. Terhardt, "Calculating virtual pitch", *Hearing Research* 1(2), 155-182, 1979. The threshold-in-quiet formula.
- H. Traunmüller, "Analytical expressions for the tonotopic sensory scale", *Journal of the Acoustical Society of America* 88(1), 97-100, 1990.
- B. R. Glasberg and B. C. J. Moore, "Derivation of auditory filter shapes from notched-noise data", *Hearing Research* 47(1-2), 103-138, 1990.
- E. Zwicker and H. Fastl, *Psychoacoustics: Facts and Models*, Springer. Critical bands and masking.
- M. R. Schroeder, B. S. Atal and J. L. Hall, "Optimizing digital speech coders by exploiting masking properties of the human ear", *Journal of the Acoustical Society of America* 66(6), 1647-1652, 1979.
- J. D. Johnston, "Transform coding of audio signals using perceptual noise criteria", *IEEE Journal on Selected Areas in Communications* 6(2), 314-323, 1988.
- S. S. Stevens, "On the psychophysical law", *Psychological Review* 64(3), 153-181, 1957.
- ITU-T Recommendation G.711, *Pulse code modulation (PCM) of voice frequencies*.
- ISO/IEC 11172-3, *Coding of moving pictures and associated audio for digital storage media at up to about 1.5 Mbit/s, Part 3: Audio* (MPEG-1 audio, including Layer III).
- J.-M. Valin, K. Vos and T. Terriberry, *Definition of the Opus Audio Codec*, RFC 6716, IETF, 2012; J.-M. Valin, G. Maxwell, T. B. Terriberry and K. Vos, "High-quality, low-delay music coding in the Opus codec", *135th Audio Engineering Society Convention*, 2013. CELT's low-overlap MDCT and energy coding.
- AES17, *AES standard method for digital audio engineering: measurement of digital audio equipment*, Audio Engineering Society (the definition of dBFS as the RMS of a full-scale sine).

## See also

- [`hz_to_mel`](crate::utils::audio_math::hz_to_mel),
  [`mel_to_hz`](crate::utils::audio_math::mel_to_hz),
  [`hz_to_bark`](crate::codecs::perceptual::bands::hz_to_bark),
  [`hz_to_erb`](crate::codecs::perceptual::bands::hz_to_erb),
  [`hz_to_erb_number`](crate::codecs::perceptual::bands::hz_to_erb_number)
- [`absolute_threshold_of_hearing`](crate::absolute_threshold_of_hearing),
  [`spreading_attenuation`](crate::spreading_attenuation),
  [`PsychoacousticConfig`](crate::PsychoacousticConfig),
  [`BandLayout`](crate::BandLayout)
- [`PerceptualCodec`](crate::PerceptualCodec), [`codecs::encode`](crate::codecs::encode),
  [`codecs::decode`](crate::codecs::decode), [`codecs::opus`](crate::codecs::opus)
- [`AudioProcessing::mu_expand`](crate::AudioProcessing::mu_expand)
- Previous chapter: [13. Onsets and tempo](13_onsets_tempo.md);
  quantisation basics: [02. Quantisation and dither](02_quantisation_dither.md);
  the mel scale in practice: [10. Cepstrum, mel and MFCC](10_cepstrum_mel_mfcc.md)
- Lab: `examples/tutorial_14_psychoacoustics_codecs.rs`
  (`cargo run --release --example tutorial_14_psychoacoustics_codecs --features educational,full_no_plotting`)
