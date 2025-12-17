Feature Roadmap: Missing librosa-like functionality and where to add it
=====================================================================

This document lists higher-level features that librosa provides but which are
missing or incomplete in `audio_samples`. For each feature it recommends:

- a short rationale,
- suggested API (config + trait/signature),
- the module/file where to implement it,
- feature-gating and dependency guidance,
- testing and example guidance.

Keep implementation style consistent with the crate: small config types in
`src/operations/types.rs`, trait methods in `src/operations/traits.rs` (or new
trait where appropriate), implementations in `src/operations/<feature>.rs`,
re-export in `src/operations/mod.rs`, and examples under `examples/`.

~~**1) Harmonic/Percussive Source Separation (HPSS)**~~ ✅ **COMPLETED**

- ~~Why: Widely used preprocessing step (improves pitch/onset/beat detection).~~
- ~~Suggested config: `HpssConfig<F>` fields: `win_size`, `hop_size`, `median_filter_harmonic`, `median_filter_percussive`, `mask_softness`.~~
- ~~Suggested API:~~
  - ~~Add to `AudioProcessing` or new trait `AudioDecomposition`:~~
    - ~~`fn hpss<F: RealFloat>(&self, cfg: &HpssConfig<F>) -> AudioSampleResult<(AudioSamples<'static, T>, AudioSamples<'static, T>)>`~~
      - ~~returns `(harmonic, percussive)`.~~
- ~~Implementation file: `src/operations/hpss.rs`.~~
- ~~Feature gate: `spectral-analysis` required; add optional feature `hpss` that depends on `spectral-analysis`.~~
- ~~Notes: Implementation uses STFT magnitude median filtering across time/frequency. Use ndarray views and fast-path contiguous buffers. Provide a `soft-mask` option.~~
- ~~Tests/examples: `examples/hpss.rs`, unit tests using synthetic sine+impulse mixtures.~~

~~**2) Chroma / Chromagram (STFT & CQT variants)**~~ ✅ **COMPLETED**

- ~~Why: Useful for pitch-class features, key estimation, chord recognition.~~
- ~~Suggested config: `ChromaConfig<F>`: `method` (STFT|CQT), `n_chroma` (12), `window_size`, `hop_size`, `sample_rate`, `norm` option.~~
- ~~Suggested API:~~
  - ~~Add to `AudioTransforms`:~~
    - ~~`fn chromagram<F: RealFloat>(&self, cfg: &ChromaConfig<F>) -> AudioSampleResult<ndarray::Array2<F>>`~~
      - ~~Returns shape `(n_chroma, n_frames)` as `Array2`.~~
- ~~Implementation: Enhanced existing chroma in `src/operations/transforms.rs`.~~
- ~~Feature gate: `chroma` feature (depends on `spectral-analysis`).~~
- ~~Notes: Implemented both STFT and CQT variants with helper methods. Maintains backward compatibility with existing `chroma()` method.~~
- ~~Tests/examples: `examples/chroma.rs` with comprehensive demonstrations including chord analysis.~~

**3) Spectral-Contrast, Spectral-Flatness, Tonnetz**

- Why: Additional spectral features for timbre and tonality analyses.
- Suggested config: `SpectralContrastConfig<F>`: `window_size`, `hop_size`, `bands`.
- Suggested API:
  - Extend `AudioStatistics` or create `AudioSpectralFeatures` trait:
    - `fn spectral_contrast<F: RealFloat>(&self, cfg: &SpectralContrastConfig<F>) -> AudioSampleResult<Array2<F>>`
    - `fn spectral_flatness<F: RealFloat>(&self, window: usize, hop: usize) -> AudioSampleResult<Vec<F>>`
- Implementation file: `src/operations/spectral_features.rs`.
- Feature gate: `spectral-analysis`.
- Tests/examples: `examples/spectral_features.rs` comparing contrast across synthetic signals.

**4) DTW (Dynamic Time Warping) & Sequence Alignment**

- Why: Template matching, alignment, and evaluation tasks (e.g., score-audio alignment).
- Suggested config: `DtwConfig<F>`: `distance` (euclidean/cosine), `global_constraint` options.
- Suggested API:
  - New trait `AudioSequenceAlignment` in `src/operations/traits.rs` (or under `operations/sequence.rs`):
    - `fn dtw<F: RealFloat>(&self, other: &Self, cfg: &DtwConfig<F>) -> AudioSampleResult<(ndarray::Array2<F>, Vec<(usize,usize)>)>`
      - Returns cost matrix plus optimal warping path (list of index pairs).
- Implementation file: `src/operations/dtw.rs`.
- Feature gate & deps: optional `dtw` feature. Consider using a small, permissively-licensed DTW crate (or implement efficient native Rust DP with optional parallel backend).
- Tests/examples: `examples/dtw.rs` aligning two simple sequences.

**5) Probabilistic Pitch Tracking (pYIN) / Voicing Probability**

- Why: More robust pitch tracking (per-frame confidence/probabilities) used in music and speech research.
- Suggested config: `PyinConfig<F>`: `fmin`, `fmax`, `frame_size`, `hop_size`, `thresholds`.
- Suggested API:
  - Add to `AudioPitchTracking` trait (new) or extend existing pitch trait:
    - `fn pyin<F: RealFloat>(&self, cfg: &PyinConfig<F>) -> AudioSampleResult<Vec<Option<F>>>` or return `(Vec<Option<F>>, Vec<F>)` where second vector is voicing probability.
- Implementation file: `src/operations/pyin.rs`.
- Feature gate: `spectral-analysis` + optional `pyin` feature. pYIN is non-trivial; provide a simplified initial implementation or wrap an existing algorithm reference implementation.
- Tests/examples: `examples/pyin.rs` on synthetic chirps with ground-truth.

**6) Beat-synchronous feature helpers & beat-synchronous aggregation**

- Why: Syncing features to beats is very common (music info retrieval pipelines).
- Suggested config: `BeatSyncConfig` with aggregation method (mean/median) and `hop_size`.
- Suggested API:
  - Extend beat-detection module (existing) or new trait `AudioBeatSync`:
    - `fn sync_to_beats<F: RealFloat>(&self, features: &ndarray::Array2<F>, beats: &[usize], cfg: &BeatSyncConfig) -> AudioSampleResult<ndarray::Array2<F>>`
- Implementation file: `src/operations/beat_sync.rs` or extend `operations/beats.rs`.
- Feature gate: `beat-detection` (already feature-gated); add `beat-sync` convenience feature if needed.
- Tests/examples: `examples/beat_sync.rs`.

**7) High-level time-stretch & pitch-shift wrappers**

- Why: User-friendly wrappers over phase-vocoder/PSOLA implementations.
- Suggested API:
  - Add to `AudioProcessing` or new `AudioEffects` trait:
    - `fn time_stretch<F: RealFloat>(&self, rate: F, cfg: &TimeStretchConfig<F>) -> AudioSampleResult<AudioSamples<'static, T>>`
    - `fn pitch_shift<F: RealFloat>(&self, semitones: F, cfg: &PitchShiftConfig<F>) -> AudioSampleResult<AudioSamples<'static, T>>`
- Implementation file: `src/operations/effects.rs` (uses `PitchShiftMethod` enum already in `types.rs`).
- Feature gate & deps: `fft` for phase vocoder; `simd`/`parallel-processing` optional for performance.
- Tests/examples: `examples/effects.rs` showing stretch/shift on a tone.

**8) Display / plotting convenience (librosa.display-like helpers)**

- Why: Make frequent plots quick (waveform, spectrogram, chromagram) for demos and debugging.
- Suggested API:
  - Add helper builders under `operations/plotting.rs` (already exists) or add a `display` module in `utils` that produces ready-to-render plot objects.
- Implementation file: extend `src/operations/plotting.rs` and `src/utils/display.rs`.
- Feature gate: `plotting` (already present).
- Tests/examples: `examples/display_*` scripts.

**9) Small utility wrappers & conversions**

- Why: librosa bundles many small helpers (mel/hz conversions, frame/time conversions, note<>midi conversions, amplitude/db conversions) that are convenient and widely used.
- Suggested additions: put in `src/utils/audio_math.rs` and re-export from crate root (consistent with `seconds_to_samples` and `samples_to_seconds`). Examples:
  - `hz_to_mel`, `mel_to_hz`, `amplitude_to_db`, `db_to_amplitude`, `frames_to_time`, `time_to_frames`, `note_to_midi`, `midi_to_hz`.
- Tests/examples: `examples/utils_demo.rs`.

API conventions and placement summary
-----------------------------------

- Config types: `src/operations/types.rs` alongside existing config structs. Follow style and docs.
- Traits: Extend `src/operations/traits.rs` for existing trait categories (e.g., `AudioTransforms`, `AudioStatistics`, `AudioProcessing`) or add focused traits for logically distinct functionality (`AudioDecomposition`, `AudioChroma`, `AudioSequenceAlignment`, `AudioPitchTracking`, `AudioEffects`).
- Implementations: New files under `src/operations/` (e.g., `hpss.rs`, `chroma.rs`, `dtw.rs`, `pyin.rs`, `beat_sync.rs`, `spectral_features.rs`, `effects.rs`). Keep code behind appropriate `#[cfg(feature = "...")]` gates.
- Feature flags: Add features in `Cargo.toml` as logical groups (e.g., `hpss`, `chroma`, `dtw`, `pyin`, `effects`) and ensure heavy dependencies are optional and gated.
- Re-exports: Add `pub mod <feature>` in `src/operations/mod.rs` with `#[cfg(feature = "...")]`, and re-export trait(s) for convenience at crate root if appropriate.
- Error handling: Return `AudioSampleResult<T>` and use `AudioSampleError::Feature` when a requested operation requires missing features (consistent with existing patterns).

Performance & testing notes
--------------------------

- Follow existing performance conventions: prefer `ndarray` views, contiguous fast-paths, in-place ops when possible, and return `LayoutError::NonContiguous` if a fast path is required.
- Expose optional `parallel-processing` acceleration hooks (use `rayon`) for expensive per-frame computations (HPSS, chroma over many frames, DTW for large matrices).
- Add unit tests in each new module under `#[cfg(test)]` and small example binaries under `examples/`.

Suggested implementation priority (quick wins first)
--------------------------------------------------

1. ~~HPSS (harmonic/percussive)~~ ✅ **COMPLETED** — ~~high impact, straightforward to implement using STFT and median filters.~~
2. **Utility wrappers (mel/hz, db conversions, frame/time helpers) — IN PROGRESS** — quick to add and improve ergonomics, foundational for other features.
3. ~~Chroma (STFT + CQT variants)~~ ✅ **COMPLETED** — ~~widely used in MIR, leverages existing transforms.~~
4. Spectral contrast & flatness — straightforward spectral features.
5. Beat-synchronous helpers — builds on existing beat detection.
6. DTW — useful for alignment; optional dependency.
7. pYIN — more complex; implement after basic pitch tools are solid.
8. Time-stretch/pitch-shift wrappers — if `PitchShiftMethod` internals exist, expose easy wrappers.
9. Display helpers and example datasets.
