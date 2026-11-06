//! Golden-reference tests against librosa.
//!
//! The fixtures in `tests/fixtures/librosa/` are produced by
//! `tests/reference/librosa/generate_librosa_fixtures.py`, which runs librosa
//! (version pinned in `tests/reference/librosa/pyproject.toml`) on the shared
//! deterministic signals in `signals.json`. Each submodule checks one area of
//! the librosa-compatible API against its fixture file. Every tolerance is
//! justified next to the assertion that uses it.
//!
//! Regenerate the fixtures with
//! `uv run --project tests/reference/librosa tests/reference/librosa/generate_librosa_fixtures.py`.

#![allow(dead_code)]

mod common;

// One module per fixture area (added as the areas land).
mod analysis_fixes;
mod array_ops;
mod constant_q;
mod effects;
mod factorization;
mod features;
mod inverse;
mod linalg;
mod mellin;
mod non_finite;
mod notation;
mod option_values;
mod options;
mod pitch;
mod pitch_filterbank;
mod resample;
mod rhythm;
mod segment;
mod sequence;
mod spectrum;
mod tonal;
