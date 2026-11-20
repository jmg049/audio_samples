//! Golden-fixture tests for the analysis API.
//!
//! The fixtures in `tests/fixtures/analysis/` are produced by
//! `tests/reference/analysis/generate_analysis_fixtures.py` from the shared
//! deterministic signals in `signals.json` (see the generator's README for the
//! reference implementation and its pinned version). Each submodule checks one
//! area of the API against its fixture file. Every tolerance is
//! justified next to the assertion that uses it.
//!
//! Regenerate the fixtures with
//! `uv run --project tests/reference/analysis tests/reference/analysis/generate_analysis_fixtures.py`.

#![allow(dead_code)]

mod common;

// One module per fixture area.
mod array_ops;
mod autocorrelation_yin_notes;
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
