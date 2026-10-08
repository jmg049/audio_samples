# Filter reference fixtures

`tests/filter_reference.rs` checks audio_samples' filters against golden
fixtures in `tests/fixtures/filters/`. This directory holds the generator for
those fixtures.

## References

| Reference | Language | Used for |
|---|---|---|
| SciPy `scipy.signal` | C / Cython | Primary oracle: `butter`/`cheby1`/`cheby2`/`ellip`/`bessel` SOS designs, `sosfreqz`, `sosfilt`, `sosfiltfilt`, `lfilter` |
| FFmpeg `af_biquads` (`precision=f64`) | C | RBJ biquads: equalizer, low/high shelf, low/high/band-pass, notch |
| DSPFilters (Vinnie Falco) | C++ (double) | Independent Butterworth / Chebyshev I/II / elliptic / Bessel designs and RBJ biquads |
| liquid-dsp | C (float32) | Independent IIR designs, well-conditioned specifications only |
| RBJ Audio EQ Cookbook | formulas | Biquad coefficients, transcribed in the generator |

The generator checks the references against each other before it writes
anything. A fixture is only produced when the independent implementations
agree within their precision:

- DSPFilters ≤ 1e-6 dB (elliptic ≤ 2e-3 dB; its Bairstow root finder stops at
  1e-6)
- liquid-dsp ≤ 5e-3-5e-2 dB
- FFmpeg ≤ 1e-9 in the time domain

The measured agreement is stored in `iir_designs.json` → `cross_check_summary`.

## Regenerating

```sh
tests/reference/c/build.sh                                   # optional: C/C++ harness (see c/README.md)
uv run --project tests/reference tests/reference/generate_filter_fixtures.py
uv run --project tests/reference tests/reference/generate_filter_fixtures.py --check   # verify only
```

You need `uv` and `ffmpeg` on `PATH`. If the C/C++ harness isn't built, the
generator leaves out the DSPFilters / liquid-dsp cross references (and
`--check` ignores them). Pass `--require-harness` to treat a missing harness
as an error.

The `pre-commit` hooks (`.pre-commit-config.yaml`, scripts in `hooks/`) do two things:

- run the reference tests whenever filter code or fixtures change;
- run `--check` whenever the generator or fixtures change.
