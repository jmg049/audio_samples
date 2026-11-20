# Analysis reference fixtures

`tests/analysis_reference/` checks the analysis API (spectra, features,
rhythm, pitch, effects, resampling and their helpers) against golden fixtures
in `tests/fixtures/analysis/`. The reference values are computed with librosa
1.0. This directory holds the generator for those fixtures, as a separate `uv`
project pinning librosa and its numeric dependencies (`pyproject.toml`; exact pins because `*.lock` files
are not committed in this repository).

## Layout

- `generate_analysis_fixtures.py`: driver. Builds the shared deterministic test
  signals (`signals.json`, stored as 16-bit PCM codes so they are exact in
  every sample type), then runs every case module and writes one JSON file per
  module.
- `cases/<area>.py`: one module per area. Each defines `NAME` and
  `build(signals, sr) -> dict`.

## Regenerating

```sh
uv run --project tests/reference/analysis tests/reference/analysis/generate_analysis_fixtures.py
uv run --project tests/reference/analysis tests/reference/analysis/generate_analysis_fixtures.py --check   # verify only
```

`--check` compares with a relative tolerance of 1e-9 because numba and BLAS
reductions round differently on different machines. The generator sets every thread
count to 1 before NumPy loads, because the rounding of a float32 BLAS product
depends on how the work is split between threads.

The `pre-commit` hooks (`.pre-commit-config.yaml`, scripts in `../hooks/`) run
the reference tests whenever the analysis code or the fixtures change,
and run `--check` whenever the generator or the fixtures change.
