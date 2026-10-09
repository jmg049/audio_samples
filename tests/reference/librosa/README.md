# librosa reference fixtures

`tests/librosa_reference/` checks audio_samples' librosa-compatible API against
golden fixtures in `tests/fixtures/librosa/`. This directory holds the
generator for those fixtures, as a separate `uv` project pinning librosa and
its numeric dependencies (`pyproject.toml`; exact pins because `*.lock` files
are not committed in this repository).

## Layout

- `generate_librosa_fixtures.py`: driver. Builds the shared deterministic test
  signals (`signals.json`, stored as 16-bit PCM codes so they are exact in
  every sample type), then runs every case module and writes one JSON file per
  module.
- `cases/<area>.py`: one module per area. Each defines `NAME` and
  `build(signals, sr) -> dict`.

## Regenerating

```sh
uv run --project tests/reference/librosa tests/reference/librosa/generate_librosa_fixtures.py
uv run --project tests/reference/librosa tests/reference/librosa/generate_librosa_fixtures.py --check   # verify only
```

`--check` compares with a relative tolerance of 1e-9: librosa's numba and BLAS
reductions are not bit-identical across machines.

The `pre-commit` hooks (`.pre-commit-config.yaml`, scripts in `../hooks/`) run
the reference tests whenever the librosa-compatible code or the fixtures change,
and run `--check` whenever the generator or the fixtures change.
