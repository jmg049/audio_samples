"""Fixture case modules, one per area.

Each module defines ``NAME`` (the fixture file stem) and
``build(signals: dict[str, numpy.ndarray], sr: int) -> dict`` returning a
JSON-serialisable dict. See ``generate_librosa_fixtures.py``.
"""
