#!/usr/bin/env python3
"""Runs the crate's examples for the gallery chapter and keeps what they produce.

Builds every example with the `full` and `educational` features, then runs each
one that does not need a person at the screen and writes to src/figures/gallery/:

* <example>.txt   its standard output;
* <plot>.html     the plots named by a `<!-- plot: ... -->` marker of the chapter
                  (examples/support/plots.rs writes them).

Fails if an example does not compile, exits with an error or takes too long, so
the chapter cannot show the output of a broken example. An example is run again
only when its binary is newer than its saved output.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Compiled but not run: `educational` opens a browser, `librosa_bench` is the
# benchmark harness and needs its signal files.
NOT_RUN = {"educational", "librosa_bench"}
TIMEOUT_S = 600


def main() -> int:
    book = Path(__file__).resolve().parent
    root = book.parent
    out = book / "src" / "figures" / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    target = root / "target" / "gallery"
    cargo = ["cargo", "build", "--quiet", "--examples", "--features", "full,educational", "--target-dir", str(target)]
    subprocess.run(cargo, cwd=root, check=True)

    import tomllib

    manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    names = [e["name"] for e in manifest["example"]]
    chapter = (book / "src" / "gallery.md").read_text(encoding="utf-8")
    wanted_plots = set(re.findall(r"<!--\s*plot:\s*([A-Za-z0-9_]+)", chapter))

    failed = []
    ran = 0
    for name in names:
        if name in NOT_RUN:
            continue
        binary = target / "debug" / "examples" / name
        saved = out / f"{name}.txt"
        if saved.exists() and saved.stat().st_mtime > binary.stat().st_mtime:
            continue
        ran += 1
        with tempfile.TemporaryDirectory() as work:
            plots = Path(work) / "plots"
            env = dict(os.environ, AUDIO_SAMPLES_EXAMPLE_PLOTS=str(plots))
            try:
                result = subprocess.run(
                    [str(binary)], cwd=work, env=env, capture_output=True, text=True, timeout=TIMEOUT_S
                )
            except subprocess.TimeoutExpired:
                failed.append(f"{name}: took longer than {TIMEOUT_S} s")
                continue
            if result.returncode != 0:
                failed.append(f"{name}: exit {result.returncode}\n{result.stderr[-2000:]}")
                continue
            saved.write_text(result.stdout, encoding="utf-8")
            if plots.is_dir():
                for html in plots.glob("*.html"):
                    if html.stem in wanted_plots:
                        shutil.copy(html, out / html.name)
    print(f"gallery: ran {ran} example(s), {len(names) - len(NOT_RUN)} have saved output")
    for problem in failed:
        print(f"gallery: {problem}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
