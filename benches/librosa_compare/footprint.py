"""Install footprint: wheel sizes, fresh environments, dependency counts.

    footprint.py --python-repo ../audio_samples_python --out FILE.json [--work DIR]

Builds this checkout's wheel with maturin, then installs it and librosa into
fresh virtual environments with uv (network needed for numpy, librosa and its
dependencies). Byte-compiled as ``pip install`` does (``--compile-bytecode``).
Sizes are apparent bytes of each environment's site-packages; an empty
environment's are subtracted.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request


def run(*cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def tree_size(path):
    total = 0
    for d, _, names in os.walk(path):
        for n in names:
            p = os.path.join(d, n)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total


def site_packages(venv):
    return glob.glob(os.path.join(venv, "lib", "python*", "site-packages"))[0]


def make_env(work, name, py, *install):
    venv = os.path.join(work, name)
    run("uv", "venv", "-q", "--python", py, venv)
    env = dict(os.environ, VIRTUAL_ENV=venv)
    if install:
        run("uv", "pip", "install", "-q", "--compile-bytecode", *install, env=env)
    return venv, env


def packages(env):
    rows = json.loads(run("uv", "pip", "list", "--format", "json", env=env))
    return {r["name"]: r["version"] for r in rows}


def per_package(site):
    sizes = {}
    for entry in os.listdir(site):
        p = os.path.join(site, entry)
        sizes[entry] = tree_size(p) if os.path.isdir(p) else os.path.getsize(p)
    return dict(sorted(sizes.items(), key=lambda kv: -kv[1])[:12])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=f"{sys.version_info.major}.{sys.version_info.minor}")
    ap.add_argument("--work", default=None)
    args = ap.parse_args()

    work = args.work or tempfile.mkdtemp(prefix="footprint")
    wheels = os.path.join(work, "wheels")
    shutil.rmtree(wheels, ignore_errors=True)
    os.makedirs(wheels)
    repo = os.path.abspath(args.python_repo)
    run("maturin", "build", "--release", "-o", wheels, "-i", sys.executable, cwd=repo)
    wheel = glob.glob(os.path.join(wheels, "*.whl"))[0]

    pypi = json.load(urllib.request.urlopen("https://pypi.org/pypi/librosa/1.0.0/json", timeout=30))
    librosa_wheel = next(u for u in pypi["urls"] if u["packagetype"] == "bdist_wheel")

    empty, _ = make_env(work, "empty", args.python)
    base = tree_size(site_packages(empty))
    numpy_env, nenv = make_env(work, "numpy_only", args.python, "numpy")
    ours, oenv = make_env(work, "ours", args.python, wheel)
    theirs, lenv = make_env(work, "librosa", args.python, "librosa==1.0.0")

    result = {
        "python": args.python,
        "ours_wheel": {"file": os.path.basename(wheel), "bytes": os.path.getsize(wheel)},
        "librosa_wheel": {"file": librosa_wheel["filename"], "bytes": librosa_wheel["size"]},
        "empty_env_bytes": base,
    }
    for name, venv, env in (("numpy_only", numpy_env, nenv), ("ours", ours, oenv), ("librosa", theirs, lenv)):
        site = site_packages(venv)
        pk = packages(env)
        result[name] = {
            "site_packages_bytes": tree_size(site) - base,
            "package_count": len(pk),
            "packages": pk,
            "largest": per_package(site),
        }
    check = {}
    for name, venv, code in (
        ("ours", ours, "import audio_samples as a, numpy as np; a.AudioSamples.new_mono(np.zeros(4096, 'f'), 22050).mfcc_frames()"),
        ("librosa", theirs, "import librosa, numpy as np; librosa.feature.mfcc(y=np.zeros(4096, 'f'), sr=22050)"),
    ):
        run(os.path.join(venv, "bin", "python"), "-W", "ignore", "-c", code)
        check[name] = "imports and computes MFCCs"
    result["checked"] = check
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if x in ("bytes", "file", "site_packages_bytes", "package_count")}) for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()
