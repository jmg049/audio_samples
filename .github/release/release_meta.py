#!/usr/bin/env python3
"""Release checks shared by the workflows of the audio_samples repositories.

Standard library only; needs Python 3.11 or later for tomllib.

    release_meta.py version              print the version in Cargo.toml
    release_meta.py name                 print the package name in Cargo.toml
    release_meta.py check-tag v1.2.3     fail unless the tag is v<Cargo.toml version>
    release_meta.py check-manifest       fail on what crates.io or a clean CI checkout rejects
    release_meta.py notes 1.2.3          print that version's section of CHANGELOG.md
    release_meta.py published            exit 0 if the version is on crates.io, 1 if not,
                                         3 if the crate has never been published
    release_meta.py check-size a.crate   fail if the file is over the crates.io upload limit

`notes` accepts `1.2.3` or `v1.2.3`, and these headings at level 1 or 2:

    ## [1.2.3] - 2026-07-08     ## 1.2.3     ## v1.2.3     ## [1.2.3](url)     ## 1.2.3 (date)
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

INDEX_URL = "https://index.crates.io"
USER_AGENT = "audio_samples-release-check (https://github.com/jmg049/audio_samples)"
# The default per-crate limit on crates.io. `cargo publish --dry-run` does not check it.
MAX_CRATE_BYTES = 10 * 1024 * 1024
# The limits crates.io applies on upload. `cargo publish --dry-run` does not check them.
MAX_KEYWORDS = 5
KEYWORD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_+-]{0,19}$")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_NEVER_PUBLISHED = 3


class ReleaseError(Exception):
    """A check failed; the message says what to fix."""


def strip_v(version: str) -> str:
    return version[1:] if version.startswith("v") else version


def read_package(manifest: Path) -> dict:
    with manifest.open("rb") as f:
        data = tomllib.load(f)
    if "package" not in data:
        raise ReleaseError(f"{manifest}: no [package] table")
    return data


def package_version(manifest: Path) -> str:
    version = read_package(manifest)["package"].get("version")
    if not isinstance(version, str):
        raise ReleaseError(f"{manifest}: package.version is not a literal string")
    return version


def check_tag(tag: str, manifest: Path) -> None:
    if not tag.startswith("v"):
        raise ReleaseError(f"tag {tag!r} does not start with v")
    version = package_version(manifest)
    if tag != f"v{version}":
        raise ReleaseError(f"tag {tag} does not match {manifest} version {version}")


def check_manifest(manifest: Path) -> list[str]:
    """Problems a dry run does not report but a real publish or a clean checkout hits."""
    data = read_package(manifest)
    package = data["package"]
    problems = []
    if package.get("publish") is False:
        problems.append("package.publish is false")
    for field in ("description", "license"):
        if not package.get(field) and not (field == "license" and package.get("license-file")):
            problems.append(f"package.{field} is missing; crates.io rejects the upload")
    keywords = package.get("keywords", [])
    if len(keywords) > MAX_KEYWORDS:
        problems.append(f"{len(keywords)} keywords; crates.io allows {MAX_KEYWORDS}")
    for keyword in keywords:
        if not KEYWORD.match(keyword):
            problems.append(f"keyword {keyword!r} is not accepted by crates.io")
    for registry, patches in data.get("patch", {}).items():
        for name, spec in patches.items():
            if isinstance(spec, dict) and "path" in spec:
                problems.append(
                    f"[patch.{registry}] {name} uses path {spec['path']}: a clean checkout "
                    "cannot build it and the published crate would not use it"
                )
    for table in ("dependencies", "build-dependencies"):
        for name, spec in data.get(table, {}).items():
            if not (isinstance(spec, dict) and "path" in spec):
                continue
            if "version" not in spec:
                problems.append(f"[{table}] {name} has a path and no version")
            if spec["path"].startswith(".."):
                problems.append(f"[{table}] {name} uses path {spec['path']} outside this repository")
    return problems


def _heading_pattern(version: str) -> re.Pattern[str]:
    escaped = re.escape(version)
    # The version must end at `]`, whitespace or the end of the line, so 1.2.3
    # matches neither 1.2.30 nor 1.2.3-rc.1.
    return re.compile(rf"^#{{1,2}}\s+(?:\[v?{escaped}\](?:\([^)]*\))?|v?{escaped})(?:\s|$)")


def changelog_section(text: str, version: str) -> str:
    """The body of a version's section, without its heading."""
    version = strip_v(version)
    pattern = _heading_pattern(version)
    body: list[str] = []
    found = 0
    inside = False
    fence: str | None = None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker.group(1)[0]
            elif marker.group(1)[0] == fence:
                fence = None
        elif fence is None:
            if pattern.match(line):
                found += 1
                inside = True
                continue
            if re.match(r"^#{1,2}\s", line):
                inside = False
                continue
        if inside:
            body.append(line)
    if found == 0:
        raise ReleaseError(f"no section for version {version} in the changelog")
    if found > 1:
        raise ReleaseError(f"{found} sections for version {version} in the changelog")
    # Link reference definitions at the end of the file belong to no section.
    while body and (not body[-1].strip() or re.match(r"^\[[^\]]+\]:\s", body[-1])):
        body.pop()
    while body and not body[0].strip():
        body.pop(0)
    if not body:
        raise ReleaseError(f"the section for version {version} in the changelog is empty")
    return "\n".join(body) + "\n"


def index_path(crate: str) -> str:
    """The path of a crate's file in the crates.io sparse index."""
    name = crate.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def _fetch_index(crate: str) -> str | None:
    request = urllib.request.Request(
        f"{INDEX_URL}/{index_path(crate)}", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        # The sparse index answers 403 or 404 for a crate it does not have.
        if err.code in (403, 404):
            return None
        raise ReleaseError(f"crates.io index: HTTP {err.code}") from err
    except urllib.error.URLError as err:
        raise ReleaseError(f"crates.io index: {err.reason}") from err


def published_state(crate: str, version: str) -> int:
    text = _fetch_index(crate)
    if text is None:
        return EXIT_NEVER_PUBLISHED
    return EXIT_OK if f'"vers":"{version}"' in text else EXIT_FAIL


def check_size(path: Path) -> int:
    size = path.stat().st_size
    if size > MAX_CRATE_BYTES:
        raise ReleaseError(
            f"{path.name} is {size / 2**20:.1f} MiB; crates.io accepts {MAX_CRATE_BYTES / 2**20:.0f} MiB"
        )
    return size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("Cargo.toml"))
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version")
    sub.add_parser("name")
    sub.add_parser("check-manifest")
    sub.add_parser("published")
    sub.add_parser("check-tag").add_argument("tag")
    sub.add_parser("notes").add_argument("version")
    sub.add_parser("check-size").add_argument("crate_file", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.command == "version":
            print(package_version(args.manifest))
        elif args.command == "name":
            print(read_package(args.manifest)["package"]["name"])
        elif args.command == "check-tag":
            check_tag(args.tag, args.manifest)
            print(f"{args.tag} matches {args.manifest}")
        elif args.command == "check-manifest":
            problems = check_manifest(args.manifest)
            for problem in problems:
                print(f"error: {problem}", file=sys.stderr)
            if problems:
                return EXIT_FAIL
            print(f"{args.manifest} can be published")
        elif args.command == "notes":
            sys.stdout.write(changelog_section(args.changelog.read_text("utf-8"), args.version))
        elif args.command == "published":
            crate = read_package(args.manifest)["package"]["name"]
            version = package_version(args.manifest)
            state = published_state(crate, version)
            message = {
                EXIT_OK: f"{crate} {version} is on crates.io",
                EXIT_FAIL: f"{crate} {version} is not on crates.io",
                EXIT_NEVER_PUBLISHED: f"{crate} has never been published to crates.io",
            }[state]
            print(message)
            return state
        elif args.command == "check-size":
            size = check_size(args.crate_file)
            print(f"{args.crate_file.name}: {size / 2**20:.1f} MiB")
    except (ReleaseError, OSError, tomllib.TOMLDecodeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
