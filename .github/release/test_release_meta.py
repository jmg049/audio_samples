"""Tests for release_meta.py. Run: python3 -m unittest discover -s .github/release"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import release_meta as rm  # noqa: E402

MANIFEST = """
[package]
name = "demo_crate"
version = "1.2.3"
description = "A demo."
license = "MIT"
keywords = ["audio"]
"""

CHANGELOG = """# Changelog

Intro text.

## [Unreleased]

### Added
- Not released.

## [1.2.30] - 2026-02-01

Not this one.

## [1.2.3-rc.1] - 2026-01-20

Not this either.

## [1.2.3] - 2026-01-15

### Fixed
- The fix.

```sh
## a comment in a fence, not a heading
cargo build
```

### Added
- A feature.

## 1.2.2

Plain heading.

## v1.2.1

V heading.

## [1.2.0](https://example.org/1.2.0) (2025-01-01)

Linked heading.

## 1.1.0 - 2024-12-01

Dated plain heading.

# 1.0.0

Top level heading.

## [0.9.0] - 2024-01-01

Last section.

[Unreleased]: https://example.org/compare
[1.2.3]: https://example.org/1.2.3
"""


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = rm.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class ChangelogSection(unittest.TestCase):
    def test_bracketed_with_date_keeps_subsections_and_fences(self):
        body = rm.changelog_section(CHANGELOG, "1.2.3")
        self.assertTrue(body.startswith("### Fixed\n- The fix."))
        self.assertIn("## a comment in a fence, not a heading", body)
        self.assertTrue(body.endswith("- A feature.\n"))
        self.assertNotIn("Plain heading", body)

    def test_exact_version_only(self):
        self.assertEqual(rm.changelog_section(CHANGELOG, "1.2.30"), "Not this one.\n")
        self.assertEqual(rm.changelog_section(CHANGELOG, "1.2.3-rc.1"), "Not this either.\n")

    def test_heading_styles(self):
        cases = {
            "1.2.2": "Plain heading.\n",
            "1.2.1": "V heading.\n",
            "1.2.0": "Linked heading.\n",
            "1.1.0": "Dated plain heading.\n",
            "1.0.0": "Top level heading.\n",
        }
        for version, expected in cases.items():
            with self.subTest(version):
                self.assertEqual(rm.changelog_section(CHANGELOG, version), expected)

    def test_v_prefix_on_the_request(self):
        self.assertEqual(rm.changelog_section(CHANGELOG, "v1.2.2"), "Plain heading.\n")

    def test_link_definitions_are_dropped(self):
        self.assertEqual(rm.changelog_section(CHANGELOG, "0.9.0"), "Last section.\n")

    def test_missing_version(self):
        with self.assertRaisesRegex(rm.ReleaseError, "no section"):
            rm.changelog_section(CHANGELOG, "9.9.9")

    def test_dots_are_literal(self):
        with self.assertRaisesRegex(rm.ReleaseError, "no section"):
            rm.changelog_section("## 1x2x3\n\ntext\n", "1.2.3")

    def test_empty_section(self):
        with self.assertRaisesRegex(rm.ReleaseError, "empty"):
            rm.changelog_section("## 1.0.0\n\n## 0.9.0\n\ntext\n", "1.0.0")

    def test_duplicate_section(self):
        with self.assertRaisesRegex(rm.ReleaseError, "2 sections"):
            rm.changelog_section("## 1.0.0\n\na\n\n## [1.0.0]\n\nb\n", "1.0.0")

    def test_unreleased_is_never_a_version(self):
        with self.assertRaisesRegex(rm.ReleaseError, "no section"):
            rm.changelog_section("## [Unreleased]\n\ntext\n", "1.0.0")

    def test_heading_shaped_line_in_a_fence_does_not_match(self):
        text = textwrap.dedent(
            """\
            ## 2.0.0

            ```
            ## 1.0.0
            ```
            """
        )
        with self.assertRaisesRegex(rm.ReleaseError, "no section"):
            rm.changelog_section(text, "1.0.0")


class WithFiles(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.manifest = self.dir / "Cargo.toml"
        self.manifest.write_text(MANIFEST)
        self.changelog = self.dir / "CHANGELOG.md"
        self.changelog.write_text(CHANGELOG)

    def cli(self, *argv: str) -> tuple[int, str, str]:
        return run("--manifest", str(self.manifest), "--changelog", str(self.changelog), *argv)


class Tag(WithFiles):
    def test_match(self):
        self.assertEqual(self.cli("check-tag", "v1.2.3")[0], 0)

    def test_mismatch(self):
        code, _, err = self.cli("check-tag", "v1.2.4")
        self.assertEqual(code, 2)
        self.assertIn("does not match", err)

    def test_branch_name_is_not_a_tag(self):
        code, _, err = self.cli("check-tag", "main")
        self.assertEqual(code, 2)
        self.assertIn("does not start with v", err)

    def test_tag_without_v_is_rejected(self):
        self.assertEqual(self.cli("check-tag", "1.2.3")[0], 2)

    def test_version_inherited_from_a_workspace_is_reported(self):
        self.manifest.write_text('[package]\nname = "a"\nversion.workspace = true\n')
        code, _, err = self.cli("check-tag", "v1.2.3")
        self.assertEqual(code, 2)
        self.assertIn("not a literal string", err)

    def test_version_and_name(self):
        self.assertEqual(self.cli("version")[1], "1.2.3\n")
        self.assertEqual(self.cli("name")[1], "demo_crate\n")

    def test_no_package_table(self):
        self.manifest.write_text("[workspace]\n")
        self.assertEqual(self.cli("version")[0], 2)

    def test_missing_manifest(self):
        self.manifest.unlink()
        self.assertEqual(self.cli("version")[0], 2)


class Notes(WithFiles):
    def test_prints_section(self):
        code, out, _ = self.cli("notes", "v1.2.2")
        self.assertEqual((code, out), (0, "Plain heading.\n"))

    def test_missing_section_fails(self):
        self.assertEqual(self.cli("notes", "3.0.0")[0], 2)

    def test_missing_changelog_fails(self):
        self.changelog.unlink()
        self.assertEqual(self.cli("notes", "1.2.3")[0], 2)


class Manifest(WithFiles):
    def problems(self, extra: str) -> list[str]:
        self.manifest.write_text(MANIFEST + extra)
        return rm.check_manifest(self.manifest)

    def test_clean(self):
        self.assertEqual(self.problems(""), [])
        self.assertEqual(self.cli("check-manifest")[0], 0)

    def test_path_patch(self):
        problems = self.problems('[patch.crates-io]\nfoo = { path = "../foo" }\n')
        self.assertEqual(len(problems), 1)
        self.assertIn("[patch.crates-io] foo", problems[0])
        self.assertEqual(self.cli("check-manifest")[0], 1)

    def test_versioned_patch_from_git_is_allowed(self):
        self.assertEqual(self.problems('[patch.crates-io]\nfoo = { git = "https://x/y" }\n'), [])

    def test_path_dependency_without_version(self):
        self.assertIn("no version", self.problems('[dependencies]\nfoo = { path = "foo" }\n')[0])

    def test_path_dependency_outside_the_repository(self):
        problems = self.problems('[dependencies]\nfoo = { version = "1", path = "../foo" }\n')
        self.assertIn("outside this repository", problems[0])

    def test_path_dependency_inside_the_repository_with_version(self):
        self.assertEqual(self.problems('[dependencies]\nfoo = { version = "1", path = "foo" }\n'), [])

    def test_dev_dependency_path_is_allowed(self):
        self.assertEqual(self.problems('[dev-dependencies]\nfoo = { path = "../foo" }\n'), [])

    def test_missing_description_and_license(self):
        self.manifest.write_text('[package]\nname = "a"\nversion = "1.0.0"\n')
        problems = rm.check_manifest(self.manifest)
        self.assertEqual(len(problems), 2)

    def test_license_file_counts_as_a_licence(self):
        self.manifest.write_text(
            '[package]\nname = "a"\nversion = "1.0.0"\ndescription = "d"\nlicense-file = "L"\n'
        )
        self.assertEqual(rm.check_manifest(self.manifest), [])

    def test_publish_false(self):
        self.manifest.write_text(MANIFEST.replace('name =', 'publish = false\nname ='))
        self.assertIn("publish is false", rm.check_manifest(self.manifest)[0])

    def test_keywords(self):
        self.manifest.write_text(MANIFEST.replace('["audio"]', '["a","b","c","d","e","f"]'))
        self.assertIn("6 keywords", rm.check_manifest(self.manifest)[0])
        self.manifest.write_text(MANIFEST.replace('["audio"]', '["not valid"]'))
        self.assertIn("not accepted", rm.check_manifest(self.manifest)[0])


class Published(WithFiles):
    INDEX = '{"name":"demo_crate","vers":"1.2.3"}\n{"name":"demo_crate","vers":"1.2.4"}\n'

    def test_index_path(self):
        self.assertEqual(rm.index_path("a"), "1/a")
        self.assertEqual(rm.index_path("ab"), "2/ab")
        self.assertEqual(rm.index_path("abc"), "3/a/abc")
        self.assertEqual(rm.index_path("audio_samples"), "au/di/audio_samples")
        self.assertEqual(rm.index_path("Audio_Samples"), "au/di/audio_samples")

    def test_states_through_the_cli(self):
        for text, code in ((self.INDEX, 0), ('{"vers":"1.2.30"}', 1), (None, 3)):
            with self.subTest(code), mock.patch.object(rm, "_fetch_index", lambda _c: text):
                self.assertEqual(self.cli("published")[0], code)


class Size(WithFiles):
    def test_limit(self):
        crate = self.dir / "a.crate"
        crate.write_bytes(b"0" * 1024)
        self.assertEqual(self.cli("check-size", str(crate))[0], 0)
        with crate.open("wb") as f:
            f.truncate(rm.MAX_CRATE_BYTES)
        self.assertEqual(self.cli("check-size", str(crate))[0], 0)
        with crate.open("wb") as f:
            f.truncate(rm.MAX_CRATE_BYTES + 1)
        code, _, err = self.cli("check-size", str(crate))
        self.assertEqual(code, 2)
        self.assertIn("crates.io accepts", err)


if __name__ == "__main__":
    unittest.main()
