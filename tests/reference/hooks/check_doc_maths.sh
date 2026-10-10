#!/usr/bin/env bash
# Guard the maths in the documentation.
#
# * Doc comments in src/ keep their rustdoc escaping, and the guide chapters
#   stay plain LaTeX (check_doc_math.py, fast, always run).
# * If the guide or its rustdoc pipeline changed: the escaping done by build.rs
#   round-trips through pulldown-cmark (tests/guide_math_roundtrip.rs), and the
#   maths in the HTML that `cargo doc` produces equals the maths in the source
#   (check_guide_rustdoc_math.py).
#
# Usage (pre-commit passes the changed files):
#   tests/reference/hooks/check_doc_maths.sh [changed files...]
# With no arguments everything is checked.
set -euo pipefail
cd "$(dirname "$0")/../../.."
FEATURES="educational,full_no_plotting"

python3 tests/reference/hooks/check_doc_math.py

guide_changed=$([ "$#" -eq 0 ] && echo yes || echo no)
for f in "$@"; do
    case "$f" in
        documentation/guide/*|src/guide.rs|build.rs|build_support/*|tests/guide_math_roundtrip.rs|tests/reference/hooks/check_guide_rustdoc_math.py)
            guide_changed=yes ;;
    esac
done
if [ "$guide_changed" = yes ]; then
    cargo test --quiet --features "$FEATURES" --test guide_math_roundtrip
    python3 tests/reference/hooks/check_guide_rustdoc_math.py --build
fi
