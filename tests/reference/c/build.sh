#!/usr/bin/env bash
# Build the independent IIR filter ground-truth harness `filter_ref`.
#
# Third-party sources are fetched into ./third_party (pinned below) and
# everything is built into ./build. Re-running is idempotent: existing
# checkouts are reused (and re-pinned), existing builds are re-used by
# make/cmake.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$HERE/third_party"
BUILD="$HERE/build"

# --- pinned versions ---------------------------------------------------------
DSPFILTERS_URL="https://github.com/vinniefalco/DSPFilters.git"
DSPFILTERS_COMMIT="acc49170e79a94fcb9c04b8a2116e9f8dffd1c7d"   # master HEAD (2022), repo is unmaintained

LIQUID_URL="https://github.com/jgaeddert/liquid-dsp.git"
LIQUID_TAG="v1.8.3"
LIQUID_COMMIT="10041f70cebbe3b97887e75bb41e48b73dda1b23"       # = tag v1.8.3

JOBS="$(nproc 2>/dev/null || echo 4)"

mkdir -p "$TP" "$BUILD"

fetch() { # url dir commit
  local url="$1" dir="$2" commit="$3"
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet "$url" "$dir"
  fi
  if [ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]; then
    git -C "$dir" fetch --quiet --tags origin || true
    git -C "$dir" checkout --quiet "$commit"
  fi
}

fetch "$DSPFILTERS_URL" "$TP/DSPFilters" "$DSPFILTERS_COMMIT"
fetch "$LIQUID_URL"     "$TP/liquid-dsp" "$LIQUID_COMMIT"

# --- liquid-dsp: static library, no SIMD / FFTW (deterministic scalar code) ---
LIQ_BUILD="$BUILD/liquid-dsp"
if [ ! -f "$LIQ_BUILD/libliquid.a" ]; then
  cmake -S "$TP/liquid-dsp" -B "$LIQ_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON \
    -DBUILD_EXAMPLES=OFF -DBUILD_AUTOTESTS=OFF -DBUILD_BENCHMARKS=OFF \
    -DBUILD_SANDBOX=OFF -DBUILD_DOC=OFF \
    -DENABLE_SIMD=OFF -DFIND_SIMD=OFF -DFIND_FFTW=OFF \
    -DENABLE_COLOR=OFF -DENABLE_TIMESTAMPS=OFF > "$BUILD/liquid-cmake.log"
  cmake --build "$LIQ_BUILD" -j "$JOBS" > "$BUILD/liquid-build.log"
fi
LIQ_LIB="$(find "$LIQ_BUILD" -name 'libliquid.a' | head -n1)"
LIQ_INC="$LIQ_BUILD"   # contains liquid/liquid.h (configured copy)

# --- DSPFilters: compile the library sources directly ------------------------
# -DNDEBUG is required: debug builds contain exact floating-point equality
# asserts on conjugate pairs in the band transforms that can abort.
DSP_SRC="$TP/DSPFilters/shared/DSPFilters"
DSP_OBJ="$BUILD/dspfilters"
mkdir -p "$DSP_OBJ"
for src in "$DSP_SRC"/source/*.cpp; do
  obj="$DSP_OBJ/$(basename "${src%.cpp}").o"
  if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
    g++ -std=c++17 -O2 -DNDEBUG -fno-fast-math -w -I"$DSP_SRC/include" -c "$src" -o "$obj"
  fi
done
ar rcs "$DSP_OBJ/libDSPFilters.a" "$DSP_OBJ"/*.o

# --- harness ------------------------------------------------------------------
g++ -std=c++17 -O2 -DNDEBUG -fno-fast-math -Wall -Wextra \
  -isystem "$DSP_SRC/include" -isystem "$LIQ_INC" \
  "$HERE/filter_ref.cpp" \
  "$DSP_OBJ/libDSPFilters.a" "$LIQ_LIB" -lm -lpthread \
  -o "$BUILD/filter_ref"

echo "built $BUILD/filter_ref"
