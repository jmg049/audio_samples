#!/usr/bin/env bash
# host_setup.sh — one-time bench-host preparation.
#
# Idempotent. Safe to re-run any number of times. Must run as root (sudo).
# Designed for Hetzner Ubuntu 26.04 but the steps apply to most Debian-family
# hosts.
#
# What it does:
#   1. /etc/sysctl.d/99-bench-perf.conf  → perf_event_paranoid = 1
#      (persists across reboots; PMU benches need this)
#   2. CPU governor → performance (disables frequency scaling for stable
#      wall-clock numbers).
#   3. Apt-installs linux-tools-common and the running-kernel tools package
#      if cpupower / perf aren't already present.
#
# Things it does NOT touch (these are host-policy decisions):
#   - SMT / hyperthreading            (leave to operator)
#   - CPU mitigations / turbo boost   (workload-dependent)
#   - NUMA balancing                  (off on single-socket hosts already)
#   - irqbalance                      (default daemon is fine)
#
# Usage:
#   sudo bash bench_suite/scripts/host_setup.sh

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: host_setup.sh must run as root (re-invoke with sudo)" >&2
    exit 1
fi

echo "============================================================"
echo " audio_samples bench_suite — host setup"
echo "------------------------------------------------------------"
echo " kernel:    $(uname -srm)"
echo " os:        $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || echo unknown)"
echo " cpu:       $(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo)"
echo " cpu count: $(nproc)"
echo "============================================================"

# --------------------------------------------------------------------------
# 1) perf_event_paranoid — required for PMU benches without CAP_PERFMON.
# --------------------------------------------------------------------------

PEP_FILE=/etc/sysctl.d/99-bench-perf.conf
PEP_LINE="kernel.perf_event_paranoid = 1"

echo
echo "[1/3] perf_event_paranoid"
CURRENT_PEP=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "?")
echo "  current runtime value: $CURRENT_PEP"

if [[ -f "$PEP_FILE" ]] && grep -qF "$PEP_LINE" "$PEP_FILE"; then
    echo "  $PEP_FILE already contains the setting (no change)."
else
    echo "  writing $PEP_FILE …"
    cat > "$PEP_FILE" <<EOF
# Written by bench_suite/scripts/host_setup.sh
# Allow non-root perf_event_open() so the PMU bench targets can read
# instructions / cycles / cache-misses / branch-misses without CAP_PERFMON.
$PEP_LINE
EOF
fi

sysctl --system > /dev/null
NEW_PEP=$(cat /proc/sys/kernel/perf_event_paranoid)
echo "  new runtime value:     $NEW_PEP"
if [[ "$NEW_PEP" -gt 1 ]]; then
    echo "  WARN: value didn't drop to ≤ 1 — another sysctl drop-in may be overriding."
fi

# --------------------------------------------------------------------------
# 2) Install cpupower / perf if missing.
# --------------------------------------------------------------------------

echo
echo "[2/3] cpupower / perf tools"
NEED_APT=0
command -v cpupower >/dev/null 2>&1 || NEED_APT=1
command -v perf     >/dev/null 2>&1 || NEED_APT=1

if [[ $NEED_APT -eq 1 ]]; then
    echo "  installing linux-tools-common + linux-tools-$(uname -r) …"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    # The kernel-specific tools package name varies; try the exact match
    # first, fall back to -generic for the running stream.
    if ! apt-get install -y -qq linux-tools-common "linux-tools-$(uname -r)" 2>/dev/null; then
        apt-get install -y -qq linux-tools-common linux-tools-generic || true
    fi
else
    echo "  cpupower + perf already present."
fi

# --------------------------------------------------------------------------
# 3) CPU governor → performance.
# --------------------------------------------------------------------------

echo
echo "[3/3] CPU frequency governor"
if command -v cpupower >/dev/null 2>&1; then
    cpupower frequency-set -g performance >/dev/null 2>&1 \
        && echo "  set via cpupower: governor=performance" \
        || echo "  cpupower failed; falling back to direct sysfs write"
fi

# Direct sysfs write as belt-and-braces (some kernels have cpupower but no
# matching backend; sysfs always works if the driver supports manual scaling).
if [[ -d /sys/devices/system/cpu/cpu0/cpufreq ]]; then
    set +e
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > "$f" 2>/dev/null
    done
    set -e
fi

CUR_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "n/a")
echo "  cpu0 governor: $CUR_GOV"
if [[ "$CUR_GOV" == "n/a" ]]; then
    echo "  NOTE: cpufreq subsystem not available on this host."
    echo "        Hetzner virtualised instances often pin the host clock —"
    echo "        the benches will still run; numbers may be slightly noisier."
fi

# Persist governor across reboot via systemd if possible (so a long bench
# survives an unexpected restart). Skip if systemd isn't running.
if command -v systemctl >/dev/null 2>&1 && [[ -d /etc/systemd/system ]]; then
    cat > /etc/systemd/system/bench-cpu-governor.service <<'EOF'
[Unit]
Description=Pin CPU governor to performance for audio_samples benches
After=cpufrequtils.service

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$f" 2>/dev/null || true; done'
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now bench-cpu-governor.service > /dev/null 2>&1 || true
    echo "  installed systemd unit: bench-cpu-governor.service (re-applies governor on boot)"
fi

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------

echo
echo "============================================================"
echo " host setup complete"
echo "------------------------------------------------------------"
echo " perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
echo " cpu0 governor:       $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
echo " systemd unit:        bench-cpu-governor.service ($(systemctl is-enabled bench-cpu-governor.service 2>/dev/null || echo n/a))"
echo "============================================================"
