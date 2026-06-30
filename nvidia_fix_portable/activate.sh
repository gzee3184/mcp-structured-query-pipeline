#!/usr/bin/env bash
# nvidia_fix_portable/activate.sh
#
# Fixes the NVML "Driver/library version mismatch" error WITHOUT root or reboot.
# Usage:  source /path/to/nvidia_fix_portable/activate.sh
#
# After sourcing, use gpu_mem.py instead of nvidia-smi:
#   python /path/to/nvidia_fix_portable/gpu_mem.py

# ── Optional: fill this in if you already have the libs extracted somewhere ──
# Leave empty ("") to auto-extract the bundled .deb files on first run.
PREEXTRACTED_LIB_DIR=""
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Resolve lib dir ───────────────────────────────────────────────────────────
if [[ -n "$PREEXTRACTED_LIB_DIR" ]]; then
    LIB_DIR="$PREEXTRACTED_LIB_DIR"
    DRIVER_VER="$(ls "$LIB_DIR"/libnvidia-ml.so.* 2>/dev/null | grep -oP '\d+\.\d+$' | head -1)"
    if [[ -z "$DRIVER_VER" ]]; then
        echo "ERROR: No libnvidia-ml.so.<ver> found in PREEXTRACTED_LIB_DIR=$LIB_DIR" >&2
        return 1
    fi
else
    # Auto-extract bundled debs
    DEB_COMPUTE="$(ls "$SCRIPT_DIR/debs/libnvidia-compute-"*.deb 2>/dev/null | head -1)"
    if [[ -z "$DEB_COMPUTE" ]]; then
        echo "ERROR: No libnvidia-compute .deb found in $SCRIPT_DIR/debs/" >&2
        return 1
    fi
    DRIVER_VER="$(basename "$DEB_COMPUTE" | grep -oP '\d+\.\d+(?=-\d)')"
    LIB_DIR="$SCRIPT_DIR/extracted/usr/lib/x86_64-linux-gnu"

    if [[ ! -f "$LIB_DIR/libnvidia-ml.so.$DRIVER_VER" ]]; then
        echo "nvidia_fix: extracting driver $DRIVER_VER libs to $SCRIPT_DIR/extracted ..."
        mkdir -p "$SCRIPT_DIR/extracted"
        for deb in "$SCRIPT_DIR/debs/"*.deb; do
            dpkg-deb -x "$deb" "$SCRIPT_DIR/extracted"
        done
        echo "nvidia_fix: extraction complete."
    fi
fi

# ── Activate ─────────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"
export NVIDIA_FIX_LIB_DIR="$LIB_DIR"
export NVIDIA_FIX_DRIVER_VER="$DRIVER_VER"

echo "nvidia_fix: activated driver $DRIVER_VER  (libs: $LIB_DIR)"
echo "nvidia_fix: run 'python $SCRIPT_DIR/gpu_mem.py' to check GPU memory"
