#!/usr/bin/env bash
# Launch SGLang for Qwen3.6-27B-FP8, run the user-supplied pipeline command,
# then GUARANTEE SGLang shutdown via trap — even on Ctrl+C or non-zero exit.
#
# Usage:
#   scripts/run_with_sglang.sh -- ./venv/bin/python3 src/test_gorilla/test_enhanced_pipeline.py \
#       --provider local --tool-schema v2_qwen ...
#
# The `--` separator divides this script's flags from the inner command.
#
# This is the safety feature for unattended N=1315 runs. It addresses the
# concern that a manually-launched SGLang process holds ~50 GB of GPU 1 VRAM
# indefinitely if you forget to stop it.
#
# Behavior:
#   1. Resolves the FP8 model path and SGLang venv.
#   2. Launches SGLang on port 30000 with the prod flag set
#      (qwen parser, qwen3 reasoning_parser, mem-fraction 0.50).
#   3. Waits up to 5 min for /health to return 200.
#   4. Runs whatever you put after `--`.
#   5. ALWAYS stops SGLang on exit (trap on EXIT, INT, TERM).
#   6. Verifies GPU 1 freed; warns if not.
#
# Env overrides:
#   SGLANG_PORT      default 30000
#   SGLANG_GPU       default 1   (do NOT set to 0 — user's research_wm runs there)
#   SGLANG_LOG_DIR   default logs/sglang_$(date +%s)
#   SGLANG_HEALTH_TIMEOUT_S  default 300

set -euo pipefail

PORT="${SGLANG_PORT:-30000}"
GPU="${SGLANG_GPU:-1}"
LOG_DIR="${SGLANG_LOG_DIR:-logs/sglang_$(date +%s)}"
HEALTH_TIMEOUT_S="${SGLANG_HEALTH_TIMEOUT_S:-300}"

mkdir -p "$LOG_DIR"
SGLANG_LOG="$LOG_DIR/server.log"

# Locate the FP8 snapshot.
FP8_PATH="$(ls -d /export/scratch/abrar008/hf_cache_home/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::')"
if [[ -z "$FP8_PATH" ]]; then
    echo "ERROR: Qwen3.6-27B-FP8 snapshot not found in HF cache" >&2
    exit 2
fi

# Find the `--` separator.
SEP_FOUND=0
INNER=()
for arg in "$@"; do
    if [[ "$SEP_FOUND" -eq 1 ]]; then
        INNER+=("$arg")
    elif [[ "$arg" == "--" ]]; then
        SEP_FOUND=1
    fi
done
if [[ "$SEP_FOUND" -eq 0 ]] || [[ "${#INNER[@]}" -eq 0 ]]; then
    echo "Usage: $0 -- <command to run>" >&2
    exit 2
fi

SGLANG_PID=""

_CLEANED_UP=0
cleanup() {
    local rc=$?
    # Guard against double-run (trap fires on both signal AND the subsequent EXIT).
    if [[ "$_CLEANED_UP" -eq 1 ]]; then return; fi
    _CLEANED_UP=1
    if [[ -n "$SGLANG_PID" ]]; then
        echo ">>> Stopping SGLang (PID $SGLANG_PID)..." >&2
        if kill -0 "$SGLANG_PID" 2>/dev/null; then
            kill "$SGLANG_PID" 2>/dev/null || true
            for _ in $(seq 1 30); do
                if ! kill -0 "$SGLANG_PID" 2>/dev/null; then break; fi
                sleep 1
            done
            if kill -0 "$SGLANG_PID" 2>/dev/null; then
                echo ">>> SGLang did not exit on SIGTERM; sending SIGKILL." >&2
                kill -9 "$SGLANG_PID" 2>/dev/null || true
                sleep 1
            fi
        fi
    fi
    # Belt-and-suspenders: sweep any orphaned SGLang workers (scheduler /
    # detokenizer / tokenizer subprocesses) for THIS port, regardless of PID.
    # SGLang spawns children that can outlive the tracked parent PID.
    pkill -9 -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
    # Verify GPU freed. nvidia-smi is broken on this box (driver-version
    # mismatch 580.142 vs 580.159), so prefer the ctypes-NVML replacement.
    sleep 1
    if [[ -f /export/scratch/abrar008/research_wm/scripts/gpu_mem.py ]]; then
        LD_LIBRARY_PATH="/export/scratch/abrar008/nvidia_local/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
            python /export/scratch/abrar008/research_wm/scripts/gpu_mem.py 2>/dev/null \
            | sed 's/^/>>> [gpu] /' >&2 || true
    elif command -v nvidia-smi >/dev/null 2>&1; then
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id="$GPU" 2>/dev/null || echo "?")
        echo ">>> GPU $GPU memory.used after shutdown: ${mem_used} MiB" >&2
    fi
    echo ">>> SGLang cleanup complete (rc=$rc)." >&2
    exit "$rc"
}
trap cleanup EXIT INT TERM

echo ">>> Launching SGLang on port $PORT (GPU $GPU); log: $SGLANG_LOG"
PATH=/export/scratch/abrar008/cuda-12.9/bin:/export/scratch/abrar008/sglang-venv/bin:$PATH \
CUDA_HOME=/export/scratch/abrar008/cuda-12.9 \
LD_LIBRARY_PATH=/export/scratch/abrar008/nvidia_local/root/usr/lib/x86_64-linux-gnu:/export/scratch/abrar008/cuda-12.9/lib64:${LD_LIBRARY_PATH:-} \
CUDA_VISIBLE_DEVICES="$GPU" HF_HOME=/export/scratch/abrar008/hf_cache_home \
TRITON_CACHE_DIR=/export/scratch/abrar008/triton_cache \
nohup /export/scratch/abrar008/sglang-venv/bin/python -m sglang.launch_server \
    --model-path "$FP8_PATH" \
    --port "$PORT" --host 127.0.0.1 \
    --tool-call-parser qwen \
    --reasoning-parser qwen3 \
    --enable-custom-logit-processor \
    --mem-fraction-static 0.50 \
    --max-running-requests 16 \
    --tp 1 \
    --attention-backend triton \
    --sampling-backend pytorch \
    > "$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!
echo ">>> SGLang PID: $SGLANG_PID"

echo ">>> Waiting up to ${HEALTH_TIMEOUT_S}s for SGLang /health..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_S ))
while (( $(date +%s) < deadline )); do
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
        echo "ERROR: SGLang process exited prematurely. Last log:" >&2
        tail -40 "$SGLANG_LOG" >&2
        exit 3
    fi
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo ">>> SGLang ready."
        break
    fi
    sleep 5
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: SGLang did not become healthy within ${HEALTH_TIMEOUT_S}s." >&2
    tail -40 "$SGLANG_LOG" >&2
    exit 3
fi

echo ">>> Running inner command:"
printf '    %q ' "${INNER[@]}"; echo
# Standard production env: thinking off (qwen parser handles tool calls directly),
# 1500 token output cap, stop-at-tool-call disabled (qwen parser handles multi-call).
# LM_AGENT_LOOP_OPENAI=1: routes the refine loop through OpenAIAdapter (chat.completions)
# instead of BedrockAdapter — required for local Qwen/SGLang since there is no .converse().
# Without this the refine loop silently crashes on every query (AttributeError swallowed).
LM_BASE_URL="http://127.0.0.1:$PORT/v1" \
LM_DISABLE_THINKING="${LM_DISABLE_THINKING:-1}" \
LM_MAX_TOKENS="${LM_MAX_TOKENS:-1500}" \
LM_AGENT_LOOP_OPENAI="${LM_AGENT_LOOP_OPENAI:-1}" \
"${INNER[@]}"
INNER_RC=$?
echo ">>> Inner command exit code: $INNER_RC"
exit "$INNER_RC"
