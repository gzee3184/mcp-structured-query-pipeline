#!/bin/bash
# ============================================================================
# Paper Comparison — Full Evaluation Suite (N=1584)
# ============================================================================
# All conditions run on the FULL dataset (1315 BIRD + 269 WG for pipeline/blind,
# 1315 BIRD for SOTA).
#
# Usage:
#   bash eval/paper_comparison/run_all.sh              # All conditions
#   bash eval/paper_comparison/run_all.sh pipeline_v2  # Just pipeline V2
#   bash eval/paper_comparison/run_all.sh pipeline_v1  # Just pipeline V1
#   bash eval/paper_comparison/run_all.sh blind        # Just blind baselines
#   bash eval/paper_comparison/run_all.sh sota         # Re-run SOTA on full N=1315
#   bash eval/paper_comparison/run_all.sh smoke        # Quick N=5 validation
#
# Estimated wall time (sequential):
#   - Pipeline V2 × 3 models: ~10h (1584q × ~7s/q × 3)
#   - Pipeline V1 × 3 models: ~10h
#   - Blind × 3 models: ~7h (1584q × ~5s/q × 3)
#   - SOTA re-run: DAIL-SQL ~2h, DIN-SQL ~16h, CHESS ~24h
#   - Total: ~69h sequential. Parallelize models for ~24h.
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."  # cd to project root (gorilla_2/gorilla/)

source venv/bin/activate
source .env 2>/dev/null || true

RESULTS_DIR="eval/paper_comparison/results"
mkdir -p "$RESULTS_DIR"

# Model registry: short_name|provider|model_id|delay
declare -A MODELS=(
    [sonnet]="bedrock|anthropic.claude-sonnet-4-6|1.0"
    [qwen]="bedrock|qwen.qwen3-next-80b-a3b|1.5"
    [llama]="bedrock|meta.llama4-maverick-17b-instruct-v1:0|1.5"
)

CONDITION="${1:-all}"

run_pipeline() {
    local MODEL_KEY=$1 SCHEMA=$2
    IFS='|' read -r PROVIDER MODEL_ID DELAY <<< "${MODELS[$MODEL_KEY]}"

    local OUTFILE="$RESULTS_DIR/pipeline_${MODEL_KEY}_${SCHEMA}_n1584_test.json"
    local LOGFILE="$RESULTS_DIR/pipeline_${MODEL_KEY}_${SCHEMA}_n1584_test.log"

    if [ -f "$OUTFILE" ]; then
        echo "[SKIP] Pipeline $MODEL_KEY $SCHEMA — exists: $OUTFILE"
        return
    fi

    echo "[RUN] Pipeline $MODEL_KEY schema=$SCHEMA started=$(date '+%H:%M:%S')"
    python -u src/test_gorilla/test_enhanced_pipeline.py \
        --all --split test --delay "$DELAY" --relaxed-match \
        --provider "$PROVIDER" --model "$MODEL_ID" \
        --tool-schema "$SCHEMA" \
        --output-dir "$RESULTS_DIR" \
        2>&1 | tee "$LOGFILE"
    echo "[DONE] Pipeline $MODEL_KEY $SCHEMA finished=$(date '+%H:%M:%S')"
}

run_blind() {
    local MODEL_KEY=$1
    IFS='|' read -r PROVIDER MODEL_ID DELAY <<< "${MODELS[$MODEL_KEY]}"

    local OUTFILE="$RESULTS_DIR/blind_${MODEL_KEY}_n1584_test.json"
    local LOGFILE="$RESULTS_DIR/blind_${MODEL_KEY}_n1584_test.log"

    if [ -f "$OUTFILE" ]; then
        echo "[SKIP] Blind $MODEL_KEY — exists: $OUTFILE"
        return
    fi

    echo "[RUN] Blind $MODEL_KEY started=$(date '+%H:%M:%S')"
    python -u eval/scripts/blind_llm_baseline.py \
        --all --split test --delay "$DELAY" \
        --provider "$PROVIDER" --model "$MODEL_ID" \
        --output-dir "$RESULTS_DIR" \
        2>&1 | tee "$LOGFILE"
    echo "[DONE] Blind $MODEL_KEY finished=$(date '+%H:%M:%S')"
}

run_sota() {
    local SYSTEM=$1
    local SOTA_DIR="/export/scratch/abrar008/llm_rag/sota_comparison"

    local OUTFILE="$SOTA_DIR/results/${SYSTEM}_predictions_full.json"

    if [ -f "$OUTFILE" ]; then
        echo "[SKIP] SOTA $SYSTEM — exists: $OUTFILE"
        return
    fi

    echo "[RUN] SOTA $SYSTEM on full N=1315 started=$(date '+%H:%M:%S')"
    pushd "$SOTA_DIR" > /dev/null

    case "$SYSTEM" in
        dail_sql)
            python -u run_dail_sql.py \
                --out "results/dail_sql_predictions_full.json" \
                --delay 0.5 \
                2>&1 | tee "results/dail_sql_full.log"
            ;;
        din_sql)
            python -u run_din_sql.py \
                --out "results/din_sql_predictions_full.json" \
                --delay 0.5 \
                2>&1 | tee "results/din_sql_full.log"
            ;;
        chess)
            python -u run_chess.py \
                --out "results/chess_predictions_full.json" \
                2>&1 | tee "results/chess_full.log"
            ;;
    esac

    popd > /dev/null
    echo "[DONE] SOTA $SYSTEM finished=$(date '+%H:%M:%S')"
}

run_smoke() {
    echo "=== SMOKE TEST (N=5 per model) ==="
    for MODEL_KEY in sonnet qwen llama; do
        IFS='|' read -r PROVIDER MODEL_ID DELAY <<< "${MODELS[$MODEL_KEY]}"
        echo "  Testing $MODEL_KEY..."
        python -u src/test_gorilla/test_enhanced_pipeline.py \
            --n 5 --split test --delay "$DELAY" --relaxed-match \
            --provider "$PROVIDER" --model "$MODEL_ID" \
            --tool-schema v2 \
            --output-dir "$RESULTS_DIR" \
            2>&1 | tail -5
        echo ""
    done
    echo "Smoke test complete. Check results for latency_s and input_tokens fields."
}

echo "============================================================"
echo "PAPER COMPARISON: FULL EVALUATION (N=1584)"
echo "Condition: $CONDITION | Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results dir: $RESULTS_DIR/"
echo "============================================================"

case "$CONDITION" in
    all)
        # Priority order: pipeline V2 (primary), then blind, then V1, then SOTA
        echo -e "\n=== PIPELINE V2 (N=1584) ==="
        for M in sonnet qwen llama; do run_pipeline "$M" "v2"; done

        echo -e "\n=== BLIND BASELINE (N=1584) ==="
        for M in sonnet qwen llama; do run_blind "$M"; done

        echo -e "\n=== PIPELINE V1 (N=1584) ==="
        for M in sonnet qwen llama; do run_pipeline "$M" "v1"; done

        echo -e "\n=== SOTA RE-RUN (N=1315 BIRD) ==="
        for S in dail_sql din_sql chess; do run_sota "$S"; done
        ;;
    pipeline_v2)
        echo -e "\n=== PIPELINE V2 (N=1584) ==="
        for M in sonnet qwen llama; do run_pipeline "$M" "v2"; done
        ;;
    pipeline_v1)
        echo -e "\n=== PIPELINE V1 (N=1584) ==="
        for M in sonnet qwen llama; do run_pipeline "$M" "v1"; done
        ;;
    blind)
        echo -e "\n=== BLIND BASELINE (N=1584) ==="
        for M in sonnet qwen llama; do run_blind "$M"; done
        ;;
    sota)
        echo -e "\n=== SOTA RE-RUN (N=1315 BIRD) ==="
        for S in dail_sql din_sql chess; do run_sota "$S"; done
        ;;
    smoke)
        run_smoke
        ;;
    # Single model runs
    sonnet_v2) run_pipeline "sonnet" "v2" ;;
    sonnet_v1) run_pipeline "sonnet" "v1" ;;
    qwen_v2) run_pipeline "qwen" "v2" ;;
    qwen_v1) run_pipeline "qwen" "v1" ;;
    llama_v2) run_pipeline "llama" "v2" ;;
    llama_v1) run_pipeline "llama" "v1" ;;
    blind_sonnet) run_blind "sonnet" ;;
    blind_qwen) run_blind "qwen" ;;
    blind_llama) run_blind "llama" ;;
    sota_dail) run_sota "dail_sql" ;;
    sota_din) run_sota "din_sql" ;;
    sota_chess) run_sota "chess" ;;
    *)
        echo "Unknown condition: $CONDITION"
        echo "Valid: all, pipeline_v2, pipeline_v1, blind, sota, smoke"
        echo "       sonnet_v2, sonnet_v1, qwen_v2, qwen_v1, llama_v2, llama_v1"
        echo "       blind_sonnet, blind_qwen, blind_llama"
        echo "       sota_dail, sota_din, sota_chess"
        exit 1
        ;;
esac

echo -e "\n============================================================"
echo "COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results: $(ls "$RESULTS_DIR/"*.json 2>/dev/null | wc -l) JSON files in $RESULTS_DIR/"
echo "============================================================"
