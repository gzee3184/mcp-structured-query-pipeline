#!/bin/bash
# ============================================================================
# run_multi_model.sh — Full pipeline (N=1584) with each LLM backbone
# ============================================================================
# Usage:
#   bash eval/scripts/run_multi_model.sh              # Run all models
#   bash eval/scripts/run_multi_model.sh sonnet        # Run just Claude Sonnet
#   bash eval/scripts/run_multi_model.sh qwen llama    # Run Qwen + Llama
#
# Results saved to: eval/results/multi_model/
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."  # cd to project root (gorilla_2/gorilla/)

source venv/bin/activate
source .env 2>/dev/null || true

RESULTS_DIR="eval/results/multi_model"
mkdir -p "$RESULTS_DIR"

# Model registry: short_name|provider|model_id|delay
declare -A MODELS=(
    [sonnet]="bedrock|anthropic.claude-sonnet-4-6|1.0"
    [opus]="bedrock|anthropic.claude-opus-4-6-v1|1.5"
    [qwen]="bedrock|qwen.qwen3-next-80b-a3b|1.0"
    [llama]="bedrock|meta.llama4-maverick-17b-instruct-v1:0|1.0"
    [nim]="openai|openai/gpt-oss-120b|2.0"
)

# Determine which models to run
if [ $# -gt 0 ]; then
    RUN_MODELS=("$@")
else
    RUN_MODELS=("sonnet" "opus" "qwen" "llama")
fi

echo "============================================================"
echo "MULTI-MODEL FULL PIPELINE EVALUATION"
echo "Models: ${RUN_MODELS[*]}"
echo "Results: $RESULTS_DIR/"
echo "============================================================"

for MODEL_KEY in "${RUN_MODELS[@]}"; do
    if [ -z "${MODELS[$MODEL_KEY]+x}" ]; then
        echo "ERROR: Unknown model key '$MODEL_KEY'. Valid: ${!MODELS[*]}"
        continue
    fi

    IFS='|' read -r PROVIDER MODEL_ID DELAY <<< "${MODELS[$MODEL_KEY]}"

    # Build safe filename
    SAFE_NAME="${MODEL_KEY}"
    LOG_FILE="$RESULTS_DIR/pipeline_bedrock_${SAFE_NAME}_full_n1584_test.log"

    # Skip if JSON result already exists
    JSON_FILE="$RESULTS_DIR/pipeline_bedrock_${SAFE_NAME}_full_n1584_test.json"
    if [ "$PROVIDER" = "openai" ]; then
        LOG_FILE="$RESULTS_DIR/pipeline_nim_gptoss_full_n1584_test.log"
        JSON_FILE="$RESULTS_DIR/pipeline_nim_gptoss_full_n1584_test.json"
    fi

    if [ -f "$JSON_FILE" ]; then
        echo "[SKIP] $MODEL_KEY — result already exists: $JSON_FILE"
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "Running: $MODEL_KEY ($PROVIDER / $MODEL_ID)"
    echo "  Delay: ${DELAY}s | Log: $LOG_FILE"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"

    python -u src/test_gorilla/test_enhanced_pipeline.py \
        --all --split test --delay "$DELAY" --relaxed-match \
        --provider "$PROVIDER" --model "$MODEL_ID" \
        2>&1 | tee "$LOG_FILE"

    echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

echo "============================================================"
echo "ALL RUNS COMPLETE"
echo "Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"*.json 2>/dev/null || echo "(no JSON results yet)"
echo "============================================================"
