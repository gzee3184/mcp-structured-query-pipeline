#!/bin/bash
# ============================================================================
# run_scale_experiment.sh — Figure 1(a): Performance vs Dataset Scale
# ============================================================================
# Runs both the pipeline AND blind baseline at each DB count (2,4,6,8,12)
# to show that the pipeline degrades less as the haystack grows.
#
# For each scale point, runs:
#   1. Full pipeline (our method)
#   2. Embedding-only baseline (pipeline without KG)
#   3. Blind LLM baseline (no pipeline, all schemas at once)
#
# Usage:
#   bash eval/scripts/run_scale_experiment.sh                    # All models
#   bash eval/scripts/run_scale_experiment.sh sonnet             # Just Sonnet
#   bash eval/scripts/run_scale_experiment.sh sonnet qwen        # Sonnet + Qwen
#
# Results saved to: eval/results/scale_exp/
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."  # cd to project root

source venv/bin/activate
source .env 2>/dev/null || true

RESULTS_DIR="eval/results/scale_exp"
mkdir -p "$RESULTS_DIR"

DB_COUNTS=(2 4 6 8 12)

# Model registry: short_name|provider|model_id|delay
declare -A MODELS=(
    [sonnet]="bedrock|anthropic.claude-sonnet-4-6|1.0"
    [qwen]="bedrock|qwen.qwen3-next-80b-a3b|1.0"
    [nim]="openai|openai/gpt-oss-120b|2.0"
)

# Determine which models to run
if [ $# -gt 0 ]; then
    RUN_MODELS=("$@")
else
    RUN_MODELS=("sonnet" "qwen")
fi

echo "============================================================"
echo "SCALE EXPERIMENT — Figure 1(a)"
echo "Models: ${RUN_MODELS[*]}"
echo "DB counts: ${DB_COUNTS[*]}"
echo "Results: $RESULTS_DIR/"
echo "============================================================"

for MODEL_KEY in "${RUN_MODELS[@]}"; do
    if [ -z "${MODELS[$MODEL_KEY]+x}" ]; then
        echo "ERROR: Unknown model key '$MODEL_KEY'. Valid: ${!MODELS[*]}"
        continue
    fi

    IFS='|' read -r PROVIDER MODEL_ID DELAY <<< "${MODELS[$MODEL_KEY]}"

    for N_DBS in "${DB_COUNTS[@]}"; do
        echo ""
        echo "============================================================"
        echo "  $MODEL_KEY | $N_DBS databases"
        echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================"

        # --- 1. Full pipeline ---
        LOG="$RESULTS_DIR/pipeline_${MODEL_KEY}_full_dbs${N_DBS}.log"
        echo "  [1/3] Full pipeline..."
        python -u src/test_gorilla/test_enhanced_pipeline.py \
            --all --split test --delay "$DELAY" --relaxed-match \
            --provider "$PROVIDER" --model "$MODEL_ID" \
            --max-dbs "$N_DBS" \
            2>&1 | tee "$LOG"

        # --- 2. Embedding-only baseline (no KG, no rerank, no values, no adaptive) ---
        LOG="$RESULTS_DIR/pipeline_${MODEL_KEY}_embed_only_dbs${N_DBS}.log"
        echo "  [2/3] Embedding-only baseline..."
        python -u src/test_gorilla/test_enhanced_pipeline.py \
            --all --split test --delay "$DELAY" --relaxed-match \
            --provider "$PROVIDER" --model "$MODEL_ID" \
            --max-dbs "$N_DBS" \
            --no-kg --no-rerank --no-values --no-adaptive \
            2>&1 | tee "$LOG"

        # --- 3. Blind LLM baseline (no pipeline) ---
        LOG="$RESULTS_DIR/blind_${MODEL_KEY}_dbs${N_DBS}.log"
        echo "  [3/3] Blind LLM baseline..."
        python -u eval/scripts/blind_llm_baseline.py \
            --all --split test --delay "$DELAY" \
            --provider "$PROVIDER" --model "$MODEL_ID" \
            --max-dbs "$N_DBS" \
            2>&1 | tee "$LOG"

        echo "  Finished $N_DBS DBs at $(date '+%Y-%m-%d %H:%M:%S')"
    done
done

echo ""
echo "============================================================"
echo "SCALE EXPERIMENT COMPLETE"
echo "Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"*.json 2>/dev/null | wc -l
echo " result files generated"
echo "============================================================"
