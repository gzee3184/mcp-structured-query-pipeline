#!/bin/bash
# ============================================================================
# run_ablation_matrix.sh — Cross-model ablation study
# ============================================================================
# Runs 5 ablation configs for 3 model backbones at N=200.
#
# Configs: full, --no-kg, --no-embedding, --no-correction, --no-adaptive
# Models:  NIM (GPT-OSS-120B), Claude Sonnet 4.6, Qwen 3 Next 80B
#
# Usage:
#   bash eval/scripts/run_ablation_matrix.sh              # All
#   bash eval/scripts/run_ablation_matrix.sh sonnet        # Just Sonnet
#
# Results saved to: eval/results/ablation_models/
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

source venv/bin/activate
source .env 2>/dev/null || true

RESULTS_DIR="eval/results/ablation_models"
mkdir -p "$RESULTS_DIR"

N=200

# Model registry
declare -A MODELS=(
    [sonnet]="bedrock|anthropic.claude-sonnet-4-6|1.0"
    [qwen]="bedrock|qwen.qwen3-next-80b-a3b|1.0"
    [nim]="openai|openai/gpt-oss-120b|2.0"
)

# Ablation configs: name|flags
declare -A CONFIGS=(
    [full]=""
    [no_kg]="--no-kg"
    [no_embedding]="--no-embedding"
    [no_correction]="--no-correction"
    [no_adaptive]="--no-adaptive"
)

# Config order (bash associative arrays don't preserve order)
CONFIG_ORDER=("full" "no_kg" "no_embedding" "no_correction" "no_adaptive")

if [ $# -gt 0 ]; then
    RUN_MODELS=("$@")
else
    RUN_MODELS=("sonnet" "qwen" "nim")
fi

echo "============================================================"
echo "CROSS-MODEL ABLATION MATRIX"
echo "Models: ${RUN_MODELS[*]}"
echo "Configs: ${CONFIG_ORDER[*]}"
echo "N=$N per run"
echo "Results: $RESULTS_DIR/"
echo "============================================================"

for MODEL_KEY in "${RUN_MODELS[@]}"; do
    if [ -z "${MODELS[$MODEL_KEY]+x}" ]; then
        echo "ERROR: Unknown model key '$MODEL_KEY'. Valid: ${!MODELS[*]}"
        continue
    fi

    IFS='|' read -r PROVIDER MODEL_ID DELAY <<< "${MODELS[$MODEL_KEY]}"

    for CONFIG_KEY in "${CONFIG_ORDER[@]}"; do
        FLAGS="${CONFIGS[$CONFIG_KEY]}"
        LOG="$RESULTS_DIR/${MODEL_KEY}_${CONFIG_KEY}_n${N}.log"

        # Skip if result already exists
        # The pipeline names ablation results differently, check for it
        if ls "$RESULTS_DIR/"*"${MODEL_KEY}"*"${CONFIG_KEY}"*".json" &>/dev/null; then
            echo "[SKIP] $MODEL_KEY / $CONFIG_KEY — result exists"
            continue
        fi

        echo ""
        echo "------------------------------------------------------------"
        echo "  $MODEL_KEY | config=$CONFIG_KEY | flags='$FLAGS'"
        echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "------------------------------------------------------------"

        python -u src/test_gorilla/test_enhanced_pipeline.py \
            --n "$N" --split test --delay "$DELAY" --relaxed-match \
            --provider "$PROVIDER" --model "$MODEL_ID" \
            $FLAGS \
            2>&1 | tee "$LOG"

        echo "  Finished at $(date '+%Y-%m-%d %H:%M:%S')"
    done
done

echo ""
echo "============================================================"
echo "ABLATION MATRIX COMPLETE"
echo "Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"*.json 2>/dev/null || echo "(no results yet)"
echo "============================================================"
