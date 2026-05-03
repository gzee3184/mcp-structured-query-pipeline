# Schema Discovery Pipeline for Structured Query Generation

A two-phase pipeline that discovers relevant database schemas from large catalogs and generates structured queries — without knowing which schema to target upfront.

## What It Does

```
"Find customers who paid in euro with monthly consumption over 1000"
                    │
    ┌───────────────┴───────────────┐
    │ Phase 1: DISCOVERY            │  No LLM needed
    │ Embed question → rank 143     │  Uses local sentence-transformers
    │ collections → KG boost →      │  + knowledge graph + reranking
    │ return top-5 candidates       │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │ Phase 2: GENERATION           │  1 LLM call
    │ Send 5 schemas + question     │  Supports Bedrock, OpenAI, Ollama
    │ → LLM returns structured      │
    │ tool call (JSON)              │
    └───────────────┬───────────────┘
                    │
                    ▼
    {
      "collection_name": "DebitCardSpecializingYearmonth",
      "filters": [
        {"property_name": "Currency", "operator": "=", "value": "EUR"},
        {"property_name": "Consumption", "operator": ">", "value": 1000}
      ],
      "output_properties": [{"property_name": "CustomerID", "aggregation": "COUNT"}]
    }
```

## Documentation

Start here depending on your goal:

| If you want to... | Read |
|-------------------|------|
| Understand how the code is organized | [`docs/CODE_MAP.md`](docs/CODE_MAP.md) |
| Understand the pipeline architecture | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Understand what each metric means | [`docs/METRICS.md`](docs/METRICS.md) |
| Understand what the pipeline cannot do | [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) |
| Set up the datasets | [`docs/DATA_SETUP.md`](docs/DATA_SETUP.md) |
| Read the literature survey | [`docs/RESEARCH_FINDINGS.md`](docs/RESEARCH_FINDINGS.md) |

For detailed evaluation results, see [`eval/paper_comparison/DEEP_DIVE_ANALYSIS.md`](eval/paper_comparison/DEEP_DIVE_ANALYSIS.md) (reviewer walkthrough) and [`eval/paper_comparison/RESULTS_EXPLAINED.md`](eval/paper_comparison/RESULTS_EXPLAINED.md) (outsider-friendly guide).

## Setup

### 1. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the datasets

See [`docs/DATA_SETUP.md`](docs/DATA_SETUP.md) for download instructions. You'll need:
- BIRD benchmark (queries + 11 SQLite databases)
- Weaviate Gorilla benchmark
- BIRD-to-Weaviate schema conversions

Place everything under `data/` at the project root.

### 3. Configure API credentials

The pipeline needs an LLM provider. Pick one:

**AWS Bedrock (Claude, Qwen, Llama):**
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
```

**NVIDIA NIM:**
```bash
export NVIDIA_API_KEY=...
```

**OpenAI:**
```bash
export OPENAI_API_KEY=...
```

**Ollama (local):** no keys needed, just make sure `ollama serve` is running.

## Quick Start

```bash
# Run pipeline on 50 queries with Claude Sonnet via Bedrock
python src/test_gorilla/test_enhanced_pipeline.py \
    --n 50 --split test --delay 1.0 \
    --provider bedrock --model anthropic.claude-sonnet-4-6 \
    --tool-schema v2

# Run blind baseline for comparison (no discovery — gives LLM all schemas)
python eval/scripts/blind_llm_baseline.py \
    --n 50 --split test --delay 1.0 \
    --provider bedrock --model anthropic.claude-sonnet-4-6

# Score BIRD execution accuracy on pipeline results
python eval/scripts/bird_ex_v2.py <result_json>
python eval/scripts/bird_ex_v2.py <result_json> --gold-guided  # With translator fixes
```

Results are saved to `eval/results/` or a directory you specify with `--output-dir`.

## Project Structure

```
src/                            # Pipeline core (17 Python files)
├── test_gorilla/
│   └── test_enhanced_pipeline.py   # Pipeline class + evaluation harness
├── mcp/                            # Schema server + validation
├── lm/                             # Multi-provider LLM client
├── utils/                          # KG, embeddings, scoring, tool schemas
└── models.py                       # Pydantic data models

eval/                           # Evaluation scripts (14 Python files)
├── scripts/                        # BIRD EX scorer, blind baseline, figures
├── paper_comparison/               # Cross-system analysis + result docs
└── eai/                            # MongoDB evaluation

docs/                           # Documentation
└── ...                             # See table above
```

See [`docs/CODE_MAP.md`](docs/CODE_MAP.md) for a file-by-file navigation guide.

## Supported LLMs

The pipeline is model-agnostic — any LLM supporting tool-call/function-calling works. Tested with:

- Claude Sonnet 4.6, Qwen 3 80B, Llama 4 Maverick 17B (via AWS Bedrock)
- openai/gpt-oss-120b (via NVIDIA NIM)
- Local models (via Ollama)

## License

TBD.
