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

## Key Results

| Domain | Benchmark | Discovery | Accuracy | Tokens/q |
|--------|-----------|-----------|----------|----------|
| **Relational** | BIRD (N=1315) | 97.9% | 93.1% relaxed | ~7,000 |
| **NoSQL** | EAI/MongoDB (N=200) | 91.5% | 0.723 OPS | ~7,000 |
| **VectorDB** | Weaviate Gorilla (N=269) | 100% | 86.1% AST | ~5,500 |
| **Blind baseline** | (all 143 schemas) | n/a | 93.5% relaxed | ~31,500 |

**78% fewer tokens** than the blind approach at equivalent accuracy.
**More model-robust** than SOTA: 2.3pp variance across 3 LLMs vs DAIL-SQL's 11.5pp.

## Setup

```bash
# Clone and install
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download data (BIRD benchmark + Weaviate Gorilla schemas)
# See docs/DATA_SETUP.md for instructions

# Set environment variables
export AWS_ACCESS_KEY_ID=...       # For Bedrock
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
# OR
export NVIDIA_API_KEY=...          # For NVIDIA NIM
```

## Quick Start

```bash
# Run pipeline on 50 queries
python src/test_gorilla/test_enhanced_pipeline.py --n 50 --split test \
    --provider bedrock --model anthropic.claude-sonnet-4-6 \
    --tool-schema v2 --delay 1.0

# Run blind baseline for comparison
python eval/scripts/blind_llm_baseline.py --n 50 --split test \
    --provider bedrock --model anthropic.claude-sonnet-4-6 --delay 1.0

# Score BIRD execution accuracy
python eval/scripts/bird_ex_v2.py <result_json>             # Standard
python eval/scripts/bird_ex_v2.py <result_json> --gold-guided  # With translator fixes
```

## Project Structure

```
src/
├── lm/
│   ├── lm.py                  # LMService: Bedrock, OpenAI/NIM, Ollama
│   └── db_gorilla_prompts.py  # System prompts (V1 + V2)
├── mcp/
│   ├── server.py              # MCPServer: schema registry, collection search
│   └── sandbox.py             # StructuredSandbox: validation + correction
├── utils/
│   ├── field_kg.py            # Knowledge graph (FK, shared keys, name patterns)
│   ├── embeddings.py          # Sentence-transformer embeddings
│   ├── ast_scoring.py         # AST scoring + V2→V1 adapter
│   ├── value_stats.py         # Column value statistics for boosting
│   ├── weaviate_fc_utils.py   # Tool schema builders (V1 + V2)
│   ├── schema_cache.py        # Token-aware schema cache
│   ├── learning_cache.py      # Keyword→collection learning
│   └── json_extraction.py     # Robust JSON extraction from LLM output
├── models.py                  # Pydantic models (WeaviateQuery, V2 types)
└── test_gorilla/
    └── test_enhanced_pipeline.py  # Pipeline class + evaluation harness

eval/
├── scripts/
│   ├── bird_ex_v2.py          # BIRD execution accuracy (with gold-guided mode)
│   ├── blind_llm_baseline.py  # Blind baseline (all schemas, no discovery)
│   └── ...                    # Additional scoring/analysis scripts
├── eai/                       # EAI/MongoDB evaluation
└── paper_comparison/          # Cross-system comparison analysis
```

## Pipeline Components

| Component | File | Purpose |
|-----------|------|---------|
| **MCPServer** | `src/mcp/server.py` | Schema registry — loads 143 collections, provides search/lookup |
| **Embeddings** | `src/utils/embeddings.py` | Local all-MiniLM-L6-v2 for semantic similarity |
| **Knowledge Graph** | `src/utils/field_kg.py` | FK/shared-key edges, multi-hop traversal, adaptive depth |
| **Value Stats** | `src/utils/value_stats.py` | Column value matching for discovery boosting |
| **LMService** | `src/lm/lm.py` | Multi-provider LLM client (Bedrock, OpenAI, Ollama) |
| **Tool Schema** | `src/utils/weaviate_fc_utils.py` | V2 structured output schema (filters, output_properties, order_by) |
| **Sandbox** | `src/mcp/sandbox.py` | Validates generated queries against schema |
| **AST Scoring** | `src/utils/ast_scoring.py` | Query structure scoring with V2→V1 auto-adapter |

## Supported LLMs

Tested with:
- **Claude Sonnet 4.6** via AWS Bedrock
- **Qwen 3 80B** via AWS Bedrock
- **Llama 4 Maverick 17B** via AWS Bedrock
- **openai/gpt-oss-120b** via NVIDIA NIM

The pipeline is model-agnostic — any LLM supporting tool-call/function-calling works.

## Evaluation

See `eval/paper_comparison/DEEP_DIVE_ANALYSIS.md` for comprehensive results including:
- Per-database accuracy breakdowns
- Failure taxonomy and confusion pairs
- Token efficiency analysis
- Ablation study (KG, embeddings, reranking, etc.)
- BIRD EX with gold-guided translator fixes
- Cross-system DAIL-SQL/DIN-SQL/CHESS comparison
