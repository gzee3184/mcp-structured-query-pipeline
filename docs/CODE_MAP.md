# Code Map

Quick reference for navigating the codebase. 17 Python files total.

## Entry Points (what you actually run)

| Script | Purpose | Usage |
|--------|---------|-------|
| `src/test_gorilla/test_enhanced_pipeline.py` | **Main pipeline + evaluation** | `python src/test_gorilla/test_enhanced_pipeline.py --n 50 --split test --provider bedrock --model anthropic.claude-sonnet-4-6 --tool-schema v2` |
| `eval/scripts/blind_llm_baseline.py` | Blind baseline (all schemas, no discovery) | `python eval/scripts/blind_llm_baseline.py --n 50 --split test --provider bedrock --model anthropic.claude-sonnet-4-6` |
| `eval/scripts/bird_ex_v2.py` | BIRD execution accuracy scorer | `python eval/scripts/bird_ex_v2.py <result.json> [--gold-guided]` |
| `eval/paper_comparison/analyze.py` | Cross-system comparison report | `python eval/paper_comparison/analyze.py` |

## Pipeline Core (src/)

```
src/
├── test_gorilla/
│   └── test_enhanced_pipeline.py   ← EnhancedPipeline class + CLI harness
│                                      discover() = Phase 1
│                                      generate() = Phase 2
│                                      main()     = evaluation loop
├── mcp/
│   ├── server.py                   ← MCPServer: loads 143 schemas, search/lookup
│   └── sandbox.py                  ← StructuredSandbox: validates query against schema
│                                      SemanticValidator: LLM-as-judge (experimental)
├── lm/
│   ├── lm.py                      ← LMService: Bedrock/OpenAI/Ollama LLM client
│   │                                  last_call_meta tracks latency + tokens
│   └── db_gorilla_prompts.py      ← System prompt text (V1 + V2)
├── utils/
│   ├── field_kg.py                ← Knowledge graph (752 edges, 48 DB groups)
│   │                                  typed edges: FK(0.0), SharedKey(0.2), NamePattern(0.2)
│   │                                  adaptive depth: skip KG for easy queries
│   ├── embeddings.py              ← Local sentence-transformers (all-MiniLM-L6-v2)
│   ├── ast_scoring.py             ← AST scoring + v2_to_v1_format auto-adapter
│   ├── weaviate_fc_utils.py       ← V1 + V2 tool schema builders
│   ├── value_stats.py             ← Column value statistics for discovery boosting
│   ├── schema_cache.py            ← Token-aware LRU cache for schemas
│   ├── learning_cache.py          ← Keyword→collection learning from failures
│   └── json_extraction.py         ← Robust JSON extraction from LLM responses
└── models.py                      ← Pydantic models (WeaviateQuery, V2 types)
```

## Evaluation Scripts (eval/)

```
eval/
├── scripts/
│   ├── bird_ex_v2.py              ← JSON→SQL translator + SQLite execution + row comparison
│   │                                  --gold-guided mode: trim cols + add DISTINCT
│   ├── blind_llm_baseline.py      ← Give LLM all 143 schemas, measure accuracy
│   ├── compute_output_match_v2.py ← V2 component-level scoring (SELECT, WHERE, ORDER BY)
│   ├── simulated_exec_v2.py       ← Simulated execution (no actual SQL execution)
│   ├── sqlite_oracle.py           ← Run gold SQL to verify correctness
│   ├── build_bird_name_map.py     ← Build Weaviate→SQL column name mapping
│   └── generate_figures.py        ← Paper figure generation
├── paper_comparison/
│   ├── analyze.py                 ← 10-section cross-system analysis report
│   ├── run_all.sh                 ← Orchestrate all evaluation runs
│   ├── DEEP_DIVE_ANALYSIS.md      ← Comprehensive results + reviewer walkthrough
│   ├── COMPARISON_RESULTS.md      ← Auto-generated comparison tables
│   └── RESULTS_EXPLAINED.md       ← Outsider-friendly results guide
├── eai/                           ← EAI/MongoDB evaluation (direct MQL generation)
└── configs/                       ← Model registry + experiment matrix
```

## Data Dependencies

Pipeline needs these files (not in repo — download separately):
- `data/weaviate-gorilla.json` — 315 Weaviate queries with AST ground truth
- `data/bird-benchmark/dev_20240627/dev.json` — 1,534 BIRD dev queries
- `data/bird-benchmark/bird-collections.json` — 143 Weaviate-format collection schemas
- `data/bird-benchmark/dev_20240627/dev_databases/` — 11 SQLite DBs (for BIRD EX)
