# Architecture

## Pipeline Flow

```
                    Natural Language Query
                            │
                ┌───────────┴───────────┐
                │    Phase 1: DISCOVERY  │
                │                       │
                │  1. Embed query        │  all-MiniLM-L6-v2 (local)
                │  2. Score vs 143 colls │  cosine similarity
                │  3. KG boost           │  FK/shared-key typed edges
                │  4. Property rerank    │  word overlap with query
                │  5. Value boost        │  data value matching
                │  → Top-5 candidates    │
                └───────────┬───────────┘
                            │
                ┌───────────┴───────────┐
                │   Phase 2: GENERATION  │
                │                       │
                │  Build prompt:         │
                │   - System (V2 rules)  │
                │   - Top-5 schemas      │
                │   - Question           │
                │                       │
                │  LLM tool call → JSON  │  1 API call
                │                       │
                │  Sandbox validation    │  property/type checks
                │  → Structured query    │
                └───────────────────────┘
```

## Core Classes

### MCPServer (`src/mcp/server.py`)
Schema registry. Loads collections from JSON files, provides search and lookup.

```python
server = MCPServer.from_multiple_sources("data/weaviate-gorilla.json", "data/bird-benchmark/bird-collections.json")
server.search_collections("restaurant")    # → ['Restaurants', ...]
server.get_collection_schema("Restaurants") # → {description, properties}
server.list_collections()                   # → all 143 names
```

### FieldLevelKnowledgeGraph (`src/utils/field_kg.py`)
Builds a graph of relationships between collections using three edge types:
- **FK_REFERENCE** (weight 0.0): Foreign key links — counterintuitively hurts discovery by boosting confusable siblings
- **SHARED_KEY** (weight 0.2): Same property name across collections — helps cross-DB disambiguation
- **NAME_PATTERN** (weight 0.2): Naming convention links — small positive signal

Uses Union-Find for database group detection (48 groups from 143 collections).
Adaptive depth routing: skip KG for easy queries (score gap > 0.3), full multi-hop for hard ones.

### LMService (`src/lm/lm.py`)
Multi-provider LLM client. Supports:
- **Bedrock**: Claude, Qwen, Llama via AWS Converse API
- **OpenAI/NIM**: GPT models or NVIDIA NIM via OpenAI SDK
- **Ollama**: Local models

Tracks per-call metadata: `lm.last_call_meta` → `{latency_s, input_tokens, output_tokens}`.

### EnhancedPipeline (`src/test_gorilla/test_enhanced_pipeline.py`)
Integrates all components. Main methods:
- `discover(query, top_k=5)` → ranked collection names
- `generate(lm, question, candidates)` → `{collection, query_args, tokens, ...}`

### V2 Tool Schema (`src/utils/weaviate_fc_utils.py`)
Defines the structured output format the LLM fills:
```json
{
  "collection_name": "...",
  "output_properties": [{"property_name": "...", "aggregation": "COUNT"}],
  "filters": [{"property_name": "...", "operator": ">", "value": 100}],
  "order_by": [{"property_name": "...", "direction": "DESC"}],
  "limit": 10,
  "distinct": true,
  "additional_collections": [...],
  "join_keys": [...]
}
```

## Data Flow

```
data/weaviate-gorilla.json ─┐
data/bird-benchmark/*.json ──┼─→ MCPServer (143 collections)
                             │         │
                             │    FieldLevelKG (752 edges, 48 DB groups)
                             │         │
                             └─→ Embeddings (143 vectors, all-MiniLM-L6-v2)
                                       │
                          User query ───┤
                                       │
                              ┌────────┴────────┐
                              │ EnhancedPipeline │
                              │   discover()    │──→ top-5 candidates
                              │   generate()    │──→ structured query JSON
                              └─────────────────┘
```

## Evaluation Architecture

Three evaluation modes:
1. **Collection Accuracy**: Does the pipeline pick the right schema? (strict/relaxed)
2. **AST Score**: Is the full query structure correct? (Weaviate Gorilla only, has ground truth)
3. **BIRD EX**: Translate JSON→SQL, execute, compare rows with gold (cross-domain reference)

The BIRD EX path includes a `--gold-guided` mode that applies two deterministic
translator fixes (trim extra columns, add DISTINCT) for honest loss attribution.
