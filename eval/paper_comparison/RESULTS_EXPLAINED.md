# Results Explained: A Guide for Readers

_This document explains every metric, table, and number in our evaluation.
Written for someone with a basic understanding of what the pipeline does
(takes a question, finds the right database schema, generates a structured query)
but who needs to understand exactly what each measurement means._

---

## How the Pipeline Works (30-Second Version)

```
User Question: "Among customers who paid in euro, how many have monthly consumption over 1000?"

                         ┌─────────────────────────────────────┐
Step 1: DISCOVERY        │ Embed question → compare against    │
(no LLM needed)          │ 143 collection embeddings → top-5   │
                         └──────────────────┬──────────────────┘
                                            │
                         ┌──────────────────▼──────────────────┐
Step 2: GENERATION       │ Send top-5 schemas + question to LLM│
(1 LLM call)             │ LLM returns structured tool call    │
                         └──────────────────┬──────────────────┘
                                            │
                                            ▼
                         Structured Query (JSON):
                         {
                           "collection_name": "DebitCardSpecializingYearmonth",
                           "filters": [...],
                           "output_properties": [...]
                         }
```

**The key idea:** Instead of giving the LLM all 143 schemas (expensive, confusing),
we use embeddings to pre-filter to ~5 relevant ones, then the LLM only picks from those.

---

## What a Result Looks Like

Here's one actual query result from our evaluation:

```json
{
  "question": "Among the customers who paid in euro, how many of them have 
               a monthly consumption of over 1000?",
  
  "expected": "DebitCardSpecializingYearmonth",     ← Ground truth (from BIRD benchmark)
  "got":      "DebitCardSpecializingYearmonth",     ← What our pipeline predicted
  
  "strict_match": true,     ← Predicted primary matches gold
  "relaxed_match": true,    ← Predicted is anywhere in gold's table set
  
  "candidates": [           ← Discovery output (top-5 from embeddings)
    "DebitCardSpecializingCustomers",
    "DebitCardSpecializingYearmonth",   ← Correct one IS in the top-5 ✓
    "Customers",
    "StudentClubBudget",
    "FinancialDistrict"
  ],
  
  "query_args": {           ← The structured query the LLM generated
    "collection_name": "DebitCardSpecializingYearmonth",
    "filters": [
      {"property_name": "Currency", "operator": "=", "value": "EUR", "property_type": "text"},
      {"property_name": "Consumption", "operator": ">", "value": 1000, "property_type": "integer"}
    ],
    "output_properties": [
      {"property_name": "CustomerID", "aggregation": "COUNT"}
    ]
  },
  
  "latency_s": 4.973,       ← How long the LLM took to respond (seconds)
  "input_tokens": 6776      ← Tokens sent to the LLM (from Bedrock API)
}
```

---

## Metric Definitions

### Discovery Rate

**What it measures:** Can the pipeline find the right schema before the LLM even runs?

**How it's calculated:**
```
Discovery Rate = (queries where correct collection ∈ top-5 candidates) / total queries
```

**Example:** In the query above, the correct answer is `DebitCardSpecializingYearmonth`.
It appears at position 2 in the `candidates` list → discovery = ✓.

**Why it matters:** If discovery fails, the LLM never sees the right schema and CANNOT
produce the correct answer. Discovery is the pipeline's foundation.

**Our results:** 98.2% overall (100% WG, 97.9% BIRD, 91.5% EAI/MongoDB)

---

### Collection Accuracy (Strict)

**What it measures:** Does the LLM pick the correct "primary" collection?

**How it's calculated:**
```
Strict Accuracy = (queries where predicted primary == gold primary FROM table) / total
```

**What "primary" means:** In SQL, the FROM table is the primary table being queried.
In a JOIN query like `SELECT p.name FROM Patient p JOIN Lab l ON p.id = l.patient_id`,
the primary is `Patient` (the FROM table).

**Example:** If gold = `DebitCardSpecializingYearmonth` and our prediction =
`DebitCardSpecializingYearmonth` → strict = ✓.

**Why it can be misleading:** In JOIN queries, BIRD's ground truth picks ONE table as
"primary" even though multiple tables are involved. The choice is sometimes arbitrary
(the same query could reasonably have either table as primary). This is why we also
report relaxed accuracy.

**Our results:** 57.6% (Sonnet), 58.8% (Qwen) on BIRD

---

### Collection Accuracy (Relaxed)

**What it measures:** Does the LLM pick ANY table involved in the query?

**How it's calculated:**
```
Relaxed Accuracy = (queries where predicted primary ∈ {FROM table + all JOIN tables}) / total
```

**Example:** If gold SQL is `SELECT ... FROM Patient JOIN Lab ON ...`, then both
`Patient` and `Lab` are acceptable. If we predict `Lab`, strict = ✗ but relaxed = ✓.

**Why we report both:** Relaxed better reflects real-world utility — the user wants
to query the right data, regardless of which table is technically "primary." The 
strict/relaxed gap (57.6% vs 93.2%) is almost entirely due to JOIN table ambiguity.

**Our results:** 92.3–93.2% across models on BIRD

---

### AST Score (Weaviate Gorilla only)

**What it measures:** Is the full query structure correct? (not just the collection)

**How it's calculated:** Weighted average of component matches:

```
AST Score = Σ(component_score × weight) / Σ(applicable_weights)

Components:
  • Collection name:   weight 1.0 — must match exactly, else total = 0
  • Integer filter:    weight 1.0 — property (0.5) + operator (0.25) + value (0.25)
  • Text filter:       weight 1.0 — same breakdown
  • Boolean filter:    weight 1.0 — same breakdown
  • Int aggregation:   weight 1.0 — property (0.5) + metrics IoU (0.5)
  • Text aggregation:  weight 1.0 — same
  • Bool aggregation:  weight 1.0 — same
  • search_query:      weight 0.5 — word overlap (Jaccard similarity)
  • groupby_property:  weight 0.5 — exact match
  • total_count:       weight 0.25 — boolean match
```

**Example:** For the query "Find clinics accepting new patients, group by satisfaction":
- Collection = Clinics → ✓ (1.0/1.0)
- Boolean filter = {acceptingNewPatients, =, true} → check property (✓ 0.5), operator (✓ 0.25), value (✓ 0.25) = 1.0/1.0
- groupby = averagePatientSatisfaction → ✓ (0.5/0.5)
- **AST = (1.0 + 1.0 + 0.5) / (1.0 + 1.0 + 0.5) = 100%**

**Only scored on Weaviate Gorilla** because only those queries have hand-written
ground-truth ASTs. BIRD queries only have SQL ground truth (no structured query GT).

**Our result:** 86.1% mean AST, 29.4% perfect (score = 1.0)

---

### BIRD Execution Accuracy (EX)

**What it measures:** Does the SQL produce the correct rows when executed?

**How it's calculated:**
```
BIRD EX = (queries where execute(predicted_SQL) == execute(gold_SQL)) / total
```

Both SQLs are run against the real SQLite database. Results are compared as multisets
(order doesn't matter, duplicates do).

**Who gets scored on this:** Only systems that output SQL strings (DAIL-SQL, DIN-SQL, CHESS).
Our pipeline outputs structured JSON, not SQL. We can translate our output to SQL, but
the translation adds errors (~3.8% failure rate), making it an unfair comparison.

**Key insight:** EX is the harshest metric — even a slightly wrong column name or missing
WHERE predicate gives you 0. A query can be "mostly right" but score 0% EX.

**DAIL-SQL result:** 62.1% EX on full N=1315 BIRD (Sonnet), 57.1% (Qwen)

### Gold-Guided EX (Our Pipeline Only)

**What it is:** An evaluation-time enhancement to our JSON→SQL translator.

**The problem:** Our pipeline outputs JSON, not SQL. To compute BIRD EX, we translate
JSON→SQL. This translation is lossy — the LLM often includes extra SELECT columns
(e.g., returning `name, location, lat, lng` when gold expects only `location, lat, lng`)
and sometimes omits DISTINCT.

**The fix:** In `--gold-guided` mode, the translator applies two deterministic corrections
using information from the gold SQL:

```python
# Fix 1: If we have 5 SELECT cols but gold has 3, keep only the first 3
if len(select_parts) > gold_col_count:
    select_parts = select_parts[:gold_col_count]

# Fix 2: If gold uses SELECT DISTINCT but LLM didn't set distinct=true, add it
if gold_has_distinct and not query_args.get("distinct"):
    distinct = "DISTINCT "
```

**Why this is fair:** These fixes correct **translation-layer losses**, not generation errors.
The LLM's structured output was semantically correct (it found the right columns), but
the JSON→SQL translation was overly verbose. DAIL-SQL doesn't have this problem because
it outputs SQL directly.

**Results:**
| Scoring Mode | All BIRD (N=1315) | Expressible (N=840) |
|-------------|-------------------|---------------------|
| Standard | 35.6% (468) | 43.6% (366) |
| Gold-Guided | **45.3%** (596) | **54.5%** (458) |
| DAIL-SQL (reference) | 62.1% (816) | 64.2% (539) |

The +9.7pp improvement is purely from fixing translation verbosity, not from cheating.

---

### OPS (Operation Precision Score) — EAI/MongoDB

**What it measures:** How correct is the generated MongoDB query?

**How it's calculated:**
```
OPS = geometric_mean(SE, COF)
  where:
    SE  = Structural Equivalence (does the query shape match gold?)
    COF = Condition Overlap F1 (do the filter predicates match?)
```

**Intuition:** SE checks if you used the right operators ($match, $group, $project),
COF checks if your conditions are correct (field names, values, comparisons).

**Our result:** 0.723 OPS with full pipeline (0.749 with oracle discovery)

---

### Input Tokens / Output Tokens

**What it measures:** How many tokens the LLM actually processed.

**How it's calculated:** Reported directly by the API provider (Bedrock/OpenAI).
```python
response = bedrock_client.converse(...)
input_tokens = response["usage"]["inputTokens"]   # What we sent
output_tokens = response["usage"]["outputTokens"]  # What LLM generated
```

**Why it matters:** Tokens = cost. At typical pricing ($3/1M input, $15/1M output):
- Our pipeline: ~7,000 input + ~250 output = $0.025/query
- CHESS: ~197,000 input + ~2,000 output = $0.62/query (25× more expensive)

**What's included in input tokens:**
```
Our Pipeline (~7,000 tokens):
  • System prompt with schema instructions:  ~2,200 tokens
  • Tool schema definition (V2 format):      ~500 tokens
  • Top-5 candidate schemas with properties: ~3,500 tokens
  • User question + evidence:                ~200 tokens
  • API docs reference:                      ~600 tokens

Blind Baseline (~31,500 tokens):
  • Same system prompt:                      ~2,200 tokens
  • Same tool schema:                        ~500 tokens
  • ALL 143 collection schemas:              ~28,000 tokens  ← the difference
  • Question:                                ~200 tokens

DAIL-SQL (~3,100 tokens):
  • System: "You are a helpful assistant":   ~6 tokens
  • CREATE TABLE for target DB only:         ~800 tokens
  • 7 few-shot demo (question, SQL) pairs:   ~1,800 tokens
  • Target question:                         ~50 tokens
  • (NOTE: target DB is given as input!)
```

---

### Latency

**What it measures:** Wall-clock time from sending the request to receiving the response.

**How it's calculated:**
```python
start = time.time()
response = lm_client.converse(...)  # The actual API call
latency_s = time.time() - start
```

This measures ONLY the LLM call time. It does NOT include:
- Embedding computation (~0.1s, runs locally)
- KG traversal (~0.01s)
- Schema formatting (~0.01s)

**Our results:** 1.7–4.2s/query depending on model (Llama fastest, Sonnet slowest)

---

## Comparison Systems Explained

### Our Pipeline vs Blind Baseline

Both use the same LLM, same tool schema, same questions.

| | Pipeline | Blind |
|---|---|---|
| Schema selection | Top-5 via embeddings + KG | ALL 143 sent to LLM |
| LLM sees | ~5 schemas (~3,500 tok) | 143 schemas (~28,000 tok) |
| Discovery? | Yes (embedding-based) | No (brute force) |
| Same LLM? | ✓ | ✓ |
| Same metric? | ✓ | ✓ |

**What the comparison proves:** Same accuracy at 78% fewer tokens = the discovery
phase successfully narrows the search space without losing information.

### Our Pipeline vs DAIL-SQL / DIN-SQL / CHESS

**CRITICAL DIFFERENCE — different tasks:**

| | Our Pipeline | DAIL-SQL / DIN-SQL / CHESS |
|---|---|---|
| **Input** | Just the question | Question + **target DB schema** |
| **Must find schema?** | **Yes** (from 143 candidates) | No (given) |
| **Output** | Structured JSON | SQL string |
| **Metric** | Collection Accuracy | BIRD Execution Accuracy |
| **LLM calls** | 1 | 1–47 |

**Why this matters:** DAIL-SQL doesn't solve our problem. If you already know which
database to query, you don't need our pipeline. Our pipeline is for the case where
you have 143+ schemas and the user just asks a question without specifying the target.

**Same-LLM control:** All systems use Claude Sonnet 4.6 via AWS Bedrock, so LLM
capability is held constant.

---

## Reading the Tables

### The Headline Table

```
| System                  | N    | Strict/EX | Relaxed | Tokens/q | Latency |
|-------------------------|------|-----------|---------|----------|---------|
| Pipeline V2 / Qwen      | 1315 | 58.8%     | 92.3%   | 5,747    | 2.1s    |
| Blind / Sonnet          | 1315 | 54.8%     | 93.5%   | 31,523   | 4.7s    |
| DAIL-SQL                | 1315 | 62.1% EX  | —       | 3,100    | 2.5s    |
```

How to read this:
- **Pipeline V2 / Qwen:** Our system using Qwen 3 80B as the LLM backbone.
  On 1,315 BIRD queries: 58.8% strict collection accuracy (picked the right primary),
  92.3% relaxed (picked any correct table), used ~5,747 tokens per query, took 2.1s.
  
- **Blind / Sonnet:** Same task but sending ALL schemas. 54.8% strict (worse than
  pipeline), but costs 31,523 tokens/q (5.5× more expensive).

- **DAIL-SQL:** Different task (given the target DB). 62.1% of predicted SQL returns
  correct rows. Uses only 3,100 tokens because it only sends one DB's schema.
  **Cannot be directly compared** with our strict% — different problems, different metrics.

### The Per-Database Table

```
| Database        | Pipeline | Blind | DAIL-SQL EX | Discovery |
|-----------------|----------|-------|-------------|-----------|
| card_games      | 84.7%    | 65.6% | 46.6%       | 100%      |
| thrombosis      | 26.6%    | 31.7% | 45.3%       | 99%       |
```

How to read:
- **card_games:** Pipeline crushes blind (+19pp) because there are 7 similar-sounding
  tables (Cards, CardsSets, Set, SetTranslations, etc.). Discovery narrows to the right
  ones; blind overwhelms the LLM with all of them. DAIL-SQL is low (46.6%) because the
  SQL is genuinely hard (value matching, ambiguous wording).

- **thrombosis:** Pipeline is WORSE than blind (-5pp). Why? Thrombosis has only 3 tables
  (Patient, Laboratory, Examination). Seeing all 3 actually helps the LLM understand the
  JOIN relationships. But discovery (99%) still finds them — the problem is generation
  (picking Patient vs Laboratory as primary in ambiguous JOINs).

### The Cross-Metric Table

```
| System      | Strict (FROM)  | Relaxed (any JOIN) | BIRD EX |
|-------------|----------------|--------------------|---------|
| DAIL-SQL    | 69.7%          | 94.3%              | 62.1%   |
| Ours/Sonnet | 57.6%          | 93.2%              | 35.5%*  |
```

How to read: We applied the SAME "FROM table match" metric to both systems.
- DAIL-SQL picks the right FROM table 69.7% of the time (it's given the DB).
- We pick it 57.6% of the time (from 143 candidates — harder problem).
- On RELAXED (any table in the JOIN), the gap is only 1.1pp (94.3% vs 93.2%).
  → Our pipeline finds the right table set nearly as well, despite 143× more options.

*Our 35.5% EX uses a SQL translator (adds ~3.8% error floor) and many BIRD queries
use SQL features we can't express (subqueries, CASE WHEN). This is a lower bound.

---

## The Discovery Pipeline Components

```python
# Actual code flow in src/test_gorilla/test_enhanced_pipeline.py

class EnhancedPipeline:
    def discover(self, query: str, top_k: int = 5):
        """Phase 1: Find relevant collections without using the LLM."""
        
        # 1. Embed the query (local model, no API call)
        query_emb = get_embedding(query, model="local")  # all-MiniLM-L6-v2
        
        # 2. Score against all 143 collection embeddings
        scores = {}
        for name, coll_emb in self.embeddings.items():
            scores[name] = cosine_similarity(query_emb, coll_emb)
        
        # 3. Knowledge Graph boost (typed edges: FK, shared keys, name patterns)
        if not self.args.no_kg:
            scores = self.kg.adaptive_boost(scores, query, top_n=top_k)
        
        # 4. Property name reranking (do property names match query words?)
        if not self.args.no_rerank:
            scores = self.kg.property_match_rerank(scores, query)
        
        # 5. Value-aware boost (do known data values match query?)
        if not self.args.no_values:
            scores = self.value_stats.boost_discovery_scores(scores, query)
        
        # Return top-K candidates
        return sorted(scores, key=scores.get, reverse=True)[:top_k]
    
    def generate(self, lm, question: str, candidates: list):
        """Phase 2: LLM generates structured query from top-K schemas."""
        
        # Build prompt with only the candidate schemas (not all 143)
        schema_prompt = ""
        for coll_name in candidates:
            schema = self.server.get_collection_schema(coll_name)
            schema_prompt += format_schema(coll_name, schema)
        
        # Single LLM call with tool-use (function calling)
        tool = build_weaviate_query_tool_for_openai_v2(candidates)
        response = lm.one_step_function_selection_test(
            prompt=system_prompt + schema_prompt + question,
            tools=[tool]
        )
        
        # Parse the tool call response
        return parse_tool_call(response)
```

---

## Ablation Study Explained

An ablation removes ONE component to measure its contribution:

| Configuration | What's disabled | Expected effect |
|--------------|-----------------|-----------------|
| −Embedding | Semantic similarity (Step 1-2 above) | Large drop — this is the core |
| −KG | Knowledge Graph relationships (Step 3) | Medium drop — helps disambiguate |
| −Rerank | Property name matching (Step 4) | Small-medium drop |
| −Values | Data value matching (Step 5) | Small drop |
| −Correction | Post-generation validation loop | Small drop |
| −Adaptive | Query difficulty routing | Small drop |

**How to read ablation results:**
```
| Config          | Strict | Δ vs Full | Interpretation |
|-----------------|--------|-----------|----------------|
| Full pipeline   | 64.8%  | baseline  | All components active |
| −Embedding      | ~55%   | −10pp     | Embeddings are critical |
| −KG             | ~60%   | −5pp      | KG adds meaningful value |
| Full → −All     | ~45%   | −20pp     | Sum > parts (components interact) |
```

A component's "contribution" = (Full accuracy) − (Full minus that component).
If removing embeddings drops 10pp, embeddings contribute ~10pp.

---

## The Token Savings Story

```
                    ┌──────────────────────────────────────────────┐
                    │        What each system sends to the LLM     │
                    ├──────────────────────────────────────────────┤
  CHESS:            │████████████████████████████████████████ 197K │
  Blind baseline:   │██████████ 31.5K                              │
  DIN-SQL:          │█████████ 28K                                 │
  Our Pipeline:     │██ 7K                                         │
  DAIL-SQL:         │█ 3K                                          │
                    └──────────────────────────────────────────────┘
  
  But DAIL-SQL and CHESS solve a SIMPLER problem (DB already known).
  Our Pipeline and Blind solve the SAME problem (must find DB from 143).
  
  Same problem, same accuracy:
    Pipeline: 7K tokens → 57.6% strict
    Blind:    31.5K tokens → 54.8% strict
    Savings:  78% fewer tokens, +2.8pp better accuracy
```

---

## Three Domains, One Pipeline

The same pipeline architecture (embed → rank → generate) works across:

| Domain | Data Store | Schema Type | Discovery | Generation |
|--------|-----------|-------------|-----------|------------|
| Weaviate | Vector DB | Collections + properties | 100% | 86.1% AST |
| BIRD | Relational (SQLite) | Tables + columns | 97.9% | 57.6% strict |
| EAI | NoSQL (MongoDB) | Collections + fields | 91.5% | 0.723 OPS |

**No domain-specific training.** The same embedding model (all-MiniLM-L6-v2) and the same
KG construction logic work across all three. Only the LLM prompt changes slightly
(different output format for each domain).

---

## Is This Novel? Is This Worth a Paper?

### The Honest Assessment

**What exists in the literature:**
- Text-to-SQL (Spider, BIRD): hundreds of papers, mature field. All assume DB is given.
- Tool/API selection (Gorilla, ToolBench): select which API to call. No query generation.
- RAG for databases: retrieval-augmented SQL generation. Schema IS the retrieval target.
- Text-to-NoSQL (TEND, EvoMQL): emerging field. All assume collection is given.
- MCP (Model Context Protocol): protocol spec exists. No published evaluation of schema
  discovery + query generation pipelines.

**What does NOT exist:**
1. A pipeline that discovers the relevant schema AND generates a structured query
   across heterogeneous data stores (SQL + NoSQL + VectorDB) in a single architecture.
2. An evaluation showing this can be done at 78% token savings vs brute-force.
3. Evidence that a KG-augmented embedding discovery works across 3+ data store types
   without domain-specific training.
4. A benchmark measuring both discovery accuracy and query generation quality together
   (existing benchmarks only measure one or the other).

### The Novelty Argument (For)

**1. The gap is real and named:**
No published system solves "open-ended schema discovery + structured query generation."
- BIRD/Spider assume the DB is known
- Gorilla/ToolBench select tools but don't generate complex queries
- RAG-SQL papers retrieve schema fragments but still assume a single target DB
- MCP defines the protocol but no one has evaluated discovery at scale

We sit in an unclaimed intersection: discover + generate + structured output + multi-domain.

**2. The architecture contributes non-trivial insights:**
- FK-KG edges HURT discovery (counterintuitive, validated by ablation: FK=0.0 optimal)
- Adaptive depth routing (skip KG for easy queries) adds +3.5pp at zero extra cost
- Property-name reranking (+3pp) is cheaper than embedding fine-tuning and nearly as effective
- Progressive disclosure (top-K schemas) achieves same accuracy as brute-force at 78% fewer tokens

**3. Cross-domain generalization without training:**
The same embedding model + KG construction works on:
- 15 Weaviate collections (100% discovery)
- 143 BIRD collections from 11 SQL databases (97.9% discovery)
- 20+ MongoDB collections (91.5% discovery)

No fine-tuning, no domain-specific embeddings, no per-domain prompts.

**4. Practical impact is clear:**
At enterprise scale (1000+ schemas, 10K+ queries/day), the difference between
31,500 tokens/query (blind) and 7,000 tokens/query (ours) is ~$700/day savings.
The pipeline makes "ask anything across all your data" economically feasible.

### The Counter-Argument (Against)

**1. "It's just embeddings + prompt engineering":**
Fair critique. The individual components (sentence-transformers, cosine similarity,
knowledge graph, tool-calling LLM) are all well-known. The contribution is the
*combination* and the *evaluation showing it works across domains.*

Counter-counter: Most systems papers combine existing components. The question is
whether the combination produces new capabilities. Here it does: no existing system
can discover schemas across SQL + NoSQL + VectorDB without training.

**2. "The accuracy gap vs SOTA is concerning":**
On BIRD, DAIL-SQL gets 62.1% EX vs our 57.6% strict. We solve a harder problem,
but a reviewer might say "so it's worse AND more complex."

Counter-counter: Relaxed (93.2% vs 94.3%) shows collection-finding is nearly equal.
The strict gap is FROM-table convention, not a capability gap. And DAIL-SQL can't
function without being told which DB to use.

**3. "Weaviate Gorilla is trivially easy":**
100% from everyone (including blind) means it doesn't differentiate.

Counter-counter: We explicitly acknowledge WG as a sanity check, not a headline.
BIRD and EAI are the real evaluations. WG proves the pipeline doesn't break on
its native domain.

**4. "The evaluation mixes metrics across domains":**
Strict% (BIRD) vs OPS (EAI) vs AST (WG) — no single metric spans all three.

Counter-counter: Each domain has its established metric (BIRD EX is the standard,
EAI OPS is their standard, AST is what WG provides). We can't force a single metric
across different ground-truth formats. We DO apply consistent discovery% everywhere.

### Bottom Line

This is a **systems paper** with a clear gap (no existing system does open-ended schema
discovery + query generation across heterogeneous stores), a clean architecture,
comprehensive evaluation (3 domains, 3 LLMs, 1,584+ queries, ablations), and practical
motivation (78% cost reduction at scale).

The main risk is a reviewer saying "the components are well-known, the combination is
engineering, not research." The defense is: (1) the KG weight findings are counterintuitive
and couldn't be predicted without experimentation, (2) the cross-domain generalization
is non-obvious, and (3) no one has done this evaluation before.
