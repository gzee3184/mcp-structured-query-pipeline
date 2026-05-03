# Pipeline Limitations on BIRD Queries

This document catalogs the specific limitations of our V2 tool schema when evaluated
against the BIRD text-to-SQL benchmark. These are architectural constraints, not bugs —
they define the boundary of what our structured query format can express.

## The Fundamental Constraint

Our pipeline outputs **structured JSON tool calls**, not SQL. The V2 schema can express:
```
collection_name, output_properties[], filters[], order_by[],
group_by_properties[], limit, distinct, join_keys[]
```

It **cannot** express arbitrary SQL. Translating our JSON to SQL for BIRD evaluation
introduces a lossy step that accounts for a significant portion of our EX gap.

## Inexpressible SQL Features (38% of BIRD queries)

| Feature | BIRD Queries | % | Example | Why We Can't Express It |
|---------|-------------|---|---------|------------------------|
| **Arithmetic** | 223 | 17.0% | `col1 * 100 / col2` | No expression language in filters or output |
| **CAST** | 125 | 9.5% | `CAST(price AS REAL)` | No type coercion field |
| **Subqueries** | 113 | 8.6% | `WHERE id IN (SELECT ...)` | No recursive query nesting |
| **strftime** | 97 | 7.4% | `strftime('%Y', date)` | No date extraction functions |
| **CASE WHEN** | 94 | 7.1% | `CASE WHEN x>0 THEN 'yes'` | No conditional expressions |
| **String functions** | 47 | 3.6% | `SUBSTR(name, 1, 3)` | No string manipulation |
| **IIF** | 30 | 2.3% | `IIF(a>b, a, b)` | No inline conditionals |
| **HAVING** | 17 | 1.3% | `HAVING COUNT(*)>5` | V2 has having_filters but rarely used |
| **BETWEEN** | 64 | 4.9% | `WHERE x BETWEEN 1 AND 10` | Expressible as >= AND <= but LLM often doesn't |

A single query can have multiple inexpressible features. 505 unique queries are affected.

### What This Means for BIRD EX

Even with a perfect LLM and perfect translator, our ceiling on BIRD EX is ~62%
(the 64% of queries we CAN express). Current: 45.3% gold-guided = 71% of ceiling.

## Translation Layer Losses

Our JSON→SQL translator (`eval/scripts/bird_ex_v2.py`) introduces additional errors:

| Issue | Count | Impact |
|-------|-------|--------|
| **Extra SELECT columns** | 109 queries | LLM adds columns not in gold SQL. Gold-guided mode trims these. |
| **Missing DISTINCT** | 17 queries | LLM omits `distinct: true`. Gold-guided mode recovers these. |
| **Table scope ambiguity** | ~20 queries | Column exists on 2+ tables; translator picks wrong one |
| **Runtime errors** | 33 queries | SQL syntax errors from edge-case translation |
| **Translation failures** | 7 queries | Can't translate at all (malformed query_args) |

### Gold-Guided Mode

`bird_ex_v2.py --gold-guided` applies two deterministic fixes using gold SQL metadata:
1. Trim SELECT columns to match gold column count
2. Add DISTINCT when gold SQL uses it

This recovers 128 queries (+9.7pp), bringing EX from 35.6% to 45.3%.

These fixes correct **translation-layer verbosity**, not generation errors. The LLM's
structured output had the right data — the SQL translation was overly inclusive.

## Generation Quality Gaps (on Expressible Queries)

Even on the 840 queries our schema CAN express, we miss some due to LLM generation errors:

| Issue | % of Failures | Description |
|-------|---------------|-------------|
| **Extra columns (trim doesn't help)** | 26% | Different column values, not just extra cols |
| **Too few rows (extra filter)** | 11% | LLM adds a WHERE predicate not in gold |
| **Wrong values (same shape)** | 10% | Right structure, wrong filter values |
| **Missing filter** | 2% | LLM omits a WHERE predicate from gold |
| **Runtime errors** | 3% | Translation-layer SQL syntax failures |

## Comparison with DAIL-SQL: Why the 10pp Gap?

On the expressible subset (N=840), our gold-guided EX is 54.5% vs DAIL-SQL's 64.2%.
This 9.7pp gap comes from DAIL-SQL's three structural advantages:

| Advantage | Estimated Impact | Description |
|-----------|-----------------|-------------|
| **7 few-shot demos** | ~5-7pp | kNN retrieves similar solved examples from 9,428 training queries |
| **Known DB scope** | ~3-5pp | DAIL-SQL only sees the target DB; we send 5 candidate schemas |
| **Native SQL output** | ~1pp | DAIL-SQL outputs SQL directly; we translate JSON→SQL |

Our pipeline's zero-shot approach is more **model-robust** (2.3pp variance vs DAIL-SQL's
11.5pp across 3 LLMs), but less precise on BIRD-specific conventions.

## The FROM Convention Problem

BIRD labels one table as "primary" (the FROM table) in JOIN queries. Our pipeline often
picks a different valid table (e.g., the filter table instead of the output table).

- **Impact on strict accuracy:** 36% of queries (473/1315) are "FROM swaps"
- **Impact on BIRD EX:** Only 8.5pp (FROM swap queries score 36.8% EX vs 45.3% for strict-correct)
- **Impact on real-world utility:** None — both tables are valid query targets

This is why we report both **strict** (57.5%) and **relaxed** (93.1%) collection accuracy.
Relaxed better reflects practical correctness.

## Discovery Limitations

| Domain | Discovery Rate | Main Miss Cause |
|--------|---------------|-----------------|
| Weaviate Gorilla | 100% | (no misses) |
| BIRD SQL | 97.9% | Embedding vocabulary mismatch (financial, codebase) |
| EAI MongoDB | 91.5% | Generic collection names (less descriptive than SQL tables) |

19 BIRD queries (1.4%) miss at top-5. These concentrate in:
- `financial` (4 misses) — vocabulary overlap with DebitCard
- `codebase_community` (6 misses) — generic tech terms
- `student_club` (4 misses) — overlaps with other education schemas

## What Would Close the Gap

| Improvement | Estimated Impact | Effort |
|-------------|-----------------|--------|
| **Expand schema** (CAST + arithmetic) | +10-15pp EX | Medium — schema + translator changes |
| **Add few-shot demos** (kNN retrieval) | +5-7pp EX | Medium — build training pool, retriever |
| **Better output discipline** (prompt engineering) | +3pp EX | Low — prompt changes only |
| **Schema linking pre-step** (separate LLM call) | +3-5pp EX | Medium — adds latency |
| **Execution-based selection** (generate + execute + pick) | +2-3pp EX | High — needs SQLite execution loop |
| **Fine-tune embeddings** | +2-5pp discovery | High — needs contrastive training |
