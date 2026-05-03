# Evaluation Metrics

## Discovery Rate

Can the pipeline find the correct schema before the LLM runs?

```
Discovery Rate = (queries where correct collection ∈ top-5 candidates) / total
```

If discovery fails, the LLM never sees the right schema. This is the pipeline's foundation.

**Results:** 100% WG, 97.9% BIRD, 91.5% EAI

## Collection Accuracy (Strict)

Does the LLM pick the correct primary collection?

```
Strict = (predicted primary == gold FROM table) / total
```

In JOIN queries, BIRD picks one table as "primary" (the FROM table). This choice is
sometimes arbitrary — the same query could reasonably use either JOIN participant.

**Result:** 57.5% on BIRD

## Collection Accuracy (Relaxed)

Does the LLM pick ANY table involved in the query?

```
Relaxed = (predicted primary ∈ {FROM + all JOIN tables}) / total
```

Relaxed better reflects real-world utility. The 57.5%→93.1% gap is entirely from
JOIN-table ambiguity, not discovery failures.

**Result:** 93.1% on BIRD

## AST Score (Weaviate Gorilla only)

Is the full query structure correct? Weighted component match:

| Component | Weight | Scoring |
|-----------|--------|---------|
| Collection name | 1.0 | Exact match (wrong = total 0) |
| Each filter | 1.0 | property (0.5) + operator (0.25) + value (0.25) |
| Each aggregation | 1.0 | property (0.5) + metrics IoU (0.5) |
| search_query | 0.5 | Word-overlap Jaccard |
| groupby_property | 0.5 | Exact match |
| total_count | 0.25 | Boolean match |

Only components present in ground truth are scored.

**Result:** 86.1% mean (V1 and V2 score identically via auto-adapter)

## BIRD Execution Accuracy (EX)

Does the translated SQL return the correct rows?

```
EX = (queries where execute(predicted_SQL) == execute(gold_SQL)) / total
```

Our pipeline outputs JSON, not SQL. We translate for BIRD evaluation only.
Gold-guided mode applies two deterministic translator fixes.

| Mode | Score | Fixes Applied |
|------|-------|---------------|
| Standard | 35.6% | None |
| Gold-guided | 45.3% | Trim extra cols + add DISTINCT |

## OPS (EAI/MongoDB)

```
OPS = geometric_mean(SE, COF)
  SE = Structural Equivalence (query shape match)
  COF = Condition Overlap F1 (filter predicate match)
```

**Result:** 0.723 (with discovery), 0.749 (oracle discovery)

## Token Measurement

- **input_tokens**: From API provider (`response.usage.inputTokens`) — real cost
- **output_tokens**: From API provider — LLM generation cost
- **tokens** (legacy): `len(prompt) / 4` — rough estimate, used in early work only
