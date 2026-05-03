#!/usr/bin/env python3
"""
Paper Comparison — Comprehensive Analysis Script

Joins per-query results from:
  - Pipeline runs (V1/V2 × 3 LLMs)
  - Blind LLM baselines (× 3 LLMs)
  - SOTA text-to-SQL systems (DIN-SQL, DAIL-SQL, CHESS)
  - BIRD dev metadata (difficulty, gold SQL)

Produces: eval/paper_comparison/COMPARISON_RESULTS.md  (10-section deep dive)

Usage:
    python eval/paper_comparison/analyze.py [--output FILE]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # gorilla_2/gorilla/
RESULTS_DIR = PROJECT_ROOT / "eval" / "paper_comparison" / "results"
SOTA_DIR = Path("/export/scratch/abrar008/llm_rag/sota_comparison/results")
BIRD_DEV = PROJECT_ROOT / "data" / "bird-benchmark" / "dev_20240627" / "dev.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "paper_comparison" / "COMPARISON_RESULTS.md"

# Model short-name mapping (for display)
MODEL_DISPLAY = {
    "sonnet": "Sonnet 4.6",
    "qwen": "Qwen 3 80B",
    "llama": "Llama 4 Maverick",
}

# Source field -> BIRD db_id mapping
SOURCE_TO_DB = {
    "bird_california_schools": "california_schools",
    "bird_card_games": "card_games",
    "bird_codebase_community": "codebase_community",
    "bird_debit_card_specializing": "debit_card_specializing",
    "bird_european_football_2": "european_football_2",
    "bird_financial": "financial",
    "bird_formula_1": "formula_1",
    "bird_student_club": "student_club",
    "bird_superhero": "superhero",
    "bird_thrombosis_prediction": "thrombosis_prediction",
    "bird_toxicology": "toxicology",
}

DB_SHORT = {
    "california_schools": "CA Schools",
    "card_games": "Card Games",
    "codebase_community": "Codebase",
    "debit_card_specializing": "Debit Card",
    "european_football_2": "Euro Football",
    "financial": "Financial",
    "formula_1": "Formula 1",
    "student_club": "Student Club",
    "superhero": "Superhero",
    "thrombosis_prediction": "Thrombosis",
    "toxicology": "Toxicology",
}

ALL_DBS = sorted(DB_SHORT.keys())

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_json(path: Path | str) -> Any:
    """Load JSON, return None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [WARN] Cannot load {path}: {e}", file=sys.stderr)
        return None


def load_bird_metadata() -> dict[str, dict]:
    """Load BIRD dev.json → dict keyed by normalized question text.

    Returns {norm_question: {question_id, db_id, difficulty, sql, evidence}}.
    """
    data = load_json(BIRD_DEV)
    if not data:
        return {}
    out = {}
    for item in data:
        key = normalize_question(item["question"])
        out[key] = {
            "question_id": item.get("question_id"),
            "db_id": item["db_id"],
            "difficulty": item.get("difficulty", "unknown"),
            "sql": item.get("SQL", ""),
            "evidence": item.get("evidence", ""),
        }
    return out


def normalize_question(q: str) -> str:
    """Strip, lowercase, collapse whitespace for matching."""
    return re.sub(r"\s+", " ", q.strip().lower())


def extract_db_from_source(source: str) -> str | None:
    """Extract BIRD db_id from a pipeline/blind source string like 'bird_formula_1'."""
    if not source:
        return None
    if source in SOURCE_TO_DB:
        return SOURCE_TO_DB[source]
    # Fallback: strip 'bird_' prefix
    if source.startswith("bird_"):
        return source[5:]
    return None


# ---------------------------------------------------------------------------
# Result loading: pipeline, blind, SOTA
# ---------------------------------------------------------------------------


def find_pipeline_files() -> list[dict]:
    """Discover all pipeline result files and parse metadata from filenames.

    Returns list of {path, model, schema, n, label, results[]}.
    """
    pattern = str(RESULTS_DIR / "pipeline_*_test.json")
    found = []
    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath)
        # pipeline_sonnet_v2_n3_test.json  or  pipeline_sonnet_v2_n1584_test.json
        m = re.match(r"pipeline_(\w+)_(v[12])_n(\d+)_test\.json", fname)
        if not m:
            continue
        model, schema, n = m.group(1), m.group(2), int(m.group(3))
        data = load_json(fpath)
        if data is None:
            continue
        results = data.get("results", [])
        found.append({
            "path": fpath,
            "model": model,
            "schema": schema,
            "n": n,
            "label": f"Pipeline {schema.upper()} / {MODEL_DISPLAY.get(model, model)}",
            "system": "pipeline",
            "results": results,
            "meta": data,
        })
    return found


def find_blind_files() -> list[dict]:
    """Discover blind baseline result files."""
    pattern = str(RESULTS_DIR / "blind_*_test.json")
    found = []
    for fpath in sorted(glob.glob(pattern)):
        fname = os.path.basename(fpath)
        m = re.match(r"blind_(\w+)_n(\d+)_test\.json", fname)
        if not m:
            continue
        model, n = m.group(1), int(m.group(2))
        data = load_json(fpath)
        if data is None:
            continue
        results = data.get("results", [])
        found.append({
            "path": fpath,
            "model": model,
            "schema": "blind",
            "n": n,
            "label": f"Blind / {MODEL_DISPLAY.get(model, model)}",
            "system": "blind",
            "results": results,
            "meta": data,
        })
    return found


def find_sota_files() -> list[dict]:
    """Discover SOTA result files (_scored.json preferred, fallback to _predictions.json).

    Checks both full (N=1315) and original (N=555) scored files.
    """
    systems = ["dail_sql", "din_sql", "chess"]
    found = []
    for sys_name in systems:
        # Prefer full scored, then full predictions, then original scored
        candidates = [
            SOTA_DIR / f"{sys_name}_predictions_full_scored.json",
            SOTA_DIR / f"{sys_name}_predictions_full.json",
            SOTA_DIR / f"{sys_name}_predictions_scored.json",
            SOTA_DIR / f"{sys_name}_predictions.json",
        ]
        data = None
        chosen_path = None
        for cand in candidates:
            data = load_json(cand)
            if data is not None:
                chosen_path = str(cand)
                break
        if data is None:
            continue

        results = data if isinstance(data, list) else data.get("results", [])
        found.append({
            "path": chosen_path,
            "model": "Sonnet 4.6",
            "schema": "sql",
            "n": len(results),
            "label": _sota_display(sys_name),
            "system": sys_name,
            "results": results,
            "meta": data,
        })
    return found


def _sota_display(sys_name: str) -> str:
    return {
        "dail_sql": "DAIL-SQL",
        "din_sql": "DIN-SQL",
        "chess": "CHESS",
    }.get(sys_name, sys_name)


# ---------------------------------------------------------------------------
# SQL feature detection (regex on gold SQL)
# ---------------------------------------------------------------------------

SQL_FEATURES = {
    "JOIN": r"\bJOIN\b",
    "WHERE+AND": r"\bWHERE\b.*\bAND\b",
    "WHERE+OR": r"\bWHERE\b.*\bOR\b",
    "ORDER BY": r"\bORDER\s+BY\b",
    "LIMIT": r"\bLIMIT\b",
    "DISTINCT": r"\bDISTINCT\b",
    "COUNT": r"\bCOUNT\s*\(",
    "GROUP BY": r"\bGROUP\s+BY\b",
    "HAVING": r"\bHAVING\b",
    "Subquery": r"\(\s*SELECT\b",
    "CASE WHEN": r"\bCASE\s+WHEN\b",
    "LIKE": r"\bLIKE\b",
    "BETWEEN": r"\bBETWEEN\b",
    "IN (subq)": r"\bIN\s*\(\s*SELECT\b",
    "AGG (SUM/AVG/MAX/MIN)": r"\b(SUM|AVG|MAX|MIN)\s*\(",
}


def detect_sql_features(sql: str) -> set[str]:
    """Return set of feature names detected in a SQL string."""
    if not sql:
        return set()
    feats = set()
    for name, pattern in SQL_FEATURES.items():
        if re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
            feats.add(name)
    return feats


def count_joins(sql: str) -> int:
    """Count number of JOIN clauses in SQL."""
    if not sql:
        return 0
    return len(re.findall(r"\bJOIN\b", sql, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Unified per-query record builder
# ---------------------------------------------------------------------------


def build_unified_records(
    pipeline_files: list[dict],
    blind_files: list[dict],
    sota_files: list[dict],
    bird_meta: dict[str, dict],
) -> list[dict]:
    """Build a flat list of per-query records across all systems.

    Each record has: system, model, schema, question, norm_question, db_id,
    difficulty, gold_sql, strict, relaxed, ex, latency_s, input_tokens,
    output_tokens, tokens, candidates, corrections, got, expected,
    has_join, sql_features, n_joins, source_file.
    """
    records = []

    # Pipeline
    for pf in pipeline_files:
        for r in pf["results"]:
            nq = normalize_question(r["question"])
            meta = bird_meta.get(nq, {})
            db_id = r.get("db_id") or extract_db_from_source(r.get("source", "")) or meta.get("db_id")
            gold_sql = r.get("sql") or meta.get("sql", "")
            records.append({
                "system": "pipeline",
                "system_label": pf["label"],
                "model": pf["model"],
                "schema": pf["schema"],
                "question": r["question"],
                "norm_question": nq,
                "db_id": db_id,
                "difficulty": meta.get("difficulty", "unknown"),
                "gold_sql": gold_sql,
                "strict": r.get("strict_match", False),
                "relaxed": r.get("relaxed_match", False),
                "ex": None,  # pipeline does not have BIRD EX by default
                "latency_s": r.get("latency_s", 0.0),
                "input_tokens": r.get("input_tokens", 0),
                "output_tokens": r.get("output_tokens", 0),
                "tokens": r.get("tokens", 0),
                "candidates": r.get("candidates"),
                "corrections": r.get("corrections", 0),
                "got": r.get("got", ""),
                "expected": r.get("expected", ""),
                "has_join": r.get("has_join", False),
                "sql_features": detect_sql_features(gold_sql),
                "n_joins": count_joins(gold_sql),
                "source": r.get("source", ""),
                "is_bird": r.get("source", "").startswith("bird_"),
            })

    # Blind
    for bf in blind_files:
        for r in bf["results"]:
            nq = normalize_question(r["question"])
            meta = bird_meta.get(nq, {})
            db_id = extract_db_from_source(r.get("source", "")) or meta.get("db_id")
            gold_sql = meta.get("sql", "")
            records.append({
                "system": "blind",
                "system_label": bf["label"],
                "model": bf["model"],
                "schema": bf["schema"],
                "question": r["question"],
                "norm_question": nq,
                "db_id": db_id,
                "difficulty": meta.get("difficulty", "unknown"),
                "gold_sql": gold_sql,
                "strict": r.get("strict_match", False),
                "relaxed": r.get("relaxed_match", False),
                "ex": None,
                "latency_s": r.get("latency_s", 0.0),
                "input_tokens": r.get("input_tokens", 0),
                "output_tokens": r.get("output_tokens", 0),
                "tokens": r.get("input_tokens", 0) + r.get("output_tokens", 0),
                "candidates": None,
                "corrections": 0,
                "got": r.get("got_collection", ""),
                "expected": r.get("expected", ""),
                "has_join": r.get("has_join", False),
                "sql_features": detect_sql_features(gold_sql),
                "n_joins": count_joins(gold_sql),
                "source": r.get("source", ""),
                "is_bird": r.get("source", "").startswith("bird_"),
            })

    # SOTA
    for sf in sota_files:
        for r in sf["results"]:
            nq = normalize_question(r["question"])
            meta = bird_meta.get(nq, {})
            gold_sql = r.get("gold_sql", "") or meta.get("sql", "")
            ev = r.get("eval", {})
            records.append({
                "system": sf["system"],
                "system_label": sf["label"],
                "model": "sonnet",
                "schema": "sql",
                "question": r["question"],
                "norm_question": nq,
                "db_id": r.get("db_id") or meta.get("db_id"),
                "difficulty": meta.get("difficulty", "unknown"),
                "gold_sql": gold_sql,
                "strict": ev.get("ex", False) if ev else False,
                "relaxed": ev.get("ex", False) if ev else False,
                "ex": ev.get("ex", None) if ev else None,
                "latency_s": r.get("wall_time_s", 0.0),
                "input_tokens": r.get("tokens_in", 0),
                "output_tokens": r.get("tokens_out", 0),
                "tokens": r.get("tokens_in", 0) + r.get("tokens_out", 0),
                "candidates": None,
                "corrections": 0,
                "got": r.get("predicted_sql", ""),
                "expected": gold_sql,
                "has_join": "JOIN" in gold_sql.upper() if gold_sql else False,
                "sql_features": detect_sql_features(gold_sql),
                "n_joins": count_joins(gold_sql),
                "source": f"bird_{r.get('db_id', '')}",
                "is_bird": True,
            })

    return records


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def md_table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    """Build a markdown table string.

    align: list of 'l', 'c', 'r' per column. Default left.
    """
    if not rows:
        return "_No data available._\n"

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    if align is None:
        align = ["l"] * len(headers)

    def fmt_row(cells):
        parts = []
        for i, cell in enumerate(cells):
            w = col_widths[i] if i < len(col_widths) else len(str(cell))
            parts.append(f" {str(cell).ljust(w)} ")
        return "|" + "|".join(parts) + "|"

    def sep_row():
        parts = []
        for i, a in enumerate(align):
            w = col_widths[i] if i < len(col_widths) else 3
            if a == "r":
                parts.append("-" * (w + 1) + ":")
            elif a == "c":
                parts.append(":" + "-" * w + ":")
            else:
                parts.append("-" * (w + 2))
        return "|" + "|".join(parts) + "|"

    lines = [fmt_row(headers), sep_row()]
    for row in rows:
        # Pad row to match headers length
        padded = list(row) + [""] * (len(headers) - len(row))
        lines.append(fmt_row(padded))
    return "\n".join(lines) + "\n"


def pct(num: int | float, denom: int | float) -> str:
    """Format as percentage string."""
    if denom == 0:
        return "-"
    return f"{100.0 * num / denom:.1f}%"


def pct_val(num: int | float, denom: int | float) -> float:
    """Return percentage as float."""
    if denom == 0:
        return 0.0
    return 100.0 * num / denom


def fmt_float(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "-"
    return f"{v:.{decimals}f}"


def fmt_int(v: int | float | None) -> str:
    if v is None:
        return "-"
    return f"{int(v):,}"


# ---------------------------------------------------------------------------
# Per-system grouping helpers
# ---------------------------------------------------------------------------


def group_by_system(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by system_label."""
    groups = defaultdict(list)
    for r in records:
        groups[r["system_label"]].append(r)
    return dict(groups)


def bird_only(records: list[dict]) -> list[dict]:
    """Filter to BIRD queries only."""
    return [r for r in records if r.get("is_bird", False)]


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


def section_1_headline(records: list[dict]) -> str:
    """Section 1: Headline Comparison Table."""
    lines = ["## Section 1: Headline Comparison Table\n"]
    lines.append("All systems on BIRD queries. Pipeline and Blind may include Weaviate Gorilla queries; "
                 "SOTA systems run on BIRD only.\n\n")

    grouped = group_by_system(records)

    headers = ["System", "N (BIRD)", "Strict/EX %", "Relaxed %",
               "Avg Input Tok/q", "Avg Latency/q (s)", "Avg Calls/q"]
    align = ["l", "r", "r", "r", "r", "r", "r"]
    rows = []

    # Sort: pipeline first (by schema desc, model), then blind, then SOTA
    def sort_key(label):
        if "Pipeline V2" in label:
            return (0, label)
        if "Pipeline V1" in label:
            return (1, label)
        if "Blind" in label:
            return (2, label)
        return (3, label)

    for label in sorted(grouped.keys(), key=sort_key):
        recs = grouped[label]
        bird_recs = bird_only(recs)
        n_bird = len(bird_recs)
        if n_bird == 0:
            continue

        n_strict = sum(1 for r in bird_recs if r["strict"])
        n_relaxed = sum(1 for r in bird_recs if r["relaxed"])
        input_toks = [r["input_tokens"] for r in bird_recs if r["input_tokens"] > 0]
        latencies = [r["latency_s"] for r in bird_recs if r["latency_s"] > 0]

        # Calls/q: pipeline = 1 (single tool call), SOTA varies
        if "Pipeline" in label or "Blind" in label:
            calls_q = "1"
        elif "CHESS" in label:
            calls_q = "~47"
        elif "DIN" in label:
            calls_q = "4"
        elif "DAIL" in label:
            calls_q = "1"
        else:
            calls_q = "-"

        rows.append([
            label,
            str(n_bird),
            pct(n_strict, n_bird),
            pct(n_relaxed, n_bird),
            fmt_int(statistics.mean(input_toks)) if input_toks else "-",
            fmt_float(statistics.mean(latencies)) if latencies else "-",
            calls_q,
        ])

    lines.append(md_table(headers, rows, align))
    return "\n".join(lines) + "\n"


def section_2_v1_v2(records: list[dict]) -> str:
    """Section 2: V1 vs V2 Schema Comparison."""
    lines = ["## Section 2: V1 vs V2 Schema Comparison\n"]
    lines.append("Per-model comparison of pipeline V1 vs V2 tool schemas on BIRD queries.\n\n")

    # Group pipeline records by model
    models = sorted(set(r["model"] for r in records if r["system"] == "pipeline"))
    if not models:
        lines.append("_No pipeline data available._\n")
        return "\n".join(lines)

    headers = ["Model", "Schema", "N", "Strict %", "Relaxed %",
               "JOIN Strict %", "Single Strict %", "Avg Input Tok"]
    align = ["l", "l", "r", "r", "r", "r", "r", "r"]
    rows = []

    for model in models:
        for schema in ["v1", "v2"]:
            recs = [r for r in records
                    if r["system"] == "pipeline" and r["model"] == model
                    and r["schema"] == schema and r["is_bird"]]
            if not recs:
                rows.append([MODEL_DISPLAY.get(model, model), schema.upper(),
                             "[MISSING]", "-", "-", "-", "-", "-"])
                continue

            n = len(recs)
            strict = sum(1 for r in recs if r["strict"])
            relaxed = sum(1 for r in recs if r["relaxed"])
            join_recs = [r for r in recs if r["has_join"]]
            single_recs = [r for r in recs if not r["has_join"]]
            join_strict = sum(1 for r in join_recs if r["strict"])
            single_strict = sum(1 for r in single_recs if r["strict"])
            input_toks = [r["input_tokens"] for r in recs if r["input_tokens"] > 0]

            rows.append([
                MODEL_DISPLAY.get(model, model),
                schema.upper(),
                str(n),
                pct(strict, n),
                pct(relaxed, n),
                pct(join_strict, len(join_recs)) if join_recs else "-",
                pct(single_strict, len(single_recs)) if single_recs else "-",
                fmt_int(statistics.mean(input_toks)) if input_toks else "-",
            ])

    lines.append(md_table(headers, rows, align))

    # Delta summary
    lines.append("\n**Deltas (V2 - V1):**\n")
    for model in models:
        v1 = [r for r in records if r["system"] == "pipeline" and r["model"] == model
              and r["schema"] == "v1" and r["is_bird"]]
        v2 = [r for r in records if r["system"] == "pipeline" and r["model"] == model
              and r["schema"] == "v2" and r["is_bird"]]
        if v1 and v2:
            s1 = pct_val(sum(1 for r in v1 if r["strict"]), len(v1))
            s2 = pct_val(sum(1 for r in v2 if r["strict"]), len(v2))
            r1 = pct_val(sum(1 for r in v1 if r["relaxed"]), len(v1))
            r2 = pct_val(sum(1 for r in v2 if r["relaxed"]), len(v2))
            lines.append(f"- **{MODEL_DISPLAY.get(model, model)}**: "
                         f"Strict {s2-s1:+.1f}pp, Relaxed {r2-r1:+.1f}pp\n")
        else:
            lines.append(f"- **{MODEL_DISPLAY.get(model, model)}**: [MISSING one schema]\n")

    return "\n".join(lines) + "\n"


def section_3_pipeline_vs_blind(records: list[dict]) -> str:
    """Section 3: Pipeline Lift Over Blind."""
    lines = ["## Section 3: Pipeline Lift Over Blind Baseline\n"]
    lines.append("Per-model comparison: pipeline (best schema) vs blind baseline. "
                 "Shows the value of the discovery phase.\n\n")

    models = sorted(set(r["model"] for r in records if r["system"] in ("pipeline", "blind")))

    headers = ["Model", "System", "N", "Strict %", "Relaxed %",
               "Avg Input Tok", "Token Savings"]
    align = ["l", "l", "r", "r", "r", "r", "r"]
    rows = []

    for model in models:
        # Best pipeline schema (prefer v2, fallback v1)
        for schema in ["v2", "v1"]:
            p_recs = [r for r in records
                      if r["system"] == "pipeline" and r["model"] == model
                      and r["schema"] == schema and r["is_bird"]]
            if p_recs:
                break
        else:
            p_recs = []

        b_recs = [r for r in records
                  if r["system"] == "blind" and r["model"] == model and r["is_bird"]]

        for label, recs, sys_name in [("Pipeline", p_recs, "pipeline"), ("Blind", b_recs, "blind")]:
            if not recs:
                rows.append([MODEL_DISPLAY.get(model, model), label,
                             "[MISSING]", "-", "-", "-", "-"])
                continue
            n = len(recs)
            strict = sum(1 for r in recs if r["strict"])
            relaxed = sum(1 for r in recs if r["relaxed"])
            input_toks = [r["input_tokens"] for r in recs if r["input_tokens"] > 0]
            avg_in = statistics.mean(input_toks) if input_toks else 0

            savings = "-"
            if label == "Blind":
                # Compare to pipeline
                p_toks = [r["input_tokens"] for r in p_recs if r["input_tokens"] > 0]
                if p_toks and avg_in > 0:
                    p_avg = statistics.mean(p_toks)
                    savings = f"{100*(1 - p_avg/avg_in):.1f}%"

            rows.append([
                MODEL_DISPLAY.get(model, model),
                label + (f" ({schema.upper()})" if label == "Pipeline" and p_recs else ""),
                str(n),
                pct(strict, n),
                pct(relaxed, n),
                fmt_int(avg_in) if avg_in > 0 else "-",
                savings,
            ])

    lines.append(md_table(headers, rows, align))
    return "\n".join(lines) + "\n"


def section_4_per_database(records: list[dict]) -> str:
    """Section 4: Per-Database Breakdown."""
    lines = ["## Section 4: Per-Database Breakdown (BIRD)\n"]
    lines.append("Strict accuracy (%) by database across all systems.\n\n")

    grouped = group_by_system(records)

    # Build system labels sorted
    def sort_key(label):
        if "Pipeline V2" in label:
            return (0, label)
        if "Pipeline V1" in label:
            return (1, label)
        if "Blind" in label:
            return (2, label)
        return (3, label)

    sys_labels = sorted(grouped.keys(), key=sort_key)

    # Filter to systems that have BIRD data
    sys_labels = [s for s in sys_labels if any(r["is_bird"] for r in grouped[s])]
    if not sys_labels:
        lines.append("_No data available._\n")
        return "\n".join(lines)

    # Truncate system labels for column headers
    short_labels = []
    for s in sys_labels:
        short = s.replace("Pipeline ", "P").replace("Blind / ", "B/")
        short = short.replace("Sonnet 4.6", "Son").replace("Qwen 3 80B", "Qw")
        short = short.replace("Llama 4 Maverick", "Ll")
        short_labels.append(short)

    headers = ["Database", "N"] + short_labels
    align = ["l", "r"] + ["r"] * len(sys_labels)
    rows = []

    totals = {s: {"correct": 0, "total": 0} for s in sys_labels}

    for db in ALL_DBS:
        row = [DB_SHORT.get(db, db)]
        # Count N from first system that has data for this db
        db_n = 0
        for s in sys_labels:
            db_recs = [r for r in grouped[s] if r["is_bird"] and r["db_id"] == db]
            if db_recs:
                db_n = max(db_n, len(db_recs))
        row.append(str(db_n) if db_n > 0 else "-")

        for s in sys_labels:
            db_recs = [r for r in grouped[s] if r["is_bird"] and r["db_id"] == db]
            if not db_recs:
                row.append("-")
            else:
                n_correct = sum(1 for r in db_recs if r["strict"])
                row.append(pct(n_correct, len(db_recs)))
                totals[s]["correct"] += n_correct
                totals[s]["total"] += len(db_recs)
        rows.append(row)

    # Total row
    total_row = ["**TOTAL**", ""]
    for s in sys_labels:
        t = totals[s]
        total_row.append(f"**{pct(t['correct'], t['total'])}**" if t["total"] > 0 else "-")
    rows.append(total_row)

    lines.append(md_table(headers, rows, align))

    # Best/worst per system
    lines.append("\n**Per-system best and worst databases:**\n")
    for s in sys_labels:
        db_accs = []
        for db in ALL_DBS:
            db_recs = [r for r in grouped[s] if r["is_bird"] and r["db_id"] == db]
            if len(db_recs) >= 5:
                acc = pct_val(sum(1 for r in db_recs if r["strict"]), len(db_recs))
                db_accs.append((db, acc, len(db_recs)))
        if db_accs:
            db_accs.sort(key=lambda x: x[1], reverse=True)
            best = db_accs[0]
            worst = db_accs[-1]
            lines.append(f"- **{s}**: Best = {DB_SHORT.get(best[0], best[0])} ({best[1]:.1f}%), "
                         f"Worst = {DB_SHORT.get(worst[0], worst[0])} ({worst[1]:.1f}%)\n")

    return "\n".join(lines) + "\n"


def section_5_per_collection(records: list[dict]) -> str:
    """Section 5: Per-Collection Accuracy (pipeline only)."""
    lines = ["## Section 5: Per-Collection Accuracy (Pipeline Only)\n"]
    lines.append("Grouped by database. Shows accuracy for each collection across pipeline models.\n\n")

    p_recs = [r for r in records if r["system"] == "pipeline" and r["is_bird"]]
    if not p_recs:
        lines.append("_No pipeline data available._\n")
        return "\n".join(lines)

    # Group by (model, schema) combos
    combos = sorted(set((r["model"], r["schema"]) for r in p_recs))
    combo_labels = [f"{MODEL_DISPLAY.get(m, m)} {s.upper()}" for m, s in combos]

    # Group by expected collection
    coll_db = {}  # collection -> db_id
    coll_data = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for r in p_recs:
        coll = r["expected"]
        if not coll:
            continue
        combo_key = (r["model"], r["schema"])
        coll_data[coll][combo_key]["total"] += 1
        if r["strict"]:
            coll_data[coll][combo_key]["correct"] += 1
        if coll not in coll_db and r["db_id"]:
            coll_db[coll] = r["db_id"]

    if not coll_data:
        lines.append("_No collection-level data._\n")
        return "\n".join(lines)

    # Sort collections by database then name
    sorted_colls = sorted(coll_data.keys(), key=lambda c: (coll_db.get(c, "zzz"), c))

    # Truncate combo labels for headers
    short_combo = []
    for m, s in combos:
        short = MODEL_DISPLAY.get(m, m)[:3] + " " + s.upper()
        short_combo.append(short)

    headers = ["Database", "Collection", "N"] + short_combo
    align = ["l", "l", "r"] + ["r"] * len(combos)
    rows = []

    for coll in sorted_colls:
        db = coll_db.get(coll, "?")
        # N = max total across combos
        n = max(coll_data[coll][c]["total"] for c in combos if coll_data[coll][c]["total"] > 0) \
            if any(coll_data[coll][c]["total"] > 0 for c in combos) else 0
        row_cells = [DB_SHORT.get(db, db)[:10], coll[:35], str(n)]
        for c in combos:
            d = coll_data[coll][c]
            if d["total"] > 0:
                row_cells.append(pct(d["correct"], d["total"]))
            else:
                row_cells.append("-")
        rows.append(row_cells)

    lines.append(md_table(headers, rows, align))

    # Hardest collections (lowest accuracy across all combos)
    lines.append("\n**Hardest collections (lowest average strict accuracy, N >= 5):**\n")
    coll_avg = []
    for coll in coll_data:
        accs = []
        for c in combos:
            d = coll_data[coll][c]
            if d["total"] >= 5:
                accs.append(pct_val(d["correct"], d["total"]))
        if accs:
            coll_avg.append((coll, statistics.mean(accs), coll_db.get(coll, "?")))
    coll_avg.sort(key=lambda x: x[1])
    for coll, avg_acc, db in coll_avg[:15]:
        lines.append(f"- {coll} ({DB_SHORT.get(db, db)}): {avg_acc:.1f}% avg strict\n")

    # Top confusion pairs
    lines.append("\n**Top confusion pairs (predicted -> expected):**\n")
    confusions = Counter()
    for r in p_recs:
        if not r["strict"] and r["got"] and r["expected"]:
            confusions[(r["got"], r["expected"])] += 1
    for (got, exp), cnt in confusions.most_common(20):
        lines.append(f"- {got} -> {exp}: {cnt}x\n")

    return "\n".join(lines) + "\n"


def section_6_by_difficulty(records: list[dict]) -> str:
    """Section 6: By Difficulty."""
    lines = ["## Section 6: Accuracy by Query Difficulty\n"]
    lines.append("BIRD difficulty levels: simple, moderate, challenging.\n\n")

    grouped = group_by_system(records)
    difficulties = ["simple", "moderate", "challenging"]

    def sort_key(label):
        if "Pipeline V2" in label:
            return (0, label)
        if "Pipeline V1" in label:
            return (1, label)
        if "Blind" in label:
            return (2, label)
        return (3, label)

    sys_labels = sorted(grouped.keys(), key=sort_key)
    sys_labels = [s for s in sys_labels if any(r["is_bird"] for r in grouped[s])]

    headers = ["System"] + [d.capitalize() for d in difficulties] + ["Overall"]
    align = ["l"] + ["r"] * (len(difficulties) + 1)
    rows = []

    for s in sys_labels:
        bird = bird_only(grouped[s])
        if not bird:
            continue
        row = [s]
        for diff in difficulties:
            diff_recs = [r for r in bird if r["difficulty"] == diff]
            if diff_recs:
                row.append(f"{pct(sum(1 for r in diff_recs if r['strict']), len(diff_recs))} (n={len(diff_recs)})")
            else:
                row.append("-")
        # Overall
        row.append(pct(sum(1 for r in bird if r["strict"]), len(bird)))
        rows.append(row)

    lines.append(md_table(headers, rows, align))

    # Where does pipeline advantage concentrate?
    lines.append("\n**Difficulty-level advantage analysis:**\n")
    # Find best pipeline and best SOTA
    p_labels = [s for s in sys_labels if "Pipeline" in s]
    sota_labels = [s for s in sys_labels if s in ("DAIL-SQL", "DIN-SQL", "CHESS")]
    if p_labels and sota_labels:
        for diff in difficulties:
            p_best = 0.0
            p_best_n = 0
            s_best = 0.0
            s_best_n = 0
            for s in p_labels:
                diff_recs = [r for r in bird_only(grouped[s]) if r["difficulty"] == diff]
                if diff_recs:
                    acc = pct_val(sum(1 for r in diff_recs if r["strict"]), len(diff_recs))
                    if acc > p_best:
                        p_best = acc
                        p_best_n = len(diff_recs)
            for s in sota_labels:
                diff_recs = [r for r in bird_only(grouped[s]) if r["difficulty"] == diff]
                if diff_recs:
                    acc = pct_val(sum(1 for r in diff_recs if r["strict"]), len(diff_recs))
                    if acc > s_best:
                        s_best = acc
                        s_best_n = len(diff_recs)
            delta = p_best - s_best
            caveat = f" [small N={p_best_n}]" if p_best_n < 30 else ""
            lines.append(f"- **{diff.capitalize()}**: Pipeline best {p_best:.1f}% (n={p_best_n}){caveat} vs "
                         f"SOTA best {s_best:.1f}% (n={s_best_n}) ({delta:+.1f}pp)\n")

    return "\n".join(lines) + "\n"


def section_7_by_sql_feature(records: list[dict]) -> str:
    """Section 7: By SQL Feature."""
    lines = ["## Section 7: Accuracy by SQL Feature\n"]
    lines.append("Queries tagged by features detected via regex on gold SQL.\n"
                 "A query can have multiple features.\n\n")

    grouped = group_by_system(records)

    def sort_key(label):
        if "Pipeline V2" in label:
            return (0, label)
        if "Pipeline V1" in label:
            return (1, label)
        if "Blind" in label:
            return (2, label)
        return (3, label)

    sys_labels = sorted(grouped.keys(), key=sort_key)
    sys_labels = [s for s in sys_labels if any(r["is_bird"] for r in grouped[s])]

    # Feature prevalence (from any system)
    all_bird = bird_only(records)
    feat_counts = Counter()
    for r in all_bird:
        for f in r["sql_features"]:
            feat_counts[f] += 1
    # Deduplicate: count unique questions per feature
    feat_questions = defaultdict(set)
    for r in all_bird:
        for f in r["sql_features"]:
            feat_questions[f].add(r["norm_question"])

    # Select top features by prevalence
    top_features = [f for f, _ in sorted(feat_questions.items(), key=lambda x: -len(x[1]))]
    # Limit to features with reasonable prevalence
    top_features = [f for f in top_features if len(feat_questions[f]) >= 10][:12]

    if not top_features:
        lines.append("_Insufficient SQL metadata for feature analysis._\n")
        return "\n".join(lines)

    # Truncate system labels
    short_sys = []
    for s in sys_labels:
        short = s.replace("Pipeline ", "P").replace("Blind / ", "B/")
        short = short.replace("Sonnet 4.6", "Son").replace("Qwen 3 80B", "Qw")
        short = short.replace("Llama 4 Maverick", "Ll")
        short_sys.append(short)

    headers = ["Feature", "N (uniq q)"] + short_sys
    align = ["l", "r"] + ["r"] * len(sys_labels)
    rows = []

    for feat in top_features:
        n_unique = len(feat_questions[feat])
        row = [feat, str(n_unique)]
        for s in sys_labels:
            feat_recs = [r for r in bird_only(grouped[s]) if feat in r["sql_features"]]
            if feat_recs:
                row.append(pct(sum(1 for r in feat_recs if r["strict"]), len(feat_recs)))
            else:
                row.append("-")
        rows.append(row)

    lines.append(md_table(headers, rows, align))

    # Feature interaction: JOIN + ORDER BY vs JOIN alone
    lines.append("\n**Feature interactions (pipeline V2 / best model, BIRD only):**\n")
    # Find best pipeline V2
    best_p = None
    best_p_acc = -1
    for s in sys_labels:
        if "Pipeline V2" in s:
            recs = bird_only(grouped[s])
            if recs:
                acc = pct_val(sum(1 for r in recs if r["strict"]), len(recs))
                if acc > best_p_acc:
                    best_p_acc = acc
                    best_p = s

    if best_p:
        bp_recs = bird_only(grouped[best_p])
        interactions = [
            ("JOIN only", lambda r: "JOIN" in r["sql_features"] and "ORDER BY" not in r["sql_features"]
             and "Subquery" not in r["sql_features"]),
            ("JOIN + ORDER BY", lambda r: "JOIN" in r["sql_features"] and "ORDER BY" in r["sql_features"]),
            ("JOIN + Subquery", lambda r: "JOIN" in r["sql_features"] and "Subquery" in r["sql_features"]),
            ("No JOIN", lambda r: "JOIN" not in r["sql_features"]),
            ("COUNT + GROUP BY", lambda r: "COUNT" in r["sql_features"] and "GROUP BY" in r["sql_features"]),
            ("WHERE+AND (no JOIN)", lambda r: "WHERE+AND" in r["sql_features"] and "JOIN" not in r["sql_features"]),
        ]
        for label, pred in interactions:
            matching = [r for r in bp_recs if pred(r)]
            if len(matching) >= 5:
                acc = pct_val(sum(1 for r in matching if r["strict"]), len(matching))
                lines.append(f"- {label}: {acc:.1f}% (n={len(matching)})\n")

    return "\n".join(lines) + "\n"


def section_8_pattern_analysis(records: list[dict]) -> str:
    """Section 8: Pattern Analysis."""
    lines = ["## Section 8: Pattern Analysis\n"]

    # --- Top 20 confusion pairs across all pipeline runs ---
    lines.append("### Top 20 Confusion Pairs (all pipeline runs)\n")
    lines.append("Counts how often collection X was predicted when Y was expected.\n\n")

    p_recs = [r for r in records if r["system"] == "pipeline"]
    confusions = Counter()
    for r in p_recs:
        if not r["strict"] and r["got"] and r["expected"]:
            confusions[(r["got"], r["expected"])] += 1

    if confusions:
        headers = ["Predicted", "Expected", "Count"]
        rows = []
        for (got, exp), cnt in confusions.most_common(20):
            rows.append([got, exp, str(cnt)])
        lines.append(md_table(headers, rows, ["l", "l", "r"]))
    else:
        lines.append("_No confusion data._\n")

    # --- Universal hard cases (all systems fail) ---
    lines.append("\n### Universal Hard Cases\n")
    lines.append("Queries where ALL systems with data fail (strict=False for every system).\n\n")

    grouped = group_by_system(records)
    sys_labels = list(grouped.keys())
    bird_sys_labels = [s for s in sys_labels if any(r["is_bird"] for r in grouped[s])]

    if len(bird_sys_labels) >= 2:
        # Build question -> {system_label: strict}
        q_results = defaultdict(dict)
        for s in bird_sys_labels:
            for r in bird_only(grouped[s]):
                q_results[r["norm_question"]][s] = r["strict"]

        # Universal failures: question appeared in ALL systems and all failed
        universal_fails = []
        for nq, sys_map in q_results.items():
            if len(sys_map) >= 2 and not any(sys_map.values()):
                # Get example record for display
                for s in bird_sys_labels:
                    for r in bird_only(grouped[s]):
                        if r["norm_question"] == nq:
                            universal_fails.append({
                                "question": r["question"],
                                "db_id": r["db_id"],
                                "difficulty": r["difficulty"],
                                "n_systems": len(sys_map),
                            })
                            break
                    else:
                        continue
                    break

        if universal_fails:
            lines.append(f"Found **{len(universal_fails)}** queries where all {len(bird_sys_labels)} systems fail.\n\n")
            # Show first 20
            for i, uf in enumerate(universal_fails[:20]):
                lines.append(f"{i+1}. [{uf['db_id']}] [{uf['difficulty']}] {uf['question']}\n")
            if len(universal_fails) > 20:
                lines.append(f"\n... and {len(universal_fails) - 20} more.\n")
        else:
            lines.append("_No queries where all systems fail._\n")
    else:
        lines.append("_Need at least 2 systems with data for cross-system analysis._\n")

    # --- Pipeline succeeds but SOTA fails ---
    lines.append("\n### Queries Where Pipeline Succeeds but SOTA Fails\n")

    sota_labels = [s for s in bird_sys_labels if s in ("DAIL-SQL", "DIN-SQL", "CHESS")]
    pipeline_labels = [s for s in bird_sys_labels if "Pipeline" in s]

    if sota_labels and pipeline_labels:
        q_results_pipe = defaultdict(bool)  # norm_q -> any pipeline strict
        q_results_sota = defaultdict(bool)   # norm_q -> any SOTA strict
        q_info = {}

        # Collect q_info from ALL systems (so we get db_id even for SOTA-only questions)
        for s in bird_sys_labels:
            for r in bird_only(grouped[s]):
                if r["norm_question"] not in q_info:
                    q_info[r["norm_question"]] = {"question": r["question"],
                                                   "db_id": r["db_id"],
                                                   "difficulty": r["difficulty"]}

        for s in pipeline_labels:
            for r in bird_only(grouped[s]):
                if r["strict"]:
                    q_results_pipe[r["norm_question"]] = True

        for s in sota_labels:
            for r in bird_only(grouped[s]):
                if r["strict"]:
                    q_results_sota[r["norm_question"]] = True

        # Questions in both
        common_qs = set(q_results_pipe.keys()) | set(q_results_sota.keys())
        pipe_only = [nq for nq in common_qs if q_results_pipe.get(nq, False) and not q_results_sota.get(nq, False)]
        sota_only = [nq for nq in common_qs if q_results_sota.get(nq, False) and not q_results_pipe.get(nq, False)]

        lines.append(f"**{len(pipe_only)}** queries where at least one pipeline run succeeds but no SOTA system does.\n")
        for i, nq in enumerate(pipe_only[:15]):
            info = q_info.get(nq, {})
            lines.append(f"{i+1}. [{info.get('db_id', '?')}] [{info.get('difficulty', '?')}] "
                         f"{info.get('question', nq)[:120]}\n")
        if len(pipe_only) > 15:
            lines.append(f"\n... and {len(pipe_only) - 15} more.\n")

        # Per-DB breakdown of unique strengths/weaknesses
        pipe_only_dbs = Counter(q_info.get(nq, {}).get("db_id", "?") for nq in pipe_only)
        sota_only_dbs = Counter(q_info.get(nq, {}).get("db_id", "?") for nq in sota_only)

        if pipe_only_dbs:
            lines.append(f"\n_Pipeline-unique successes by DB:_ ")
            lines.append(", ".join(f"{DB_SHORT.get(db, db)} ({cnt})"
                                   for db, cnt in pipe_only_dbs.most_common(10)))
            lines.append("\n")

        lines.append(f"\n### Queries Where SOTA Succeeds but Pipeline Fails\n")
        lines.append(f"**{len(sota_only)}** queries where at least one SOTA system succeeds but no pipeline run does.\n")
        for i, nq in enumerate(sota_only[:15]):
            info = q_info.get(nq, {})
            lines.append(f"{i+1}. [{info.get('db_id', '?')}] [{info.get('difficulty', '?')}] "
                         f"{info.get('question', nq)[:120]}\n")
        if len(sota_only) > 15:
            lines.append(f"\n... and {len(sota_only) - 15} more.\n")

        if sota_only_dbs:
            lines.append(f"\n_SOTA-unique successes by DB:_ ")
            lines.append(", ".join(f"{DB_SHORT.get(db, db)} ({cnt})"
                                   for db, cnt in sota_only_dbs.most_common(11)))
            lines.append("\n")
    else:
        lines.append("_Need both pipeline and SOTA data for comparison._\n")

    return "\n".join(lines) + "\n"


def section_9_latency(records: list[dict]) -> str:
    """Section 9: Latency Analysis."""
    lines = ["## Section 9: Latency Analysis\n"]

    grouped = group_by_system(records)

    def sort_key(label):
        if "Pipeline V2" in label:
            return (0, label)
        if "Pipeline V1" in label:
            return (1, label)
        if "Blind" in label:
            return (2, label)
        return (3, label)

    sys_labels = sorted(grouped.keys(), key=sort_key)

    # --- Distribution table ---
    lines.append("### Latency Distribution (seconds/query)\n\n")

    headers = ["System", "N", "Mean", "Median", "P95", "P99", "Min", "Max"]
    align = ["l", "r", "r", "r", "r", "r", "r", "r"]
    rows = []

    for s in sys_labels:
        lats = [r["latency_s"] for r in grouped[s] if r["latency_s"] > 0]
        if not lats:
            rows.append([s, "0", "-", "-", "-", "-", "-", "-"])
            continue
        lats_sorted = sorted(lats)
        n = len(lats_sorted)
        p95_idx = min(int(n * 0.95), n - 1)
        p99_idx = min(int(n * 0.99), n - 1)
        rows.append([
            s,
            str(n),
            fmt_float(statistics.mean(lats)),
            fmt_float(statistics.median(lats)),
            fmt_float(lats_sorted[p95_idx]),
            fmt_float(lats_sorted[p99_idx]),
            fmt_float(lats_sorted[0]),
            fmt_float(lats_sorted[-1]),
        ])

    lines.append(md_table(headers, rows, align))

    # --- Latency by database ---
    lines.append("\n### Latency by Database (mean seconds, pipeline only)\n\n")

    p_labels = [s for s in sys_labels if "Pipeline" in s]
    if p_labels:
        # Truncate labels
        short_p = []
        for s in p_labels:
            short = s.replace("Pipeline ", "P").replace("Sonnet 4.6", "Son").replace("Qwen 3 80B", "Qw").replace("Llama 4 Maverick", "Ll")
            short_p.append(short)

        headers = ["Database"] + short_p
        align = ["l"] + ["r"] * len(p_labels)
        rows = []

        for db in ALL_DBS:
            row = [DB_SHORT.get(db, db)]
            for s in p_labels:
                db_lats = [r["latency_s"] for r in grouped[s]
                           if r["is_bird"] and r["db_id"] == db and r["latency_s"] > 0]
                row.append(fmt_float(statistics.mean(db_lats)) if db_lats else "-")
            rows.append(row)
        lines.append(md_table(headers, rows, align))

    # --- Latency by difficulty ---
    lines.append("\n### Latency by Difficulty (mean seconds)\n\n")

    headers = ["System", "Simple", "Moderate", "Challenging"]
    align = ["l", "r", "r", "r"]
    rows = []

    for s in sys_labels:
        bird = bird_only(grouped[s])
        if not bird:
            continue
        row = [s]
        for diff in ["simple", "moderate", "challenging"]:
            lats = [r["latency_s"] for r in bird if r["difficulty"] == diff and r["latency_s"] > 0]
            row.append(fmt_float(statistics.mean(lats)) if lats else "-")
        rows.append(row)

    lines.append(md_table(headers, rows, align))

    return "\n".join(lines) + "\n"


def section_10_token_efficiency(records: list[dict]) -> str:
    """Section 10: Token Efficiency Deep Dive."""
    lines = ["## Section 10: Token Efficiency Deep Dive\n"]

    grouped = group_by_system(records)

    def sort_key(label):
        if "Pipeline V2" in label:
            return (0, label)
        if "Pipeline V1" in label:
            return (1, label)
        if "Blind" in label:
            return (2, label)
        return (3, label)

    sys_labels = sorted(grouped.keys(), key=sort_key)

    # --- Distribution stats ---
    lines.append("### Token Distribution (BIRD queries)\n\n")

    headers = ["System", "N", "Mean In", "Median In", "P95 In",
               "Mean Out", "Median Out", "P95 Out", "Mean Total"]
    align = ["l"] + ["r"] * 8
    rows = []

    for s in sys_labels:
        bird = bird_only(grouped[s])
        in_toks = [r["input_tokens"] for r in bird if r["input_tokens"] > 0]
        out_toks = [r["output_tokens"] for r in bird if r["output_tokens"] > 0]
        total_toks = [r["input_tokens"] + r["output_tokens"] for r in bird
                      if r["input_tokens"] > 0 or r["output_tokens"] > 0]
        if not in_toks:
            rows.append([s, str(len(bird)), "-", "-", "-", "-", "-", "-", "-"])
            continue

        in_sorted = sorted(in_toks)
        out_sorted = sorted(out_toks) if out_toks else [0]
        n = len(in_sorted)
        p95_in = in_sorted[min(int(n * 0.95), n - 1)]
        p95_out = out_sorted[min(int(len(out_sorted) * 0.95), len(out_sorted) - 1)] if out_sorted else 0

        rows.append([
            s,
            str(n),
            fmt_int(statistics.mean(in_toks)),
            fmt_int(statistics.median(in_toks)),
            fmt_int(p95_in),
            fmt_int(statistics.mean(out_toks)) if out_toks else "-",
            fmt_int(statistics.median(out_toks)) if out_toks else "-",
            fmt_int(p95_out),
            fmt_int(statistics.mean(total_toks)) if total_toks else "-",
        ])

    lines.append(md_table(headers, rows, align))

    # --- Tokens by number of JOINs ---
    lines.append("\n### Tokens by Number of JOINs in Gold SQL\n\n")

    # Use best pipeline V2 as reference
    best_p = None
    for s in sys_labels:
        if "Pipeline V2" in s:
            best_p = s
            break
    if not best_p:
        for s in sys_labels:
            if "Pipeline" in s:
                best_p = s
                break

    if best_p:
        bird = bird_only(grouped[best_p])
        join_groups = defaultdict(list)
        for r in bird:
            nj = r["n_joins"]
            join_groups[min(nj, 3)].append(r)  # group 3+ together

        headers = ["# JOINs", "N", "Mean Input Tok", "Strict %"]
        align = ["l", "r", "r", "r"]
        rows = []
        for nj in sorted(join_groups.keys()):
            recs = join_groups[nj]
            in_toks = [r["input_tokens"] for r in recs if r["input_tokens"] > 0]
            label = f"{nj}+" if nj == 3 else str(nj)
            rows.append([
                label,
                str(len(recs)),
                fmt_int(statistics.mean(in_toks)) if in_toks else "-",
                pct(sum(1 for r in recs if r["strict"]), len(recs)),
            ])
        lines.append(f"_Reference system: {best_p}_\n\n")
        lines.append(md_table(headers, rows, align))

    # --- Cost comparison ---
    lines.append("\n### Cost Comparison ($3/1M input, $15/1M output)\n\n")
    lines.append("Estimated cost for N=1,000 queries at Bedrock Sonnet 4.6 pricing.\n\n")

    COST_IN = 3.0 / 1_000_000
    COST_OUT = 15.0 / 1_000_000

    headers = ["System", "Avg In Tok", "Avg Out Tok", "Cost/Query", "Cost/1000q", "Relative"]
    align = ["l", "r", "r", "r", "r", "r"]
    rows = []

    costs = {}
    for s in sys_labels:
        bird = bird_only(grouped[s])
        in_toks = [r["input_tokens"] for r in bird if r["input_tokens"] > 0]
        out_toks = [r["output_tokens"] for r in bird if r["output_tokens"] > 0]
        if not in_toks:
            continue
        avg_in = statistics.mean(in_toks)
        avg_out = statistics.mean(out_toks) if out_toks else 0
        cost_per_q = avg_in * COST_IN + avg_out * COST_OUT
        costs[s] = cost_per_q

    if costs:
        min_cost = min(costs.values())
        for s in sys_labels:
            if s not in costs:
                continue
            bird = bird_only(grouped[s])
            in_toks = [r["input_tokens"] for r in bird if r["input_tokens"] > 0]
            out_toks = [r["output_tokens"] for r in bird if r["output_tokens"] > 0]
            avg_in = statistics.mean(in_toks)
            avg_out = statistics.mean(out_toks) if out_toks else 0
            c = costs[s]
            rows.append([
                s,
                fmt_int(avg_in),
                fmt_int(avg_out),
                f"${c:.4f}",
                f"${c * 1000:.2f}",
                f"{c / min_cost:.1f}x" if min_cost > 0 else "-",
            ])
        lines.append(md_table(headers, rows, align))
    else:
        lines.append("_No token data available._\n")

    # --- Token savings summary ---
    lines.append("\n### Token Savings Summary\n\n")
    lines.append("Comparing pipeline (progressive disclosure) to blind (all 143 schemas) "
                 "and SOTA systems.\n\n")

    # Find all pipeline and blind combos for same model
    models = sorted(set(r["model"] for r in records if r["system"] in ("pipeline", "blind")))
    for model in models:
        p_recs = [r for r in records if r["system"] == "pipeline" and r["model"] == model
                  and r["schema"] in ("v2", "v1") and r["is_bird"]]
        b_recs = [r for r in records if r["system"] == "blind" and r["model"] == model and r["is_bird"]]
        if p_recs and b_recs:
            p_in = statistics.mean([r["input_tokens"] for r in p_recs if r["input_tokens"] > 0]) \
                if any(r["input_tokens"] > 0 for r in p_recs) else 0
            b_in = statistics.mean([r["input_tokens"] for r in b_recs if r["input_tokens"] > 0]) \
                if any(r["input_tokens"] > 0 for r in b_recs) else 0
            if b_in > 0 and p_in > 0:
                lines.append(f"- **{MODEL_DISPLAY.get(model, model)}**: Pipeline {p_in:.0f} tok/q vs "
                             f"Blind {b_in:.0f} tok/q = **{100*(1-p_in/b_in):.1f}% savings**\n")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Paper comparison analysis")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output markdown file path")
    args = parser.parse_args()

    print("=" * 60)
    print("Paper Comparison Analysis")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/4] Loading BIRD metadata...")
    bird_meta = load_bird_metadata()
    print(f"  BIRD dev questions: {len(bird_meta)}")

    print("\n[2/4] Loading result files...")
    pipeline_files = find_pipeline_files()
    blind_files = find_blind_files()
    sota_files = find_sota_files()

    print(f"  Pipeline files: {len(pipeline_files)}")
    for pf in pipeline_files:
        print(f"    {pf['label']} (n={len(pf['results'])})")
    print(f"  Blind files:    {len(blind_files)}")
    for bf in blind_files:
        print(f"    {bf['label']} (n={len(bf['results'])})")
    print(f"  SOTA files:     {len(sota_files)}")
    for sf in sota_files:
        print(f"    {sf['label']} (n={len(sf['results'])})")

    if not pipeline_files and not blind_files and not sota_files:
        print("\n[ERROR] No result files found. Run evaluations first.")
        sys.exit(1)

    # --- Build unified records ---
    print("\n[3/4] Building unified records...")
    records = build_unified_records(pipeline_files, blind_files, sota_files, bird_meta)
    print(f"  Total records: {len(records)}")
    print(f"  BIRD records:  {sum(1 for r in records if r['is_bird'])}")
    print(f"  Systems: {sorted(set(r['system_label'] for r in records))}")

    # --- Generate report ---
    print("\n[4/4] Generating report...")

    sections = []

    # Header
    sections.append("# Paper Comparison Results\n")
    sections.append(f"_Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    # Data inventory
    sections.append("\n## Data Inventory\n")
    sections.append(f"- BIRD metadata: {len(bird_meta)} questions\n")
    sections.append(f"- Total per-query records: {len(records)}\n")
    sections.append(f"- Systems loaded:\n")
    for label in sorted(set(r["system_label"] for r in records)):
        n_total = sum(1 for r in records if r["system_label"] == label)
        n_bird = sum(1 for r in records if r["system_label"] == label and r["is_bird"])
        sections.append(f"  - {label}: {n_total} total ({n_bird} BIRD)\n")
    sections.append("\n---\n\n")

    # Generate sections
    section_generators = [
        ("Section 1", section_1_headline),
        ("Section 2", section_2_v1_v2),
        ("Section 3", section_3_pipeline_vs_blind),
        ("Section 4", section_4_per_database),
        ("Section 5", section_5_per_collection),
        ("Section 6", section_6_by_difficulty),
        ("Section 7", section_7_by_sql_feature),
        ("Section 8", section_8_pattern_analysis),
        ("Section 9", section_9_latency),
        ("Section 10", section_10_token_efficiency),
    ]

    for name, gen_fn in section_generators:
        print(f"  Generating {name}...")
        try:
            sections.append(gen_fn(records))
            sections.append("\n---\n\n")
        except Exception as e:
            sections.append(f"## {name}\n\n_Error generating section: {e}_\n\n---\n\n")
            print(f"  [WARN] {name} failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    # --- Write output ---
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(sections))

    print(f"\n{'=' * 60}")
    print(f"Report written to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
