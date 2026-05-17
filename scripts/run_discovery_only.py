#!/usr/bin/env python3
"""Run ONLY the discovery phase and export results to CSV.

Skips the LLM generation step. Useful for giving discovery output to other
text-to-SQL systems (DAIL-SQL, DIN-SQL, CHESS) as a fair substitute for
the ground-truth db_id they normally receive.

Usage:
    python scripts/run_discovery_only.py --out discovery_test.csv --split test
    python scripts/run_discovery_only.py --n 100 --top-k 5 --out sample.csv

CSV columns:
    bird_index        BIRD dev.json index (-1 for non-BIRD queries)
    question          Natural language question
    gold_db_id        Ground truth BIRD database (snake_case)
    gold_table        Ground truth primary FROM table (snake_case)
    gold_collection   Ground truth Weaviate-style name (PascalCase)
    discovery_hit     1 if any acceptable collection in top-K, else 0
    top1_db, top1_table, top1_collection
    top2_db, top2_table, top2_collection
    ...
    topK_db, topK_table, topK_collection
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mcp.server import MCPServer
from src.utils.embeddings import get_embedding, cosine_similarity
from src.utils.field_kg import FieldLevelKnowledgeGraph
from src.utils.value_stats import ValueStats
from src.test_gorilla.test_enhanced_pipeline import (
    load_all_queries, split_queries, EnhancedPipeline
)


def decode_collection_name(coll_name: str, server: MCPServer) -> tuple[str, str]:
    """Reverse-map a Weaviate collection name to (db_id, table_name).

    E.g. 'CaliforniaSchoolsSchools' -> ('california_schools', 'schools').
    Matches BIRD dev_tables.json to resolve ambiguous splits (e.g.,
    'CardGamesSetTranslations' -> ('card_games', 'set_translations')).
    """
    bird_dbs = [
        "california_schools", "card_games", "codebase_community",
        "debit_card_specializing", "european_football_2", "financial",
        "formula_1", "student_club", "superhero",
        "thrombosis_prediction", "toxicology",
    ]

    def pascal(n: str) -> str:
        return "".join(p.capitalize() for p in n.split("_"))

    # Try matching each BIRD db as prefix
    for db in bird_dbs:
        db_pascal = pascal(db)
        if coll_name.startswith(db_pascal):
            remainder = coll_name[len(db_pascal):]
            if not remainder:
                continue
            # Get actual table names for this db from the schemas registry
            # by checking which table name matches the remainder
            schema = server.schemas.get(coll_name)
            if schema and "sql_table" in schema:
                return db, schema["sql_table"]
            # Fallback: split CamelCase back to snake_case
            table = re.sub(r"([a-z])([A-Z])", r"\1_\2", remainder).lower()
            return db, table

    # Not a BIRD collection (e.g., Weaviate Gorilla's Restaurants)
    return "", coll_name


def main():
    parser = argparse.ArgumentParser(
        description="Run discovery only and export results as CSV"
    )
    parser.add_argument("--out", type=str, default="discovery_results.csv",
                        help="Output CSV path")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of queries (default: all in split)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "dev", "test", "all"])
    parser.add_argument("--top-k", type=int, default=5,
                        help="How many candidates to export per query (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    # Discovery always runs against the full collection pool (preserves
    # distractor pressure across BIRD + Weaviate Gorilla + auxiliary schemas).
    # These flags filter only at CSV-write time.
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--bird-only", action="store_true",
                        help="Only export BIRD queries (exclude Weaviate Gorilla)")
    subset.add_argument("--weaviate-only", action="store_true",
                        help="Only export Weaviate Gorilla queries (exclude BIRD)")
    parser.add_argument("--no-kg", action="store_true")
    parser.add_argument("--no-embedding", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-values", action="store_true")
    parser.add_argument("--no-adaptive", action="store_true")
    parser.add_argument("--no-multi-hop", action="store_true")
    parser.add_argument("--no-joins", action="store_true")
    parser.add_argument("--split-ratios", type=str, default="0.05,0.10,0.85")
    args = parser.parse_args()

    # Flags the pipeline's discover() path never uses but __init__ checks —
    # set sensible defaults so EnhancedPipeline can be constructed without an LLM
    args.no_correction = True         # no generation = no correction
    args.no_primary_rerank = True     # generation-only option
    args.no_relaxed_match = False
    args.no_schema_link = True
    args.no_db_context = True
    args.primary_fix = "none"
    args.tool_schema = "v2"
    args.relaxed_match = True
    args.semantic_correction = False
    args.learn_weights = False
    args.verbose = False
    args.persistent = False
    args.value_enrichment = False
    args.max_dbs = None

    args.split_ratios = tuple(float(x) for x in args.split_ratios.split(","))
    random.seed(args.seed)

    # Load queries (reusing pipeline's loader for consistency).
    # We do NOT filter by --bird-only / --weaviate-only here so that discovery
    # runs against the same query pool regardless of the export subset; the
    # filter is applied just before CSV write.
    all_queries = load_all_queries(include_joins=not args.no_joins)
    all_queries = split_queries(all_queries, seed=args.seed, split=args.split,
                                ratios=args.split_ratios)

    random.shuffle(all_queries)
    queries = all_queries if args.n is None else all_queries[:args.n]
    n = len(queries)

    # Build BIRD dev_index lookup so we can emit bird_index in the CSV
    bird_dev_path = Path("data/bird-benchmark/dev_20240627/dev.json")
    bird_index_by_q = {}
    if bird_dev_path.exists():
        bird_dev = json.loads(bird_dev_path.read_text())
        for i, item in enumerate(bird_dev):
            bird_index_by_q[item["question"].strip().lower()] = i

    # Load MCPServer with all known schema sources
    print("Loading MCPServer...", flush=True)
    source_files = [
        "data/weaviate-gorilla.json",
        "data/bird-processor/bird-to-weaviate.json",
        "data/retail-world-weaviate.json",
        "data/movie-3-weaviate.json",
        "data/student-loan-weaviate.json",
        "data/chicago-crime-weaviate.json",
        "data/university-weaviate.json",
        "data/bird-benchmark/bird-collections.json",
    ]
    existing = [f for f in source_files if Path(f).exists()]
    server = MCPServer.from_multiple_sources(*existing)
    print(f"  {len(server.schemas)} collections loaded", flush=True)

    # Build a pipeline instance solely to reuse its discover() method
    # (we never call generate(), so no LLM is needed)
    pipeline = EnhancedPipeline(server, learning=None, args=args, lm=None)

    # Run discovery for each query
    print(f"\nRunning discovery on {n} queries...", flush=True)
    rows = []
    hits = 0
    for i, q in enumerate(queries):
        question = q["question"]
        acceptable = set(q.get("acceptable_collections", [q["expected_collection"]]))
        gold_coll = q["expected_collection"]
        gold_db = q.get("db_id", "")
        gold_sql = q.get("sql", "")
        has_join = q.get("has_join", False)

        # Extract gold table from SQL (primary FROM table).
        # Handles unquoted, "double-quoted", `back-ticked`, and [bracketed]
        # identifiers — BIRD uses all three for reserved-word table names
        # like `order` and "Match".
        gold_table = ""
        if gold_sql:
            m = re.search(r'FROM\s+["`\[]?(\w+)["`\]]?', gold_sql,
                          re.IGNORECASE)
            if m:
                gold_table = m.group(1)

        # Run discovery (top-K with optional cluster expansion)
        candidates = pipeline.discover(question, top_k=args.top_k,
                                       expand_cluster=False)

        hit = int(any(c in acceptable for c in candidates[:args.top_k]))
        hits += hit

        row = {
            "bird_index": bird_index_by_q.get(question.strip().lower(), -1),
            "question": question,
            "gold_db_id": gold_db,
            "gold_table": gold_table,
            "gold_collection": gold_coll,
            "source": q["source"],
            "has_join": int(has_join),
            "discovery_hit": hit,
        }

        # Flatten top-K candidates into columns
        for rank, coll in enumerate(candidates[:args.top_k], start=1):
            db, table = decode_collection_name(coll, server)
            row[f"top{rank}_collection"] = coll
            row[f"top{rank}_db"] = db
            row[f"top{rank}_table"] = table

        # Pad empty ranks if discovery returned fewer than top-K
        for rank in range(len(candidates) + 1, args.top_k + 1):
            row[f"top{rank}_collection"] = ""
            row[f"top{rank}_db"] = ""
            row[f"top{rank}_table"] = ""

        rows.append(row)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] hit_rate={hits/(i+1)*100:.1f}%", flush=True)

    # Overall hit rate across the full discovery pool (before any subset filter)
    print(f"\nDiscovery hit rate over full pool (correct in top-{args.top_k}): "
          f"{hits}/{n} = {hits/n*100:.1f}%")

    # Apply export-time subset filter (--bird-only / --weaviate-only)
    if args.bird_only:
        export_rows = [r for r in rows if r["source"].startswith("bird_")]
        subset_label = "bird-only"
    elif args.weaviate_only:
        export_rows = [r for r in rows if r["source"] == "weaviate_gorilla"]
        subset_label = "weaviate-only"
    else:
        export_rows = rows
        subset_label = "all"

    # Write CSV
    fieldnames = ["bird_index", "question", "gold_db_id", "gold_table",
                  "gold_collection", "source", "has_join", "discovery_hit"]
    for k in range(1, args.top_k + 1):
        fieldnames += [f"top{k}_db", f"top{k}_table", f"top{k}_collection"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(export_rows)

    print(f"\nWrote {len(export_rows)} rows to {out_path} (subset: {subset_label})")
    if export_rows:
        subset_hits = sum(r["discovery_hit"] for r in export_rows)
        print(f"Discovery hit rate on exported subset (top-{args.top_k}): "
              f"{subset_hits}/{len(export_rows)} = "
              f"{subset_hits/len(export_rows)*100:.1f}%")


if __name__ == "__main__":
    main()
