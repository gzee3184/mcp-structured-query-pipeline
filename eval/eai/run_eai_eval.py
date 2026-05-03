#!/usr/bin/env python3
"""
run_eai_eval.py — End-to-end EAI MongoDB evaluation using our pipeline.

Runs: discovery → V2 generation (Sonnet via Bedrock) → MongoDB translation →
mongomock execution → EvoMQL metric computation.

Usage:
    python -u eval/eai/run_eai_eval.py --n 10 --delay 1.0  # smoke test
    python -u eval/eai/run_eai_eval.py --all --delay 1.0    # full run

Security: Uses ast.literal_eval (safe — only parses literals) for parsing
expected.result strings from the EAI CSV. No arbitrary code execution.
"""

import sys
import os
import json
import ast  # ast.literal_eval is safe: only parses dicts/lists/strings/numbers
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval/eai"))
os.chdir(PROJECT_ROOT)

from src.lm.lm import LMService
from src.utils.embeddings import get_embedding, cosine_similarity
from src.utils.weaviate_fc_utils import build_weaviate_query_tool_for_openai_v2

from eai_schema_loader import load_eai_schemas, load_eai_queries
from eai_executor import MongoExecutor, compute_evomql_metrics
from mongo_translator import translate_v2_to_mql


def discover_eai(query: str, schemas: dict, embeddings: dict, top_k: int = 5) -> list:
    """Embedding-based discovery over EAI collections."""
    query_emb = get_embedding(query, model="local")
    scores = []
    for name, emb in embeddings.items():
        sim = cosine_similarity(list(query_emb), list(emb))
        scores.append((name, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scores[:top_k]]


def build_prompt(question: str, candidates: list, schemas: dict) -> str:
    """Build the generation prompt from discovered candidates."""
    prompt = f"Q: {question}\n\nCollections:\n\n"
    for coll in candidates:
        schema = schemas.get(coll, {})
        props = schema.get("properties", {})
        desc = schema.get("description", coll)
        n_docs = schema.get("n_docs", "?")

        prompt += f"## {coll} ({n_docs} documents)\n"
        prompt += f"  {desc}\n"

        if isinstance(props, dict):
            for name, info in list(props.items())[:40]:
                ptype = info.get("type", "?") if isinstance(info, dict) else "?"
                prompt += f"  - {name} ({ptype})\n"
            if len(props) > 40:
                prompt += f"  ... ({len(props) - 40} more properties)\n"
        prompt += "\n"

    return prompt


def parse_expected_result(result_str: str):
    """Parse the expected.result string from the EAI CSV.

    Uses ast.literal_eval (safe: only parses Python literals — dicts, lists,
    strings, numbers, booleans, None. Does NOT execute arbitrary code).
    """
    if not result_str or not result_str.strip():
        return None
    try:
        # ast.literal_eval is safe — only parses literal expressions
        return ast.literal_eval(result_str)
    except (ValueError, SyntaxError):
        try:
            return json.loads(result_str.replace("'", '"'))
        except Exception:
            return None


def main():
    parser = argparse.ArgumentParser(description="EAI MongoDB Evaluation")
    parser.add_argument("--n", type=int, default=10, help="Number of queries")
    parser.add_argument("--all", action="store_true", help="Run all queries")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between queries")
    parser.add_argument("--provider", type=str, default="bedrock")
    parser.add_argument("--model", type=str, default="anthropic.claude-sonnet-4-6")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("EAI MongoDB Evaluation")
    print(f"Provider: {args.provider} | Model: {args.model}")
    print("=" * 70)

    # Load schemas and queries
    print("\nLoading EAI schemas...")
    schemas = load_eai_schemas()
    print(f"  {len(schemas)} collections loaded")

    print("Loading EAI queries...")
    queries = load_eai_queries()
    if not args.all:
        queries = queries[:args.n]
    print(f"  {len(queries)} queries to evaluate")

    # Build embeddings for discovery
    print("Computing embeddings for EAI collections...")
    embeddings = {}
    for name, schema in schemas.items():
        desc = schema.get("description", name)
        props = schema.get("properties", {})
        prop_names = list(props.keys())[:20] if isinstance(props, dict) else []
        text = f"{name}: {desc} Properties: {', '.join(prop_names)}"
        embeddings[name] = get_embedding(text, model="local")
    print(f"  {len(embeddings)} embeddings computed")

    # Initialize LM
    print("\nInitializing LLM...")
    if args.provider == "bedrock":
        api_key = None
    elif args.provider == "openai":
        api_key = os.environ.get("NVIDIA_API_KEY")
    else:
        api_key = os.environ.get(f"{args.provider.upper()}_API_KEY")

    lm = LMService(args.provider, args.model, api_key=api_key)
    lm.connection_test()

    # Initialize executor
    print("Loading MongoDB data into mongomock...")
    executor = MongoExecutor()
    executor.load_data()
    print("  Data loaded")

    # Run evaluation
    print(f"\nRunning evaluation on {len(queries)} queries...")
    results = []
    agg_metrics = defaultdict(float)
    n_scored = 0
    n_gen_errors = 0
    n_translate_errors = 0
    start_time = time.time()

    for i, q in enumerate(queries):
        question = q["question"]
        expected_coll = q["expected_collection"]
        expected_mql = q["expected_mql"]
        expected_result_str = q.get("expected_result", "")
        db_name = q["db_name"]
        complexity = q.get("complexity", "?")

        # Step 1: Discovery
        candidates = discover_eai(question, schemas, embeddings, top_k=5)
        discovery_hit = expected_coll in candidates

        # Step 2: Build prompt + generate V2 tool call
        prompt = build_prompt(question, candidates, schemas)
        tool = build_weaviate_query_tool_for_openai_v2(prompt, candidates)

        try:
            response = lm.one_step_function_selection_test(
                question, [tool], False, tool_schema="v2"
            )
            if response is None:
                raise ValueError("No tool call returned")
            query_args = json.loads(response[0].function.arguments)
        except Exception as e:
            results.append({
                "question": question, "expected_collection": expected_coll,
                "db_name": db_name, "complexity": complexity,
                "discovery_hit": discovery_hit, "error": f"generation: {e}",
                "metrics": {"se": 0, "cof": 0, "neo": 0, "ro": 0, "ops": 0},
            })
            n_gen_errors += 1
            if args.delay > 0:
                time.sleep(args.delay)
            continue

        # Step 3: Collection accuracy
        got_coll = query_args.get("collection_name", "")
        strict_match = got_coll == expected_coll

        # Step 4: Translate to MongoDB
        pred_mql = translate_v2_to_mql(query_args, db_name)
        if pred_mql is None:
            results.append({
                "question": question, "expected_collection": expected_coll,
                "got_collection": got_coll, "strict_match": strict_match,
                "discovery_hit": discovery_hit, "query_args": query_args,
                "error": "translation_error",
                "metrics": {"se": 0, "cof": 0, "neo": 0, "ro": 0, "ops": 0},
            })
            n_translate_errors += 1
            if args.delay > 0:
                time.sleep(args.delay)
            continue

        # Step 5: Execute predicted query
        pred_result, pred_error = executor.execute_mql(db_name, pred_mql)

        # Step 6: Get gold results
        gold_result, gold_error = executor.execute_mql(db_name, expected_mql)
        if gold_error and expected_result_str:
            gold_result = parse_expected_result(expected_result_str)

        # Step 7: Compute EvoMQL metrics
        metrics = compute_evomql_metrics(gold_result, pred_result, pred_error)

        for k, v in metrics.items():
            agg_metrics[k] += v
        n_scored += 1

        # Progress
        elapsed = time.time() - start_time
        sym = "+" if metrics['cof'] > 0.5 else ("~" if metrics['se'] > 0 else "-")
        if i < 5 or (i + 1) % 20 == 0 or args.verbose:
            print(f"  {sym} [{i+1}/{len(queries)}] {db_name:20s} "
                  f"COF:{metrics['cof']:.2f} OPS:{metrics['ops']:.2f} "
                  f"strict:{strict_match} disc:{discovery_hit} "
                  f"({elapsed/(i+1):.1f}s/q)", flush=True)

        results.append({
            "question": question, "expected_collection": expected_coll,
            "got_collection": got_coll, "strict_match": strict_match,
            "discovery_hit": discovery_hit, "candidates": candidates,
            "query_args": query_args, "pred_mql": pred_mql,
            "pred_error": pred_error, "metrics": metrics,
            "complexity": complexity, "db_name": db_name,
        })

        if args.delay > 0:
            time.sleep(args.delay)

    # Summary
    elapsed = time.time() - start_time
    n = len(queries)
    strict_correct = sum(1 for r in results if r.get("strict_match"))
    discovery_hits = sum(1 for r in results if r.get("discovery_hit"))

    print(f"\n{'=' * 70}")
    print("EAI EVALUATION RESULTS")
    print(f"{'=' * 70}")
    print(f"  Queries:            {n}")
    print(f"  Scored:             {n_scored}")
    print(f"  Generation errors:  {n_gen_errors}")
    print(f"  Translation errors: {n_translate_errors}")
    print(f"  Time:               {elapsed:.0f}s ({elapsed/n:.1f}s/q)")
    print()
    print(f"  Discovery rate:     {discovery_hits}/{n} ({100*discovery_hits/n:.1f}%)")
    print(f"  Strict accuracy:    {strict_correct}/{n} ({100*strict_correct/n:.1f}%)")
    print()

    if n_scored > 0:
        print(f"  EvoMQL Metrics (N={n_scored} scored):")
        for metric in ['se', 'cof', 'neo', 'ro', 'ops']:
            avg = agg_metrics[metric] / n_scored
            print(f"    {metric.upper():>4s}: {avg:.3f}")

        print(f"\n  Per-complexity OPS:")
        for complexity in ['simple', 'moderate', 'complex']:
            scored = [r for r in results if r.get('complexity') == complexity and 'metrics' in r]
            if scored:
                avg_ops = sum(r['metrics']['ops'] for r in scored) / len(scored)
                print(f"    {complexity}: OPS={avg_ops:.3f} (N={len(scored)})")

    # Comparison context
    print(f"\n  Comparison to EvoMQL Table 1 (EAI ID):")
    print(f"    GPT-4o:              COF=0.700 OPS=0.784")
    print(f"    Qwen3-30B-Thinking:  COF=0.671 OPS=0.746")
    print(f"    EvoMQL (3B, RL):     COF=0.766 OPS=0.821")
    if n_scored > 0:
        our_cof = agg_metrics['cof'] / n_scored
        our_ops = agg_metrics['ops'] / n_scored
        print(f"    Ours (pipeline+{args.model.split('.')[-1]}): COF={our_cof:.3f} OPS={our_ops:.3f}")

    # Save results
    results_dir = Path("eval/eai/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.split('.')[-1] if '.' in args.model else args.model
    results_file = results_dir / f"eai_{model_short}_n{n}.json"

    results_file.write_text(json.dumps({
        "provider": args.provider,
        "model": args.model,
        "n": n,
        "n_scored": n_scored,
        "metrics": {k: v / n_scored if n_scored > 0 else 0 for k, v in agg_metrics.items()},
        "strict_accuracy": strict_correct / n if n > 0 else 0,
        "discovery_rate": discovery_hits / n if n > 0 else 0,
        "elapsed_seconds": elapsed,
        "results": results,
    }, indent=2, default=str))

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
