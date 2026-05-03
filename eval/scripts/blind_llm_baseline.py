#!/usr/bin/env python3
"""Blind LLM baseline: all 143 schemas in one prompt, no pipeline assistance.

Measures raw LLM collection-selection ability without embeddings, KG, or
progressive disclosure. Comparison target for the pipeline -- demonstrates
the value of the discovery phase. Also supports --max-dbs for scale
experiments (accuracy vs. number of databases).
"""

import sys
import os
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.mcp.server import MCPServer
from src.lm.lm import LMService
from src.utils.weaviate_fc_utils import build_weaviate_query_tool_for_openai
from src.test_gorilla.test_enhanced_pipeline import load_all_queries, split_queries


def build_all_schemas_description(server: MCPServer, collections: list[str]) -> str:
    """Build a text description of all collection schemas for the blind prompt."""
    lines = []
    for coll_name in sorted(collections):
        schema = server.get_collection_schema(coll_name, compressed=False)
        if schema is None:
            continue
        desc = schema.get("description", "No description")
        props = schema.get("properties", {})
        if isinstance(props, dict):
            prop_items = list(props.items())[:10]
            prop_list = ", ".join(
                f"{pname} ({pinfo.get('type', '?') if isinstance(pinfo, dict) else '?'})"
                for pname, pinfo in prop_items
            )
            if len(props) > 10:
                prop_list += f", ... (+{len(props) - 10} more)"
        else:
            prop_list = str(props)[:100]
        lines.append(f"- **{coll_name}**: {desc}\n  Properties: {prop_list}")
    return "\n".join(lines)


def build_blind_prompt(question: str, schemas_description: str, n_collections: int) -> str:
    """Build the blind evaluation prompt: question + all schemas."""
    return f"""You have access to {n_collections} database collections. Based on the question below,
select the MOST RELEVANT collection to query by calling the tool.

AVAILABLE COLLECTIONS:
{schemas_description}

QUESTION: {question}

You MUST call the tool with the collection_name that best answers this question.
Pick the single most relevant collection. Do NOT explain — just call the tool."""


def main():
    parser = argparse.ArgumentParser(description="Blind LLM Baseline (no pipeline)")
    parser.add_argument("--n", type=int, default=50, help="Number of queries")
    parser.add_argument("--all", action="store_true", help="Test ALL queries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "dev", "test", "all"])
    parser.add_argument("--split-ratios", type=str, default="0.05,0.10,0.85")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between queries")
    parser.add_argument("--provider", type=str, required=True,
                        choices=["openai", "bedrock", "anthropic", "together", "ollama"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ollama-host", type=str, default=None)
    parser.add_argument("--max-dbs", type=int, default=None,
                        help="Limit to first N databases (for scale experiment)")
    parser.add_argument("--relaxed-match", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for results JSON")
    args = parser.parse_args()
    args.split_ratios = tuple(float(x) for x in args.split_ratios.split(','))
    random.seed(args.seed)

    # --- Load queries ---
    all_queries = load_all_queries(include_joins=True)
    all_queries = split_queries(all_queries, seed=args.seed, split=args.split,
                                ratios=args.split_ratios)

    # --- Load server ---
    print("Loading MCPServer...")
    source_files = [
        "data/weaviate-gorilla.json",
        "data/bird-processor/bird-to-weaviate.json",
        "data/retail-world-weaviate.json",
        "data/movie-3-weaviate.json",
        "data/student-loan-weaviate.json",
        "data/chicago-crime-weaviate.json",
        "data/university-weaviate.json",
        "data/bird-benchmark/bird-collections.json"
    ]
    existing_files = [f for f in source_files if Path(f).exists()]
    server = MCPServer.from_multiple_sources(*existing_files)
    all_collections = sorted(server.schemas.keys())
    print(f"  {len(all_collections)} collections loaded")

    # --- Scale experiment: filter by max-dbs ---
    if args.max_dbs:
        allowed_sources = sorted(set(q['source'] for q in all_queries))[:args.max_dbs]
        all_queries = [q for q in all_queries if q['source'] in allowed_sources]
        # Also filter collections to only those from allowed databases
        allowed_prefixes = set()
        for src in allowed_sources:
            db_name = src.replace("bird_", "").replace("weaviate_", "")
            # Find collections belonging to this DB
            for coll in all_collections:
                # Collection names typically start with the DB name in PascalCase
                if coll.lower().startswith(db_name.replace("_", "").lower()) or \
                   any(coll.lower().startswith(part) for part in db_name.split("_")):
                    allowed_prefixes.add(coll)
        # Simpler approach: keep collections that are expected by any query in our set
        expected_colls = set()
        for q in all_queries:
            expected_colls.add(q["expected_collection"])
            expected_colls.update(q.get("acceptable_collections", []))
        # Include all collections from same DB groups
        # Use source->collection mapping from queries
        source_colls = defaultdict(set)
        all_raw_queries = load_all_queries(include_joins=True)
        for q in all_raw_queries:
            if q["source"] in allowed_sources:
                source_colls[q["source"]].add(q["expected_collection"])
                source_colls[q["source"]].update(q.get("acceptable_collections", []))
        db_collections = set()
        for colls in source_colls.values():
            db_collections.update(colls)
        # Also include weaviate_gorilla collections if that source is allowed
        if "weaviate_gorilla" in allowed_sources:
            for q in all_raw_queries:
                if q["source"] == "weaviate_gorilla":
                    db_collections.add(q["expected_collection"])
        # Use full collection list (not filtered) to simulate realistic "haystack"
        # But for scale experiment, only include collections from the allowed DBs
        filtered_collections = sorted(db_collections) if db_collections else all_collections
        print(f"  [Scale experiment] {args.max_dbs} databases, {len(filtered_collections)} collections, {len(all_queries)} queries")
    else:
        filtered_collections = all_collections

    random.shuffle(all_queries)
    test_queries = all_queries if args.all else all_queries[:args.n]
    n = len(test_queries)

    # --- Initialize LLM ---
    if args.provider == "bedrock":
        api_key = None
    elif args.provider == "ollama":
        api_key = None
        if args.ollama_host:
            os.environ["OLLAMA_HOST"] = args.ollama_host
    elif args.provider == "openai":
        api_key = os.environ.get("NVIDIA_API_KEY") if "/" in args.model else os.environ.get("OPENAI_API_KEY")
    else:
        api_key = os.environ.get(f"{args.provider.upper()}_API_KEY")

    lm = LMService(args.provider, args.model, api_key=api_key)

    # --- Build blind prompt components ---
    schemas_description = build_all_schemas_description(server, filtered_collections)
    tool = build_weaviate_query_tool_for_openai(
        collections_description=schemas_description,
        collections_list=filtered_collections
    )
    n_collections = len(filtered_collections)

    # Token count estimate
    schema_tokens = len(schemas_description.split())  # rough word count
    print(f"\nBlind baseline: {n_collections} collections, ~{schema_tokens} words in schema description")

    print("=" * 70)
    print("BLIND LLM BASELINE (NO PIPELINE)")
    print(f"Provider: {args.provider} | Model: {args.model}")
    print(f"Collections: {n_collections} | Queries: {n} | Split: {args.split}")
    if args.max_dbs:
        print(f"Scale: {args.max_dbs} databases")
    print("=" * 70)

    # --- Run evaluation ---
    correct = 0
    relaxed_correct = 0
    join_total = 0
    join_correct = 0
    coll_in_output = 0
    per_db = defaultdict(lambda: {"correct": 0, "relaxed_correct": 0, "total": 0})
    results = []
    prompts_saved = []
    start = time.time()

    for i, q in enumerate(test_queries):
        question = q["question"]
        expected = q["expected_collection"]
        acceptable = q.get("acceptable_collections", [expected])
        has_join = q.get("has_join", False)
        source = q["source"]
        db = source.replace("bird_", "")

        per_db[db]["total"] += 1
        if has_join:
            join_total += 1

        # Build the blind prompt for this query
        blind_prompt = build_blind_prompt(question, schemas_description, n_collections)

        # Save prompt (first 10 + every 100th for reference)
        if i < 10 or i % 100 == 0:
            prompts_saved.append({
                "query_index": i,
                "question": question,
                "expected_collection": expected,
                "prompt_length_chars": len(blind_prompt),
                "prompt": blind_prompt
            })

        # Call LLM with tool
        got_coll = None
        got_args = {}
        _t0 = time.time()
        try:
            tool_response = lm.one_step_function_selection_test(
                prompt=blind_prompt,
                tools=[tool],
                parallel_tool_calls=False
            )

            if tool_response is not None:
                # Handle different return formats
                if isinstance(tool_response, list) and len(tool_response) > 0:
                    tc = tool_response[0]
                    if hasattr(tc, 'function'):
                        got_args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                    else:
                        got_args = tc if isinstance(tc, dict) else {}
                elif isinstance(tool_response, dict):
                    got_args = tool_response

                got_coll = got_args.get("collection_name")
        except Exception as e:
            print(f"    [ERROR] Query {i}: {e}")
        _elapsed = time.time() - _t0
        call_meta = lm.last_call_meta.copy()

        # Rate limit
        if args.delay > 0:
            time.sleep(args.delay)

        # Evaluate
        strict_match = got_coll == expected
        relaxed_match = got_coll in acceptable if args.relaxed_match else strict_match
        output_match = got_coll is not None and expected in str(got_args)

        if strict_match:
            correct += 1
            per_db[db]["correct"] += 1
        if relaxed_match:
            relaxed_correct += 1
            per_db[db]["relaxed_correct"] += 1
        if has_join and relaxed_match:
            join_correct += 1
        if output_match or strict_match:
            coll_in_output += 1

        result_entry = {
            "question": question,
            "expected": expected,
            "acceptable": acceptable,
            "got_collection": got_coll,
            "strict_match": strict_match,
            "relaxed_match": relaxed_match,
            "has_join": has_join,
            "source": source,
            "latency_s": round(_elapsed, 3),
            "input_tokens": call_meta.get("input_tokens", 0),
            "output_tokens": call_meta.get("output_tokens", 0),
        }
        results.append(result_entry)

        # Progress
        sym = "+" if strict_match else ("~" if relaxed_match else "-")
        elapsed = time.time() - start
        print(f"  {sym} [{i+1}/{n}] {db:25s} got={got_coll or 'None':30s} exp={expected:30s} "
              f"Strict:{correct/(i+1)*100:.1f}% Rlx:{relaxed_correct/(i+1)*100:.1f}% "
              f"({elapsed/(i+1):.1f}s/q)", flush=True)

    elapsed = time.time() - start

    # --- Print results ---
    print(f"\n{'=' * 70}")
    print("BLIND BASELINE RESULTS")
    print(f"{'=' * 70}")
    print(f"Collection Accuracy: {correct}/{n} ({correct/n*100:.1f}%)  [strict]")
    print(f"Relaxed Accuracy:    {relaxed_correct}/{n} ({relaxed_correct/n*100:.1f}%)")
    print(f"Coll-in-Output:      {coll_in_output}/{n} ({coll_in_output/n*100:.1f}%)")
    print(f"Time:                {elapsed:.1f}s ({elapsed/n:.1f}s/query)")

    if join_total > 0:
        non_join_total = n - join_total
        non_join_correct = correct - sum(1 for r in results if r["has_join"] and r["strict_match"])
        print(f"\n  Single-table:      {non_join_correct}/{non_join_total} ({non_join_correct/non_join_total*100:.1f}%)" if non_join_total > 0 else "")
        print(f"  JOIN (strict):     {sum(1 for r in results if r['has_join'] and r['strict_match'])}/{join_total} ({sum(1 for r in results if r['has_join'] and r['strict_match'])/join_total*100:.1f}%)")
        print(f"  JOIN (relaxed):    {join_correct}/{join_total} ({join_correct/join_total*100:.1f}%)")

    print(f"\nPer-Database Accuracy (Strict / Relaxed):")
    for db in sorted(per_db.keys()):
        d = per_db[db]
        total = d["total"]
        sc = d["correct"]
        rc = d["relaxed_correct"]
        spct = sc / total * 100 if total > 0 else 0
        rpct = rc / total * 100 if total > 0 else 0
        gap_str = f" gap:{rpct - spct:.0f}%" if rpct > spct else ""
        print(f"  {db:30s} {sc}/{total} ({spct:.1f}%) rlx:{rpct:.1f}%{gap_str}")
    print(f"{'=' * 70}")

    # --- Save results ---
    import statistics as _stats

    model_short_map = {
        "openai/gpt-oss-120b": "nim_gptoss",
        "anthropic.claude-sonnet-4-6": "bedrock_sonnet46",
        "anthropic.claude-opus-4-6-v1": "bedrock_opus46",
        "qwen.qwen3-next-80b-a3b": "bedrock_qwen3_80b",
        "meta.llama4-maverick-17b-instruct-v1:0": "bedrock_llama4_maverick",
    }
    paper_model_map = {
        "openai/gpt-oss-120b": "nim",
        "anthropic.claude-sonnet-4-6": "sonnet",
        "anthropic.claude-opus-4-6-v1": "opus",
        "qwen.qwen3-next-80b-a3b": "qwen",
        "meta.llama4-maverick-17b-instruct-v1:0": "llama",
    }
    model_short = model_short_map.get(args.model,
                                       args.model.replace("/", "_").replace(".", "_").replace(":", "_"))
    split_suffix = f"_{args.split}" if args.split != "all" else ""
    dbs_suffix = f"_dbs{args.max_dbs}" if args.max_dbs else ""

    if getattr(args, 'output_dir', None):
        results_dir = Path(args.output_dir)
        paper_model = paper_model_map.get(args.model, model_short)
        results_file = results_dir / f"blind_{paper_model}_n{n}{split_suffix}.json"
        prompts_file = results_dir / f"blind_{paper_model}_n{n}{split_suffix}_prompts.json"
    elif args.max_dbs:
        results_dir = Path("eval/results/scale_exp")
        results_file = results_dir / f"blind_{model_short}{dbs_suffix}_n{n}{split_suffix}.json"
        prompts_file = results_dir / f"blind_{model_short}{dbs_suffix}_n{n}{split_suffix}_prompts.json"
    else:
        results_dir = Path("eval/results/baselines")
        results_file = results_dir / f"blind_{model_short}{dbs_suffix}_n{n}{split_suffix}.json"
        prompts_file = results_dir / f"blind_{model_short}{dbs_suffix}_n{n}{split_suffix}_prompts.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Compute latency/token aggregates
    latencies = [r["latency_s"] for r in results if r.get("latency_s", 0) > 0]
    input_toks = [r["input_tokens"] for r in results if r.get("input_tokens", 0) > 0]
    output_toks = [r["output_tokens"] for r in results if r.get("output_tokens", 0) > 0]

    results_file.write_text(json.dumps({
        "experiment": "blind_llm_baseline",
        "provider": args.provider,
        "model": args.model,
        "model_short": model_short,
        "split": args.split,
        "max_dbs": args.max_dbs,
        "n_collections": n_collections,
        "n": n,
        "strict_accuracy": correct / n,
        "relaxed_accuracy": relaxed_correct / n,
        "coll_in_output": coll_in_output / n,
        "join_total": join_total,
        "join_strict": sum(1 for r in results if r["has_join"] and r["strict_match"]),
        "join_relaxed": join_correct,
        "single_table_accuracy": (correct - sum(1 for r in results if r["has_join"] and r["strict_match"])) / (n - join_total) if (n - join_total) > 0 else None,
        "per_database": {db: dict(per_db[db]) for db in per_db},
        "elapsed_seconds": elapsed,
        "schema_description_length": len(schemas_description),
        "avg_latency_s": _stats.mean(latencies) if latencies else 0,
        "median_latency_s": _stats.median(latencies) if latencies else 0,
        "avg_input_tokens": _stats.mean(input_toks) if input_toks else 0,
        "avg_output_tokens": _stats.mean(output_toks) if output_toks else 0,
        "results": results,
    }, indent=2))
    print(f"\nResults saved to {results_file}")

    # Save prompts separately (they're large)
    prompts_file.write_text(json.dumps({
        "experiment": "blind_llm_baseline",
        "model": args.model,
        "n_collections": n_collections,
        "max_dbs": args.max_dbs,
        "total_queries": n,
        "prompts_saved": len(prompts_saved),
        "schema_description": schemas_description[:2000] + "..." if len(schemas_description) > 2000 else schemas_description,
        "prompts": prompts_saved,
    }, indent=2))
    print(f"Prompts saved to {prompts_file}")


if __name__ == "__main__":
    main()
