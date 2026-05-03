#!/usr/bin/env python3
"""
run_eai_direct_mql.py — EAI evaluation with direct MongoDB query generation.

Instead of V2 tool schema → translator (lossy), this path:
1. Uses our discovery phase to find the right collection
2. Shows the LLM the schema WITH sample documents (critical for nested arrays)
3. Asks the LLM to generate the mongosh query directly

This keeps the main pipeline untouched — this is an EAI-specific eval path.

Security note: Uses ast.literal_eval (safe — only parses Python literal
expressions: dicts, lists, strings, numbers, booleans, None) for parsing
the expected.result strings from EAI CSV. Does NOT execute code.

Usage:
    python -u eval/eai/run_eai_direct_mql.py --n 10 --delay 1.0
    python -u eval/eai/run_eai_direct_mql.py --all --delay 1.0
"""

import sys
import os
import json
import re
import ast  # ast.literal_eval: safe literal parser (no code execution)
import time
import argparse
import bson
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval/eai"))
os.chdir(PROJECT_ROOT)

from src.lm.lm import LMService
from src.utils.embeddings import get_embedding, cosine_similarity

from eai_schema_loader import load_eai_schemas, load_eai_queries
from eai_executor import compute_evomql_metrics
from real_mongo_executor import RealMongoExecutor

EAI_DB_DIR = Path("/export/scratch/abrar008/llm_rag/datasets/eai-mongosh/databases")


def load_sample_documents(db_name: str, coll_name: str, n: int = 2) -> list:
    """Load N sample documents from BSON for context."""
    bson_path = EAI_DB_DIR / db_name / f"{coll_name}.bson"
    if not bson_path.exists():
        return []
    with open(bson_path, 'rb') as f:
        docs = bson.decode_all(f.read())
    return docs[:n]


def truncate_doc(doc: dict, max_list: int = 2, max_str: int = 60) -> dict:
    """Truncate a document for prompt inclusion (show structure, not data).

    Critical: datetime objects are rendered as ISODate("...") so the LLM
    knows to use ISODate() in its generated queries.
    """
    from datetime import datetime as dt
    from bson import ObjectId, Decimal128

    def _trunc(obj, depth=0):
        if depth > 4:
            return "..."
        if isinstance(obj, dt):
            return f'ISODate("{obj.isoformat()}")'
        if isinstance(obj, ObjectId):
            return f'ObjectId("{obj}")'
        if isinstance(obj, Decimal128):
            return float(str(obj))
        if isinstance(obj, dict):
            return {k: _trunc(v, depth+1) for k, v in list(obj.items())[:15]}
        if isinstance(obj, list):
            items = [_trunc(v, depth+1) for v in obj[:max_list]]
            if len(obj) > max_list:
                items.append(f"... ({len(obj)} items total)")
            return items
        if isinstance(obj, str) and len(obj) > max_str:
            return obj[:max_str] + "..."
        return obj
    return _trunc(doc)


def build_mql_prompt(question: str, db_name: str, coll_name: str,
                     schema: dict, sample_docs: list) -> str:
    """Build a prompt giving the LLM full context to generate native MQL."""
    props = schema.get("properties", {})
    n_docs = schema.get("n_docs", "?")

    prompt = f"""You are a MongoDB query expert. Generate a single mongosh query to answer the question.

Database: {db_name}
Collection: {coll_name} ({n_docs} documents)

Schema (field_name: type):
"""
    for prop_name, info in sorted(props.items()):
        ptype = info.get("type", "?") if isinstance(info, dict) else "?"
        prompt += f"  {prop_name}: {ptype}\n"

    if sample_docs:
        prompt += "\nSample document:\n```json\n"
        truncated = truncate_doc(sample_docs[0])
        prompt += json.dumps(truncated, indent=2, default=str)
        prompt += "\n```\n"

    prompt += f"""
Question: {question}

Rules:
- Return ONLY the mongosh query (db.{coll_name}.find(...) or db.{coll_name}.aggregate([...]))
- IMPORTANT: Date fields are stored as Date objects. You MUST use ISODate() for ALL date comparisons. Example: {{ saleDate: {{ $gte: ISODate("2017-01-01") }} }}
- NEVER compare date fields to plain strings — always wrap in ISODate()
- Use $unwind when you need to filter/group inside embedded arrays (e.g., $unwind: "$transactions" before matching on transactions.date)
- Include $project to return only the relevant fields
- For aggregations, use .aggregate() with a pipeline array
- No explanations — output ONLY the query

Query:"""

    return prompt


def extract_mql_from_response(response: str) -> str:
    """Extract the MongoDB query from the LLM response.

    Uses balanced-parenthesis matching to handle nested aggregate pipelines
    which contain many () characters.
    """
    if not response:
        return None

    # Strip markdown code blocks
    response = re.sub(r'```(?:javascript|js|mongodb)?\n?', '', response)
    response = response.replace('```', '').strip()

    # Find the start: db.<coll>.<method>(
    match = re.search(r'db\.\w+\.(find|findOne|aggregate)\s*\(', response)
    if not match:
        # If the whole response looks like a query
        if response.strip().startswith('db.'):
            return response.strip()
        return None

    # From the opening paren, find the matching closing paren
    start = match.start()
    paren_start = match.end() - 1  # position of the '('
    depth = 0
    i = paren_start

    while i < len(response):
        ch = response[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                # Found matching close paren
                result = response[start:i+1]
                # Also grab any chained methods: .sort(...).limit(...)
                remainder = response[i+1:]
                chain_match = re.match(r'(\s*\.\w+\([^)]*\))+', remainder)
                if chain_match:
                    result += chain_match.group(0)
                return result
        i += 1

    # If we ran out without finding the close, return from start to end
    return response[start:]


def discover_eai(query: str, schemas: dict, embeddings: dict, top_k: int = 5) -> list:
    """Embedding-based discovery over EAI collections."""
    query_emb = get_embedding(query, model="local")
    scores = []
    for name, emb in embeddings.items():
        sim = cosine_similarity(list(query_emb), list(emb))
        scores.append((name, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scores[:top_k]]


def parse_expected_result(result_str: str):
    """Parse expected.result from EAI CSV using ast.literal_eval (safe)."""
    if not result_str or not result_str.strip():
        return None
    try:
        # ast.literal_eval only parses literals — no code execution
        return ast.literal_eval(result_str)
    except (ValueError, SyntaxError):
        try:
            return json.loads(result_str.replace("'", '"'))
        except Exception:
            return None


def main():
    parser = argparse.ArgumentParser(description="EAI Direct MQL Evaluation")
    parser.add_argument("--n", type=int, default=10, help="Number of queries")
    parser.add_argument("--all", action="store_true", help="Run all queries")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--provider", type=str, default="bedrock")
    parser.add_argument("--model", type=str, default="anthropic.claude-sonnet-4-6")
    parser.add_argument("--top-k", type=int, default=3, help="Discovery top-K")
    parser.add_argument("--oracle-collection", action="store_true",
                        help="Skip discovery, use gold collection (upper bound)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("EAI Direct MQL Evaluation (LLM generates MongoDB natively)")
    print(f"Provider: {args.provider} | Model: {args.model}")
    print(f"Oracle collection: {args.oracle_collection}")
    print("=" * 70)

    # Load data
    print("\nLoading EAI schemas...")
    schemas = load_eai_schemas(sample_docs=5)
    print(f"  {len(schemas)} collections")

    print("Loading queries...")
    queries = load_eai_queries()
    if not args.all:
        queries = queries[:args.n]
    print(f"  {len(queries)} queries")

    # Embeddings for discovery
    if not args.oracle_collection:
        print("Computing embeddings...")
        embeddings = {}
        for name, schema in schemas.items():
            desc = schema.get("description", name)
            props = schema.get("properties", {})
            prop_names = list(props.keys())[:20] if isinstance(props, dict) else []
            text = f"{name}: {desc} Properties: {', '.join(prop_names)}"
            embeddings[name] = get_embedding(text, model="local")
        print(f"  {len(embeddings)} embeddings")
    else:
        embeddings = {}

    # Initialize LM
    print("\nInitializing LLM...")
    if args.provider == "bedrock":
        api_key = None
    elif args.provider == "openai":
        api_key = os.environ.get("NVIDIA_API_KEY")
    else:
        api_key = os.environ.get(f"{args.provider.upper()}_API_KEY")

    lm = LMService(args.provider, args.model, api_key=api_key)

    # Initialize executor (real MongoDB on port 27117)
    print("Connecting to real MongoDB (port 27117)...")
    executor = RealMongoExecutor()
    print("  Connected\n")

    # Run
    results = []
    agg_metrics = defaultdict(float)
    n_scored = 0
    n_gen_errors = 0
    n_exec_errors = 0
    start_time = time.time()

    for i, q in enumerate(queries):
        question = q["question"]
        expected_coll = q["expected_collection"]
        expected_mql = q["expected_mql"]
        expected_result_str = q.get("expected_result", "")
        db_name = q["db_name"]
        coll_name = q["collection_name"]
        complexity = q.get("complexity", "?")

        # Step 1: Discovery (or oracle)
        if args.oracle_collection:
            chosen_coll = expected_coll
            discovery_hit = True
        else:
            candidates = discover_eai(question, schemas, embeddings, top_k=args.top_k)
            discovery_hit = expected_coll in candidates
            chosen_coll = candidates[0] if candidates else expected_coll

        chosen_db = chosen_coll.split('.')[0] if '.' in chosen_coll else db_name
        chosen_coll_name = chosen_coll.split('.')[1] if '.' in chosen_coll else coll_name

        # Step 2: Load sample docs for context
        sample_docs = load_sample_documents(chosen_db, chosen_coll_name, n=2)

        # Step 3: Build prompt and generate MQL
        schema = schemas.get(chosen_coll, schemas.get(expected_coll, {}))
        prompt = build_mql_prompt(question, chosen_db, chosen_coll_name, schema, sample_docs)

        try:
            response = lm.generate(prompt, output_model=None)
            pred_mql = extract_mql_from_response(response)
            if not pred_mql:
                raise ValueError(f"No MQL in response: {(response or '')[:100]}")
        except Exception as e:
            results.append({
                "question": question, "expected_collection": expected_coll,
                "chosen_collection": chosen_coll, "discovery_hit": discovery_hit,
                "complexity": complexity, "db_name": db_name,
                "error": f"generation: {e}",
                "metrics": {"se": 0, "cof": 0, "neo": 0, "ro": 0, "ops": 0},
            })
            n_gen_errors += 1
            if args.delay > 0:
                time.sleep(args.delay)
            continue

        # Step 4: Execute predicted query
        pred_result, pred_error = executor.execute_mql(chosen_db, pred_mql)
        if pred_error:
            n_exec_errors += 1

        # Step 5: Get gold results
        gold_result, gold_error = executor.execute_mql(db_name, expected_mql)
        if gold_error and expected_result_str:
            gold_result = parse_expected_result(expected_result_str)

        # Step 6: Compute metrics
        metrics = compute_evomql_metrics(gold_result, pred_result, pred_error)

        for k, v in metrics.items():
            agg_metrics[k] += v
        n_scored += 1

        strict_match = chosen_coll == expected_coll

        # Progress
        elapsed = time.time() - start_time
        sym = "+" if metrics['cof'] > 0.5 else ("~" if metrics['se'] > 0 else "-")
        if i < 5 or (i + 1) % 20 == 0 or args.verbose:
            avg_cof = agg_metrics['cof'] / n_scored
            avg_ops = agg_metrics['ops'] / n_scored
            print(f"  {sym} [{i+1}/{len(queries)}] "
                  f"COF:{metrics['cof']:.2f} OPS:{metrics['ops']:.2f} "
                  f"(avg COF:{avg_cof:.3f} OPS:{avg_ops:.3f}) "
                  f"disc:{discovery_hit}"
                  + (f" err:{pred_error[:40]}" if pred_error else "")
                  + f" ({elapsed/(i+1):.1f}s/q)", flush=True)

        results.append({
            "question": question, "expected_collection": expected_coll,
            "chosen_collection": chosen_coll, "strict_match": strict_match,
            "discovery_hit": discovery_hit, "complexity": complexity,
            "db_name": db_name, "pred_mql": pred_mql,
            "expected_mql": expected_mql, "pred_error": pred_error,
            "metrics": metrics,
        })

        if args.delay > 0:
            time.sleep(args.delay)

    # Summary
    elapsed = time.time() - start_time
    n = len(queries)
    strict_correct = sum(1 for r in results if r.get("strict_match"))
    discovery_hits = sum(1 for r in results if r.get("discovery_hit"))

    print(f"\n{'=' * 70}")
    print("EAI DIRECT MQL RESULTS")
    print(f"{'=' * 70}")
    print(f"  Model:              {args.model}")
    print(f"  Oracle collection:  {args.oracle_collection}")
    print(f"  Queries:            {n}")
    print(f"  Scored:             {n_scored}")
    print(f"  Generation errors:  {n_gen_errors}")
    print(f"  Execution errors:   {n_exec_errors}")
    print(f"  Time:               {elapsed:.0f}s ({elapsed/n:.1f}s/q)")
    print()
    print(f"  Discovery rate:     {discovery_hits}/{n} ({100*discovery_hits/n:.1f}%)")
    print(f"  Strict coll. acc.:  {strict_correct}/{n} ({100*strict_correct/n:.1f}%)")
    print()

    if n_scored > 0:
        print(f"  EvoMQL Metrics (N={n_scored}):")
        for metric in ['se', 'cof', 'neo', 'ro', 'ops']:
            avg = agg_metrics[metric] / n_scored
            print(f"    {metric.upper():>4s}: {avg:.3f}")

        print(f"\n  Per-complexity:")
        for c in ['simple', 'moderate', 'complex']:
            scored = [r for r in results if r.get('complexity') == c and 'metrics' in r]
            if scored:
                ops = sum(r['metrics']['ops'] for r in scored) / len(scored)
                cof = sum(r['metrics']['cof'] for r in scored) / len(scored)
                se = sum(r['metrics']['se'] for r in scored) / len(scored)
                print(f"    {c:10s}: SE={se:.3f} COF={cof:.3f} OPS={ops:.3f} (N={len(scored)})")

    print(f"\n  Comparison (EvoMQL Table 1, EAI ID):")
    print(f"    GPT-4o:              COF=0.700 OPS=0.784")
    print(f"    Qwen3-30B-Thinking:  COF=0.671 OPS=0.746")
    print(f"    EvoMQL (3B, RL):     COF=0.766 OPS=0.821")
    if n_scored > 0:
        our_cof = agg_metrics['cof'] / n_scored
        our_ops = agg_metrics['ops'] / n_scored
        model_short = args.model.split('.')[-1] if '.' in args.model else args.model
        print(f"    Ours ({model_short}): COF={our_cof:.3f} OPS={our_ops:.3f}")

    # Save
    results_dir = Path("eval/eai/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model.split('.')[-1] if '.' in args.model else args.model
    mode = "oracle" if args.oracle_collection else "disc"
    fname = f"eai_directmql_{model_short}_{mode}_n{n}.json"
    results_path = results_dir / fname

    results_path.write_text(json.dumps({
        "provider": args.provider,
        "model": args.model,
        "oracle_collection": args.oracle_collection,
        "n": n, "n_scored": n_scored,
        "metrics": {k: v / n_scored if n_scored > 0 else 0 for k, v in agg_metrics.items()},
        "strict_accuracy": strict_correct / n if n > 0 else 0,
        "discovery_rate": discovery_hits / n if n > 0 else 0,
        "elapsed_seconds": elapsed,
        "results": results,
    }, indent=2, default=str))
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
