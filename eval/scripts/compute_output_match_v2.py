#!/usr/bin/env python3
"""
compute_output_match_v2.py — Direct comparison of declared output_properties
against SQL SELECT columns, for V2 tool-schema runs.

This is the Phase-1 successor to compute_output_match.py. Where v1 inferred
output scope from the primary collection's property set (a proxy), V2 runs
have explicit output_properties in the tool call, so we compare them
directly to what the SQL SELECTs — the honest SELECT-equivalence check.

Metrics produced:
  strict                  : predicted primary == SQL FROM table
  relaxed                 : primary ∈ FROM/JOIN tables
  output_exact            : declared output_properties SET == SQL SELECT SET
                            (collection, property, aggregation tuples match exactly)
  output_cols_covered     : every SQL SELECT column appears in declared
                            output_properties (allows extras in prediction)
  output_cols_subset      : every declared output property appears in SQL SELECT
                            (allows missing cols but penalizes extras)
  output_iou              : average Jaccard overlap on the (col, prop, agg) sets

Usage:
    python eval/scripts/compute_output_match_v2.py <v2_result_json>
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

try:
    import sqlglot
    from sqlglot import exp
except ImportError:
    print("ERROR: sqlglot not installed. Run: pip install sqlglot", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def pascal(n):
    return "".join(p.capitalize() for p in n.split("_"))


def convert_name(db, table):
    return f"{pascal(db)}{pascal(table)}"


# ---- BIRD Weaviate<->SQL property name mapping -------------------------------
# Built by eval/scripts/build_bird_name_map.py from BIRD's own dev_tables.json.
# Lets us canonicalize predicted output_properties against SQL SELECT columns
# without being penalized for Weaviate-side descriptive naming.

_NAME_MAP_CACHE = None


def _load_name_map():
    global _NAME_MAP_CACHE
    if _NAME_MAP_CACHE is not None:
        return _NAME_MAP_CACHE
    path = PROJECT_ROOT / "data/bird-benchmark/bird_property_name_map.json"
    if path.exists():
        _NAME_MAP_CACHE = json.loads(path.read_text())
    else:
        _NAME_MAP_CACHE = {}
    return _NAME_MAP_CACHE


def _norm_key(s):
    """Same normalization the name-map builder uses."""
    if s is None:
        return ""
    import re as _re
    return _re.sub(r"[\s_\-]+", "", str(s).lower())


def canonicalize_to_sql(collection: str, prop_name: str) -> str:
    """Map a Weaviate-side property name to its BIRD-SQL canonical form, if known."""
    m = _load_name_map()
    if not m:
        return prop_name
    w2s = m.get("weaviate_to_sql", {}).get(collection, {})
    sql = w2s.get(_norm_key(prop_name))
    return sql if sql else prop_name


def norm_prop(s):
    """Normalize a property name (strip case/spaces/underscores)."""
    if s is None:
        return ""
    return str(s).lower().replace("_", "").replace(" ", "").replace("-", "")


def norm_agg(s):
    if not s or str(s).upper() in ("NONE", ""):
        return "NONE"
    return str(s).upper()


def parse_sql_select(sql, db_id):
    """Parse SQL and extract the SELECT list as {(collection, norm_prop, agg_fn), ...}.

    For SQL aggregate functions like COUNT(T1.id) or SUM(T2.points), detect the
    aggregation on the column itself. Literal SELECT columns map to agg=NONE.
    """
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        try:
            parsed = sqlglot.parse_one(sql)
        except Exception:
            return None

    if parsed is None:
        return None

    alias_to_table = {}
    for table in parsed.find_all(exp.Table):
        tname = table.name
        alias = table.alias or tname
        alias_to_table[alias.lower()] = tname

    select_node = parsed.find(exp.Select)
    if select_node is None:
        return None

    # Map sqlglot aggregate nodes to our agg enum
    agg_node_map = {
        "Count": "COUNT", "Sum": "SUM", "Min": "MIN", "Max": "MAX",
        "Avg": "MEAN", "Median": "MEDIAN",
    }

    select_tuples = set()
    for projection in select_node.expressions:
        # Detect the outermost aggregate function wrapping this projection
        proj_agg = "NONE"
        proj_root = projection.unalias() if hasattr(projection, 'unalias') else projection
        if type(proj_root).__name__ in agg_node_map:
            proj_agg = agg_node_map[type(proj_root).__name__]

        # Walk every Column reference inside the projection
        for col in projection.find_all(exp.Column):
            alias = (col.table or "").lower()
            real = alias_to_table.get(alias)
            if not real and len(alias_to_table) == 1:
                real = next(iter(alias_to_table.values()))
            if not real:
                continue
            coll = convert_name(db_id, real)

            # If an inner aggregate wraps this column, detect it too
            col_agg = proj_agg
            parent = col.parent
            while parent is not None and parent is not projection:
                if type(parent).__name__ in agg_node_map:
                    col_agg = agg_node_map[type(parent).__name__]
                    break
                parent = parent.parent

            select_tuples.add((coll, norm_prop(col.name), col_agg))

    return select_tuples


def extract_v2_output_tuples(query_args, primary_collection):
    """From V2 output_properties list, return {(collection, norm_prop, agg_fn), ...}.

    If an entry has no `collection` field, default to the primary_collection.
    Property names are canonicalized through the BIRD weaviate->sql name map
    before normalization, so `"White blood cell"` and `"WBC"` compare equal.
    """
    out = set()
    for entry in query_args.get("output_properties") or []:
        if not isinstance(entry, dict):
            continue
        prop = entry.get("property_name")
        if not prop:
            continue
        coll = entry.get("collection") or primary_collection
        # Canonicalize to SQL-side name when the mapping knows this property.
        sql_prop = canonicalize_to_sql(coll, prop)
        agg = norm_agg(entry.get("aggregation"))
        out.add((coll, norm_prop(sql_prop), agg))
    return out


def score_one(result, bird_dev):
    """Score a single per-query record. Returns a dict of per-metric booleans
    (None if indeterminate) and the raw sets for debugging.
    """
    question = result.get("question", "")
    sql = result.get("sql")
    db_id = result.get("db_id")

    # Fallback: look up from BIRD dev by question if the run pre-dates sql storage
    if not sql and bird_dev and question in bird_dev:
        sql = bird_dev[question].get("SQL")
        db_id = bird_dev[question].get("db_id")

    got = result.get("got") or (result.get("query_args", {}) or {}).get("collection_name", "")
    is_strict = bool(result.get("strict_match"))
    is_relaxed = bool(result.get("relaxed_match"))

    if not sql or not db_id:
        return {
            "strict": is_strict, "relaxed": is_relaxed,
            "output_exact": None, "output_covered": None,
            "output_subset": None, "output_iou": None,
            "indeterminate": True,
        }

    sql_tuples = parse_sql_select(sql, db_id)
    if not sql_tuples:
        return {
            "strict": is_strict, "relaxed": is_relaxed,
            "output_exact": None, "output_covered": None,
            "output_subset": None, "output_iou": None,
            "indeterminate": True,
        }

    qa = result.get("query_args", {}) or {}
    pred_tuples = extract_v2_output_tuples(qa, got)

    if not pred_tuples:
        return {
            "strict": is_strict, "relaxed": is_relaxed,
            "output_exact": False, "output_covered": False,
            "output_subset": False, "output_iou": 0.0,
            "indeterminate": False,
            "sql_tuples": sql_tuples, "pred_tuples": pred_tuples,
        }

    # Strict exact match of the (collection, property, aggregation) sets
    output_exact = sql_tuples == pred_tuples

    # Does the predicted set contain every required SQL SELECT column?
    output_covered = sql_tuples.issubset(pred_tuples)

    # Are all predicted entries legitimate SQL SELECT columns?
    output_subset = pred_tuples.issubset(sql_tuples)

    # Jaccard
    iou = len(sql_tuples & pred_tuples) / max(1, len(sql_tuples | pred_tuples))

    return {
        "strict": is_strict, "relaxed": is_relaxed,
        "output_exact": output_exact, "output_covered": output_covered,
        "output_subset": output_subset, "output_iou": iou,
        "indeterminate": False,
        "sql_tuples": sql_tuples, "pred_tuples": pred_tuples,
    }


def load_bird_dev():
    import glob
    cands = glob.glob(str(PROJECT_ROOT / "data/bird-benchmark/**/dev.json"), recursive=True)
    if not cands:
        return {}
    dev = json.loads(Path(cands[0]).read_text())
    return {q["question"]: q for q in dev}


def main():
    if len(sys.argv) < 2:
        print("Usage: compute_output_match_v2.py <result_json> [--sample N] [--per-db]")
        sys.exit(1)

    path = Path(sys.argv[1])
    sample_n = 0
    show_per_db = False
    if "--sample" in sys.argv:
        sample_n = int(sys.argv[sys.argv.index("--sample") + 1])
    if "--per-db" in sys.argv:
        show_per_db = True

    data = json.loads(path.read_text())
    if data.get("tool_schema") != "v2":
        print(f"WARNING: tool_schema is '{data.get('tool_schema')}' — this script is for v2 runs.", file=sys.stderr)

    results = data["results"]
    bird_dev = load_bird_dev()
    n = len(results)

    counters = Counter()
    per_db = defaultdict(Counter)
    iou_total = 0.0
    iou_count = 0
    indeterminate = 0
    bird_n = 0  # number of BIRD queries for BIRD-only rates

    # Cross-metric disagreement buckets
    artifact_saves = 0    # strict=F, output_covered=T
    real_breaks = 0       # strict=T, output_covered=F
    samples_by_bucket = defaultdict(list)

    for r in results:
        s = score_one(r, bird_dev)
        db = (r.get("db_id") or r.get("source", "").replace("bird_", "") or "?")
        is_bird = bool(r.get("db_id")) or r.get("source", "").startswith("bird_")

        counters["total"] += 1
        per_db[db]["total"] += 1
        if is_bird:
            bird_n += 1
            counters["bird_total"] += 1

        if s["strict"]:
            counters["strict"] += 1; per_db[db]["strict"] += 1
            if is_bird: counters["bird_strict"] += 1
        if s["relaxed"]:
            counters["relaxed"] += 1; per_db[db]["relaxed"] += 1
            if is_bird: counters["bird_relaxed"] += 1

        if s["indeterminate"]:
            indeterminate += 1
            per_db[db]["indeterminate"] += 1
            continue

        if s["output_exact"]:
            counters["output_exact"] += 1; per_db[db]["output_exact"] += 1
            if is_bird: counters["bird_output_exact"] += 1
        if s["output_covered"]:
            counters["output_covered"] += 1; per_db[db]["output_covered"] += 1
            if is_bird: counters["bird_output_covered"] += 1
        if s["output_subset"]:
            counters["output_subset"] += 1; per_db[db]["output_subset"] += 1
            if is_bird: counters["bird_output_subset"] += 1

        iou_total += s["output_iou"]; iou_count += 1

        # Disagreement analysis
        if s["output_covered"] and not s["strict"]:
            artifact_saves += 1
            if sample_n and len(samples_by_bucket["artifact"]) < sample_n:
                samples_by_bucket["artifact"].append((r, s))
        elif s["strict"] and not s["output_covered"]:
            real_breaks += 1
            if sample_n and len(samples_by_bucket["break"]) < sample_n:
                samples_by_bucket["break"].append((r, s))

    def pct(key):
        c = counters.get(key, 0)
        return 100 * c / n if n else 0.0

    def line(label, key):
        c = counters.get(key, 0)
        delta = pct(key) - pct("strict")
        delta_s = f"{delta:+.2f}pp" if key != "strict" else "(baseline)"
        print(f"  {label:<40s} {c:>5d}/{n}  {pct(key):>6.2f}%  {delta_s:>10s}")

    print("=" * 78)
    print(f"V2 OUTPUT-MATCH EVAL: {path.name}")
    print(f"Model: {data.get('provider')}/{data.get('model')}  tool_schema: {data.get('tool_schema')}")
    print(f"N: {n}   Indeterminate (no SQL/parse fail): {indeterminate}")
    print("=" * 78)
    line("strict (primary == FROM)", "strict")
    line("relaxed (primary in FROM/JOINs)", "relaxed")
    line("output_exact (full set equality)", "output_exact")
    line("output_covered (predicted ⊇ SQL SELECT)", "output_covered")
    line("output_subset (predicted ⊆ SQL SELECT)", "output_subset")
    if iou_count:
        print(f"  {'output_iou (avg Jaccard)':<40s} {iou_total/iou_count*100:>21.2f}%")

    print()
    print(f"Artifact saves (strict=F, output_covered=T):  {artifact_saves}")
    print(f"Real breaks    (strict=T, output_covered=F):  {real_breaks}")

    # BIRD-only rates (excludes Weaviate Gorilla which has no SQL)
    if counters["bird_total"]:
        bird_n_ = counters["bird_total"]
        print()
        print("-" * 78)
        print(f"BIRD-ONLY (N={bird_n_}; excludes Weaviate Gorilla, which has no SQL)")
        print("-" * 78)
        for label, key in [
            ("strict", "bird_strict"),
            ("relaxed", "bird_relaxed"),
            ("output_exact", "bird_output_exact"),
            ("output_covered", "bird_output_covered"),
            ("output_subset", "bird_output_subset"),
        ]:
            c = counters.get(key, 0)
            print(f"  {label:<40s} {c:>5d}/{bird_n_}  {100*c/bird_n_:>6.2f}%")

    if show_per_db and per_db:
        print()
        print("=" * 78)
        print("PER-DATABASE")
        print("=" * 78)
        print(f"  {'DB':<30s} {'N':>5s} {'strict':>8s} {'out_exact':>10s} {'out_cov':>9s} {'Δ_cov':>7s}")
        print("  " + "-" * 74)
        for db in sorted(per_db.keys()):
            c = per_db[db]
            tot = c.get("total", 0)
            if tot == 0:
                continue
            st = c.get("strict", 0)
            ex = c.get("output_exact", 0)
            cv = c.get("output_covered", 0)
            delta = cv - st
            print(f"  {db:<30s} {tot:>5d} "
                  f"{st:>4d} ({100*st/tot:>3.0f}%) "
                  f"{ex:>4d} ({100*ex/tot:>3.0f}%) "
                  f"{cv:>4d} ({100*cv/tot:>3.0f}%) "
                  f"{delta:>+5d}")

    if sample_n:
        for bucket, label in [("artifact", "ARTIFACT SAVES  (strict=FAIL, output_covered=PASS)"),
                              ("break", "REAL BREAKS  (strict=PASS, output_covered=FAIL)")]:
            if not samples_by_bucket[bucket]:
                continue
            print()
            print("=" * 78)
            print(label)
            print("=" * 78)
            for i, (r, s) in enumerate(samples_by_bucket[bucket][:sample_n], 1):
                print(f"\n[{i}]  DB={r.get('db_id')}  Q: {r.get('question','')[:90]}")
                print(f"    SQL: {(r.get('sql') or '')[:150]}")
                print(f"    got_primary: {r.get('got')}  expected_primary: {r.get('expected')}")
                print(f"    pred output: {sorted(s.get('pred_tuples', set()))}")
                print(f"    SQL SELECT:  {sorted(s.get('sql_tuples', set()))}")


if __name__ == "__main__":
    main()
