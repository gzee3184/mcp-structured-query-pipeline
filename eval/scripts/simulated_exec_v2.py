#!/usr/bin/env python3
"""
simulated_exec_v2.py — Phase-2 "simulated execution" scorer for V2 runs.

For each saved V2 prediction, parse the BIRD ground-truth SQL into its
components (SELECT, WHERE, ORDER BY, LIMIT, DISTINCT, GROUP BY) and compare
each component set-wise against the V2 query_args. Output per-component
accuracy *and* a combined simulated_exec_pass metric that requires every
component to match above a threshold.

What this scores:
  - select_match    : SELECT columns match (IoU >= 0.8)
  - where_match     : WHERE predicates match (IoU >= 0.8; predicate-level canonicalization)
  - groupby_match   : GROUP BY columns match
  - orderby_match   : ORDER BY sequence matches (order + direction)
  - limit_match     : LIMIT values match
  - distinct_match  : DISTINCT boolean matches
  - simulated_exec_pass : all above components pass

Predicted property names are canonicalized through the BIRD namespace map
(weaviate -> SQL) before comparison, so `White blood cell` matches `WBC`.

Usage:
    python eval/scripts/simulated_exec_v2.py <v2_result_json>
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter

try:
    import sqlglot
    from sqlglot import exp
except ImportError:
    print("ERROR: sqlglot not installed.", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def pascal(n):
    return "".join(p.capitalize() for p in n.split("_"))


def convert_name(db, table):
    return f"{pascal(db)}{pascal(table)}"


def norm(s):
    if s is None:
        return ""
    return re.sub(r"[\s_\-]+", "", str(s).lower())


# ---- Name map (same as compute_output_match_v2.py) -------------------------

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


def canon_to_sql(collection: str, prop_name: str) -> str:
    m = _load_name_map()
    if not m:
        return prop_name
    w2s = m.get("weaviate_to_sql", {}).get(collection, {})
    return w2s.get(norm(prop_name), prop_name)


# ---- BIRD dev lookup -------------------------------------------------------


def load_bird_dev():
    import glob
    cands = glob.glob(str(PROJECT_ROOT / "data/bird-benchmark/**/dev.json"), recursive=True)
    if not cands:
        raise FileNotFoundError("BIRD dev.json not found")
    dev = json.loads(Path(cands[0]).read_text())
    return {q["question"]: q for q in dev}


# ---- SQL parsing into components -------------------------------------------


_AGG_NODE_MAP = {"Count": "COUNT", "Sum": "SUM", "Min": "MIN", "Max": "MAX",
                 "Avg": "MEAN", "Median": "MEDIAN"}


def _resolve_col(col, alias_to_table, db_id):
    alias = (col.table or "").lower()
    real = alias_to_table.get(alias)
    if not real and len(alias_to_table) == 1:
        real = next(iter(alias_to_table.values()))
    return convert_name(db_id, real) if real else None


def parse_sql_components(sql: str, db_id: str) -> dict:
    """Return a dict of component sets / values extracted from the SQL."""
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        try:
            parsed = sqlglot.parse_one(sql)
        except Exception:
            return {}

    if parsed is None:
        return {}

    alias_to_table = {}
    for t in parsed.find_all(exp.Table):
        tn = t.name
        al = t.alias or tn
        alias_to_table[al.lower()] = tn

    sel = parsed.find(exp.Select)
    if sel is None:
        return {}

    # ---- SELECT columns with agg
    select_tuples = set()
    for projection in sel.expressions:
        # outermost agg
        proj_agg = "NONE"
        proj_root = projection.unalias() if hasattr(projection, "unalias") else projection
        if type(proj_root).__name__ in _AGG_NODE_MAP:
            proj_agg = _AGG_NODE_MAP[type(proj_root).__name__]

        for col in projection.find_all(exp.Column):
            coll = _resolve_col(col, alias_to_table, db_id)
            if not coll:
                continue
            col_agg = proj_agg
            parent = col.parent
            while parent is not None and parent is not projection:
                if type(parent).__name__ in _AGG_NODE_MAP:
                    col_agg = _AGG_NODE_MAP[type(parent).__name__]
                    break
                parent = parent.parent
            select_tuples.add((coll, norm(col.name), col_agg))

    # ---- DISTINCT
    is_distinct = bool(sel.args.get("distinct"))

    # ---- LIMIT
    limit = None
    lim_node = sel.args.get("limit")
    if lim_node is not None:
        lim_expr = lim_node.expression
        if hasattr(lim_expr, "this"):
            try:
                limit = int(lim_expr.this)
            except Exception:
                pass
        elif hasattr(lim_expr, "to_py"):
            try:
                limit = int(lim_expr.to_py())
            except Exception:
                pass

    # ---- ORDER BY (sequence)
    order_seq = []
    order_node = sel.args.get("order")
    if order_node:
        for ord_exp in order_node.expressions:
            direction = "DESC" if ord_exp.args.get("desc") else "ASC"
            # Inner column reference
            col = ord_exp.find(exp.Column)
            if col:
                coll = _resolve_col(col, alias_to_table, db_id)
                # detect aggregation
                agg = "NONE"
                parent = col.parent
                while parent is not None and parent is not ord_exp:
                    if type(parent).__name__ in _AGG_NODE_MAP:
                        agg = _AGG_NODE_MAP[type(parent).__name__]
                        break
                    parent = parent.parent
                order_seq.append((coll, norm(col.name), agg, direction))

    # ---- GROUP BY
    group_by = set()
    group_node = sel.args.get("group")
    if group_node:
        for ge in group_node.expressions:
            for col in ge.find_all(exp.Column):
                coll = _resolve_col(col, alias_to_table, db_id)
                if coll:
                    group_by.add((coll, norm(col.name)))

    # ---- WHERE predicates (flattened — ignore AND/OR structure for set compare)
    where_preds = set()
    where_node = sel.args.get("where")
    if where_node:
        # walk comparison nodes
        comp_types = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.Is, exp.In, exp.Between)
        for n in where_node.find_all(*comp_types):
            # get column(s) in this comparison
            cols = [c for c in n.find_all(exp.Column)]
            if not cols:
                continue
            col = cols[0]  # primary side
            coll = _resolve_col(col, alias_to_table, db_id)
            if not coll:
                continue
            op_name = type(n).__name__
            # canonicalize op
            op_map = {
                "EQ": "=", "NEQ": "!=", "GT": ">", "GTE": ">=", "LT": "<", "LTE": "<=",
                "Like": "LIKE", "Is": "IS", "In": "IN", "Between": "BETWEEN",
            }
            op = op_map.get(op_name, op_name.upper())
            where_preds.add((coll, norm(col.name), op))

    return {
        "select": select_tuples,
        "where": where_preds,
        "group_by": group_by,
        "order_by": order_seq,
        "limit": limit,
        "distinct": is_distinct,
    }


# ---- V2 prediction -> components -------------------------------------------


def extract_v2_components(query_args: dict) -> dict:
    """Parallel extractor for predicted V2 args — same shape as parse_sql_components."""
    primary = query_args.get("collection_name")

    def _canon_prop(coll, prop):
        # weaviate -> sql; our stored args should already be weaviate-named,
        # so canonicalize to sql for apples-to-apples compare.
        sql = canon_to_sql(coll or primary, prop) if prop else prop
        return norm(sql)

    # SELECT = output_properties
    select_tuples = set()
    for e in query_args.get("output_properties") or []:
        if not isinstance(e, dict):
            continue
        prop = e.get("property_name")
        if not prop:
            continue
        coll = e.get("collection") or primary
        agg = (e.get("aggregation") or "NONE").upper()
        select_tuples.add((coll, _canon_prop(coll, prop), agg))

    # WHERE = filters list
    where_preds = set()
    for f in query_args.get("filters") or []:
        if not isinstance(f, dict):
            continue
        prop = f.get("property_name")
        if not prop:
            continue
        coll = f.get("collection") or primary
        op = (f.get("operator") or "").upper()
        # align op naming with SQL parser output
        op_alias = {"IS_NULL": "IS", "IS_NOT_NULL": "IS"}
        op = op_alias.get(op, op)
        where_preds.add((coll, _canon_prop(coll, prop), op))

    # GROUP BY
    group_by = set()
    for gp in query_args.get("group_by_properties") or []:
        if isinstance(gp, str):
            group_by.add((primary, _canon_prop(primary, gp)))

    # ORDER BY (sequence)
    order_seq = []
    for ob in query_args.get("order_by") or []:
        if not isinstance(ob, dict):
            continue
        prop = ob.get("property_name")
        if not prop:
            continue
        coll = ob.get("collection") or primary
        agg = (ob.get("aggregation") or "NONE").upper()
        direction = (ob.get("direction") or "ASC").upper()
        order_seq.append((coll, _canon_prop(coll, prop), agg, direction))

    # LIMIT and DISTINCT
    limit = query_args.get("limit")
    try:
        limit = int(limit) if limit is not None else None
    except Exception:
        limit = None
    distinct = bool(query_args.get("distinct"))

    return {
        "select": select_tuples,
        "where": where_preds,
        "group_by": group_by,
        "order_by": order_seq,
        "limit": limit,
        "distinct": distinct,
    }


# ---- Scoring ---------------------------------------------------------------


def iou(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_one(pred: dict, gt: dict, threshold: float = 0.8) -> dict:
    """Score pred vs gt. Each component is a pass/fail boolean at `threshold` IoU,
    plus raw IoU for softer analysis. Combined 'pass' requires all components
    to pass."""
    # SELECT / WHERE / GROUP BY are set IoU
    select_iou = iou(pred["select"], gt["select"])
    where_iou = iou(pred["where"], gt["where"])
    gb_iou = iou(pred["group_by"], gt["group_by"])

    # ORDER BY is a sequence — exact match or not
    ob_match = pred["order_by"] == gt["order_by"]

    # LIMIT and DISTINCT are scalars
    limit_match = pred["limit"] == gt["limit"]
    distinct_match = pred["distinct"] == gt["distinct"]

    # Component-level pass
    select_pass = select_iou >= threshold
    where_pass = where_iou >= threshold
    # If gt has no GROUP BY and pred has none, trivial pass
    gb_pass = gb_iou >= threshold if (pred["group_by"] or gt["group_by"]) else True

    simulated_exec_pass = all([select_pass, where_pass, gb_pass, ob_match, limit_match, distinct_match])

    return {
        "select_iou": select_iou,
        "where_iou": where_iou,
        "gb_iou": gb_iou,
        "ob_match": ob_match,
        "limit_match": limit_match,
        "distinct_match": distinct_match,
        "select_pass": select_pass,
        "where_pass": where_pass,
        "gb_pass": gb_pass,
        "simulated_exec_pass": simulated_exec_pass,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: simulated_exec_v2.py <v2_result_json> [--threshold 0.8]")
        sys.exit(1)

    path = Path(sys.argv[1])
    threshold = 0.8
    if "--threshold" in sys.argv:
        threshold = float(sys.argv[sys.argv.index("--threshold") + 1])

    data = json.loads(path.read_text())
    if data.get("tool_schema") != "v2":
        print(f"WARN: tool_schema is {data.get('tool_schema')!r}, script expects v2", file=sys.stderr)

    results = data["results"]
    bird_dev = load_bird_dev()
    n = len(results)

    agg = Counter()
    per_db = defaultdict(Counter)
    sum_iou = defaultdict(float)
    iou_count = 0

    total_ex = 0  # BIRD queries we could score

    for r in results:
        q = r.get("question", "")
        sql = r.get("sql")
        db_id = r.get("db_id")
        if not sql and q in bird_dev:
            sql = bird_dev[q].get("SQL"); db_id = bird_dev[q].get("db_id")
        if not sql or not db_id:
            continue

        gt = parse_sql_components(sql, db_id)
        if not gt:
            continue
        pred = extract_v2_components(r.get("query_args") or {})
        s = score_one(pred, gt, threshold=threshold)

        total_ex += 1
        per_db[db_id]["total"] += 1

        for k in ("select_pass", "where_pass", "gb_pass", "ob_match", "limit_match", "distinct_match", "simulated_exec_pass"):
            if s[k]:
                agg[k] += 1
                per_db[db_id][k] += 1

        for k in ("select_iou", "where_iou", "gb_iou"):
            sum_iou[k] += s[k]
        iou_count += 1

    def pct(k):
        return 100 * agg[k] / total_ex if total_ex else 0.0

    print("=" * 78)
    print(f"SIMULATED EXEC (V2): {path.name}")
    print(f"Model: {data.get('provider')}/{data.get('model')}   Threshold: {threshold}")
    print(f"BIRD queries scored: {total_ex} / {n} total")
    print("=" * 78)
    print(f"  {'component':<30s} {'pass':>12s} {'%':>8s}   {'avg IoU':>8s}")
    print("  " + "-" * 65)
    print(f"  {'SELECT columns':<30s} {agg['select_pass']:>7d}/{total_ex:<4d} {pct('select_pass'):>7.2f}%   {100*sum_iou['select_iou']/iou_count:>7.2f}%")
    print(f"  {'WHERE predicates':<30s} {agg['where_pass']:>7d}/{total_ex:<4d} {pct('where_pass'):>7.2f}%   {100*sum_iou['where_iou']/iou_count:>7.2f}%")
    print(f"  {'GROUP BY':<30s} {agg['gb_pass']:>7d}/{total_ex:<4d} {pct('gb_pass'):>7.2f}%   {100*sum_iou['gb_iou']/iou_count:>7.2f}%")
    print(f"  {'ORDER BY (exact)':<30s} {agg['ob_match']:>7d}/{total_ex:<4d} {pct('ob_match'):>7.2f}%")
    print(f"  {'LIMIT':<30s} {agg['limit_match']:>7d}/{total_ex:<4d} {pct('limit_match'):>7.2f}%")
    print(f"  {'DISTINCT':<30s} {agg['distinct_match']:>7d}/{total_ex:<4d} {pct('distinct_match'):>7.2f}%")
    print()
    print(f"  {'SIMULATED_EXEC_PASS (all)':<30s} {agg['simulated_exec_pass']:>7d}/{total_ex:<4d} {pct('simulated_exec_pass'):>7.2f}%")

    print()
    print("Per-DB simulated_exec_pass:")
    for db in sorted(per_db.keys()):
        c = per_db[db]
        tot = c.get("total", 0)
        if tot == 0:
            continue
        sp = c.get("simulated_exec_pass", 0)
        print(f"  {db:<30s} {sp:>4d}/{tot:<4d} ({100*sp/tot:>5.2f}%)")


if __name__ == "__main__":
    main()
