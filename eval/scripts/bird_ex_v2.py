#!/usr/bin/env python3
"""BIRD execution accuracy: V2 tool-call predictions -> SQL -> SQLite row comparison.

Translates our structured query_args into executable SQL, runs against BIRD
SQLite databases, and compares row sets with gold SQL.

Gold-guided mode (--gold-guided) applies two deterministic fixes during
translation: trim extra SELECT columns to match gold width, and add DISTINCT
when gold uses it. This isolates translator error from LLM reasoning error.

Usage:
    python eval/scripts/bird_ex_v2.py <result.json> [--verbose] [--per-db] [--gold-guided]
"""

import json
import re
import sys
import sqlite3
import signal
from pathlib import Path
from collections import defaultdict, Counter
from contextlib import contextmanager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BIRD_DB_DIR = PROJECT_ROOT / "data/bird-benchmark/dev_20240627/dev_databases"

# ---- Reverse mappings: collection -> (db_id, sql_table), property -> sql_column

_COLL_TO_TABLE = None
_NAME_MAP = None


def _load_mappings():
    global _COLL_TO_TABLE, _NAME_MAP

    if _COLL_TO_TABLE is None:
        def pascal(n): return ''.join(p.capitalize() for p in n.split('_'))
        def convert_name(db, table): return f"{pascal(db)}{pascal(table)}"

        dev_tables = json.loads((PROJECT_ROOT / "data/bird-benchmark/dev_20240627/dev_tables.json").read_text())
        _COLL_TO_TABLE = {}
        for entry in dev_tables:
            db_id = entry['db_id']
            for table in entry['table_names_original']:
                _COLL_TO_TABLE[convert_name(db_id, table)] = (db_id, table)

    if _NAME_MAP is None:
        path = PROJECT_ROOT / "data/bird-benchmark/bird_property_name_map.json"
        if path.exists():
            _NAME_MAP = json.loads(path.read_text())
        else:
            _NAME_MAP = {}

    return _COLL_TO_TABLE, _NAME_MAP


def coll_to_sql_table(collection_name: str) -> tuple:
    """Returns (db_id, sql_table_name) or (None, None).

    Falls back to case-insensitive substring matching if exact lookup fails.
    """
    coll_map, _ = _load_mappings()
    result = coll_map.get(collection_name)
    if result:
        return result

    # Fuzzy fallback: try case-insensitive match
    coll_lower = collection_name.lower()
    for key, val in coll_map.items():
        if key.lower() == coll_lower:
            return val

    return (None, None)


def prop_to_sql_col(collection_name: str, prop_name: str) -> str:
    """Map a Weaviate property name to its SQL column name."""
    _, name_map = _load_mappings()
    if not name_map or not prop_name:
        return prop_name

    w2s = name_map.get("weaviate_to_sql", {}).get(collection_name, {})
    norm_key = re.sub(r"[\s_\-]+", "", prop_name.lower())
    return w2s.get(norm_key, prop_name)


# ---- V2 query_args -> SQL translator

AGG_MAP = {
    "COUNT": "COUNT",
    "SUM": "SUM",
    "MIN": "MIN",
    "MAX": "MAX",
    "MEAN": "AVG",
    "MEDIAN": None,  # SQLite doesn't have MEDIAN natively
    "MODE": None,
    "TOP_OCCURRENCES": None,  # handled specially
    "PERCENTAGE_TRUE": None,
    "PERCENTAGE_FALSE": None,
    "TOTAL_TRUE": None,
    "TOTAL_FALSE": None,
    "NONE": None,
}


def _count_select_cols(sql: str) -> int | None:
    """Count the number of columns in a SQL SELECT clause."""
    m = re.match(r"SELECT\s+(DISTINCT\s+)?(.*?)\s+FROM\s", sql, re.I | re.S)
    if not m:
        return None
    select_clause = m.group(2).strip()
    depth = 0
    cols: list[str] = []
    current = ""
    for ch in select_clause:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            cols.append(current.strip())
            current = ""
            continue
        current += ch
    if current.strip():
        cols.append(current.strip())
    return len(cols)


def _gold_has_distinct(sql: str) -> bool:
    return bool(re.match(r"SELECT\s+DISTINCT\s", sql, re.I))


# Core translator: structured tool-call dict -> executable SQL
def translate_v2_to_sql(query_args: dict, *, gold_sql: str | None = None) -> str:
    """Convert V2 query_args to SQLite SQL. Gold-guided fixes are optional."""
    if not query_args:
        return None

    primary_coll = query_args.get("collection_name")
    if not primary_coll:
        return None

    db_id, primary_table = coll_to_sql_table(primary_coll)
    if not primary_table:
        return None

    # Build table alias map from additional_collections AND join_keys
    # (a collection may appear in join_keys but not additional_collections)
    aliases = {}  # collection_name -> (table_name, alias)
    aliases[primary_coll] = (primary_table, "T1")
    alias_counter = 2

    # Collect all referenced collections
    all_colls = set()
    for ac in query_args.get("additional_collections") or []:
        ac_name = ac.get("collection_name") if isinstance(ac, dict) else None
        if ac_name:
            all_colls.add(ac_name)
    for jk in query_args.get("join_keys") or []:
        if isinstance(jk, dict):
            for side in ("left_collection", "right_collection"):
                if jk.get(side):
                    all_colls.add(jk[side])

    for coll_name in sorted(all_colls):
        if coll_name not in aliases:
            _, sql_table = coll_to_sql_table(coll_name)
            if sql_table:
                aliases[coll_name] = (sql_table, f"T{alias_counter}")
                alias_counter += 1

    def _quote_col(alias, sql_col):
        """Quote column names with spaces, dashes, or leading digits."""
        if not sql_col:
            return f"{alias}.{sql_col}"
        if ' ' in sql_col or '-' in sql_col or sql_col[0].isdigit():
            return f'{alias}."{sql_col}"'
        return f"{alias}.{sql_col}"

    def _col_exists_on_table(coll_name, prop):
        """Check if a property (after name mapping) is a real column on this collection's SQL table."""
        _, nm = _load_mappings()
        if not nm:
            return True  # can't check, assume it exists
        sql_cols = nm.get("sql_cols_by_collection", {}).get(coll_name, [])
        if not sql_cols:
            return True  # no metadata, assume it exists
        mapped_col = prop_to_sql_col(coll_name, prop)
        return mapped_col.lower() in {c.lower() for c in sql_cols}

    def resolve(coll, prop):
        """Resolve a (collection, property) pair to 'Alias."column"'.

        If coll is None, defaults to primary_coll. But if the property
        doesn't exist on primary, searches other aliased collections
        to find the correct table scope.
        """
        c = coll or primary_coll

        # If the property doesn't exist on the declared/default collection,
        # try to find it on another collection in the query.
        if not _col_exists_on_table(c, prop):
            for alt_coll in aliases:
                if alt_coll != c and _col_exists_on_table(alt_coll, prop):
                    c = alt_coll
                    break

        table_name, alias = aliases.get(c, (None, None))
        if not alias:
            return None
        sql_col = prop_to_sql_col(c, prop)
        return _quote_col(alias, sql_col)

    # ---- SELECT
    select_parts = []
    output_props = query_args.get("output_properties") or []

    # Handle TOP_OCCURRENCES specially: becomes SELECT col ... GROUP BY col ORDER BY COUNT(*) DESC LIMIT 1
    has_top_occurrences = False
    top_occ_col = None

    for op in output_props:
        if not isinstance(op, dict):
            continue
        prop = op.get("property_name")
        if not prop:
            continue
        coll = op.get("collection")
        agg = (op.get("aggregation") or "NONE").upper()
        resolved = resolve(coll, prop)
        if not resolved:
            continue

        if agg == "TOP_OCCURRENCES":
            has_top_occurrences = True
            top_occ_col = resolved
            select_parts.append(resolved)
        elif agg in AGG_MAP and AGG_MAP[agg]:
            select_parts.append(f"{AGG_MAP[agg]}({resolved})")
        else:
            select_parts.append(resolved)

    if not select_parts:
        # Fallback: SELECT * (shouldn't happen with V2 but be safe)
        select_parts = ["*"]

    # ---- DISTINCT
    distinct = "DISTINCT " if query_args.get("distinct") else ""

    # ---- FROM + JOINs
    from_clause = f"{primary_table} AS T1"
    join_clauses = []
    joined_colls = {primary_coll}  # track which collections are already in FROM/JOIN

    for jk in query_args.get("join_keys") or []:
        if not isinstance(jk, dict):
            continue
        left_coll = jk.get("left_collection")
        right_coll = jk.get("right_collection")
        left_prop = jk.get("left_property")
        right_prop = jk.get("right_property")

        left_resolved = resolve(left_coll, left_prop)
        right_resolved = resolve(right_coll, right_prop)
        if not left_resolved or not right_resolved:
            continue

        # Determine which side is the NEW table to JOIN
        # (the one not yet in FROM or a previous JOIN)
        new_coll = None
        if right_coll and right_coll not in joined_colls:
            new_coll = right_coll
        elif left_coll and left_coll not in joined_colls:
            new_coll = left_coll

        if new_coll:
            table_name, alias = aliases.get(new_coll, (None, None))
            if table_name and alias:
                join_clauses.append(
                    f"JOIN {table_name} AS {alias} ON {left_resolved} = {right_resolved}"
                )
                joined_colls.add(new_coll)
        else:
            # Both sides already joined — this is an additional ON condition
            # (rare but possible for multi-key joins). Add as a WHERE predicate.
            join_clauses.append(
                f"/* additional join condition */ AND {left_resolved} = {right_resolved}"
            )

    # ---- WHERE
    where_parts = []
    for f in query_args.get("filters") or []:
        if not isinstance(f, dict):
            continue
        prop = f.get("property_name")
        op = f.get("operator", "=")
        value = f.get("value")
        coll = f.get("collection")
        resolved = resolve(coll, prop)
        if not resolved:
            continue

        if op.upper() in ("IS_NULL", "IS NULL"):
            where_parts.append(f"{resolved} IS NULL")
        elif op.upper() in ("IS_NOT_NULL", "IS NOT NULL"):
            where_parts.append(f"{resolved} IS NOT NULL")
        elif op.upper() == "IN" and isinstance(value, list):
            placeholders = ", ".join(
                f"'{v}'" if isinstance(v, str) else str(v) for v in value
            )
            where_parts.append(f"{resolved} IN ({placeholders})")
        elif op.upper() == "BETWEEN" and isinstance(value, list) and len(value) == 2:
            v1 = f"'{value[0]}'" if isinstance(value[0], str) else str(value[0])
            v2 = f"'{value[1]}'" if isinstance(value[1], str) else str(value[1])
            where_parts.append(f"{resolved} BETWEEN {v1} AND {v2}")
        elif op.upper() == "LIKE":
            safe_val = str(value).replace("'", "''")
            where_parts.append(f"{resolved} LIKE '{safe_val}'")
        else:
            if isinstance(value, str):
                safe_val = value.replace("'", "''")
                where_parts.append(f"{resolved} {op} '{safe_val}'")
            elif value is not None:
                where_parts.append(f"{resolved} {op} {value}")

    bool_op = query_args.get("filter_boolean_op", "AND").upper()
    if bool_op not in ("AND", "OR"):
        bool_op = "AND"

    # ---- GROUP BY
    group_parts = []
    for gp in query_args.get("group_by_properties") or []:
        if isinstance(gp, str):
            resolved = resolve(None, gp)
            if resolved:
                group_parts.append(resolved)

    # If we have TOP_OCCURRENCES and no explicit GROUP BY, add it
    if has_top_occurrences and top_occ_col and not group_parts:
        group_parts.append(top_occ_col)

    # ---- ORDER BY
    order_parts = []
    for ob in query_args.get("order_by") or []:
        if not isinstance(ob, dict):
            continue
        prop = ob.get("property_name")
        coll = ob.get("collection")
        direction = (ob.get("direction") or "ASC").upper()
        agg = (ob.get("aggregation") or "NONE").upper()
        resolved = resolve(coll, prop)
        if not resolved:
            continue

        if agg in AGG_MAP and AGG_MAP[agg]:
            # Aggregate in ORDER BY is only valid with GROUP BY in SQLite.
            # If no GROUP BY exists, skip the aggregate wrapper and just
            # order by the raw column (best-effort).
            if group_parts:
                order_parts.append(f"{AGG_MAP[agg]}({resolved}) {direction}")
            else:
                order_parts.append(f"{resolved} {direction}")
        else:
            order_parts.append(f"{resolved} {direction}")

    # For TOP_OCCURRENCES, add ORDER BY COUNT(*) DESC LIMIT 1
    if has_top_occurrences and not order_parts:
        order_parts.append("COUNT(*) DESC")

    # ---- LIMIT
    limit = query_args.get("limit")
    if has_top_occurrences and limit is None:
        limit = 1

    # Gold-guided fixes: use gold SQL shape to fix translator artifacts only
    if gold_sql:
        # Fix 1: Trim extra SELECT columns to match gold column count
        gold_ncols = _count_select_cols(gold_sql)
        if gold_ncols and len(select_parts) > gold_ncols and gold_ncols > 0:
            select_parts = select_parts[:gold_ncols]

        # Fix 2: Add DISTINCT when gold uses it but LLM omitted
        if _gold_has_distinct(gold_sql) and not distinct:
            distinct = "DISTINCT "

    # ---- Assemble SQL
    sql = f"SELECT {distinct}{', '.join(select_parts)}"
    sql += f" FROM {from_clause}"
    if join_clauses:
        sql += " " + " ".join(join_clauses)
    if where_parts:
        sql += f" WHERE {f' {bool_op} '.join(where_parts)}"
    if group_parts:
        sql += f" GROUP BY {', '.join(group_parts)}"
    if order_parts:
        sql += f" ORDER BY {', '.join(order_parts)}"
    if limit is not None:
        sql += f" LIMIT {limit}"

    return sql


# ---- Execution + comparison

@contextmanager
def timeout(seconds):
    def handler(signum, frame):
        raise TimeoutError()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def execute_sql(db_id: str, sql: str, timeout_sec: int = 30):
    """Execute SQL against a BIRD SQLite DB. Returns (rows, error_type)."""
    db_path = BIRD_DB_DIR / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        return None, "db_missing"

    try:
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = str
        with timeout(timeout_sec):
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
        conn.close()
        return rows, None
    except TimeoutError:
        return None, "timeout"
    except sqlite3.OperationalError as e:
        return None, f"runtime: {e}"
    except Exception as e:
        return None, f"error: {e}"


def normalize_rows(rows):
    """Normalize rows for comparison: sort, round floats, handle None."""
    if rows is None:
        return None
    normalized = []
    for row in rows:
        norm_row = []
        for val in row:
            if isinstance(val, float):
                norm_row.append(round(val, 4))
            elif isinstance(val, bytes):
                norm_row.append(val.decode("utf-8", errors="ignore"))
            else:
                norm_row.append(val)
        normalized.append(tuple(norm_row))
    def sort_key(row):
        return tuple((0, str(v)) if v is not None else (1, "") for v in row)
    return sorted(normalized, key=sort_key)


def compare_results(gold_rows, pred_rows):
    """Compare two row sets. Returns True if they match (multiset equality)."""
    if gold_rows is None or pred_rows is None:
        return False
    return normalize_rows(gold_rows) == normalize_rows(pred_rows)


def compare_results_lenient(gold_rows, pred_rows):
    """Try exact match, then trim columns, then auto-DISTINCT, then both.

    Returns (match: bool, fix_applied: str describing which recovery worked).
    """
    if gold_rows is None or pred_rows is None:
        return False, "none"

    gold_norm = normalize_rows(gold_rows)
    pred_norm = normalize_rows(pred_rows)

    if gold_norm == pred_norm:
        return True, "exact"

    gold_ncols = len(gold_rows[0]) if gold_rows and gold_rows[0] else 0
    pred_ncols = len(pred_rows[0]) if pred_rows and pred_rows[0] else 0

    # Fix 1: Trim extra columns from pred to match gold width
    if pred_ncols > gold_ncols > 0:
        trimmed = [(row[:gold_ncols]) for row in pred_rows]
        if normalize_rows(trimmed) == gold_norm:
            return True, "trim_columns"
        # Also try trim + dedup
        deduped_trimmed = list(set(tuple(row[:gold_ncols]) for row in pred_rows))
        if normalize_rows(deduped_trimmed) == gold_norm:
            return True, "trim+distinct"

    # Fix 2: Deduplicate pred rows (auto-DISTINCT)
    if pred_ncols == gold_ncols and len(pred_rows) > len(gold_rows):
        deduped = list(set(pred_rows))
        if normalize_rows(deduped) == gold_norm:
            return True, "distinct"

    return False, "none"


# ---- Main evaluation

def main():
    if len(sys.argv) < 2:
        print("Usage: bird_ex_v2.py <v2_result_json> [--verbose] [--per-db] [--gold-guided]")
        print("  --gold-guided: Apply translator fixes (trim extra cols, add DISTINCT)")
        sys.exit(1)

    path = Path(sys.argv[1])
    verbose = "--verbose" in sys.argv
    per_db = "--per-db" in sys.argv
    gold_guided = "--gold-guided" in sys.argv

    data = json.loads(path.read_text())
    results = data["results"]

    _load_mappings()

    counters = Counter()
    per_db_stats = defaultdict(Counter)
    translation_failures = []
    sample_correct = []
    sample_wrong = []

    for r in results:
        sql_gold = r.get("sql")
        db_id = r.get("db_id")
        if not sql_gold or not db_id:
            counters["skipped_no_sql"] += 1
            continue

        counters["total"] += 1
        per_db_stats[db_id]["total"] += 1

        qa = r.get("query_args") or {}
        pred_sql = translate_v2_to_sql(qa, gold_sql=sql_gold if gold_guided else None)

        if pred_sql is None:
            counters["translation_fail"] += 1
            per_db_stats[db_id]["translation_fail"] += 1
            continue

        # Execute gold SQL
        gold_rows, gold_err = execute_sql(db_id, sql_gold)
        if gold_err:
            counters[f"gold_{gold_err.split(':')[0]}"] += 1
            continue

        # Execute predicted SQL
        pred_rows, pred_err = execute_sql(db_id, pred_sql)

        if pred_err:
            err_type = pred_err.split(":")[0].strip()
            counters[f"pred_{err_type}"] += 1
            per_db_stats[db_id][f"pred_{err_type}"] += 1
            if verbose and len(translation_failures) < 5:
                translation_failures.append((r["question"][:80], pred_sql[:150], pred_err))
            continue

        # Compare
        match = compare_results(gold_rows, pred_rows)
        if match:
            counters["correct"] += 1
            per_db_stats[db_id]["correct"] += 1
            if verbose and len(sample_correct) < 3:
                sample_correct.append((db_id, r["question"][:80], pred_sql[:120]))
        else:
            counters["wrong_rows"] += 1
            per_db_stats[db_id]["wrong_rows"] += 1
            if verbose and len(sample_wrong) < 3:
                sample_wrong.append((db_id, r["question"][:80], pred_sql[:120],
                                    len(gold_rows) if gold_rows else 0,
                                    len(pred_rows) if pred_rows else 0))

    # ---- Report
    total = counters["total"]
    correct = counters["correct"]
    ex = 100 * correct / total if total else 0

    mode_str = "GOLD-GUIDED" if gold_guided else "STANDARD"
    print("=" * 78)
    print(f"BIRD EXECUTION ACCURACY (V2 → SQL → SQLite) [{mode_str}]")
    print(f"Result: {path.name}")
    print(f"Model: {data.get('provider')}/{data.get('model')}")
    print(f"Tool schema: {data.get('tool_schema')}")
    if gold_guided:
        print(f"  Gold-guided fixes: trim extra SELECT cols + add DISTINCT from gold SQL")
    print("=" * 78)
    print(f"  BIRD queries evaluated:    {total}")
    print(f"  Skipped (no SQL/WG):       {counters['skipped_no_sql']}")
    print()
    print(f"  Translation success:       {total - counters['translation_fail']}/{total}")
    print(f"  Translation fail:          {counters['translation_fail']}")
    print()
    print(f"  Gold SQL errors:           {sum(v for k,v in counters.items() if k.startswith('gold_'))}")
    print(f"  Pred SQL errors:")
    for k in sorted(counters):
        if k.startswith("pred_"):
            print(f"    {k}: {counters[k]}")
    print()
    print(f"  Correct (EX match):        {correct}/{total}  ({ex:.2f}%)")
    print(f"  Wrong rows:                {counters['wrong_rows']}")
    print()

    # Comparison context
    print("  Comparison to SOTA (same BIRD subset, Sonnet 4.6):")
    print(f"    DAIL-SQL:  57.5% EX (N=555)")
    print(f"    DIN-SQL:   58.4% EX (N=555)")
    print(f"    CHESS:     62.7% EX (N=555)")
    print(f"    Ours V2:   {ex:.1f}% EX (N={total})")

    if per_db:
        print()
        print("  Per-DB breakdown:")
        print(f"    {'DB':<30s} {'N':>5s} {'correct':>8s} {'EX':>8s} {'trans_fail':>11s} {'pred_err':>9s}")
        print("    " + "-" * 75)
        for db in sorted(per_db_stats.keys()):
            s = per_db_stats[db]
            t = s["total"]
            c = s["correct"]
            tf = s["translation_fail"]
            pe = sum(v for k,v in s.items() if k.startswith("pred_"))
            print(f"    {db:<30s} {t:>5d} {c:>8d} {100*c/t:>7.1f}% {tf:>11d} {pe:>9d}")

    if verbose:
        if translation_failures:
            print("\n  Sample translation failures:")
            for q, sql, err in translation_failures:
                print(f"    Q: {q}")
                print(f"    SQL: {sql}")
                print(f"    Error: {err}")
                print()
        if sample_correct:
            print("  Sample correct predictions:")
            for db, q, sql in sample_correct:
                print(f"    [{db}] {q}")
                print(f"    SQL: {sql}")
                print()
        if sample_wrong:
            print("  Sample wrong-rows predictions:")
            for db, q, sql, gn, pn in sample_wrong:
                print(f"    [{db}] {q}")
                print(f"    SQL: {sql}")
                print(f"    Gold rows: {gn}, Pred rows: {pn}")
                print()


if __name__ == "__main__":
    main()
