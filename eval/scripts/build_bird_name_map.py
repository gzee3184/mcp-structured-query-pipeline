#!/usr/bin/env python3
"""
build_bird_name_map.py — Build a deterministic mapping from our Weaviate
collection property names to BIRD's canonical SQL column names, using BIRD's
own dev_tables.json as the authoritative source.

Why this exists:
  Our bird-collections.json uses descriptive property names (e.g., "White blood cell",
  "AST glutamic oxaloacetic transaminase") that come from BIRD's column_names
  (descriptive). The BIRD SQL queries use column_names_original (e.g., "WBC", "GOT").
  When we score predicted output_properties against SQL SELECT columns, a naive
  string compare sees "White blood cell" != "WBC" as a mismatch even though they
  refer to the same column.

  dev_tables.json pairs column_names[i] <-> column_names_original[i] per table per
  database. We use this pairing to build:
      map[ "ThrombosisPredictionLaboratory" ][ "white blood cell" ] -> "WBC"
  and the reverse.

Usage:
    python eval/scripts/build_bird_name_map.py
Output:
    data/bird-benchmark/bird_property_name_map.json
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BIRD_TABLES = PROJECT_ROOT / "data/bird-benchmark/dev_20240627/dev_tables.json"
BIRD_COLLECTIONS = PROJECT_ROOT / "data/bird-benchmark/bird-collections.json"
OUT_PATH = PROJECT_ROOT / "data/bird-benchmark/bird_property_name_map.json"


def pascal(s: str) -> str:
    return "".join(p.capitalize() for p in s.split("_"))


def convert_name(db_id: str, table: str) -> str:
    """Mirror test_enhanced_pipeline.py:convert_name()."""
    return f"{pascal(db_id)}{pascal(table)}"


# Known collection-name divergences between BIRD tables and our Weaviate
# collections. BIRD table 'attribute' was renamed 'GenericAttribute' in our
# bird-collections.json to avoid reserved-word collisions. The fuzzy matcher
# can't cross that gap, so we list it explicitly.
COLLECTION_ALIASES = {
    "SuperheroAttribute": "SuperheroGenericAttribute",
}


def norm(s: str) -> str:
    """Aggressive property-name canonicalization (lower, strip spaces/underscores/dashes)."""
    if s is None:
        return ""
    return re.sub(r"[\s_\-]+", "", str(s).lower())


def main():
    dev_tables = json.loads(BIRD_TABLES.read_text())
    bc = json.loads(BIRD_COLLECTIONS.read_text())

    # Build: (weaviate_coll_name, norm(weaviate_prop)) -> sql_col_name
    # And the reverse: (weaviate_coll_name, norm(sql_col_name)) -> weaviate_prop
    weaviate_to_sql = defaultdict(dict)  # coll -> {norm_weaviate_prop: sql_col}
    sql_to_weaviate = defaultdict(dict)  # coll -> {norm_sql_col: weaviate_prop}

    # Also keep all known SQL col names per collection for membership checks
    sql_cols_by_coll = defaultdict(list)

    # Stats
    matched = 0
    unmatched_weaviate = defaultdict(list)
    unmatched_sql = defaultdict(list)

    # Build Weaviate property name lookup once
    wc_props = {}
    for c in bc.get("weaviate_collections", []):
        name = c["name"]
        props = c.get("properties", [])
        if isinstance(props, list):
            prop_names = [p["name"] for p in props if isinstance(p, dict) and p.get("name")]
        elif isinstance(props, dict):
            prop_names = list(props.keys())
        else:
            prop_names = []
        wc_props[name] = prop_names

    # For each BIRD DB entry
    for entry in dev_tables:
        db_id = entry["db_id"]
        tables = entry["table_names_original"]
        col_sql = entry["column_names_original"]       # [[table_idx, sql_col], ...]
        col_desc = entry["column_names"]               # [[table_idx, descriptive], ...]

        # Pair up index-by-index (BIRD guarantees same ordering)
        # For each table: collect (sql_name, descriptive_name) pairs.
        table_pairs = defaultdict(list)
        for (tidx, sql_col), (_, desc_col) in zip(col_sql, col_desc):
            if tidx < 0:
                continue
            tname = tables[tidx]
            table_pairs[tname].append((sql_col, desc_col))

        for table_name, pairs in table_pairs.items():
            coll = convert_name(db_id, table_name)
            # Apply known renamings
            if coll in COLLECTION_ALIASES:
                coll = COLLECTION_ALIASES[coll]
            our_props = wc_props.get(coll, [])
            our_norms = {norm(p): p for p in our_props}

            for sql_col, desc_col in pairs:
                sql_cols_by_coll[coll].append(sql_col)
                # Try match descriptive name to one of our Weaviate property names.
                # Strategy 1: exact normalized match (desc vs our prop)
                # Strategy 2: exact normalized match (sql name vs our prop)
                # Strategy 3: substring either way
                n_desc = norm(desc_col)
                n_sql = norm(sql_col)

                picked = None
                if n_desc in our_norms:
                    picked = our_norms[n_desc]
                elif n_sql in our_norms:
                    picked = our_norms[n_sql]
                else:
                    # Substring fallback (both directions, minimum length 3 to avoid noise)
                    for n_prop, orig_prop in our_norms.items():
                        if len(n_prop) >= 3 and (n_prop in n_desc or n_desc in n_prop):
                            picked = orig_prop
                            break
                    if picked is None:
                        for n_prop, orig_prop in our_norms.items():
                            if len(n_prop) >= 3 and (n_prop in n_sql or n_sql in n_prop):
                                picked = orig_prop
                                break

                if picked is not None:
                    weaviate_to_sql[coll][norm(picked)] = sql_col
                    sql_to_weaviate[coll][norm(sql_col)] = picked
                    matched += 1
                else:
                    unmatched_sql[coll].append((sql_col, desc_col))

    # Record any Weaviate props that have no SQL mapping (probably benchmark-side mismatch)
    for coll, props in wc_props.items():
        mapped_weaviate_norms = set(weaviate_to_sql.get(coll, {}).keys())
        for p in props:
            if norm(p) not in mapped_weaviate_norms:
                unmatched_weaviate[coll].append(p)

    out = {
        "description": "Mapping between our Weaviate property names and BIRD's canonical SQL column names. Built from BIRD's dev_tables.json pairing of column_names (descriptive) with column_names_original (SQL).",
        "source": {
            "bird_tables": str(BIRD_TABLES.relative_to(PROJECT_ROOT)),
            "bird_collections": str(BIRD_COLLECTIONS.relative_to(PROJECT_ROOT)),
        },
        "weaviate_to_sql": {k: dict(v) for k, v in weaviate_to_sql.items()},
        "sql_to_weaviate": {k: dict(v) for k, v in sql_to_weaviate.items()},
        "sql_cols_by_collection": {k: sorted(set(v)) for k, v in sql_cols_by_coll.items()},
        "stats": {
            "total_matched": matched,
            "unmatched_sql_columns": sum(len(v) for v in unmatched_sql.values()),
            "unmatched_weaviate_properties": sum(len(v) for v in unmatched_weaviate.values()),
            "collections_covered": len(sql_cols_by_coll),
        },
        "unmatched_sql": {k: v for k, v in unmatched_sql.items() if v},
        "unmatched_weaviate": {k: v for k, v in unmatched_weaviate.items() if v},
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Saved mapping to {OUT_PATH}")
    print(f"Collections covered: {out['stats']['collections_covered']}")
    print(f"SQL columns matched: {out['stats']['total_matched']}")
    print(f"Unmatched SQL columns: {out['stats']['unmatched_sql_columns']}")
    print(f"Unmatched Weaviate properties: {out['stats']['unmatched_weaviate_properties']}")

    # Spot-check a few interesting collections
    print()
    print("Spot-check (ThrombosisPredictionLaboratory):")
    lab_map = weaviate_to_sql.get("ThrombosisPredictionLaboratory", {})
    for n_prop, sql in sorted(lab_map.items())[:15]:
        print(f"  {n_prop!r:<40s} -> {sql!r}")


if __name__ == "__main__":
    main()
