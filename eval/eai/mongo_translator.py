#!/usr/bin/env python3
"""
mongo_translator.py — Translate V2 query_args into executable MongoDB
aggregate pipeline strings (mongosh syntax).

Our V2 schema maps to MongoDB aggregate stages:
  output_properties → $project (non-aggregated) or $group (aggregated)
  filters           → $match
  group_by          → $group
  order_by          → $sort
  limit             → $limit
  distinct          → $group (deduplicate)

We always generate .aggregate() pipelines (superset of .find() semantics).
"""

import json
import re


def translate_v2_to_mql(query_args: dict, db_name: str) -> str:
    """Translate V2 query_args into a mongosh aggregate pipeline string.

    Args:
        query_args: the V2 tool-call output dict
        db_name: MongoDB database name (e.g., "sample_guides")

    Returns:
        A mongosh command string like:
        db.planets.aggregate([{$match: ...}, {$project: ...}])
        or None if translation fails.
    """
    if not query_args:
        return None

    primary_coll = query_args.get("collection_name")
    if not primary_coll:
        return None

    # Extract the MongoDB collection name from our "db_name.collection" format
    if '.' in primary_coll:
        mongo_coll = primary_coll.split('.', 1)[1]
    else:
        mongo_coll = primary_coll

    # Analyze what kind of query this is
    output_props = query_args.get("output_properties") or []
    filters = query_args.get("filters") or []
    group_by = query_args.get("group_by_properties") or []
    order_by = query_args.get("order_by") or []
    limit = query_args.get("limit")
    distinct = query_args.get("distinct")

    has_aggregation = any(
        (op.get("aggregation") or "NONE").upper() not in ("NONE", "")
        for op in output_props if isinstance(op, dict)
    )

    pipeline = []

    # ---- Stage 1: $match (from filters)
    match_doc = _build_match(filters)
    if match_doc:
        pipeline.append({"$match": match_doc})

    # ---- Stage 2: $group (if aggregations or group_by or distinct)
    if has_aggregation or group_by:
        group_stage = _build_group(output_props, group_by)
        if group_stage:
            pipeline.append({"$group": group_stage})

            # After $group, we may need $project to rename fields
            project_after_group = _build_project_after_group(output_props, group_by)
            if project_after_group:
                pipeline.append({"$project": project_after_group})

    elif distinct:
        # DISTINCT without aggregation — use $group to deduplicate
        group_id = {}
        for op in output_props:
            if isinstance(op, dict) and op.get("property_name"):
                prop = op["property_name"]
                safe_key = prop.replace(".", "_")
                group_id[safe_key] = f"${prop}"
        if group_id:
            pipeline.append({"$group": {"_id": group_id}})
            # Project to flatten
            project = {}
            for key in group_id:
                project[key] = f"$_id.{key}"
            project["_id"] = 0
            pipeline.append({"$project": project})

    else:
        # Simple query — just $project for output columns
        project = _build_simple_project(output_props)
        if project:
            pipeline.append({"$project": project})

    # ---- Stage 3: $sort (from order_by)
    sort_doc = _build_sort(order_by)
    if sort_doc:
        pipeline.append({"$sort": sort_doc})

    # ---- Stage 4: $limit
    if limit is not None:
        try:
            pipeline.append({"$limit": int(limit)})
        except (ValueError, TypeError):
            pass

    # If pipeline is empty, at least project all fields
    if not pipeline:
        pipeline.append({"$project": {"_id": 0}})

    # Format as mongosh command
    # Convert $date objects to ISODate() calls for the executor's parser
    pipeline_str = json.dumps(pipeline, default=str)
    # Replace {"$date": "XXXX"} with ISODate("XXXX") for mongosh compat
    pipeline_str = re.sub(
        r'\{"\$date":\s*"([^"]+)"\}',
        r'ISODate("\1")',
        pipeline_str
    )
    return f"db.{mongo_coll}.aggregate({pipeline_str})"


def _build_match(filters: list) -> dict:
    """Build a $match document from V2 filters list."""
    if not filters:
        return {}

    conditions = []
    for f in filters:
        if not isinstance(f, dict):
            continue
        prop = f.get("property_name")
        op = (f.get("operator") or "=").upper()
        value = f.get("value")

        if not prop:
            continue

        # Detect date-type values and wrap in ISODate-compatible format
        # MongoDB compares dates as Date objects, not strings
        prop_type = (f.get("property_type") or "").lower()
        if prop_type == "date" and isinstance(value, str):
            value = {"$date": value}
        if prop_type == "date" and isinstance(value, list):
            value = [{"$date": v} if isinstance(v, str) else v for v in value]

        if op == "=" or op == "==":
            conditions.append({prop: value})
        elif op == "!=":
            conditions.append({prop: {"$ne": value}})
        elif op == "<":
            conditions.append({prop: {"$lt": value}})
        elif op == ">":
            conditions.append({prop: {"$gt": value}})
        elif op == "<=":
            conditions.append({prop: {"$lte": value}})
        elif op == ">=":
            conditions.append({prop: {"$gte": value}})
        elif op == "LIKE":
            pattern = str(value).replace("%", ".*").replace("_", ".")
            conditions.append({prop: {"$regex": pattern, "$options": "i"}})
        elif op == "IN" and isinstance(value, list):
            conditions.append({prop: {"$in": value}})
        elif op == "BETWEEN" and isinstance(value, list) and len(value) == 2:
            conditions.append({prop: {"$gte": value[0], "$lte": value[1]}})
        elif op in ("IS_NULL", "IS NULL"):
            conditions.append({prop: None})
        elif op in ("IS_NOT_NULL", "IS NOT NULL"):
            conditions.append({prop: {"$ne": None}})

    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _build_group(output_props: list, group_by: list) -> dict:
    """Build a $group stage from aggregated output_properties and group_by."""
    group = {}

    # _id: the group-by fields
    if group_by:
        if len(group_by) == 1:
            group["_id"] = f"${group_by[0]}"
        else:
            group["_id"] = {gb.replace(".", "_"): f"${gb}" for gb in group_by}
    else:
        group["_id"] = None  # aggregate over all documents

    # Accumulator fields from aggregated output_properties
    for op in output_props:
        if not isinstance(op, dict):
            continue
        prop = op.get("property_name")
        agg = (op.get("aggregation") or "NONE").upper()

        if not prop or agg in ("NONE", ""):
            continue

        safe_key = prop.replace(".", "_")

        agg_map = {
            "COUNT": {"$sum": 1},
            "SUM": {"$sum": f"${prop}"},
            "MIN": {"$min": f"${prop}"},
            "MAX": {"$max": f"${prop}"},
            "MEAN": {"$avg": f"${prop}"},
            "MEDIAN": {"$avg": f"${prop}"},  # approximation
        }

        if agg in agg_map:
            group[f"{safe_key}_{agg.lower()}"] = agg_map[agg]

    # Also include non-aggregated output fields as $first
    for op in output_props:
        if not isinstance(op, dict):
            continue
        prop = op.get("property_name")
        a = (op.get("aggregation") or "NONE").upper()
        if prop and a in ("NONE", ""):
            safe_key = prop.replace(".", "_")
            if safe_key not in group and f"{safe_key}_" not in str(group):
                group[safe_key] = {"$first": f"${prop}"}

    return group if len(group) > 1 else None  # >1 because _id is always there


def _build_project_after_group(output_props: list, group_by: list) -> dict:
    """Build a $project stage to clean up after $group (rename fields)."""
    project = {"_id": 0}

    for op in output_props:
        if not isinstance(op, dict):
            continue
        prop = op.get("property_name")
        agg = (op.get("aggregation") or "NONE").upper()
        if not prop:
            continue

        safe_key = prop.replace(".", "_")

        if agg not in ("NONE", ""):
            project[prop] = f"${safe_key}_{agg.lower()}"
        else:
            # Check if it's a group-by key
            if prop in group_by:
                if len(group_by) == 1:
                    project[prop] = "$_id"
                else:
                    project[prop] = f"$_id.{safe_key}"
            else:
                project[prop] = f"${safe_key}"

    return project if len(project) > 1 else None


def _build_simple_project(output_props: list) -> dict:
    """Build a $project for non-aggregated queries."""
    project = {"_id": 0}

    for op in output_props:
        if not isinstance(op, dict):
            continue
        prop = op.get("property_name")
        if prop:
            project[prop] = 1

    return project if len(project) > 1 else None


def _build_sort(order_by: list) -> dict:
    """Build a $sort document from V2 order_by."""
    sort = {}
    for ob in order_by:
        if not isinstance(ob, dict):
            continue
        prop = ob.get("property_name")
        direction = (ob.get("direction") or "ASC").upper()
        if prop:
            sort[prop] = -1 if direction == "DESC" else 1
    return sort if sort else None


if __name__ == "__main__":
    # Quick test with sample V2 args
    test_cases = [
        {
            "desc": "Simple find with filter",
            "args": {
                "collection_name": "sample_guides.planets",
                "output_properties": [
                    {"property_name": "name"},
                    {"property_name": "orderFromSun"},
                ],
                "filters": [
                    {"property_name": "hasRings", "operator": "=", "value": True}
                ],
            },
            "db": "sample_guides",
        },
        {
            "desc": "Aggregate with group + count",
            "args": {
                "collection_name": "sample_supplies.sales",
                "output_properties": [
                    {"property_name": "storeLocation"},
                    {"property_name": "_id", "aggregation": "COUNT"},
                ],
                "group_by_properties": ["storeLocation"],
                "order_by": [{"property_name": "storeLocation", "direction": "ASC"}],
            },
            "db": "sample_supplies",
        },
        {
            "desc": "Find with sort + limit",
            "args": {
                "collection_name": "sample_mflix.movies",
                "output_properties": [
                    {"property_name": "title"},
                    {"property_name": "year"},
                ],
                "filters": [
                    {"property_name": "year", "operator": ">=", "value": 2000}
                ],
                "order_by": [{"property_name": "year", "direction": "DESC"}],
                "limit": 5,
            },
            "db": "sample_mflix",
        },
    ]

    for tc in test_cases:
        mql = translate_v2_to_mql(tc["args"], tc["db"])
        print(f"=== {tc['desc']} ===")
        print(f"  MQL: {mql}")
        print()
