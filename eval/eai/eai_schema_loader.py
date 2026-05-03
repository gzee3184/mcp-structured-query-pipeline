#!/usr/bin/env python3
"""
eai_schema_loader.py — Load EAI MongoDB Atlas schemas into our pipeline's
MCPServer collection format.

Reads the .metadata.json and BSON files from the EAI dataset, introspects
document structure from sample documents, and builds a schema registry
compatible with our MCPServer.

The key challenge: MongoDB documents have nested structures (embedded arrays,
sub-documents). We represent these as flat property lists with dot-notation
paths, since our V2 schema uses flat property references.
"""

import json
import bson
from pathlib import Path
from collections import defaultdict


# EAI data lives at the repo root level, not inside gorilla_2/gorilla/
EAI_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "datasets/eai-mongosh"
EAI_DB_DIR = EAI_ROOT / "databases"


def _infer_type(value) -> str:
    """Infer a Weaviate-like type string from a Python/BSON value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "text"
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return "object[]"
        return "text[]"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "text"
    return "text"


def _flatten_schema(doc: dict, prefix: str = "", max_depth: int = 3) -> dict:
    """Flatten a MongoDB document into a flat property map with dot-notation keys.

    Returns: {property_path: type_string}

    For nested arrays of objects, we traverse one level into the array element
    structure (common in EAI: transactions.date, items.name, etc.)
    """
    props = {}
    if max_depth <= 0:
        return props

    for key, value in doc.items():
        if key == '_id':
            props['_id'] = 'text'
            continue

        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"

        if isinstance(value, dict):
            # Recurse into sub-document
            props[path] = "object"
            sub = _flatten_schema(value, prefix=path, max_depth=max_depth - 1)
            props.update(sub)
        elif isinstance(value, list) and value:
            if isinstance(value[0], dict):
                # Array of objects — traverse one element
                props[path] = "object[]"
                sub = _flatten_schema(value[0], prefix=path, max_depth=max_depth - 1)
                props.update(sub)
            else:
                props[path] = _infer_type(value)
        else:
            props[path] = _infer_type(value)

    return props


def load_eai_schemas(sample_docs: int = 5) -> dict:
    """Load all EAI collection schemas by introspecting sample documents.

    Returns a dict matching MCPServer's schema format:
    {
        "db_name.collection_name": {
            "name": "db_name.collection_name",
            "description": "...",
            "properties": { "prop_name": {"type": "...", "description": "..."} }
        }
    }
    """
    schemas = {}

    for db_dir in sorted(EAI_DB_DIR.iterdir()):
        if not db_dir.is_dir() or db_dir.name in ('admin', '.cache'):
            continue

        db_name = db_dir.name

        for bson_file in sorted(db_dir.glob('*.bson')):
            coll_name = bson_file.stem
            full_name = f"{db_name}.{coll_name}"

            # Read sample documents to infer schema
            with open(bson_file, 'rb') as f:
                raw = f.read()
            all_docs = bson.decode_all(raw)

            if not all_docs:
                continue

            # Merge schemas from multiple sample docs for better coverage
            merged_props = {}
            for doc in all_docs[:sample_docs]:
                flat = _flatten_schema(doc, max_depth=3)
                for k, v in flat.items():
                    if k not in merged_props:
                        merged_props[k] = v

            # Build MCPServer-compatible schema
            properties = {}
            for prop_path, prop_type in sorted(merged_props.items()):
                properties[prop_path] = {
                    "type": prop_type,
                    "description": f"{prop_path} field"
                }

            schemas[full_name] = {
                "name": full_name,
                "description": f"MongoDB collection {coll_name} in database {db_name}. Contains {len(all_docs)} documents.",
                "envisioned_use_case_overview": f"Query {coll_name} data from MongoDB Atlas {db_name} sample dataset.",
                "properties": properties,
                "n_docs": len(all_docs),
            }

    return schemas


def load_eai_queries() -> list:
    """Load EAI queries from the CSV benchmark file.

    Returns list of dicts with:
        question, expected_collection, expected_mql, expected_result,
        db_name, complexity, source
    """
    import csv

    csv_path = EAI_ROOT / "atlas_sample_data_benchmark.flat.csv"

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    queries = []
    for row in rows:
        db_name = row.get('input.databaseName', '')
        nl_query = row.get('input.nlQuery', '')
        expected_mql = row.get('expected.dbQuery', '')
        expected_result = row.get('expected.result', '')
        complexity = row.get('metadata.complexity', '')

        # Extract collection name from MQL: db.<collection>.<method>(
        import re
        coll_match = re.search(r'db\.(\w+)\.(find|aggregate|findOne)', expected_mql)
        coll_name = coll_match.group(1) if coll_match else ''

        if not nl_query or not coll_name:
            continue

        full_coll = f"{db_name}.{coll_name}"

        queries.append({
            "question": nl_query,
            "expected_collection": full_coll,
            "acceptable_collections": [full_coll],
            "expected_mql": expected_mql,
            "expected_result": expected_result,
            "db_name": db_name,
            "collection_name": coll_name,
            "complexity": complexity,
            "source": f"eai_{db_name}",
            "has_join": False,  # EAI is single-collection
        })

    return queries


if __name__ == "__main__":
    schemas = load_eai_schemas()
    queries = load_eai_queries()

    print(f"Schemas loaded: {len(schemas)}")
    for name, schema in schemas.items():
        n_props = len(schema.get('properties', {}))
        n_docs = schema.get('n_docs', 0)
        print(f"  {name}: {n_props} properties, {n_docs} docs")

    print(f"\nQueries loaded: {len(queries)}")
    from collections import Counter
    by_coll = Counter(q['expected_collection'] for q in queries)
    print("By collection:")
    for coll, n in by_coll.most_common():
        print(f"  {coll}: {n}")

    by_complexity = Counter(q['complexity'] for q in queries)
    print(f"\nBy complexity: {dict(by_complexity)}")
