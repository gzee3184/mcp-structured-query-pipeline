#!/usr/bin/env python3
"""
eai_executor.py — Execute MongoDB queries against EAI data using mongomock.

Provides:
1. MongoExecutor: loads EAI BSON data into mongomock, executes mongosh-style
   query strings, and returns result documents.
2. EvoMQL metric computation: SE, COF, NEO, RO, OPS per their paper's
   exact definitions.

Security note: Uses ast.literal_eval (not eval) for parsing Python-like
expressions from MQL strings. ast.literal_eval is safe — it only parses
literals (dicts, lists, strings, numbers, booleans, None) and rejects
any executable code.
"""

import json
import re
import ast
import bson
from pathlib import Path
from datetime import datetime

import mongomock

EAI_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent / "datasets/eai-mongosh"
EAI_DB_DIR = EAI_ROOT / "databases"


class MongoExecutor:
    """In-memory MongoDB executor using mongomock."""

    def __init__(self):
        self.client = mongomock.MongoClient()
        self._loaded = False

    def load_data(self):
        """Load all EAI BSON data into mongomock collections."""
        if self._loaded:
            return

        for db_dir in sorted(EAI_DB_DIR.iterdir()):
            if not db_dir.is_dir() or db_dir.name in ('admin', '.cache'):
                continue
            db = self.client[db_dir.name]
            for bson_file in sorted(db_dir.glob('*.bson')):
                coll_name = bson_file.stem
                with open(bson_file, 'rb') as f:
                    raw = f.read()
                docs = bson.decode_all(raw)
                if docs:
                    db[coll_name].insert_many(docs)

        self._loaded = True

    def execute_mql(self, db_name: str, mql_string: str) -> tuple:
        """Execute a mongosh-style query string.

        Returns (result_list, error_string_or_None).
        result_list is a list of dicts (documents), or None on error.
        """
        self.load_data()

        try:
            result = self._parse_and_execute(db_name, mql_string)
            clean = self._clean_results(result)
            return clean, None
        except Exception as e:
            return None, str(e)

    def _parse_and_execute(self, db_name: str, mql: str):
        """Parse mongosh syntax and execute against mongomock."""
        db = self.client[db_name]
        mql = mql.strip().rstrip(';')

        match = re.match(r'db\.(\w+)\.(find|findOne|aggregate)\s*\(', mql)
        if not match:
            raise ValueError(f"Cannot parse MQL pattern: {mql[:80]}")

        coll_name = match.group(1)
        method = match.group(2)
        coll = db[coll_name]

        if method == 'aggregate':
            return self._exec_aggregate(coll, mql)
        elif method == 'find':
            return self._exec_find(coll, mql)
        elif method == 'findOne':
            return self._exec_find_one(coll, mql)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _preprocess_mql(self, s: str) -> str:
        """Convert mongosh JavaScript syntax to Python-parseable literals."""
        # Remove chained methods
        s = re.sub(r'\)\s*\.(toArray|pretty|forEach|map|count|length)\s*\(\s*\)', ')', s)

        # new Date("...") / ISODate("...")
        s = re.sub(r'(?:new\s+Date|ISODate)\s*\(\s*["\']([^"\']*?)["\']\s*\)',
                    r'{"$date": "\1"}', s)
        s = re.sub(r'(?:new\s+Date|ISODate)\s*\(\s*\)', '{"$date": "now"}', s)

        # ObjectId("...")
        s = re.sub(r'ObjectId\s*\(\s*["\']([^"\']*?)["\']\s*\)',
                    r'{"$oid": "\1"}', s)

        # NumberInt / NumberLong / NumberDecimal
        s = re.sub(r'NumberInt\s*\(\s*(\d+)\s*\)', r'\1', s)
        s = re.sub(r'NumberLong\s*\(\s*(\d+)\s*\)', r'\1', s)
        s = re.sub(r'NumberDecimal\s*\(\s*["\']?([\d.]+)["\']?\s*\)', r'\1', s)

        # JS booleans/null
        s = s.replace('true', 'True').replace('false', 'False').replace('null', 'None')

        return s

    def _safe_parse(self, s: str):
        """Parse a mongosh-style expression into Python objects.

        MongoDB query syntax uses $ operators ($gte, $match, etc.) which
        aren't valid Python identifiers. We convert to JSON first (quoting
        all keys), then parse with json.loads.
        """
        s = self._preprocess_mql(s).strip()
        if not s:
            return None

        def _convert_dates(obj):
            if isinstance(obj, dict):
                if "$date" in obj:
                    date_str = obj["$date"]
                    if date_str == "now":
                        return datetime.now()
                    try:
                        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except Exception:
                        try:
                            return datetime.strptime(date_str, "%Y-%m-%d")
                        except Exception:
                            return date_str
                return {k: _convert_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert_dates(v) for v in obj]
            return obj

        # Strategy: convert the mongosh JS-like syntax to valid JSON, then parse.
        # Keys in MongoDB queries can be:
        #   - bare identifiers: purchaseMethod, storeLocation
        #   - $operators: $gte, $match, $unwind, $group
        #   - dotted paths: "review_scores.review_scores_rating"
        #   - _id
        # All need to be double-quoted for valid JSON.
        s_json = self._mql_to_json(s)

        try:
            result = json.loads(s_json)
            return _convert_dates(result)
        except json.JSONDecodeError:
            # Fallback: try ast.literal_eval on the preprocessed form
            try:
                result = ast.literal_eval(s)
                return _convert_dates(result)
            except (ValueError, SyntaxError):
                raise ValueError(f"Cannot parse expression: {s[:200]}")

    def _mql_to_json(self, s: str) -> str:
        """Convert mongosh JavaScript-like object syntax to valid JSON.

        Handles:
        - Bare identifier keys (purchaseMethod → "purchaseMethod")
        - $operator keys ($gte → "$gte")
        - Single-quoted strings → double-quoted
        - Python True/False/None → JSON true/false/null
        - Trailing commas
        """
        # Step 1: Replace single-quoted strings with double-quoted
        # (but preserve single quotes inside double-quoted strings)
        result = []
        i = 0
        in_double = False
        in_single = False

        while i < len(s):
            ch = s[i]
            if ch == '"' and not in_single:
                in_double = not in_double
                result.append(ch)
            elif ch == "'" and not in_double:
                if not in_single:
                    in_single = True
                    result.append('"')
                else:
                    in_single = False
                    result.append('"')
            else:
                result.append(ch)
            i += 1

        s = ''.join(result)

        # Step 2: Convert Python booleans/None to JSON
        s = s.replace('True', 'true').replace('False', 'false').replace('None', 'null')

        # Step 3: Quote bare identifier keys that aren't already quoted.
        # Match any bare key (word chars, $, dots) followed by a colon,
        # but not already inside quotes. We use a simple approach:
        # find all occurrences of `bareKey:` and wrap bareKey in quotes.
        def _quote_keys(text):
            # Match: optional whitespace, then a bare key (starting with $ or letter),
            # possibly containing dots and underscores, followed by :
            # But NOT if preceded by a quote (already quoted)
            return re.sub(
                r'(?<!["\w])([\$a-zA-Z_][\$\w.]*)\s*:',
                r'"\1":',
                text
            )
        s = _quote_keys(s)

        # Step 4: Remove trailing commas before ] or }
        s = re.sub(r',\s*([}\]])', r'\1', s)

        return s

    def _extract_args(self, mql: str, method_pattern: str) -> str:
        """Extract arguments from inside the first matching parentheses."""
        match = re.search(method_pattern + r'\s*\(', mql)
        if not match:
            return ""

        start = match.end()
        depth = 1
        i = start
        while i < len(mql) and depth > 0:
            if mql[i] == '(':
                depth += 1
            elif mql[i] == ')':
                depth -= 1
            i += 1

        return mql[start:i-1]

    def _split_find_args(self, args_str: str) -> list:
        """Split find arguments at top-level commas."""
        depth = 0
        parts = []
        current = []

        for ch in args_str:
            if ch in ('{', '[', '('):
                depth += 1
                current.append(ch)
            elif ch in ('}', ']', ')'):
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)

        if current:
            parts.append(''.join(current).strip())

        return parts

    def _exec_aggregate(self, coll, mql: str):
        args_str = self._extract_args(mql, r'\.aggregate')
        pipeline = self._safe_parse(args_str)

        if not isinstance(pipeline, list):
            raise ValueError(f"Aggregate pipeline must be a list, got {type(pipeline)}")

        return list(coll.aggregate(pipeline))

    def _exec_find(self, coll, mql: str):
        args_str = self._extract_args(mql, r'\.find')
        parts = self._split_find_args(args_str)

        query = self._safe_parse(parts[0]) if len(parts) > 0 and parts[0].strip() else {}
        projection = self._safe_parse(parts[1]) if len(parts) > 1 and parts[1].strip() else None

        cursor_args = {}
        if projection:
            cursor_args['projection'] = projection

        cursor = coll.find(query or {}, **cursor_args)

        # Handle chained .sort() and .limit()
        remainder = mql[mql.find(')', mql.find('.find(')) + 1:]

        sort_match = re.search(r'\.sort\s*\(', remainder)
        if sort_match:
            sort_str = self._extract_args(remainder, r'\.sort')
            sort_spec = self._safe_parse(sort_str)
            if isinstance(sort_spec, dict):
                cursor = cursor.sort(list(sort_spec.items()))

        limit_match = re.search(r'\.limit\s*\(\s*(\d+)\s*\)', remainder)
        if limit_match:
            cursor = cursor.limit(int(limit_match.group(1)))

        return list(cursor)

    def _exec_find_one(self, coll, mql: str):
        args_str = self._extract_args(mql, r'\.findOne')
        parts = self._split_find_args(args_str)

        query = self._safe_parse(parts[0]) if len(parts) > 0 and parts[0].strip() else {}
        projection = self._safe_parse(parts[1]) if len(parts) > 1 and parts[1].strip() else None

        result = coll.find_one(query or {}, projection)
        return [result] if result else []

    def _clean_results(self, results) -> list:
        """Convert MongoDB results to JSON-serializable form."""
        if results is None:
            return []

        clean = []
        for doc in results:
            if not isinstance(doc, dict):
                clean.append(doc)
                continue
            clean_doc = {}
            for k, v in doc.items():
                if k == '_id':
                    continue
                clean_doc[k] = self._serialize(v)
            clean.append(clean_doc)
        return clean

    def _serialize(self, value):
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore')
        if isinstance(value, dict):
            return {k: self._serialize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize(v) for v in value]
        if isinstance(value, float):
            return round(value, 6)
        return value


# ---- EvoMQL Metrics ----

def compute_evomql_metrics(gold_result, pred_result, pred_error) -> dict:
    """Compute SE, COF, NEO, RO per EvoMQL paper definitions.

    Returns dict with: se, cof, neo, ro, ops
    """
    se = 1.0 if pred_error is None else 0.0
    neo = 1.0 if pred_result and len(pred_result) > 0 else 0.0

    ro = 0.0
    if pred_result and len(pred_result) > 0:
        has_null_or_empty = False
        for doc in pred_result:
            if isinstance(doc, dict):
                for v in doc.values():
                    if v is None or v == "" or v == []:
                        has_null_or_empty = True
                        break
            if has_null_or_empty:
                break
        ro = 1.0 if not has_null_or_empty else 0.0

    cof = 0.0
    if pred_result is not None and gold_result is not None:
        cof = _fuzzy_result_match(gold_result, pred_result)

    ops = 0.6 * cof + 0.2 * se + 0.1 * neo + 0.1 * ro

    return {"se": se, "cof": cof, "neo": neo, "ro": ro, "ops": ops}


def _fuzzy_result_match(gold: list, pred: list) -> float:
    """Fuzzy comparison of two result sets per EvoMQL definitions."""
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0

    gold_norm = [_normalize_doc(d) for d in gold if isinstance(d, dict)]
    pred_norm = [_normalize_doc(d) for d in pred if isinstance(d, dict)]

    if not gold_norm:
        return 1.0 if not pred_norm else 0.0

    matched = 0
    pred_used = set()

    for g_doc in gold_norm:
        for j, p_doc in enumerate(pred_norm):
            if j in pred_used:
                continue
            if _docs_fuzzy_match(g_doc, p_doc):
                matched += 1
                pred_used.add(j)
                break

    return matched / len(gold_norm)


def _normalize_doc(doc: dict) -> dict:
    result = {}
    for k, v in doc.items():
        if k == '_id':
            continue
        result[k.lower()] = _normalize_value(v)
    return result


def _normalize_value(v):
    if isinstance(v, str):
        return v.lower().strip()
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, dict):
        return {k.lower(): _normalize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_normalize_value(item) for item in v]
    return v


def _docs_fuzzy_match(gold: dict, pred: dict) -> bool:
    """Check if pred doc matches gold doc (gold fields must be present)."""
    for key, gold_val in gold.items():
        if key not in pred:
            return False
        if not _values_fuzzy_equal(gold_val, pred[key]):
            return False
    return True


def _values_fuzzy_equal(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == 0 and b == 0:
            return True
        if a == 0:
            return abs(b) < 0.01
        return abs(a - b) / max(abs(a), 1e-10) < 0.01

    if isinstance(a, str) and isinstance(b, str):
        return a.lower().strip() == b.lower().strip()

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_values_fuzzy_equal(x, y) for x, y in zip(a, b))

    if isinstance(a, dict) and isinstance(b, dict):
        return _docs_fuzzy_match(a, b)

    return a == b


if __name__ == "__main__":
    from eai_schema_loader import load_eai_queries

    executor = MongoExecutor()
    executor.load_data()

    queries = load_eai_queries()

    success = 0
    fail = 0
    for q in queries[:20]:
        result, err = executor.execute_mql(q['db_name'], q['expected_mql'])
        if err:
            fail += 1
            if fail <= 3:
                print(f"  FAIL: {q['question'][:60]}")
                print(f"    MQL: {q['expected_mql'][:120]}")
                print(f"    Error: {err}")
        else:
            success += 1
            if success <= 3:
                print(f"  OK: {q['question'][:60]} -> {len(result)} docs")

    print(f"\nGold query execution: {success}/{success+fail} succeeded ({fail} failed)")
