#!/usr/bin/env python3
"""
real_mongo_executor.py — Execute MongoDB queries against real MongoDB (port 27117).

Uses pymongo instead of mongomock. Handles the same mongosh-style parsing
but executes against a real MongoDB instance with full operator support.

Security: Uses ast.literal_eval (safe literal-only parser, no code execution)
for converting MQL expression strings into Python dicts/lists.
"""

import json
import re
import ast  # ast.literal_eval: safe, parses only literals
import signal
from datetime import datetime
from contextlib import contextmanager

import pymongo

MONGO_PORT = 27117


@contextmanager
def timeout_context(seconds):
    """Context manager for query timeout."""
    def handler(signum, frame):
        raise TimeoutError(f"Query timed out after {seconds}s")
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class RealMongoExecutor:
    """Execute MongoDB queries against a real local MongoDB instance."""

    def __init__(self, port: int = MONGO_PORT):
        self.client = pymongo.MongoClient('localhost', port,
                                          serverSelectionTimeoutMS=5000)
        self.client.server_info()

    def execute_mql(self, db_name: str, mql_string: str, timeout_sec: int = 30) -> tuple:
        """Execute a mongosh-style query string.

        Returns (result_list, error_string_or_None).
        """
        try:
            with timeout_context(timeout_sec):
                result = self._parse_and_execute(db_name, mql_string)
                clean = self._clean_results(result)
                return clean, None
        except Exception as e:
            return None, str(e)

    def _parse_and_execute(self, db_name: str, mql: str):
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
        """Convert mongosh JS syntax to Python-parseable form."""
        s = re.sub(r'\)\s*\.(toArray|pretty|forEach|map|count|length)\s*\(\s*\)', ')', s)
        s = re.sub(r'(?:new\s+Date|ISODate)\s*\(\s*["\']([^"\']*?)["\']\s*\)',
                    r'{"$date": "\1"}', s)
        s = re.sub(r'(?:new\s+Date|ISODate)\s*\(\s*\)', '{"$date": "now"}', s)
        s = re.sub(r'ObjectId\s*\(\s*["\']([^"\']*?)["\']\s*\)',
                    r'{"$oid": "\1"}', s)
        s = re.sub(r'NumberInt\s*\(\s*(\d+)\s*\)', r'\1', s)
        s = re.sub(r'NumberLong\s*\(\s*(\d+)\s*\)', r'\1', s)
        s = re.sub(r'NumberDecimal\s*\(\s*["\']?([\d.]+)["\']?\s*\)', r'\1', s)
        s = s.replace('true', 'True').replace('false', 'False').replace('null', 'None')
        return s

    def _mql_to_json(self, s: str) -> str:
        """Convert preprocessed MQL to valid JSON."""
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
                in_single = not in_single
                result.append('"')
            else:
                result.append(ch)
            i += 1
        s = ''.join(result)
        s = s.replace('True', 'true').replace('False', 'false').replace('None', 'null')

        def _quote_keys(text):
            return re.sub(r'(?<!["\w])([\$a-zA-Z_][\$\w.]*)\s*:', r'"\1":', text)
        s = _quote_keys(s)
        s = re.sub(r',\s*([}\]])', r'\1', s)
        return s

    def _safe_parse(self, s: str):
        """Parse MQL expression into Python objects (safe, no code execution)."""
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

        s_json = self._mql_to_json(s)
        try:
            result = json.loads(s_json)
            return _convert_dates(result)
        except json.JSONDecodeError:
            try:
                # ast.literal_eval: safe — only parses Python literal values
                result = ast.literal_eval(self._preprocess_mql(s.strip()))
                return _convert_dates(result)
            except (ValueError, SyntaxError):
                raise ValueError(f"Cannot parse expression: {s[:200]}")

    def _extract_args(self, mql: str, method_pattern: str) -> str:
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
        cursor = coll.find(query or {}, projection)

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


if __name__ == "__main__":
    """Quick gold-query validation against real MongoDB."""
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
    from eai_schema_loader import load_eai_queries
    from collections import Counter

    executor = RealMongoExecutor()
    queries = load_eai_queries()

    results = Counter()
    for i, q in enumerate(queries[:100]):
        result, err = executor.execute_mql(q['db_name'], q['expected_mql'])
        if err:
            results['fail'] += 1
            if results['fail'] <= 3:
                print(f"  FAIL [{i}]: {err[:100]}")
        else:
            results['success'] += 1

    print(f"\nGold execution on real MongoDB (N=100):")
    print(f"  Success: {results['success']}%")
    print(f"  Fail: {results['fail']}")
