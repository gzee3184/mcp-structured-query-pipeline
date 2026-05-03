"""
Value-Aware Schema Linking

Generates and uses column value statistics to improve schema discovery.
When a query says "Find students in California", this module knows that the
'state' column contains "California" and can boost the right collection.

Addresses GAP 2 from Research.txt: The pipeline operates on schema structure only.
Real-world schema linking requires understanding data values.
"""

import json
import re
from pathlib import Path
from typing import Optional
from collections import defaultdict


class ValueStats:
    """
    Manages column value statistics for collections.

    For each collection property, stores:
    - Numeric: min, max, mean, common values
    - Text: top-K most common values, cardinality
    - Boolean: true_ratio

    These are used during discovery to match query values to collections.
    """

    def __init__(self, stats: Optional[dict] = None):
        """
        Args:
            stats: Pre-computed stats dict: {collection: {property: {min, max, top_values, ...}}}
        """
        self.stats = stats or {}
        # Inverted index: value -> [(collection, property)]
        self._value_index = defaultdict(list)
        if self.stats:
            self._build_value_index()

    def _build_value_index(self):
        """Build inverted index from values to (collection, property) pairs."""
        for collection, props in self.stats.items():
            for prop_name, prop_stats in props.items():
                # Index top text values
                for val in prop_stats.get('top_values', []):
                    val_lower = str(val).lower()
                    self._value_index[val_lower].append((collection, prop_name))

                # Index range boundaries for numeric properties
                if 'min' in prop_stats and 'max' in prop_stats:
                    self._value_index[f"__range_{collection}_{prop_name}"] = {
                        'min': prop_stats['min'],
                        'max': prop_stats['max'],
                        'collection': collection,
                        'property': prop_name
                    }

    @classmethod
    def from_schema_file(cls, schema_file: str) -> "ValueStats":
        """
        Extract value statistics from a schema file that includes value metadata.

        Some schema files include 'value_examples' or 'description' fields that
        contain value information we can parse.
        """
        stats = {}
        path = Path(schema_file)
        if not path.exists():
            return cls()

        data = json.loads(path.read_text())

        if 'weaviate_collections' in data:
            for coll in data['weaviate_collections']:
                coll_name = coll['name']
                coll_stats = {}

                for prop in coll.get('properties', []):
                    prop_name = prop['name']
                    prop_stats = {}

                    dtype = prop.get('data_type', ['unknown'])
                    if isinstance(dtype, list):
                        dtype = dtype[0]

                    # Extract value hints from description
                    desc = prop.get('description', '')
                    if desc:
                        # Look for value examples in description
                        values = _extract_values_from_description(desc, dtype)
                        if values:
                            prop_stats['top_values'] = values

                        # Look for numeric ranges
                        ranges = _extract_numeric_ranges(desc)
                        if ranges:
                            prop_stats.update(ranges)

                    if prop_stats:
                        coll_stats[prop_name] = prop_stats

                if coll_stats:
                    stats[coll_name] = coll_stats

        # Also check for queries that reveal value information
        if 'queries' in data:
            for query in data['queries']:
                coll = query.get('target_collection', '')
                if coll not in stats:
                    stats[coll] = {}

                # Extract values from filters
                for filter_key in ['integer_property_filter', 'text_property_filter']:
                    f = query.get(filter_key)
                    if f and isinstance(f, dict):
                        prop = f.get('property_name', '')
                        val = f.get('value')
                        if prop and val is not None:
                            if prop not in stats.get(coll, {}):
                                stats[coll][prop] = {}
                            existing = stats[coll][prop].get('top_values', [])
                            if val not in existing:
                                existing.append(val)
                            stats[coll][prop]['top_values'] = existing[:20]

        return cls(stats)

    @classmethod
    def from_multiple_sources(cls, *schema_files: str) -> "ValueStats":
        """Merge value stats from multiple schema files."""
        combined = {}
        for f in schema_files:
            vs = cls.from_schema_file(f)
            for coll, props in vs.stats.items():
                if coll not in combined:
                    combined[coll] = {}
                for prop, pstats in props.items():
                    if prop not in combined[coll]:
                        combined[coll][prop] = pstats
                    else:
                        # Merge top_values
                        existing = combined[coll][prop].get('top_values', [])
                        new = pstats.get('top_values', [])
                        merged = list(set(existing + new))[:20]
                        combined[coll][prop]['top_values'] = merged
                        # Merge numeric ranges
                        if 'min' in pstats:
                            combined[coll][prop]['min'] = min(
                                combined[coll][prop].get('min', float('inf')),
                                pstats['min']
                            )
                        if 'max' in pstats:
                            combined[coll][prop]['max'] = max(
                                combined[coll][prop].get('max', float('-inf')),
                                pstats['max']
                            )
        return cls(combined)

    def match_query_values(self, query: str) -> dict:
        """
        Match values mentioned in a query to collections.

        Returns:
            Dict of {collection_name: match_score}
        """
        scores = defaultdict(float)

        # Extract potential values from query
        query_lower = query.lower()

        # 1. Match text values (proper nouns, quoted strings)
        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', query) + re.findall(r"'([^']+)'", query)
        for val in quoted:
            val_lower = val.lower()
            for coll, prop in self._value_index.get(val_lower, []):
                scores[coll] += 0.3  # Strong match for exact value

        # Proper nouns (capitalized words that aren't at sentence start)
        words = query.split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and len(word) > 2:
                word_lower = word.lower()
                for coll, prop in self._value_index.get(word_lower, []):
                    scores[coll] += 0.2

        # Common words (non-stopwords, longer than 3 chars)
        stopwords = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'have',
                     'find', 'show', 'list', 'give', 'what', 'how', 'many', 'which'}
        for word in re.findall(r'\b[a-zA-Z]{4,}\b', query_lower):
            if word not in stopwords:
                for coll, prop in self._value_index.get(word, []):
                    scores[coll] += 0.1

        # 2. Match numeric values against ranges
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', query)
        for num_str in numbers:
            num = float(num_str)
            for key, info in self._value_index.items():
                if key.startswith('__range_') and isinstance(info, dict):
                    if info['min'] <= num <= info['max']:
                        scores[info['collection']] += 0.1

        return dict(scores)

    def get_value_context(self, collection: str, max_values: int = 5) -> str:
        """
        Get a value summary string for a collection to include in prompts.

        Example: "[Values: state has California, Texas, New York; price ranges 10-500]"
        """
        coll_stats = self.stats.get(collection, {})
        if not coll_stats:
            return ""

        parts = []
        for prop, pstats in list(coll_stats.items())[:5]:
            if 'top_values' in pstats:
                vals = pstats['top_values'][:max_values]
                val_str = ', '.join(str(v) for v in vals)
                parts.append(f"{prop}: {val_str}")
            elif 'min' in pstats and 'max' in pstats:
                parts.append(f"{prop}: {pstats['min']}-{pstats['max']}")

        if not parts:
            return ""
        return f"[Sample values: {'; '.join(parts)}]"

    def boost_discovery_scores(self, scores: list, query: str) -> list:
        """
        Boost discovery scores based on value matching.

        Args:
            scores: List of (collection_name, score) tuples
            query: Natural language query

        Returns:
            Re-ranked list of (collection_name, score) tuples
        """
        value_matches = self.match_query_values(query)
        if not value_matches:
            return scores

        boosted = []
        for name, score in scores:
            value_boost = value_matches.get(name, 0.0)
            boosted.append((name, score + value_boost))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def save(self, path: str):
        """Save stats to JSON file."""
        Path(path).write_text(json.dumps(self.stats, indent=2, default=str))

    @classmethod
    def load(cls, path: str) -> "ValueStats":
        """Load stats from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(data)

    def enrich_from_sql_queries(self, queries: list, db_to_collection_fn=None):
        """
        Extract actual values from SQL WHERE clauses and enrich the stats.

        This addresses the shallow value stats problem: instead of only
        parsing schema descriptions, we extract concrete filter values
        from SQL ground truth (e.g., BIRD benchmark).

        Args:
            queries: List of dicts with 'SQL', 'db_id' keys (BIRD format)
            db_to_collection_fn: Function(db_id, table_name) -> collection_name
        """
        if db_to_collection_fn is None:
            def db_to_collection_fn(db, table):
                def pascal(n): return ''.join(p.capitalize() for p in n.split('_'))
                return f"{pascal(db)}{pascal(table)}"

        values_added = 0

        for q in queries:
            sql = q.get('SQL', '')
            db_id = q.get('db_id', '')
            if not sql or not db_id:
                continue

            # Extract FROM table
            from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
            if not from_match:
                continue
            table = from_match.group(1)
            collection = db_to_collection_fn(db_id, table)

            if collection not in self.stats:
                self.stats[collection] = {}

            # Extract WHERE clause values
            # Pattern: column = 'value' or column = number
            where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|HAVING|$)', sql, re.IGNORECASE | re.DOTALL)
            if not where_match:
                continue
            where_clause = where_match.group(1)

            # Text values: column = 'value' or column LIKE '%value%'
            text_patterns = re.findall(
                r"(?:(\w+)\.)?(\w+)\s*(?:=|LIKE|!=)\s*'([^']+)'",
                where_clause, re.IGNORECASE
            )
            for alias, col, val in text_patterns:
                col_lower = col.lower()
                if col_lower not in self.stats[collection]:
                    self.stats[collection][col_lower] = {}
                existing = self.stats[collection][col_lower].get('top_values', [])
                clean_val = val.strip('%')  # Remove LIKE wildcards
                if clean_val and clean_val not in existing and len(existing) < 30:
                    existing.append(clean_val)
                    self.stats[collection][col_lower]['top_values'] = existing
                    values_added += 1

            # Numeric values: column > number, column = number
            num_patterns = re.findall(
                r"(?:(\w+)\.)?(\w+)\s*(?:=|>|<|>=|<=|!=)\s*(\d+(?:\.\d+)?)",
                where_clause
            )
            for alias, col, num_str in num_patterns:
                col_lower = col.lower()
                if col_lower not in self.stats[collection]:
                    self.stats[collection][col_lower] = {}
                num = float(num_str)
                stats = self.stats[collection][col_lower]
                stats['min'] = min(stats.get('min', num), num)
                stats['max'] = max(stats.get('max', num), num)
                values_added += 1

        # Rebuild inverted index with enriched data
        self._value_index = defaultdict(list)
        self._build_value_index()

        return values_added


def _extract_values_from_description(description: str, dtype: str) -> list:
    """Extract value examples from a property description string."""
    values = []

    # Look for patterns like "e.g., X, Y, Z" or "such as X, Y"
    patterns = [
        r'(?:e\.g\.|such as|like|including|values?:)\s*([^.]+)',
        r'(?:one of|options?:)\s*([^.]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            text = match.group(1)
            # Split on commas and clean
            parts = [p.strip().strip("'\"") for p in text.split(',')]
            values.extend(p for p in parts if p and len(p) < 50)

    return values[:10]


def _extract_numeric_ranges(description: str) -> dict:
    """Extract numeric range information from a description."""
    result = {}

    # Pattern: "ranges from X to Y" or "between X and Y"
    patterns = [
        r'(?:ranges? from|between)\s+(\d+(?:\.\d+)?)\s+(?:to|and)\s+(\d+(?:\.\d+)?)',
        r'(?:min|minimum)[:\s]+(\d+(?:\.\d+)?).*?(?:max|maximum)[:\s]+(\d+(?:\.\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            result['min'] = float(match.group(1))
            result['max'] = float(match.group(2))
            break

    return result
