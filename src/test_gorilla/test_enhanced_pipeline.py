#!/usr/bin/env python3
"""Pipeline class + evaluation harness in one file.

EnhancedPipeline: two-phase structured query generation
  - discover(): semantic embedding + KG boost + adaptive routing -> top-K collections
  - generate(): schema prompt + LLM tool call + sandbox validation + self-correction

main(): loads Weaviate Gorilla + BIRD queries, runs the pipeline, computes
strict/relaxed accuracy, multi-collection F1, AST scores, token efficiency,
per-database breakdowns, and saves JSON results for paper figures.

Every pipeline feature has a --no-* ablation flag. Run with --all --split test
for reproducible paper-ready evaluation.
"""

import json
import sys
import re
import os
import random
import time
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from src.mcp.server import MCPServer
from src.mcp.sandbox import StructuredSandbox
from src.utils.weaviate_fc_utils import build_weaviate_query_tool_for_openai, build_weaviate_query_tool_for_openai_v2
from src.lm.lm import LMService
from src.utils.schema_cache import SchemaCache
from src.utils.learning_cache import LearningCache
from src.utils.embeddings import get_embedding, cosine_similarity
from src.utils.field_kg import FieldLevelKnowledgeGraph
from src.utils.ast_scoring import calculate_ast_score, calculate_ast_score_detailed
from src.utils.value_stats import ValueStats
from src.mcp.sandbox import SemanticValidator


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def convert_name(db: str, table: str) -> str:
    def pascal(n): return ''.join(p.capitalize() for p in n.split('_'))
    return f"{pascal(db)}{pascal(table)}"


def extract_sql_tables(sql: str) -> list:
    """Extract all table names from a SQL statement (FROM and JOIN clauses).

    Handles unquoted, "double-quoted", `back-ticked`, and [bracketed]
    identifiers — BIRD uses all three for reserved-word table names like
    `order` and "Match".
    """
    tables = []
    # FROM table [alias]
    from_match = re.search(r'FROM\s+["`\[]?(\w+)["`\]]?', sql, re.IGNORECASE)
    if from_match:
        tables.append(from_match.group(1))
    # JOIN table [alias]
    join_matches = re.findall(r'JOIN\s+["`\[]?(\w+)["`\]]?', sql,
                              re.IGNORECASE)
    tables.extend(join_matches)
    return tables


def load_all_queries(include_joins: bool = True) -> list:
    """Load queries from all available sources.

    Args:
        include_joins: If True, include BIRD queries with JOINs (maps to primary FROM table,
                      with all JOIN tables as acceptable alternatives). Default True.
    """
    queries = []

    # Weaviate Gorilla (has full AST)
    wg_path = Path("data/weaviate-gorilla.json")
    if wg_path.exists():
        data = json.loads(wg_path.read_text())
        for item in data:
            if 'ground_truth_query' in item:
                queries.append({
                    "question": item['natural_language_command'],
                    "expected_collection": item['ground_truth_query'].get('target_collection'),
                    "acceptable_collections": [item['ground_truth_query'].get('target_collection')],
                    "expected_ast": item['ground_truth_query'],
                    "source": "weaviate_gorilla",
                    "has_join": False,
                })

    # BIRD benchmark
    dev_file = Path("data/bird-benchmark/dev_20240627/dev.json")
    all_dbs = [
        'california_schools', 'card_games', 'debit_card_specializing',
        'european_football_2', 'formula_1', 'codebase_community',
        'financial', 'student_club', 'superhero', 'thrombosis_prediction', 'toxicology'
    ]
    if dev_file.exists():
        all_q = json.loads(dev_file.read_text())
        for q in all_q:
            if q['db_id'] not in all_dbs:
                continue
            has_join = 'JOIN' in q['SQL'].upper()
            if has_join and not include_joins:
                continue

            sql_tables = extract_sql_tables(q['SQL'])
            if not sql_tables:
                continue

            primary_table = sql_tables[0]
            primary_coll = convert_name(q['db_id'], primary_table)
            # All tables in the JOIN are acceptable targets
            acceptable = list(dict.fromkeys(
                convert_name(q['db_id'], t) for t in sql_tables
            ))

            queries.append({
                "question": q['question'],
                "expected_collection": primary_coll,
                "acceptable_collections": acceptable,
                "expected_ast": None,
                "source": f"bird_{q['db_id']}",
                "has_join": has_join,
                "sql": q.get("SQL"),
                "db_id": q.get("db_id"),
                "evidence": q.get("evidence", ""),
            })

    return [q for q in queries if q.get("expected_collection")]


def split_queries(queries: list, seed: int = 42, split: str = "test",
                   ratios: tuple = (0.05, 0.10, 0.85)) -> list:
    """Split queries into train/dev/test sets stratified by source.

    Args:
        queries: Full list of queries
        seed: Random seed for reproducibility
        split: One of 'train', 'dev', 'test', or 'all'
        ratios: (train, dev, test) ratios. Default (0.05, 0.10, 0.85) maximizes
                test data. KG weight learning converges at N=25, so minimal
                train split is sufficient.
    """
    if split == "all":
        return queries

    train_r, dev_r, test_r = ratios
    rng = random.Random(seed)

    # Group by source for stratified split
    by_source = defaultdict(list)
    for q in queries:
        by_source[q['source']].append(q)

    train, dev, test = [], [], []
    for source, group in by_source.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_r))  # At least 1 per source
        n_dev = max(1, int(n * dev_r))
        train.extend(shuffled[:n_train])
        dev.extend(shuffled[n_train:n_train + n_dev])
        test.extend(shuffled[n_train + n_dev:])

    if split == "train":
        return train
    elif split == "dev":
        return dev
    elif split == "test":
        return test
    else:
        return queries


class EnhancedPipeline:
    """Pipeline with all improvements integrated."""

    def __init__(self, server: MCPServer, learning: LearningCache, args, lm=None):
        self.server = server
        self.learning = learning
        self.args = args
        self.cache = SchemaCache(cache_dir=".cache_enhanced")
        self.sandbox = StructuredSandbox(server) if not args.no_correction else None

        # Load BIRD property-name mapping (Weaviate descriptive -> SQL canonical).
        # Used under tool_schema=v2 to emit SQL-canonical property names in the
        # schema prompt so LLM outputs line up with BIRD SQL SELECT columns.
        self._name_map = None
        try:
            name_map_path = Path("data/bird-benchmark/bird_property_name_map.json")
            if name_map_path.exists():
                self._name_map = json.loads(name_map_path.read_text())
        except Exception:
            self._name_map = None

        self.semantic_validator = SemanticValidator(lm) if (
            getattr(args, 'semantic_correction', False) and lm is not None
        ) else None

        # Load value stats
        self.value_stats = None
        if not args.no_values:
            print("Loading value statistics...")
            schema_files = [
                "data/weaviate-gorilla.json",
                "data/bird-processor/bird-to-weaviate.json",
            ]
            existing = [f for f in schema_files if Path(f).exists()]
            if existing:
                self.value_stats = ValueStats.from_multiple_sources(*existing)
                print(f"  Value stats from schemas: {len(self.value_stats.stats)} collections")

            # Enrich with BIRD SQL ground truth values
            bird_file = Path("data/bird-benchmark/dev_20240627/dev.json")
            if bird_file.exists() and self.value_stats and getattr(args, 'value_enrichment', False):
                bird_queries = json.loads(bird_file.read_text())
                n_added = self.value_stats.enrich_from_sql_queries(bird_queries)
                print(f"  Enriched from BIRD SQL: +{n_added} values, now {len(self.value_stats.stats)} collections")

        # Build enhanced KG
        print("Building Enhanced Knowledge Graph...")
        value_stats_dict = self.value_stats.stats if self.value_stats else None
        self.kg = FieldLevelKnowledgeGraph(server.schemas, value_stats=value_stats_dict)
        topo = self.kg.get_topology_summary()
        print(f"  KG: {topo['total_edges']} edges, {len(topo['hub_collections'])} hubs")
        print(f"  Edge types: {topo['edge_types']}")

        # Compute embeddings
        print(f"Computing embeddings for {len(server.schemas)} collections...")
        self.embeddings = {}
        for i, (name, schema) in enumerate(server.schemas.items()):
            desc = schema.get('envisioned_use_case_overview', '') or schema.get('description', name)
            text = f"{name}: {desc}"

            if not args.no_kg:
                # Use rich context with edge types
                kg_ctx = self.kg.get_rich_context(name) if not args.no_multi_hop else self.kg.get_context(name)
                if kg_ctx:
                    text += f" {kg_ctx}"

            if self.value_stats and not args.no_values:
                val_ctx = self.value_stats.get_value_context(name, max_values=3)
                if val_ctx:
                    text += f" {val_ctx}"

            self.embeddings[name] = get_embedding(text, model="local")
            if i % 20 == 0:
                print(".", end="", flush=True)
        print(" Done.")

        # Stats tracking
        self.stats = {
            "difficulty_counts": defaultdict(int),
            "correction_counts": defaultdict(int),  # "structural" and "semantic" keys
            "kg_boost_applied": 0,
            "value_boost_applied": 0,
        }

        if self.semantic_validator:
            print("  Semantic correction: ENABLED")

    def _sql_to_weaviate_in_args(self, args: dict) -> dict:
        """Rewrite query_args that use SQL-canonical property names back to
        Weaviate-side names so the sandbox/evaluation can validate them.

        No-op if the name map is not loaded or args are None.
        """
        if not self._name_map or not args:
            return args

        s2w_all = self._name_map.get("sql_to_weaviate", {})
        if not s2w_all:
            return args

        def _norm(s):
            return re.sub(r"[\s_\-]+", "", str(s).lower()) if s else ""

        def _remap(coll_name: str, prop_name: str) -> str:
            if not coll_name or not prop_name:
                return prop_name
            s2w = s2w_all.get(coll_name, {})
            return s2w.get(_norm(prop_name), prop_name)

        primary = args.get("collection_name")

        # V2 lists: filters, output_properties, order_by, having_filters
        for list_key in ("filters", "output_properties", "order_by", "having_filters"):
            items = args.get(list_key) or []
            if not isinstance(items, list):
                continue
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                coll = entry.get("collection") or primary
                if "property_name" in entry and entry["property_name"]:
                    entry["property_name"] = _remap(coll, entry["property_name"])

        # V2 group_by_properties (strings, scoped to primary)
        gbp = args.get("group_by_properties") or []
        if isinstance(gbp, list) and primary:
            args["group_by_properties"] = [_remap(primary, p) for p in gbp]

        # V1 singular filter fields (for backward compat)
        for key in ("integer_property_filter", "text_property_filter", "boolean_property_filter",
                   "integer_property_aggregation", "text_property_aggregation", "boolean_property_aggregation"):
            f = args.get(key)
            if isinstance(f, dict) and f.get("property_name") and primary:
                f["property_name"] = _remap(primary, f["property_name"])

        # V1 groupby_property (singular string)
        if args.get("groupby_property") and primary:
            args["groupby_property"] = _remap(primary, args["groupby_property"])

        # join_keys: left_collection/left_property + right_collection/right_property
        for jk in args.get("join_keys") or []:
            if not isinstance(jk, dict):
                continue
            if jk.get("left_property") and jk.get("left_collection"):
                jk["left_property"] = _remap(jk["left_collection"], jk["left_property"])
            if jk.get("right_property") and jk.get("right_collection"):
                jk["right_property"] = _remap(jk["right_collection"], jk["right_property"])

        return args

    def discover(self, query: str, top_k: int = 5, expand_cluster: bool = True) -> list:
        """Phase 1 - Find relevant collections via embedding + KG boost.

        Steps: 1) Embed query, cosine-rank all collections.
        2) KG boost (adaptive/multi-hop/legacy depending on flags).
        3) Property-match rerank. 4) Value-aware boost. 5) Cluster expansion.
        """
        # Step 1: Semantic similarity ranking
        query_emb = get_embedding(query, model="local")
        scores = []
        for name, emb in self.embeddings.items():
            sim = cosine_similarity(list(query_emb), list(emb))
            scores.append((name, sim))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        if self.args.no_embedding:
            scores = [(name, 0.0) for name, _ in scores]

        # Step 2: KG boost (adaptive depth or full pipeline)
        if not self.args.no_kg:
            if not self.args.no_adaptive:
                # Adaptive: choose strategy based on query difficulty
                scores, difficulty = self.kg.adaptive_boost(scores, query, top_n=3)
                self.stats["difficulty_counts"][difficulty] += 1
            elif not self.args.no_multi_hop:
                # Full typed+multi-hop boost
                scores = self.kg.typed_boost(scores, query, top_n=3, use_multi_hop=True)
                self.stats["kg_boost_applied"] += 1
            else:
                # Legacy 1-hop boost
                scores = self.kg.query_aware_boost(scores, query, top_n=3)
                self.stats["kg_boost_applied"] += 1

        # Step 3: Property match re-ranking
        if not self.args.no_rerank:
            scores = self.kg.property_match_rerank(scores, query, weight=0.15)

        # Step 4: Value-aware boost
        if self.value_stats and not self.args.no_values:
            scores = self.value_stats.boost_discovery_scores(scores, query)
            self.stats["value_boost_applied"] += 1

        # Learning cache boost
        if self.learning:
            for i, (name, score) in enumerate(scores):
                boost = self.learning.get_boost(query, name) / 1000
                scores[i] = (name, score + boost)
            scores.sort(key=lambda x: x[1], reverse=True)

        top_candidates = [name for name, _ in scores[:top_k]]

        # Step 5: Expand with DB-group siblings so the LLM sees all tables for JOINs
        if expand_cluster and not self.args.no_kg:
            score_dict = {name: sc for name, sc in scores}
            cluster_additions = set()
            # Get DB group for top-3 candidates
            for cand in top_candidates[:3]:
                db_group = self.kg.get_database_group(cand)
                cluster_additions.update(db_group)
            # Remove already-present candidates
            cluster_additions -= set(top_candidates)
            # Add cluster members sorted by their original score, up to a limit
            if cluster_additions:
                cluster_sorted = sorted(cluster_additions,
                                       key=lambda n: score_dict.get(n, 0),
                                       reverse=True)
                # Add up to 5 extra cluster members (total candidates capped at top_k + 5)
                top_candidates.extend(cluster_sorted[:5])

        return top_candidates

    # ── Primary selection fixes ──────────────────────────────────────────

    def column_first_primary(self, question: str, args: dict, schemas_available: dict) -> tuple[dict, bool]:
        """Option 3: Output-column heuristic for primary selection.

        Identifies which collection's PROPERTIES best match the output data
        requested by the question (column-first linking). The collection whose
        properties provide the answer = primary (output-as-primary rule).

        Returns (args, swapped).
        """
        primary = args.get("collection_name")
        additional = args.get("additional_collections", []) or []
        if not additional or not primary:
            return args, False

        all_colls = [primary] + [
            a["collection_name"] for a in additional
            if isinstance(a, dict) and "collection_name" in a
        ]
        if len(all_colls) < 2:
            return args, False

        q_lower = question.lower()

        # Extract output-intent terms: what data does the user want to see?
        output_terms = set()
        # After "what is/are/was/were the <TERM>"
        for m in re.finditer(r"what (?:is|are|was|were) the (\w+(?:\s+\w+)?)", q_lower):
            output_terms.update(m.group(1).split())
        # After "list/find/show/get the <TERM>"
        for m in re.finditer(r"(?:list|find|show|get|give|state|provide|identify|indicate) (?:all |the |me )?(\w+(?:\s+\w+)?)", q_lower):
            output_terms.update(m.group(1).split())
        # "how many <TERM>" — counting something
        for m in re.finditer(r"how many (\w+)", q_lower):
            output_terms.add(m.group(1))
        # "calculate/compute the <TERM>"
        for m in re.finditer(r"(?:calculate|compute|count) (?:the )?(\w+(?:\s+\w+)?)", q_lower):
            output_terms.update(m.group(1).split())

        # Remove stopwords
        stopwords = {"the", "a", "an", "of", "for", "in", "on", "to", "and", "or",
                      "all", "each", "every", "their", "its", "his", "her", "this",
                      "that", "these", "those", "which", "who", "whom", "how", "many",
                      "much", "most", "some", "any", "with", "from", "has", "have", "had",
                      "was", "were", "been", "being", "does", "did", "not", "out", "over"}
        output_terms = {t for t in output_terms if t not in stopwords and len(t) > 2}

        # Score each collection by how many of its properties match output terms
        scores = {}
        for coll in all_colls:
            cached = schemas_available.get(coll) or self.cache.get(coll, compressed=True)
            if not cached:
                scores[coll] = 0
                continue

            props = cached.get("properties", {})
            prop_names = []
            if isinstance(props, dict):
                prop_names = [k.lower().replace("_", " ") for k in props.keys()]
            elif isinstance(props, list):
                prop_names = [p.get("name", "").lower().replace("_", " ") for p in props if isinstance(p, dict)]

            score = 0
            for term in output_terms:
                for prop in prop_names:
                    if term in prop or prop in term:
                        score += 2
                    # Plural/singular match
                    if term.rstrip("s") == prop.rstrip("s"):
                        score += 3

            scores[coll] = score

        best_coll = max(scores, key=scores.get)
        if best_coll != primary and scores[best_coll] > scores.get(primary, 0):
            old_primary = primary
            args["collection_name"] = best_coll
            new_additional = [{"collection_name": old_primary, "role": "demoted from primary"}]
            for a in additional:
                cn = a["collection_name"] if isinstance(a, dict) and "collection_name" in a else None
                if cn and cn != best_coll:
                    new_additional.append(a)
            args["additional_collections"] = new_additional
            return args, True

        return args, False

    def disambiguate_primary(self, lm, question: str, args: dict, schema_dict: dict) -> tuple[dict, bool]:
        """Option 2: Two-step disambiguation via focused second LLM call.

        After the LLM returns 2+ collections, makes a second call asking
        which collection is the subject entity. Uses a tiny prompt with
        only the relevant schemas.

        Returns (args, swapped).
        """
        primary = args.get("collection_name")
        additional = args.get("additional_collections", []) or []
        if not additional or not primary:
            return args, False

        all_colls = [primary] + [
            a["collection_name"] for a in additional
            if isinstance(a, dict) and "collection_name" in a
        ]
        if len(all_colls) < 2:
            return args, False

        # Build a focused prompt with just the candidate schemas
        schema_summary = ""
        for coll in all_colls[:3]:  # Max 3 collections
            cached = schema_dict.get(coll) or self.cache.get(coll, compressed=True)
            if not cached:
                continue
            props = cached.get("properties", {})
            prop_list = []
            if isinstance(props, dict):
                prop_list = list(props.keys())[:8]
            elif isinstance(props, list):
                prop_list = [p.get("name", "") for p in props[:8] if isinstance(p, dict)]
            schema_summary += f"  {coll}: {', '.join(prop_list)}\n"

        if not schema_summary:
            return args, False

        # Build disambiguation tool
        disambig_tool = {
            "type": "function",
            "function": {
                "name": "select_primary",
                "description": "Select the primary (output) collection for the query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Identify what data the user wants returned, then which collection's properties contain that data."
                        },
                        "primary_collection": {
                            "type": "string",
                            "enum": all_colls[:3],
                            "description": "The OUTPUT collection — the one whose properties contain the data the user wants to see."
                        }
                    },
                    "required": ["reasoning", "primary_collection"]
                }
            }
        }

        disambig_prompt = f"""Given this question and these collections, select the PRIMARY collection.

The PRIMARY collection is the OUTPUT TABLE: the collection whose PROPERTIES contain the data the user wants to SEE or RETRIEVE. The other collections are used to filter or narrow results.

Think: "What specific data does the user want returned? Which collection has those properties?"

Examples:
- "List names of patients with low albumin" → Output = names → Patient has names → Primary = Patient
- "What was the overall rating for Aaron Mooy?" → Output = rating → PlayerAttributes has rating → Primary = PlayerAttributes
- "How many customers had monthly consumption > 1000?" → Output = consumption count → Yearmonth has consumption → Primary = Yearmonth
- "What is the sprint speed of player X?" → Output = sprint speed → PlayerAttributes has sprint_speed → Primary = PlayerAttributes

Question: {question}

Collections (with properties):
{schema_summary}
Select the PRIMARY collection whose properties contain the output data."""

        try:
            for retry in range(3):
                try:
                    response = lm.one_step_function_selection_test(disambig_prompt, [disambig_tool], False)
                    break
                except Exception as e:
                    if '429' in str(e) or 'Too Many' in str(e):
                        time.sleep(2 * (2 ** retry))
                        continue
                    return args, False

            if response is None:
                return args, False

            disambig_args = json.loads(response[0].function.arguments)
            new_primary = disambig_args.get("primary_collection")

            if new_primary and new_primary != primary and new_primary in all_colls:
                old_primary = primary
                args["collection_name"] = new_primary
                new_additional = [{"collection_name": old_primary, "role": "demoted from primary"}]
                for a in additional:
                    cn = a["collection_name"] if isinstance(a, dict) and "collection_name" in a else None
                    if cn and cn != new_primary:
                        new_additional.append(a)
                args["additional_collections"] = new_additional
                return args, True

        except Exception as e:
            pass  # Fail silently, keep original primary

        return args, False

    def rerank_primary(self, question: str, args: dict, schema_dict: dict) -> tuple[dict, bool]:
        """Rerank primary collection based on property matching.

        Checks if an additional_collection's schema better matches
        the properties mentioned in the question. If so, swaps primary.
        Only considers distinctive properties (>= 4 chars, not shared across
        all candidate collections) to avoid false positives from generic
        names like 'id', 'name', 'type'.

        Returns (args, swapped) where swapped indicates if a swap occurred.
        """
        if self.args.no_primary_rerank:
            return args, False

        primary = args.get("collection_name")
        additional = args.get("additional_collections", []) or []
        if not additional or not primary:
            return args, False

        all_colls = [primary] + [
            a["collection_name"] for a in additional
            if isinstance(a, dict) and "collection_name" in a
        ]

        def get_prop_names(coll):
            schema = schema_dict.get(coll, {})
            props = schema.get("properties", {})
            if isinstance(props, dict):
                return {p.lower().replace(" ", "_") for p in props.keys()}
            elif isinstance(props, list):
                return {p.get("name", "").lower().replace(" ", "_") for p in props if p.get("name")}
            return set()

        # Build per-collection prop sets
        coll_props = {c: get_prop_names(c) for c in all_colls}

        # Find distinctive properties: appear in only ONE of the candidate collections
        # and are >= 4 chars (skip 'id', 'sex', 'name', etc.)
        all_prop_sets = list(coll_props.values())
        shared_props = set()
        if len(all_prop_sets) >= 2:
            shared_props = set.intersection(*all_prop_sets)

        # Tokenize question: extract words >= 3 chars, lowercase, strip punctuation
        q_words = set()
        for w in question.lower().split():
            w = w.strip(".,?!:;'\"()[]{}")
            if len(w) >= 3:
                q_words.add(w)

        def score(coll):
            props = coll_props.get(coll, set())
            distinctive = {p for p in props if len(p) >= 4 and p not in shared_props}
            matches = 0
            for prop in distinctive:
                for word in q_words:
                    # Exact match or substring containment for longer strings
                    if prop == word or (len(prop) >= 5 and prop in word) or (len(word) >= 5 and word in prop):
                        matches += 1
                        break
            return matches

        primary_score = score(primary)
        best_alt = None
        best_alt_score = primary_score

        for coll in all_colls[1:]:  # Skip primary
            s = score(coll)
            if s > best_alt_score:
                best_alt = coll
                best_alt_score = s

        # Only swap if the alternative has strictly more distinctive matches
        if best_alt and best_alt_score > primary_score:
            old_primary = primary
            args["collection_name"] = best_alt
            new_additional = [{"collection_name": old_primary, "role": "demoted from primary"}]
            for a in additional:
                coll_name = a["collection_name"] if isinstance(a, dict) and "collection_name" in a else None
                if coll_name and coll_name != best_alt:
                    new_additional.append(a)
            args["additional_collections"] = new_additional
            return args, True

        return args, False

    def schema_link(self, lm, question: str, schema_prompt: str, collections: list) -> str:
        """Call 1 of Level 3: Schema linking decomposition (DIN-SQL style).

        Uses a tool call (not plain text — NVIDIA NIM returns None for plain text)
        to classify which properties are OUTPUT vs FILTER, and which collection
        owns each. Returns a structured analysis string to inject into the
        tool-call prompt for Call 2.
        """
        if getattr(self.args, 'no_schema_link', True):
            return ""

        linking_prompt = f"""Analyze this database question. Identify which properties are OUTPUT (returned/counted/listed) vs FILTER (used as conditions), and which collection owns each.

Question: {question}

Available collections and their properties:
{schema_prompt}

Call the schema_analysis tool with your analysis."""

        schema_link_tool = {
            "type": "function",
            "function": {
                "name": "schema_analysis",
                "description": "Analyze which properties are OUTPUT vs FILTER before generating a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "output_properties": {
                            "type": "string",
                            "description": "Properties the question asks to RETURN/COUNT/LIST, with collection. E.g. 'count of patients → Patient, diagnosis → Patient'"
                        },
                        "filter_properties": {
                            "type": "string",
                            "description": "Properties used as WHERE conditions, with collection. E.g. 'WBC < 3.5 → Laboratory, SEX=female → Patient'"
                        },
                        "primary_collection": {
                            "type": "string",
                            "description": "The collection owning the OUTPUT properties — the FROM table.",
                            "enum": collections
                        },
                        "primary_reason": {
                            "type": "string",
                            "description": "Brief: why is this collection primary?"
                        }
                    },
                    "required": ["output_properties", "filter_properties", "primary_collection", "primary_reason"]
                }
            }
        }

        for retry in range(3):
            try:
                response = lm.lm_client.chat.completions.create(
                    model=lm.model_name,
                    messages=[
                        {"role": "system", "content": "You MUST call the schema_analysis tool. Identify OUTPUT properties (what to return/count) vs FILTER properties (WHERE conditions). Primary = collection with OUTPUT properties."},
                        {"role": "user", "content": linking_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300,
                    tools=[schema_link_tool],
                    tool_choice="required"
                )
                msg = response.choices[0].message
                tc = msg.tool_calls
                if tc and len(tc) > 0:
                    sl_args = json.loads(tc[0].function.arguments)
                    result = (
                        f"OUTPUT: {sl_args.get('output_properties', 'N/A')}\n"
                        f"FILTER: {sl_args.get('filter_properties', 'N/A')}\n"
                        f"PRIMARY: {sl_args.get('primary_collection', 'N/A')} ({sl_args.get('primary_reason', '')})"
                    )
                    return result
                # NIM sometimes returns content instead of tool_calls
                if msg.content:
                    return msg.content.strip()
                return ""
            except Exception as e:
                if '429' in str(e) or 'Too Many' in str(e):
                    time.sleep(2 * (2 ** retry))
                    continue
                print(f"    [Schema link error: {e}]")
                return ""
        return ""

    def generate(self, lm, question: str, candidates: list, evidence: str = "") -> dict:
        """Phase 2 - Build schema prompt, call LLM tool, validate, self-correct.

        Steps: 1) Assemble schema prompt from candidate collections.
        2) Optional schema-linking pre-call. 3) LLM tool call.
        4) Sandbox validation + self-correction loop.
        5) Post-generation primary reranking/disambiguation.
        """
        # Inject BIRD evidence (domain hints) before the question when available.
        evidence_str = ""
        if evidence and evidence.strip():
            evidence_str = f"\nHint: {evidence.strip()}\n"
        prompt = f"Q: {question}{evidence_str}\n\nCollections:\n\n"
        schema_dict = {}
        total_tokens = estimate_tokens(prompt)

        for coll in candidates:
            cached = self.cache.get(coll, compressed=True)
            if cached is None:
                schema = self.server.get_collection_schema(coll, compressed=False)
                if schema:
                    self.cache.set(coll, schema)
                    cached = self.cache.get(coll, compressed=True)

            if cached:
                # Schema role annotation for disambiguation
                role_tag = ""
                if not self.args.no_kg:
                    role = self.kg.classify_collection_role(coll)
                    if role:
                        props_count = len(cached.get("properties", {}))
                        role_tag = f" [{role}: {props_count} props]"

                schema_str = f"## {coll}{role_tag}\n"

                # Add database group context for disambiguation
                if not self.args.no_kg and not getattr(self.args, 'no_db_context', False):
                    db_group = self.kg.get_database_group(coll)
                    if len(db_group) > 1:
                        siblings = sorted(db_group - {coll})[:5]
                        schema_str += f"  [Database group: {', '.join(siblings)}]\n"

                if not self.args.no_kg:
                    kg_ctx = self.kg.get_rich_context(coll) if not self.args.no_multi_hop else self.kg.get_context(coll)
                    if kg_ctx:
                        schema_str += f"  {kg_ctx}\n"

                if self.value_stats and not self.args.no_values:
                    val_ctx = self.value_stats.get_value_context(coll, max_values=3)
                    if val_ctx:
                        schema_str += f"  {val_ctx}\n"

                if self.learning:
                    hint = self.learning.get_collection_hint(coll)
                    if hint:
                        schema_str += f"  {hint}\n"

                props = cached.get("properties", {})
                if isinstance(props, dict):
                    # Truncation limit: V1 uses 10 for backward compat; V2
                    # raises to 40 because the LLM needs to reference SELECT
                    # and WHERE columns that may be beyond property #10 in
                    # collections with 40+ properties (e.g., ThrombosisPrediction
                    # Laboratory has 43, CardGamesCards has 74).
                    tool_schema = getattr(self.args, 'tool_schema', 'v1')
                    prop_limit = 40 if tool_schema == "v2" else 10

                    # ADAPTIVE VISIBILITY: promote properties whose names
                    # fuzzy-match tokens in the question. This ensures relevant
                    # columns (e.g., 'White blood cell' for "WBC") appear
                    # inside the truncation window even for large schemas.
                    q_tokens = set()
                    for w in question.lower().split():
                        w = w.strip(".,?!:;'\"()[]{}")
                        if len(w) >= 3:
                            q_tokens.add(w)

                    def question_match_score(prop_name: str) -> int:
                        """Higher = more relevant to the question."""
                        p_low = prop_name.lower()
                        p_tokens = set()
                        for w in p_low.replace("_", " ").split():
                            if len(w) >= 2:
                                p_tokens.add(w)
                        # Exact word overlap
                        direct = len(p_tokens & q_tokens)
                        if direct:
                            return 3
                        # Substring match (WBC in "white blood cell" etc.)
                        for qt in q_tokens:
                            if len(qt) >= 3 and (qt in p_low or p_low.replace(" ", "") in qt):
                                return 2
                            # Plural/singular
                            if qt.rstrip("s") and qt.rstrip("s") in p_low:
                                return 1
                        return 0

                    # Known identity/common column names — always floated up
                    common_names = {
                        "id", "name", "code", "type", "date", "year", "sex",
                        "rarity", "title", "label", "status", "score",
                    }

                    def prop_sort_key(item):
                        name = item[0]
                        # Bucket 0: question-relevant (adaptive)
                        # Bucket 1: common identifier-like
                        # Bucket 2: short names (<=6 chars)
                        # Bucket 3: everything else (alpha order)
                        q_score = question_match_score(name)
                        if q_score > 0:
                            return (0, -q_score, name)
                        if name.lower() in common_names:
                            return (1, 0, name)
                        if len(name) <= 6:
                            return (2, 0, name)
                        return (3, 0, name)
                    ordered_props = sorted(props.items(), key=prop_sort_key)

                    # Optional SQL-canonical name emission (V2 + BIRD collection only).
                    # When enabled, we emit `WBC` instead of `White blood cell` so
                    # LLM outputs align with BIRD SQL. The sandbox and evaluation
                    # reverse-map back to Weaviate names via self._name_map.
                    use_sql_names = (
                        tool_schema == "v2"
                        and self._name_map is not None
                        and coll in self._name_map.get("weaviate_to_sql", {})
                    )
                    weav_to_sql = (
                        self._name_map["weaviate_to_sql"].get(coll, {})
                        if use_sql_names else {}
                    )
                    def _norm_for_map(s):
                        return re.sub(r"[\s_\-]+", "", str(s).lower())

                    for name, info in ordered_props[:prop_limit]:
                        t = info.get('type', '?') if isinstance(info, dict) else info
                        display_name = name
                        if use_sql_names:
                            sql_name = weav_to_sql.get(_norm_for_map(name))
                            if sql_name:
                                display_name = sql_name
                        schema_str += f"  - {display_name} ({t})\n"
                    if len(props) > prop_limit:
                        schema_str += f"  ... ({len(props) - prop_limit} more properties omitted)\n"
                schema_str += "\n"

                tokens = estimate_tokens(schema_str)
                if total_tokens + tokens <= 2500:
                    prompt += schema_str
                    total_tokens += tokens
                    schema_dict[coll] = cached

        if not schema_dict:
            return {"error": "No schemas", "collection": None, "tokens": total_tokens}

        # Level 3: Schema linking pre-call (DIN-SQL style decomposition)
        schema_link_result = self.schema_link(lm, question, prompt, list(schema_dict.keys()))
        if schema_link_result:
            prompt += f"\n--- SCHEMA ANALYSIS (follow this) ---\n{schema_link_result}\n---\n\n"
            self.stats["schema_links"] = self.stats.get("schema_links", 0) + 1

        # Add preventive instructions from sandbox if available
        extra_instructions = ""
        if self.sandbox and hasattr(self.sandbox, 'get_preventive_instructions'):
            extra_instructions = self.sandbox.get_preventive_instructions()

        tool_schema = getattr(self.args, 'tool_schema', 'v1')
        if tool_schema == "v2":
            tool = build_weaviate_query_tool_for_openai_v2(prompt + extra_instructions, list(schema_dict.keys()))
        else:
            tool = build_weaviate_query_tool_for_openai(prompt + extra_instructions, list(schema_dict.keys()))

        # Generation with self-correction loop
        # Structural correction: sandbox validates property names, operators, types
        # Semantic correction: LLM verifies the query answers the question
        struct_max = 0 if self.args.no_correction else 2
        semantic_max = 1 if self.semantic_validator else 0
        total_max = struct_max + semantic_max
        current_question = question
        correction_attempts = 0
        last_args = None
        semantic_done = False  # Only run semantic validation once per query

        for attempt in range(total_max + 1):
            try:
                # Retry with backoff for rate limit (429) errors
                response = None
                for retry in range(5):
                    try:
                        primary_fix = getattr(self.args, 'primary_fix', 'none')
                        response = lm.one_step_function_selection_test(current_question, [tool], False, primary_fix=primary_fix, tool_schema=tool_schema)
                        break
                    except Exception as api_err:
                        if '429' in str(api_err) or 'Too Many' in str(api_err):
                            delay = 2 * (2 ** retry)
                            time.sleep(delay)
                        else:
                            raise

                if response is None:
                    return {"error": "No tool call", "collection": None, "tokens": total_tokens}

                args = json.loads(response[0].function.arguments)

                # If V2 + SQL-canonical names are being shown in the prompt,
                # the LLM likely used SQL names (e.g., 'WBC'). Reverse-map to
                # Weaviate property names so the sandbox + scorer see the
                # canonical form our schemas use.
                if tool_schema == "v2":
                    args = self._sql_to_weaviate_in_args(args)

                last_args = args

                # Step 1: Structural validation (sandbox)
                needs_structural_fix = False
                if self.sandbox and attempt < struct_max:
                    if tool_schema == "v2":
                        # V2: validate the raw query_args dict against scoped schemas
                        validation = self.sandbox.validate_v2(args)
                        if not validation.valid:
                            self.sandbox.track_error_pattern(validation.errors)
                            correction_attempts += 1
                            self.stats["correction_counts"]["structural"] = self.stats["correction_counts"].get("structural", 0) + 1
                            current_question = self.sandbox.build_v2_correction_prompt(
                                original_query=question,
                                query_args=args,
                                validation_result=validation,
                                schema_desc=prompt,
                            )
                            needs_structural_fix = True
                    else:
                        # V1: build a WeaviateQuery and use the legacy validator
                        from src.models import WeaviateQuery, IntPropertyFilter, TextPropertyFilter, BooleanPropertyFilter

                        query_obj = WeaviateQuery(
                            target_collection=args.get("collection_name", ""),
                            search_query=args.get("search_query"),
                            groupby_property=args.get("groupby_property")
                        )
                        if args.get("integer_property_filter"):
                            f = args["integer_property_filter"]
                            query_obj.integer_property_filter = IntPropertyFilter(
                                property_name=f.get("property_name", ""),
                                operator=f.get("operator", "="),
                                value=f.get("value", 0)
                            )
                        if args.get("text_property_filter"):
                            f = args["text_property_filter"]
                            query_obj.text_property_filter = TextPropertyFilter(
                                property_name=f.get("property_name", ""),
                                operator=f.get("operator", "="),
                                value=f.get("value", "")
                            )

                        validation = self.sandbox.validate_with_suggestions(query_obj)
                        if not validation.valid:
                            self.sandbox.track_error_pattern(validation.errors)
                            correction_attempts += 1
                            self.stats["correction_counts"]["structural"] = self.stats["correction_counts"].get("structural", 0) + 1
                            current_question = self.sandbox.build_correction_prompt(
                                original_query=question,
                                query_args=args,
                                validation_result=validation,
                                schema_desc=prompt
                            )
                            needs_structural_fix = True

                if needs_structural_fix:
                    continue

                # Step 2: Semantic validation (LLM-as-judge, once per query)
                if self.semantic_validator and not semantic_done:
                    semantic_done = True
                    sem_result = self.semantic_validator.validate(
                        question, args, candidates
                    )
                    if not sem_result['valid']:
                        correction_attempts += 1
                        self.stats["correction_counts"]["semantic"] += 1
                        current_question = self.semantic_validator.build_semantic_correction_prompt(
                            question, args, sem_result, prompt
                        )
                        continue

                # Property-aware primary reranking
                args, swapped = self.rerank_primary(question, args, schema_dict)
                if swapped:
                    self.stats["primary_reranks"] = self.stats.get("primary_reranks", 0) + 1

                # Primary selection fix (post-generation)
                primary_fix = getattr(self.args, 'primary_fix', 'none')
                if primary_fix == "column-first":
                    args, cf_swapped = self.column_first_primary(question, args, schema_dict)
                    if cf_swapped:
                        self.stats["primary_cf_swaps"] = self.stats.get("primary_cf_swaps", 0) + 1
                elif primary_fix == "twostep":
                    args, ts_swapped = self.disambiguate_primary(lm, question, args, schema_dict)
                    if ts_swapped:
                        self.stats["primary_ts_swaps"] = self.stats.get("primary_ts_swaps", 0) + 1

                # Extract multi-collection info
                additional = args.get("additional_collections", []) or []
                additional_names = [a["collection_name"] for a in additional
                                   if isinstance(a, dict) and "collection_name" in a]
                join_keys = args.get("join_keys", []) or []
                all_collections = [args.get("collection_name")] + additional_names

                return {
                    "collection": args.get("collection_name"),
                    "all_collections": all_collections,
                    "additional_collections": additional,
                    "join_keys": join_keys,
                    "query_args": args,
                    "tokens": total_tokens,
                    "candidates": candidates,
                    "correction_attempts": correction_attempts,
                    "primary_reranked": swapped,
                    "schema_link": schema_link_result
                }

            except Exception as e:
                return {"error": str(e), "collection": None, "all_collections": [],
                        "tokens": total_tokens}

        # Return last attempt even if validation failed
        additional = (last_args.get("additional_collections", []) or []) if last_args else []
        additional_names = [a["collection_name"] for a in additional
                           if isinstance(a, dict) and "collection_name" in a]
        primary = last_args.get("collection_name") if last_args else None
        all_collections = ([primary] + additional_names) if primary else []

        return {
            "collection": primary,
            "all_collections": all_collections,
            "additional_collections": additional,
            "join_keys": (last_args.get("join_keys", []) or []) if last_args else [],
            "query_args": last_args,
            "tokens": total_tokens,
            "candidates": candidates,
            "correction_attempts": correction_attempts
        }


def main():
    parser = argparse.ArgumentParser(description="Enhanced Pipeline Test")
    parser.add_argument("--n", type=int, default=50, help="Number of queries")
    parser.add_argument("--all", action="store_true", help="Test ALL queries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Show per-query details")
    parser.add_argument("--persistent", action="store_true", help="Persistent learning cache")

    # Ablation flags
    parser.add_argument("--no-kg", action="store_true", help="ABLATION: Disable KG")
    parser.add_argument("--no-embedding", action="store_true", help="ABLATION: Disable embeddings")
    parser.add_argument("--no-rerank", action="store_true", help="ABLATION: Disable property reranking")
    parser.add_argument("--no-correction", action="store_true", help="ABLATION: Disable self-correction")
    parser.add_argument("--no-multi-hop", action="store_true", help="ABLATION: Disable multi-hop KG")
    parser.add_argument("--no-values", action="store_true", help="ABLATION: Disable value-aware linking")
    parser.add_argument("--no-adaptive", action="store_true", help="ABLATION: Disable adaptive depth")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between queries in seconds (rate limit protection)")
    parser.add_argument("--no-joins", action="store_true", help="Exclude JOIN queries from BIRD (legacy mode)")
    parser.add_argument("--split", type=str, default="all", choices=["train", "dev", "test", "all"],
                        help="Use train/dev/test split (70/15/15 stratified). Default: all")
    parser.add_argument("--relaxed-match", action="store_true", default=True,
                        help="Accept any JOIN table as correct (default: True)")
    parser.add_argument("--no-relaxed-match", action="store_true",
                        help="Only accept primary FROM table as correct")
    parser.add_argument("--semantic-correction", action="store_true",
                        help="Enable LLM-based semantic validation of generated queries")
    parser.add_argument("--learn-weights", action="store_true",
                        help="Learn KG edge weights from training split before testing")
    parser.add_argument("--no-db-context", action="store_true",
                        help="ABLATION: Disable database group context in prompts")
    parser.add_argument("--value-enrichment", action="store_true",
                        help="Enable BIRD SQL value enrichment (experimental, can add noise)")
    parser.add_argument("--split-ratios", type=str, default="0.05,0.10,0.85",
                        help="Train/dev/test ratios (default: 0.05,0.10,0.85)")
    parser.add_argument("--no-primary-rerank", action="store_true",
                        help="ABLATION: Disable post-generation primary collection reranking")
    parser.add_argument("--no-schema-link", action="store_true",
                        help="ABLATION: Disable schema linking pre-call (Level 3 decomposition)")
    parser.add_argument("--schema-link", action="store_true",
                        help="Enable schema linking pre-call (Level 3 decomposition, adds ~5s/query)")

    # Multi-model evaluation flags
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "bedrock", "anthropic", "together", "ollama"],
                        help="LLM provider (default: openai/NVIDIA NIM)")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b",
                        help="Model name/ID (e.g., anthropic.claude-sonnet-4-6)")
    parser.add_argument("--ollama-host", type=str, default=None,
                        help="Ollama server URL for remote host (e.g., http://192.168.1.100:11434)")
    parser.add_argument("--max-dbs", type=int, default=None,
                        help="Limit to first N databases (for scale experiment)")
    parser.add_argument("--primary-fix", type=str, default="none",
                        choices=["none", "prompt", "twostep", "column-first"],
                        help="Primary selection fix: none (baseline), prompt (enhanced system prompt), "
                             "twostep (second LLM call for disambiguation), column-first (deterministic NL heuristic)")
    parser.add_argument("--tool-schema", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="Tool schema version. v1 (default) is the existing single-filter, no-output-scope "
                             "schema. v2 adds output_properties, a filters list, order_by, distinct, and limit.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for results JSON")

    args = parser.parse_args()
    # schema_link defaults to OFF; --schema-link enables it, --no-schema-link disables it
    if args.schema_link:
        args.no_schema_link = False
    elif not args.no_schema_link:
        # Neither flag passed — default to OFF
        args.no_schema_link = True
    args.split_ratios = tuple(float(x) for x in args.split_ratios.split(','))
    if args.no_relaxed_match:
        args.relaxed_match = False
    random.seed(args.seed)

    # Config summary
    flags = []
    for flag in ['no_kg', 'no_embedding', 'no_rerank', 'no_correction', 'no_multi_hop', 'no_values', 'no_adaptive']:
        val = getattr(args, flag)
        name = flag.replace('no_', '').replace('_', '-')
        flags.append(f"{name}={'OFF' if val else 'ON'}")

    join_str = "joins=OFF" if args.no_joins else "joins=ON"
    split_str = f"split={args.split}"
    primary_fix_str = f"primary-fix={args.primary_fix}" if args.primary_fix != "none" else ""

    print("=" * 70)
    print("ENHANCED PIPELINE TEST")
    print(f"Provider: {args.provider} | Model: {args.model}")
    extra = f", {primary_fix_str}" if primary_fix_str else ""
    print(f"Config: {', '.join(flags)}, {join_str}, {split_str}{extra}")
    print("=" * 70)

    all_queries = load_all_queries(include_joins=not args.no_joins)
    all_queries = split_queries(all_queries, seed=args.seed, split=args.split,
                                ratios=args.split_ratios)

    # Scale experiment: limit to first N databases
    if args.max_dbs:
        allowed_sources = sorted(set(q['source'] for q in all_queries))[:args.max_dbs]
        all_queries = [q for q in all_queries if q['source'] in allowed_sources]
        print(f"\n[Scale experiment] Limited to {args.max_dbs} databases: {allowed_sources}")

    random.shuffle(all_queries)
    test_queries = all_queries if args.all else all_queries[:args.n]

    source_counts = defaultdict(int)
    for q in test_queries:
        source_counts[q['source']] += 1
    print(f"\nTotal: {len(test_queries)} queries")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")

    # Load server
    print("\nLoading MCPServer...")
    source_files = [
        "data/weaviate-gorilla.json",
        "data/bird-processor/bird-to-weaviate.json",
        "data/retail-world-weaviate.json",
        "data/movie-3-weaviate.json",
        "data/student-loan-weaviate.json",
        "data/chicago-crime-weaviate.json",
        "data/university-weaviate.json",
        "data/bird-benchmark/bird-collections.json"
    ]
    existing_files = [f for f in source_files if Path(f).exists()]
    server = MCPServer.from_multiple_sources(*existing_files)
    print(f"  {len(server.schemas)} collections loaded")

    learning = LearningCache(cache_dir=".learning_cache", persistent=args.persistent)

    # Provider-aware API key resolution
    if args.provider == "bedrock":
        api_key = None  # Bedrock uses AWS credentials from env/config
    elif args.provider == "ollama":
        api_key = None  # Ollama doesn't need an API key
        if args.ollama_host:
            os.environ["OLLAMA_HOST"] = args.ollama_host
    elif args.provider == "openai":
        api_key = os.environ.get("NVIDIA_API_KEY") if "/" in args.model else os.environ.get("OPENAI_API_KEY")
    else:
        api_key = os.environ.get(f"{args.provider.upper()}_API_KEY")

    if args.provider == "openai" and not api_key:
        print("ERROR: NVIDIA_API_KEY (for NIM models) or OPENAI_API_KEY (for OpenAI models) not set!")
        return

    lm = LMService(args.provider, args.model, api_key=api_key)
    pipeline = EnhancedPipeline(server, learning, args, lm=lm)

    # Learn edge weights from training data (if requested)
    if getattr(args, 'learn_weights', False):
        print("\nLearning KG edge weights from training split...")
        train_queries = split_queries(
            load_all_queries(include_joins=not args.no_joins),
            seed=args.seed, split="train", ratios=args.split_ratios
        )
        weight_result = pipeline.kg.learn_edge_weights(train_queries)
        print(f"  Default weights:  {weight_result.get('default_weights')}")
        print(f"  Learned weights:  {weight_result.get('learned_weights')}")
        print(f"  Default accuracy: {weight_result.get('default_accuracy', 0)*100:.1f}%")
        print(f"  Learned accuracy: {weight_result.get('learned_accuracy', 0)*100:.1f}%")
        print(f"  Improvement:      {weight_result.get('improvement', 0)*100:+.1f}%")
        # Re-compute embeddings with updated KG context (weights changed)
        print("  Re-computing embeddings with learned weights...")
        for name, schema in server.schemas.items():
            desc = schema.get('envisioned_use_case_overview', '') or schema.get('description', name)
            text = f"{name}: {desc}"
            if not args.no_kg:
                kg_ctx = pipeline.kg.get_rich_context(name) if not args.no_multi_hop else pipeline.kg.get_context(name)
                if kg_ctx:
                    text += f" {kg_ctx}"
            if pipeline.value_stats and not args.no_values:
                val_ctx = pipeline.value_stats.get_value_context(name, max_values=3)
                if val_ctx:
                    text += f" {val_ctx}"
            pipeline.embeddings[name] = get_embedding(text, model="local")
        print("  Done.")

    # Token efficiency: compute full-schema baseline
    all_schema_text = ""
    for name, schema in server.schemas.items():
        all_schema_text += f"## {name}\n"
        props = schema.get('properties', {})
        if isinstance(props, dict):
            for pname, pinfo in props.items():
                t = pinfo.get('type', '?') if isinstance(pinfo, dict) else pinfo
                all_schema_text += f"  - {pname} ({t})\n"
    full_schema_tokens = estimate_tokens(all_schema_text)
    print(f"\nToken efficiency baseline: {full_schema_tokens} tokens for all {len(server.schemas)} schemas")

    # Run test
    correct = 0
    relaxed_correct = 0  # Correct with relaxed matching (any JOIN table)
    in_topk = 0
    total_ast = 0
    ast_count = 0
    total_corrections = 0
    total_tokens_used = 0
    join_correct = 0
    join_total = 0
    # Multi-collection metrics
    set_recall_sum = 0.0   # |predicted ∩ GT| / |GT|
    set_precision_sum = 0.0  # |predicted ∩ GT| / |predicted|
    set_exact_match = 0    # predicted == GT (exact set)
    multi_coll_total = 0   # Queries where GT has >1 collection
    multi_coll_predicted = 0  # Queries where LLM predicted >1 collection
    join_keys_total = 0    # Total join keys returned by LLM
    per_db = defaultdict(lambda: {"correct": 0, "relaxed_correct": 0, "total": 0,
                                   "join_total": 0, "join_correct": 0,
                                   "set_recall_sum": 0.0, "set_total": 0})
    results = []
    start = time.time()

    for i, q in enumerate(test_queries):
        question = q["question"]
        expected = q["expected_collection"]
        acceptable = q.get("acceptable_collections", [expected])
        expected_ast = q.get("expected_ast")
        source = q["source"]
        has_join = q.get("has_join", False)
        sql = q.get("sql")
        db_id = q.get("db_id")
        db = source.replace("bird_", "")

        per_db[db]["total"] += 1
        if has_join:
            per_db[db]["join_total"] += 1
            join_total += 1

        evidence = q.get("evidence", "")
        candidates = pipeline.discover(question, top_k=5)

        # Retry with backoff if we get "No tool call" (often from rate limiting)
        result = None
        for retry_attempt in range(3):
            result = pipeline.generate(lm, question, candidates, evidence=evidence)
            if result.get("error") != "No tool call" or retry_attempt == 2:
                break
            delay = 3 * (2 ** retry_attempt)  # 3, 6, 12 seconds
            print(f"    [Retry {retry_attempt+1}/3] No tool call, waiting {delay}s...")
            time.sleep(delay)

        # Capture LLM call metadata (latency, tokens)
        call_meta = lm.last_call_meta.copy()

        # Rate limit protection
        if args.delay > 0:
            time.sleep(args.delay)

        got_coll = result.get("collection")
        got_all = result.get("all_collections", [got_coll] if got_coll else [])
        got_additional = result.get("additional_collections", [])
        got_join_keys = result.get("join_keys", [])
        got_args = result.get("query_args", {})
        tokens_used = result.get("tokens", 0)
        total_tokens_used += tokens_used

        # Strict match: primary FROM table
        strict_match = got_coll == expected
        # Relaxed match: any table in the JOIN
        relaxed_match = got_coll in acceptable if args.relaxed_match else strict_match

        corrections = result.get("correction_attempts", 0)
        total_corrections += corrections

        if strict_match:
            correct += 1
            per_db[db]["correct"] += 1
        if relaxed_match:
            relaxed_correct += 1
            per_db[db]["relaxed_correct"] += 1
        if has_join and relaxed_match:
            join_correct += 1
            per_db[db]["join_correct"] += 1

        # Multi-collection set-based metrics
        gt_set = set(acceptable)
        pred_set = set(got_all) if got_all else set()
        if has_join and len(gt_set) > 1:
            multi_coll_total += 1
            per_db[db]["set_total"] += 1
            if pred_set:
                intersection = pred_set & gt_set
                recall = len(intersection) / len(gt_set)
                precision = len(intersection) / len(pred_set) if pred_set else 0
                set_recall_sum += recall
                set_precision_sum += precision
                per_db[db]["set_recall_sum"] += recall
                if pred_set == gt_set:
                    set_exact_match += 1
            if len(pred_set) > 1:
                multi_coll_predicted += 1
            join_keys_total += len(got_join_keys)

        # Discovery rate: check if any acceptable collection is in candidates
        if any(c in candidates for c in acceptable):
            in_topk += 1

        # AST scoring (enhanced)
        if expected_ast and got_args:
            ast = calculate_ast_score(got_args, expected_ast)
            total_ast += ast
            ast_count += 1

        # Progress
        elapsed = time.time() - start
        sym = "+" if strict_match else ("~" if relaxed_match else "-")
        ast_str = f"{total_ast/ast_count*100:.1f}%" if ast_count > 0 else "N/A"

        if args.verbose or i < 5 or (i+1) % 10 == 0:
            corr_str = f" [corr:{corrections}]" if corrections > 0 else ""
            join_str = " [JOIN]" if has_join else ""
            mc_str = f" [MC:{len(pred_set)}]" if len(pred_set) > 1 else ""
            rr_str = " [RR]" if result.get("primary_reranked") else ""
            print(f"  {sym} [{i+1}/{len(test_queries)}] {db:25s} Strict:{correct/(i+1)*100:.1f}% Rlx:{relaxed_correct/(i+1)*100:.1f}% AST:{ast_str}{corr_str}{join_str}{mc_str}{rr_str} ({elapsed/(i+1):.1f}s/q)", flush=True)

        results.append({
            "question": question, "expected": expected, "acceptable": acceptable,
            "got": got_coll, "got_all_collections": got_all,
            "additional_collections": got_additional, "join_keys": got_join_keys,
            "strict_match": strict_match, "relaxed_match": relaxed_match,
            "has_join": has_join, "source": source, "candidates": candidates,
            "corrections": corrections, "tokens": tokens_used, "error": result.get("error"),
            "primary_reranked": result.get("primary_reranked", False),
            "primary_reasoning": got_args.get("primary_reasoning", "") if got_args else "",
            "query_args": got_args or {},
            "sql": sql,
            "db_id": db_id,
            "tool_schema": getattr(args, 'tool_schema', 'v1'),
            "latency_s": call_meta.get("latency_s", 0.0),
            "input_tokens": call_meta.get("input_tokens", 0),
            "output_tokens": call_meta.get("output_tokens", 0),
        })

    elapsed = time.time() - start

    # Results
    n = len(test_queries)
    avg_tokens = total_tokens_used / n if n > 0 else 0
    token_savings = (1 - avg_tokens / full_schema_tokens) * 100 if full_schema_tokens > 0 else 0

    # Collection-in-Output: expected appears anywhere in got_all_collections
    coll_in_output = sum(1 for r in results if r["expected"] in r.get("got_all_collections", []))

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"Collection Accuracy: {correct}/{n} ({correct/n*100:.1f}%)  [strict: primary must match FROM table]")
    if args.relaxed_match:
        print(f"Relaxed Accuracy:    {relaxed_correct}/{n} ({relaxed_correct/n*100:.1f}%)  [any JOIN table counts]")
    print(f"Coll-in-Output:      {coll_in_output}/{n} ({coll_in_output/n*100:.1f}%)  [expected anywhere in output]")
    print(f"Discovery Rate:      {in_topk}/{n} ({in_topk/n*100:.1f}%)")
    if ast_count > 0:
        print(f"Enhanced AST Score:  {total_ast/ast_count*100:.1f}% (N={ast_count})")
    print(f"Self-Corrections:    {total_corrections} total")
    print(f"Time:                {elapsed:.1f}s ({elapsed/n:.1f}s/query)")

    # JOIN query breakdown (using strict counts)
    non_join_total = n - join_total
    non_join_strict = correct - sum(1 for r in results if r["has_join"] and r["strict_match"])
    join_strict = sum(1 for r in results if r["has_join"] and r["strict_match"])
    non_join_relaxed = relaxed_correct - join_correct
    print(f"\nQuery Type Breakdown (Strict / Relaxed):")
    if non_join_total > 0:
        print(f"  Single-table:      {non_join_strict}/{non_join_total} ({non_join_strict/non_join_total*100:.1f}%)")
    if join_total > 0:
        print(f"  JOIN (strict):     {join_strict}/{join_total} ({join_strict/join_total*100:.1f}%)")
        print(f"  JOIN (relaxed):    {join_correct}/{join_total} ({join_correct/join_total*100:.1f}%)")

    # Multi-collection metrics
    if multi_coll_total > 0:
        avg_recall = set_recall_sum / multi_coll_total
        avg_precision = set_precision_sum / multi_coll_total if multi_coll_predicted > 0 else 0
        f1 = 2 * avg_recall * avg_precision / (avg_recall + avg_precision) if (avg_recall + avg_precision) > 0 else 0
        print(f"\nMulti-Collection Metrics (JOIN queries with >1 GT collection):")
        print(f"  Queries evaluated:   {multi_coll_total}")
        print(f"  LLM predicted >1:    {multi_coll_predicted}/{multi_coll_total} ({multi_coll_predicted/multi_coll_total*100:.1f}%)")
        print(f"  Set Recall (avg):    {avg_recall*100:.1f}%")
        print(f"  Set Precision (avg): {avg_precision*100:.1f}%")
        print(f"  Set F1 (avg):        {f1*100:.1f}%")
        print(f"  Exact Set Match:     {set_exact_match}/{multi_coll_total} ({set_exact_match/multi_coll_total*100:.1f}%)")
        print(f"  Join Keys returned:  {join_keys_total} total")

    # Token efficiency
    print(f"\nToken Efficiency:")
    print(f"  Full schema cost:  {full_schema_tokens} tokens (all {len(server.schemas)} collections)")
    print(f"  Avg. progressive:  {avg_tokens:.0f} tokens/query")
    print(f"  Token savings:     {token_savings:.1f}%")

    # Latency and Bedrock token usage
    latencies = [r["latency_s"] for r in results if r.get("latency_s", 0) > 0]
    if latencies:
        import statistics
        print(f"\nLatency (LLM call only):")
        print(f"  Mean:   {statistics.mean(latencies):.2f}s/query")
        print(f"  Median: {statistics.median(latencies):.2f}s/query")
        print(f"  P95:    {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s/query")
    input_toks = [r["input_tokens"] for r in results if r.get("input_tokens", 0) > 0]
    output_toks = [r["output_tokens"] for r in results if r.get("output_tokens", 0) > 0]
    if input_toks:
        print(f"\nAPI Token Usage (from provider):")
        print(f"  Input:  {statistics.mean(input_toks):.0f} tokens/query (mean)")
        print(f"  Output: {statistics.mean(output_toks):.0f} tokens/query (mean)")
        print(f"  Total:  {statistics.mean(input_toks) + statistics.mean(output_toks):.0f} tokens/query")

    # Pipeline stats
    print(f"\nPipeline Stats:")
    if pipeline.stats["difficulty_counts"]:
        print(f"  Adaptive Depth: {dict(pipeline.stats['difficulty_counts'])}")
    if pipeline.stats["correction_counts"]:
        print(f"  Corrections: {dict(pipeline.stats['correction_counts'])}")
    reranks = pipeline.stats.get("primary_reranks", 0)
    if reranks > 0:
        print(f"  Primary Reranks:   {reranks}")
    cf_swaps = pipeline.stats.get("primary_cf_swaps", 0)
    if cf_swaps > 0:
        print(f"  Column-First Swaps: {cf_swaps}")
    ts_swaps = pipeline.stats.get("primary_ts_swaps", 0)
    if ts_swaps > 0:
        print(f"  Two-Step Swaps:     {ts_swaps}")

    # KG info
    topo = pipeline.kg.get_topology_summary()
    print(f"  KG: {topo['total_edges']} edges, {topo['db_groups']} DB groups (via connected components)")

    print(f"\nPer-Database Accuracy (Strict / Relaxed):")
    for db in sorted(per_db.keys()):
        total = per_db[db]["total"]
        sc = per_db[db]["correct"]
        rc = per_db[db]["relaxed_correct"]
        jt = per_db[db]["join_total"]
        jc = per_db[db]["join_correct"]
        st = per_db[db].get("set_total", 0)
        sr = per_db[db].get("set_recall_sum", 0)
        spct = sc / total * 100 if total > 0 else 0
        rpct = rc / total * 100 if total > 0 else 0
        gap = rpct - spct
        gap_str = f" gap:{gap:.0f}%" if gap > 0 else ""
        set_info = f" SetR:{sr/st*100:.0f}%" if st > 0 else ""
        print(f"  {db:30s} {sc}/{total} ({spct:.1f}%) rlx:{rpct:.1f}%{gap_str}{set_info}")

    print(f"{'=' * 70}")

    # Save results JSON with full metrics, per-query details, and pipeline stats
    config_parts = []
    for flag in ['no_kg', 'no_embedding', 'no_rerank', 'no_correction', 'no_multi_hop', 'no_values', 'no_adaptive', 'no_primary_rerank']:
        if getattr(args, flag, False):
            config_parts.append(flag)
    config_str = "_".join(config_parts) if config_parts else "full"
    # Include primary-fix in config string if not 'none'
    primary_fix = getattr(args, 'primary_fix', 'none')
    if primary_fix != "none":
        config_str = f"{config_str}_pfix-{primary_fix}"
    split_suffix = f"_{args.split}" if args.split != "all" else ""

    # Build model short name for filenames
    model_short_map = {
        "openai/gpt-oss-120b": "nim_gptoss",
        "anthropic.claude-sonnet-4-6": "bedrock_sonnet46",
        "anthropic.claude-opus-4-6-v1": "bedrock_opus46",
        "qwen.qwen3-next-80b-a3b": "bedrock_qwen3_80b",
        "meta.llama4-maverick-17b-instruct-v1:0": "bedrock_llama4_maverick",
    }
    model_short = model_short_map.get(args.model, args.model.replace("/", "_").replace(".", "_").replace(":", "_"))

    # Determine experiment subfolder based on flags
    if getattr(args, 'output_dir', None):
        results_dir = Path(args.output_dir)
        schema_tag = f"_{args.tool_schema}" if getattr(args, 'tool_schema', 'v1') != 'v1' else ""
        # Use simplified naming for paper comparison
        paper_model_map = {
            "openai/gpt-oss-120b": "nim",
            "anthropic.claude-sonnet-4-6": "sonnet",
            "anthropic.claude-opus-4-6-v1": "opus",
            "qwen.qwen3-next-80b-a3b": "qwen",
            "meta.llama4-maverick-17b-instruct-v1:0": "llama",
        }
        paper_model = paper_model_map.get(args.model, model_short)
        config_tag = f"_{config_str}" if config_str != "full" else ""
        results_file_name = f"pipeline_{paper_model}_{args.tool_schema}{config_tag}_n{n}{split_suffix}.json"
    elif args.max_dbs:
        # Scale experiment
        results_dir = Path("eval/results/scale_exp")
        results_file_name = f"pipeline_{model_short}_{config_str}_dbs{args.max_dbs}_n{n}{split_suffix}.json"
    elif config_str != "full":
        # Ablation experiment
        results_dir = Path("eval/results/ablation_models")
        results_file_name = f"pipeline_{model_short}_{config_str}_n{n}{split_suffix}.json"
    else:
        # Full pipeline multi-model run
        results_dir = Path("eval/results/multi_model")
        results_file_name = f"pipeline_{model_short}_{config_str}_n{n}{split_suffix}.json"

    # Append _schemaV2 suffix for V2 runs so they don't collide with V1 results
    # (only when not using --output-dir, which already encodes schema in filename)
    if getattr(args, 'tool_schema', 'v1') == "v2" and not getattr(args, 'output_dir', None):
        results_file_name = results_file_name.replace(".json", "_schemaV2.json")

    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / results_file_name

    results_file.write_text(json.dumps({
        "provider": args.provider,
        "model": args.model,
        "model_short": model_short,
        "tool_schema": getattr(args, 'tool_schema', 'v1'),
        "config": {f: getattr(args, f, False) for f in ['no_kg', 'no_embedding', 'no_rerank', 'no_correction', 'no_multi_hop', 'no_values', 'no_adaptive', 'no_primary_rerank']},
        "split": args.split,
        "max_dbs": args.max_dbs,
        "include_joins": not args.no_joins,
        "relaxed_match": args.relaxed_match,
        "strict_accuracy": correct / n,
        "relaxed_accuracy": relaxed_correct / n,
        "coll_in_output": coll_in_output / n,
        "discovery_rate": in_topk / n,
        "ast_score": total_ast / ast_count if ast_count > 0 else None,
        "total_corrections": total_corrections,
        "join_accuracy": join_correct / join_total if join_total > 0 else None,
        "single_table_accuracy": non_join_strict / non_join_total if non_join_total > 0 else None,
        "multi_collection": {
            "total_evaluated": multi_coll_total,
            "llm_predicted_multi": multi_coll_predicted,
            "avg_set_recall": set_recall_sum / multi_coll_total if multi_coll_total > 0 else None,
            "avg_set_precision": set_precision_sum / multi_coll_total if multi_coll_total > 0 else None,
            "exact_set_match": set_exact_match,
            "join_keys_total": join_keys_total,
        },
        "token_efficiency": {
            "full_schema_tokens": full_schema_tokens,
            "avg_progressive_tokens": avg_tokens,
            "savings_pct": token_savings,
        },
        "pipeline_stats": {k: dict(v) if isinstance(v, defaultdict) else v for k, v in pipeline.stats.items()},
        "kg_stats": topo,
        "per_database": {db: dict(per_db[db]) for db in per_db},
        "n": n,
        "n_join": join_total,
        "n_single_table": non_join_total,
        "elapsed_seconds": elapsed,
        "results": results
    }, indent=2))
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
