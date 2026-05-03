"""Knowledge graph that infers cross-table relationships from column names.

Three edge types with learned weights (FK=0.0, SK=0.2, NP=0.2):
  FK_REFERENCE  -- explicit foreign-key naming (e.g., TableB.tablea_id)
  SHARED_KEY    -- same ID-like column in two tables
  NAME_PATTERN  -- shared PascalCase segments in the same DB group

Counter-intuitive finding: FK edges HURT discovery because they boost
confusable same-DB siblings already close in embedding space. Small
SHARED_KEY weight helps cross-DB disambiguation (validated N=500).

Key features: Union-Find for database grouping, multi-hop graph walks
with quadratic decay, adaptive depth routing (easy/medium/hard).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re
import math


class EdgeType(Enum):
    """Types of relationships between collections."""
    FK_REFERENCE = "fk_reference"      # TableA.id <-> TableB.tablea_id (strongest)
    SHARED_KEY = "shared_key"          # TableA.product_id <-> TableB.product_id
    NAME_PATTERN = "name_pattern"      # Inferred from naming conventions


@dataclass
class KGEdge:
    """A typed, weighted edge in the knowledge graph."""
    source: str
    target: str
    edge_type: EdgeType
    shared_columns: list = field(default_factory=list)
    weight: float = 1.0

    def __hash__(self):
        return hash((self.source, self.target, self.edge_type))

    def __eq__(self, other):
        return (self.source == other.source and self.target == other.target
                and self.edge_type == other.edge_type)


# Original hand-designed weights (for reference / ablation)
ORIGINAL_EDGE_TYPE_WEIGHTS = {
    EdgeType.FK_REFERENCE: 1.0,    # Direct FK: strongest relationship
    EdgeType.SHARED_KEY: 0.7,      # Shared column name: strong but might be coincidental
    EdgeType.NAME_PATTERN: 0.4,    # Name-based inference: weakest
}

# Learned via grid search (N=25 train converges). FK=0 is counterintuitive
# but correct: FK boost pushes confusable same-DB siblings above the real target.
DEFAULT_EDGE_TYPE_WEIGHTS = {
    EdgeType.FK_REFERENCE: 0.0,
    EdgeType.SHARED_KEY: 0.2,
    EdgeType.NAME_PATTERN: 0.2,
}

# Active weights (can be updated by learn_edge_weights)
EDGE_TYPE_WEIGHTS = dict(DEFAULT_EDGE_TYPE_WEIGHTS)


class FieldLevelKnowledgeGraph:
    """
    Knowledge Graph that builds relationships based on Field/Column matching.

    It solves the "Selection Error" problem where the model acts on a table
    that looks semantically relevant (e.g. "Items") but misses the specific
    relational table (e.g. "OrderItems") that is actually needed to join with "Orders".

    Enhanced with typed edges, multi-hop traversal, and topology-aware boosting.
    """

    def __init__(self, schemas: dict, value_stats: Optional[dict] = None):
        self.schemas = schemas
        self.field_index = defaultdict(list)
        self.relationships = defaultdict(set)  # Backward compat: set of neighbor names
        self.edges = defaultdict(list)         # New: typed edges per collection
        self.edge_map = {}                     # (source, target) -> list of KGEdge
        self.boost_factor = 0.2
        self.value_stats = value_stats or {}   # Collection -> property -> {min, max, top_values, cardinality}

        # Database grouping (discovered via connected components)
        self._db_groups = {}   # collection_name -> group_id
        self._db_components = {}  # group_id -> set of collection_names

        # Topology caches (computed lazily)
        self._degree_centrality = None
        self._hub_collections = None
        self._betweenness = None

        self._build_index()
        self._discover_database_groups()
        self._infer_relationships()
        self._compute_topology()

    def _build_index(self):
        """Index all columns: column_name -> [tables]"""
        for table, schema in self.schemas.items():
            props = schema.get('properties', {})

            if isinstance(props, list):
                cols = [p.get('name') for p in props if p.get('name')]
            elif isinstance(props, dict):
                cols = list(props.keys())
            else:
                continue

            for col in cols:
                self.field_index[col.lower()].append(table)

    def _discover_database_groups(self):
        """Union-Find over FK-pattern columns to discover database boundaries.

        No hardcoded DB prefixes -- new databases are auto-detected.
        Phase 1: FK-name linking, Phase 2: shared ID columns, Phase 3: PascalCase prefix fallback.
        """
        tables = list(self.schemas.keys())
        # Union-Find
        parent = {t: t for t in tables}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Phase 1: FK-pattern linking (highly specific, won't cross DB boundaries)
        for target_table in tables:
            simple_name = target_table.lower().replace('_', '')
            fk_candidates = [
                f"{target_table.lower()}_id",
                f"{simple_name}_id",
                f"{target_table.lower()}id",
                f"{simple_name}id"
            ]
            for fk in fk_candidates:
                if fk in self.field_index:
                    for source in self.field_index[fk]:
                        if source != target_table:
                            union(source, target_table)

        # Phase 2: Specific shared keys (columns with 4+ chars before _id, not just 'id')
        for col, col_tables in self.field_index.items():
            if len(col_tables) > 1 and col.endswith('_id') and len(col) > 4:
                for i, t1 in enumerate(col_tables):
                    for t2 in col_tables[i + 1:]:
                        union(t1, t2)

        # Phase 3: For still-isolated tables, cluster by longest common PascalCase prefix
        components = defaultdict(set)
        for t in tables:
            components[find(t)].add(t)

        isolated = [t for t in tables if components[find(t)] == {t}]
        if isolated:
            # Group isolated tables by longest common PascalCase prefix
            for t in isolated:
                parts = re.findall(r'[A-Z][a-z]*|\d+', t)
                # Try progressively shorter prefixes (down to 1 word)
                for length in range(len(parts) - 1, 0, -1):
                    prefix = ''.join(parts[:length])
                    if len(prefix) < 3:
                        continue  # Skip trivially short prefixes
                    # Check if any other table shares this prefix
                    matches = [o for o in tables if o != t and o.startswith(prefix)]
                    if matches:
                        for m in matches:
                            union(t, m)
                        break

        # Build final group mappings
        self._db_groups = {}
        self._db_components = defaultdict(set)
        for t in tables:
            root = find(t)
            self._db_groups[t] = root
            self._db_components[root].add(t)

    def _same_database(self, t1: str, t2: str) -> bool:
        """Check if two collections belong to the same database using connected components."""
        return self._db_groups.get(t1) == self._db_groups.get(t2)

    def get_database_group(self, collection: str) -> set:
        """Get all collections in the same database as the given collection."""
        root = self._db_groups.get(collection)
        if root is None:
            return {collection}
        return self._db_components.get(root, {collection})

    def _add_edge(self, source: str, target: str, edge_type: EdgeType, shared_cols: list = None):
        """Add a typed edge between two collections."""
        edge = KGEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            shared_columns=shared_cols or [],
            weight=EDGE_TYPE_WEIGHTS[edge_type]
        )
        self.edges[source].append(edge)
        self.relationships[source].add(target)

        key = (source, target)
        if key not in self.edge_map:
            self.edge_map[key] = []
        self.edge_map[key].append(edge)

    def _infer_relationships(self):
        """
        Infer typed links between tables based on FK patterns.

        Rules:
        1. SHARED_KEY: TableA.product_id <-> TableB.product_id (same column name, ID-like)
        2. FK_REFERENCE: TableA.id <-> TableB.tablea_id (explicit FK naming)
        3. NAME_PATTERN: Collections with overlapping name segments in same DB

        Only links tables within the SAME database.
        """
        # 1. Shared Keys (SHARED_KEY edges)
        for col, tables in self.field_index.items():
            if len(tables) > 1 and (col.endswith('_id') or col.endswith('_code') or col == 'id'):
                for t1 in tables:
                    for t2 in tables:
                        if t1 != t2 and self._same_database(t1, t2):
                            self._add_edge(t1, t2, EdgeType.SHARED_KEY, [col])

        # 2. FK Pattern (FK_REFERENCE edges - strongest)
        tables = list(self.schemas.keys())
        for target_table in tables:
            simple_name = target_table.lower().replace('_', '')
            fk_candidates = [
                f"{target_table.lower()}_id",
                f"{simple_name}_id",
                f"{target_table.lower()}id",
                f"{simple_name}id"
            ]

            for fk in fk_candidates:
                if fk in self.field_index:
                    source_tables = self.field_index[fk]
                    for source in source_tables:
                        if source != target_table and self._same_database(source, target_table):
                            self._add_edge(source, target_table, EdgeType.FK_REFERENCE, [fk])
                            self._add_edge(target_table, source, EdgeType.FK_REFERENCE, [fk])

        # 3. Name Pattern (NAME_PATTERN edges - weakest)
        # Collections sharing significant name segments (>= 4 chars) in same DB group
        # Use connected components to scope — no hardcoded prefix list needed
        name_parts = {}
        for table in tables:
            parts = set(re.findall(r'[A-Z][a-z]{3,}', table))
            name_parts[table] = parts

        # Only check pairs within the same database group (connected component)
        for group_root, group_members in self._db_components.items():
            group_list = sorted(group_members)
            for i, t1 in enumerate(group_list):
                for t2 in group_list[i + 1:]:
                    shared = name_parts.get(t1, set()) & name_parts.get(t2, set())
                    if shared and (t1, t2) not in self.edge_map:
                        self._add_edge(t1, t2, EdgeType.NAME_PATTERN, list(shared))
                        self._add_edge(t2, t1, EdgeType.NAME_PATTERN, list(shared))

    def _compute_topology(self):
        """Compute graph topology metrics for ranking."""
        n = len(self.schemas)
        if n <= 1:
            self._degree_centrality = {}
            self._hub_collections = set()
            return

        # Degree centrality: fraction of nodes a node is connected to
        self._degree_centrality = {}
        for name in self.schemas:
            neighbors = len(self.relationships.get(name, set()))
            self._degree_centrality[name] = neighbors / (n - 1) if n > 1 else 0

        # Hub detection: collections with above-average degree
        if self._degree_centrality:
            avg_degree = sum(self._degree_centrality.values()) / len(self._degree_centrality)
            threshold = avg_degree + (max(self._degree_centrality.values()) - avg_degree) * 0.5
            self._hub_collections = {
                name for name, dc in self._degree_centrality.items()
                if dc > threshold
            }
        else:
            self._hub_collections = set()

        # Approximate betweenness: for each node, count how many shortest paths go through it
        # Using simplified BFS-based approximation (exact is O(n^3))
        self._betweenness = defaultdict(float)
        sample_nodes = list(self.schemas.keys())[:min(50, n)]  # Sample for efficiency

        for source in sample_nodes:
            # BFS from source
            visited = {source: 0}
            queue = [source]
            parents = defaultdict(list)

            while queue:
                current = queue.pop(0)
                for neighbor in self.relationships.get(current, set()):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
                        parents[neighbor].append(current)
                    elif visited[neighbor] == visited[current] + 1:
                        parents[neighbor].append(current)

            # Count paths through each node
            for target in visited:
                if target == source:
                    continue
                path = target
                while path != source and parents.get(path):
                    for p in parents[path]:
                        if p != source:
                            self._betweenness[p] += 1.0
                    path = parents[path][0] if parents[path] else source

        # Normalize
        if self._betweenness:
            max_b = max(self._betweenness.values()) or 1
            for k in self._betweenness:
                self._betweenness[k] /= max_b

    def get_degree_centrality(self, collection: str) -> float:
        """Get degree centrality for a collection (0-1)."""
        return self._degree_centrality.get(collection, 0.0)

    def is_hub(self, collection: str) -> bool:
        """Check if a collection is a hub (highly connected)."""
        return collection in self._hub_collections

    def get_edge_weight(self, source: str, target: str) -> float:
        """Get the max edge weight between two collections."""
        edges = self.edge_map.get((source, target), [])
        if not edges:
            return 0.0
        return max(e.weight for e in edges)

    def get_multi_hop_neighbors(self, collection: str, max_hops: int = 2) -> dict:
        """
        Get neighbors up to max_hops away with distance-decayed weights.

        Returns:
            Dict of {neighbor_name: (hop_distance, accumulated_weight)}
        """
        visited = {collection: (0, 1.0)}
        frontier = [collection]

        for hop in range(1, max_hops + 1):
            next_frontier = []
            for current in frontier:
                for edge in self.edges.get(current, []):
                    neighbor = edge.target
                    if neighbor not in visited:
                        # Decay weight by hop distance and edge type
                        decay = 1.0 / (hop * hop)  # Quadratic decay
                        weight = edge.weight * decay
                        visited[neighbor] = (hop, weight)
                        next_frontier.append(neighbor)
            frontier = next_frontier

        # Remove the source node itself
        visited.pop(collection, None)
        return visited

    def _get_property_names(self, collection: str) -> list:
        """Get property names for a collection, handling both dict and list schema formats."""
        schema = self.schemas.get(collection, {})
        props = schema.get('properties', {})
        if isinstance(props, list):
            return [p.get('name', '').lower() for p in props if p.get('name')]
        elif isinstance(props, dict):
            return [k.lower() for k in props.keys()]
        return []

    def _get_property_types(self, collection: str) -> dict:
        """Get property name -> type mapping for a collection."""
        schema = self.schemas.get(collection, {})
        props = schema.get('properties', {})
        result = {}
        if isinstance(props, list):
            for p in props:
                name = p.get('name', '')
                dtype = p.get('data_type', ['unknown'])
                if isinstance(dtype, list):
                    dtype = dtype[0]
                result[name.lower()] = dtype
        elif isinstance(props, dict):
            for name, info in props.items():
                if isinstance(info, dict):
                    result[name.lower()] = info.get('type', 'unknown')
                else:
                    result[name.lower()] = str(info)
        return result

    @staticmethod
    def _extract_query_keywords(query: str) -> set:
        """Extract meaningful keywords from a query for matching."""
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'and', 'but', 'or', 'of', 'at', 'by', 'for', 'with', 'about',
            'to', 'from', 'in', 'on', 'how', 'what', 'which', 'who',
            'many', 'much', 'list', 'find', 'show', 'give', 'all', 'each',
            'their', 'there', 'than', 'that', 'this', 'these', 'those',
            'not', 'no', 'nor', 'only', 'very', 'just', 'more', 'most',
        }
        words = re.findall(r'[a-zA-Z]{3,}', query.lower())
        return {w for w in words if w not in stopwords}

    def _collection_name_keywords(self, name: str) -> set:
        """Extract keywords from a collection name (PascalCase splitting)."""
        parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z][a-z])|[A-Z]+$|\d+', name)
        return {p.lower() for p in parts if len(p) >= 2}

    def get_related(self, collection: str) -> list:
        """Get list of tables related via Foreign Keys (backward compat)."""
        return list(self.relationships.get(collection, set()))

    def get_typed_edges(self, collection: str) -> list:
        """Get all typed edges for a collection."""
        return self.edges.get(collection, [])

    # =========================================================================
    # BOOSTING METHODS
    # =========================================================================

    def boost_candidates(self, scores: list, top_n: int = 3) -> list:
        """
        Original static KG boost (kept for backward compatibility).
        """
        if not scores:
            return scores

        top_candidates = [s[0] for s in scores[:top_n]]
        boost = defaultdict(float)

        for coll in top_candidates:
            related = self.get_related(coll)
            for r in related:
                boost[r] += self.boost_factor

        boosted = []
        for name, score in scores:
            final_score = score + boost.get(name, 0)
            boosted.append((name, final_score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def typed_boost(self, scores: list, query: str, top_n: int = 3,
                    use_multi_hop: bool = True) -> list:
        """
        Enhanced boost using typed edges, multi-hop traversal, and topology.

        Improvements over query_aware_boost:
        1. Edge-type-aware: FK_REFERENCE edges get stronger boost than SHARED_KEY
        2. Multi-hop: 2-hop neighbors can be boosted (with decay)
        3. Hub penalty: Highly-connected hub nodes get reduced boost (they're generic)
        4. Topology-aware: Nodes on shortest paths between query-relevant nodes get boost

        Args:
            scores: List of (collection_name, score) tuples, sorted by score
            query: The natural language query
            top_n: Number of top candidates to expand
            use_multi_hop: Whether to use 2-hop expansion
        """
        if not scores:
            return scores

        query_keywords = self._extract_query_keywords(query)
        if not query_keywords:
            return self.boost_candidates(scores, top_n)

        top_candidates = [s[0] for s in scores[:top_n]]
        boost = defaultdict(float)

        for coll in top_candidates:
            if use_multi_hop:
                # Multi-hop neighbors with distance decay
                neighbors = self.get_multi_hop_neighbors(coll, max_hops=2)
                for neighbor, (hop, edge_weight) in neighbors.items():
                    neighbor_keywords = self._collection_name_keywords(neighbor)
                    neighbor_props = set(self._get_property_names(neighbor))

                    name_overlap = query_keywords & neighbor_keywords
                    prop_overlap = query_keywords & neighbor_props

                    if name_overlap or prop_overlap:
                        relevance = len(name_overlap) * 1.0 + len(prop_overlap) * 0.5

                        # Hub penalty: reduce boost for highly-connected nodes
                        hub_penalty = 0.5 if self.is_hub(neighbor) else 1.0

                        # Edge-type-weighted boost
                        type_weight = edge_weight  # Already includes type weight + decay

                        total_boost = self.boost_factor * min(relevance, 3.0) * type_weight * hub_penalty
                        boost[neighbor] += total_boost
            else:
                # 1-hop only with typed edges
                for edge in self.get_typed_edges(coll):
                    neighbor = edge.target
                    neighbor_keywords = self._collection_name_keywords(neighbor)
                    neighbor_props = set(self._get_property_names(neighbor))

                    name_overlap = query_keywords & neighbor_keywords
                    prop_overlap = query_keywords & neighbor_props

                    if name_overlap or prop_overlap:
                        relevance = len(name_overlap) * 1.0 + len(prop_overlap) * 0.5
                        hub_penalty = 0.5 if self.is_hub(neighbor) else 1.0
                        total_boost = self.boost_factor * min(relevance, 3.0) * edge.weight * hub_penalty
                        boost[neighbor] += total_boost

        boosted = []
        for name, score in scores:
            final_score = score + boost.get(name, 0)
            boosted.append((name, final_score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def query_aware_boost(self, scores: list, query: str, top_n: int = 3) -> list:
        """
        Query-aware KG boost with typed edges (backward compatible signature).
        Now delegates to typed_boost for improved accuracy.
        """
        return self.typed_boost(scores, query, top_n, use_multi_hop=True)

    def property_match_rerank(self, scores: list, query: str, weight: float = 0.15) -> list:
        """
        Re-rank candidates by counting how many query keywords match property names.
        Enhanced with value-aware matching when value_stats are available.
        """
        query_keywords = self._extract_query_keywords(query)
        if not query_keywords:
            return scores

        # Also extract potential values: numbers and capitalized words
        query_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', query))
        query_proper = set(re.findall(r'\b[A-Z][a-z]+\b', query))

        reranked = []
        for name, score in scores:
            prop_names = set(self._get_property_names(name))
            name_keywords = self._collection_name_keywords(name)

            # Keyword matches
            prop_matches = len(query_keywords & prop_names)
            name_matches = len(query_keywords & name_keywords)

            # Value-aware bonus (if value_stats available)
            value_bonus = 0.0
            if name in self.value_stats:
                coll_stats = self.value_stats[name]
                for prop, stats in coll_stats.items():
                    # Check if query numbers fall in numeric range
                    if query_numbers and 'min' in stats and 'max' in stats:
                        for num_str in query_numbers:
                            num = float(num_str)
                            if stats['min'] <= num <= stats['max']:
                                value_bonus += 0.1

                    # Check if query proper nouns match top values
                    if query_proper and 'top_values' in stats:
                        top_vals = {str(v).lower() for v in stats['top_values']}
                        for proper in query_proper:
                            if proper.lower() in top_vals:
                                value_bonus += 0.2

            bonus = (prop_matches * weight) + (name_matches * weight * 1.5) + value_bonus
            reranked.append((name, score + bonus))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    # =========================================================================
    # ADAPTIVE DEPTH ROUTING
    # =========================================================================

    def classify_query_difficulty(self, query: str, top_scores: list) -> str:
        """Route query to cheap (easy) or expensive (hard) discovery strategy."""
        if not top_scores or len(top_scores) < 2:
            return "hard"

        top_score = top_scores[0][1]
        second_score = top_scores[1][1] if len(top_scores) > 1 else 0

        # Score gap between #1 and #2
        gap = top_score - second_score

        # Check for direct name match
        query_lower = query.lower()
        top_name_lower = top_scores[0][0].lower()
        has_name_match = (top_name_lower in query_lower or
                         query_lower in top_name_lower or
                         any(w in top_name_lower for w in query_lower.split() if len(w) > 3))

        if has_name_match and gap > 0.15:
            return "easy"
        elif gap > 0.08 or top_score > 0.6:
            return "medium"
        else:
            return "hard"

    def adaptive_boost(self, scores: list, query: str, top_n: int = 3) -> tuple:
        """Skip KG (easy), use 1-hop (medium), or full multi-hop + rerank (hard)."""
        difficulty = self.classify_query_difficulty(query, scores[:5])

        if difficulty == "easy":
            # Skip KG entirely - top candidate is already clear
            return scores, "easy"
        elif difficulty == "medium":
            # Use 1-hop typed edges only (no multi-hop)
            boosted = self.typed_boost(scores, query, top_n, use_multi_hop=False)
            return boosted, "medium"
        else:
            # Full power: multi-hop + typed edges + property rerank
            boosted = self.typed_boost(scores, query, top_n, use_multi_hop=True)
            boosted = self.property_match_rerank(boosted, query, weight=0.15)
            return boosted, "hard"

    # =========================================================================
    # CONTEXT GENERATION
    # =========================================================================

    def classify_collection_role(self, collection: str) -> str:
        """Classify a collection as ENTITY, LOOKUP, or JUNCTION based on schema structure.

        Rules:
        - JUNCTION: 2-3 properties, most are FK-like (end in _id or reference other tables)
        - LOOKUP: ≤3 properties, typically id + one descriptive value
        - ENTITY: >5 properties or mixed types → main fact table
        - (default): 4-5 properties that don't match other patterns
        """
        schema = self.schemas.get(collection)
        if not schema:
            return ""

        props = schema.get('properties', {})
        if isinstance(props, list):
            prop_names = [p.get('name', '') for p in props if p.get('name')]
        elif isinstance(props, dict):
            prop_names = list(props.keys())
        else:
            return ""

        n_props = len(prop_names)
        if n_props == 0:
            return ""

        # Count FK-like properties (end in _id, id, _code, or match another table name)
        fk_like = 0
        for pn in prop_names:
            pn_lower = pn.lower().replace(' ', '_')
            if pn_lower == 'id':
                continue  # Primary key, not FK
            if pn_lower.endswith('_id') or pn_lower.endswith(' id') or pn_lower.endswith('_code'):
                fk_like += 1

        if n_props <= 3 and fk_like >= 2:
            return "JUNCTION"
        elif n_props <= 3:
            return "LOOKUP"
        elif n_props > 5:
            return "ENTITY"
        else:
            return ""

    def get_context(self, collection: str) -> str:
        """Generate a context string for the prompt (backward compat)."""
        related = self.get_related(collection)
        if not related:
            return ""
        return f"[Linked tables: {', '.join(related[:5])}]"

    def get_rich_context(self, collection: str) -> str:
        """Generate a richer context string with FK direction arrows.

        Shows directional FK hints from the collection's own properties.
        E.g., SuperheroSuperhero has 'gender id' → points to SuperheroGender.
        """
        schema = self.schemas.get(collection)
        if not schema:
            return ""

        # Extract property names
        props = schema.get('properties', {})
        if isinstance(props, list):
            prop_names = [p.get('name', '') for p in props if p.get('name')]
        elif isinstance(props, dict):
            prop_names = list(props.keys())
        else:
            prop_names = []

        # Find FK-like properties and try to resolve their targets
        db_group = self.get_database_group(collection)
        fk_arrows = []

        for pn in prop_names:
            pn_lower = pn.lower().replace(' ', '_')
            # Skip primary key 'id'
            if pn_lower == 'id':
                continue
            # Detect FK-like: ends in _id or " id"
            if not (pn_lower.endswith('_id') or pn.lower().endswith(' id')):
                continue

            # Try to resolve target collection from the FK name
            # E.g., "gender id" → look for collection containing "Gender" in same DB
            fk_stem = pn_lower.replace('_id', '').replace(' ', '')
            target = None
            best_specificity = 0
            for sibling in db_group:
                if sibling == collection:
                    continue
                # Extract the distinguishing suffix (after DB prefix) in lowercase
                # e.g., SuperheroGender → "gender", Formula1Races → "races"
                sib_lower = sibling.lower()
                # Match: fk_stem should match a word boundary in the sibling name
                # Prefer exact suffix match over substring match
                # Split camelCase into parts for matching
                sib_parts = [p.lower() for p in re.findall(r'[A-Z][a-z]*', sibling)]
                for part in sib_parts:
                    if part == fk_stem and len(part) > best_specificity:
                        target = sibling
                        best_specificity = len(part)
                    elif fk_stem.startswith(part) and len(part) >= 4 and len(part) > best_specificity:
                        target = sibling
                        best_specificity = len(part)

            if target:
                fk_arrows.append(f"{pn} → {target}")
            else:
                fk_arrows.append(f"{pn} → ?")

        # Also add explicit KG FK edges not covered by property analysis
        edges = self.get_typed_edges(collection)
        covered_targets = {a.split(' → ')[1] for a in fk_arrows if ' → ' in a and '?' not in a}

        for edge in edges:
            if edge.target in covered_targets:
                continue
            if edge.edge_type == EdgeType.FK_REFERENCE and edge.shared_columns:
                col = edge.shared_columns[0]
                if col.lower() != 'id':  # Skip generic id shared keys
                    fk_arrows.append(f"{col} → {edge.target}")
                    covered_targets.add(edge.target)

        if not fk_arrows:
            return ""

        # Limit to 4 arrows to keep prompt concise
        hub_tag = " [HUB]" if self.is_hub(collection) else ""
        display = [a for a in fk_arrows[:5] if '→ ?' not in a]
        if not display:
            return ""
        return f"[Links: {', '.join(display)}{hub_tag}]"

    def get_topology_summary(self) -> dict:
        """Get a summary of the graph topology for debugging/reporting."""
        total_edges = sum(len(edges) for edges in self.edges.values())
        edge_type_counts = defaultdict(int)
        for edges in self.edges.values():
            for edge in edges:
                edge_type_counts[edge.edge_type.value] += 1

        return {
            "total_collections": len(self.schemas),
            "total_edges": total_edges,
            "edge_types": dict(edge_type_counts),
            "hub_collections": list(self._hub_collections) if self._hub_collections else [],
            "db_groups": len(self._db_components),
            "db_group_sizes": {
                root: len(members)
                for root, members in sorted(self._db_components.items(), key=lambda x: -len(x[1]))
            },
            "avg_degree_centrality": (
                sum(self._degree_centrality.values()) / len(self._degree_centrality)
                if self._degree_centrality else 0
            ),
            "max_degree_centrality": (
                max(self._degree_centrality.values())
                if self._degree_centrality else 0
            ),
            "edge_weights": {et.value: w for et, w in EDGE_TYPE_WEIGHTS.items()},
        }

    # =========================================================================
    # LEARNED EDGE WEIGHTS
    # =========================================================================

    def learn_edge_weights(self, training_data: list, embedding_func=None) -> dict:
        """
        Learn optimal edge type weights from training data.

        Uses grid search over weight combinations to maximize discovery accuracy
        (correct collection in top-1 after KG boost).

        Args:
            training_data: List of dicts with 'question', 'expected_collection',
                          and optionally 'acceptable_collections'
            embedding_func: Function(text) -> embedding vector. If None, uses
                           pre-computed self.embeddings (if available from pipeline).

        Returns:
            Dict with learned weights and accuracy improvement
        """
        global EDGE_TYPE_WEIGHTS

        if not training_data:
            return {"error": "No training data", "weights": dict(EDGE_TYPE_WEIGHTS)}

        # Pre-compute query embeddings
        if embedding_func is None:
            try:
                from src.utils.embeddings import get_embedding, cosine_similarity
                embedding_func = lambda text: get_embedding(text, model="local")
            except ImportError:
                return {"error": "No embedding function available"}

        from src.utils.embeddings import cosine_similarity

        # We need collection embeddings — compute minimal ones (name + description)
        coll_embeddings = {}
        for name, schema in self.schemas.items():
            desc = schema.get('envisioned_use_case_overview', '') or schema.get('description', name)
            text = f"{name}: {desc}"
            coll_embeddings[name] = embedding_func(text)

        # Grid search over weight combinations
        best_weights = dict(DEFAULT_EDGE_TYPE_WEIGHTS)
        best_accuracy = 0.0

        weight_options = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        results_log = []

        for fk_w in weight_options:
            for sk_w in weight_options:
                for np_w in weight_options:
                    if fk_w == 0 and sk_w == 0 and np_w == 0:
                        continue  # Skip all-zero

                    # Temporarily set weights
                    test_weights = {
                        EdgeType.FK_REFERENCE: fk_w,
                        EdgeType.SHARED_KEY: sk_w,
                        EdgeType.NAME_PATTERN: np_w,
                    }

                    # Update edge weights in the graph
                    for edges_list in self.edges.values():
                        for edge in edges_list:
                            edge.weight = test_weights[edge.edge_type]

                    # Evaluate on training data
                    correct = 0
                    for item in training_data:
                        query = item['question']
                        expected = item['expected_collection']
                        acceptable = item.get('acceptable_collections', [expected])

                        query_emb = embedding_func(query)
                        scores = []
                        for name, emb in coll_embeddings.items():
                            sim = cosine_similarity(list(query_emb), list(emb))
                            scores.append((name, sim))
                        scores.sort(key=lambda x: x[1], reverse=True)

                        # Apply KG boost
                        boosted = self.typed_boost(scores, query, top_n=3, use_multi_hop=True)

                        # Check if top-1 is correct
                        if boosted and boosted[0][0] in acceptable:
                            correct += 1

                    accuracy = correct / len(training_data) if training_data else 0

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_weights = dict(test_weights)
                        results_log.append({
                            'fk': fk_w, 'sk': sk_w, 'np': np_w,
                            'accuracy': accuracy
                        })

        # Restore best weights
        EDGE_TYPE_WEIGHTS.update(best_weights)
        for edges_list in self.edges.values():
            for edge in edges_list:
                edge.weight = best_weights[edge.edge_type]

        # Compare with default
        default_accuracy = 0
        default_test = dict(DEFAULT_EDGE_TYPE_WEIGHTS)
        for edges_list in self.edges.values():
            for edge in edges_list:
                edge.weight = default_test[edge.edge_type]

        for item in training_data:
            query = item['question']
            acceptable = item.get('acceptable_collections', [item['expected_collection']])
            query_emb = embedding_func(query)
            scores = [(name, cosine_similarity(list(query_emb), list(emb)))
                      for name, emb in coll_embeddings.items()]
            scores.sort(key=lambda x: x[1], reverse=True)
            boosted = self.typed_boost(scores, query, top_n=3, use_multi_hop=True)
            if boosted and boosted[0][0] in acceptable:
                default_accuracy += 1
        default_accuracy /= len(training_data) if training_data else 1

        # Restore learned weights
        EDGE_TYPE_WEIGHTS.update(best_weights)
        for edges_list in self.edges.values():
            for edge in edges_list:
                edge.weight = best_weights[edge.edge_type]

        return {
            "learned_weights": {et.value: w for et, w in best_weights.items()},
            "default_weights": {et.value: w for et, w in DEFAULT_EDGE_TYPE_WEIGHTS.items()},
            "learned_accuracy": best_accuracy,
            "default_accuracy": default_accuracy,
            "improvement": best_accuracy - default_accuracy,
            "n_training": len(training_data),
        }
