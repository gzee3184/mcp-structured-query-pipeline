"""
Learning Cache for MCP Discovery

A cache that learns from discovery failures to improve future query-to-collection matching.
Uses keyword extraction and boost scoring to progressively improve discovery accuracy.

DESIGN PRINCIPLES:
1. Learn from failures - record which collections were expected for which queries
2. Keyword extraction - map keywords to collections with boost scores
3. Persistent storage - save learnings to disk for cross-session improvement
4. Explainable - can inspect which keywords map to which collections
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, List, Set
from collections import defaultdict
from threading import Lock


# Common stopwords to filter out
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until',
    'while', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'find', 'show', 'get', 'list', 'give', 'tell', 'want', 'search', 'query',
    'data', 'information', 'record', 'records', 'database', 'table', 'tables'
}


def extract_keywords(text: str, min_length: int = 3) -> Set[str]:
    """
    Extract meaningful keywords from text.
    
    Args:
        text: Input text to extract keywords from
        min_length: Minimum keyword length to include
    
    Returns:
        Set of lowercase keywords
    """
    # Lowercase and split on non-alphanumeric
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter stopwords and short words
    keywords = {w for w in words if len(w) >= min_length and w not in STOPWORDS}
    
    # Also extract numbers as potential keywords (e.g., "2024", "1000")
    numbers = re.findall(r'\b\d+\b', text)
    keywords.update(numbers)
    
    return keywords


def extract_collection_keywords(collection_name: str) -> Set[str]:
    """
    Extract keywords from a collection name (e.g., FbiCode -> {fbi, code}).
    """
    # Split on case changes and underscores
    parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z][a-z])|[A-Z]+$|\d+', collection_name)
    return {p.lower() for p in parts if len(p) >= 2}


class LearningCache:
    """
    Cache that learns from discovery failures to improve future matching.
    
    Features:
    - Records query → collection mappings from failures
    - Extracts keywords and builds boost scores
    - Persists learnings to disk
    - Provides boost scores for discovery ranking
    
    Example:
        >>> cache = LearningCache()
        >>> cache.record_failure("Find FBI crime codes", "FbiCode", "Crime")
        >>> boost = cache.get_boost("Show FBI codes", "FbiCode")
        >>> print(boost)  # Returns positive boost
    """
    
    DEFAULT_CACHE_DIR = ".learning_cache"
    DEFAULT_BOOST_INCREMENT = 10.0
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        boost_increment: float = DEFAULT_BOOST_INCREMENT,
        persistent: bool = False
    ):
        """
        Initialize the learning cache.
        
        Args:
            cache_dir: Directory for persistent storage
            boost_increment: How much to boost a collection per learned keyword
            persistent: If True, load/save from disk (cross-session learning).
                       If False (default), start fresh every session.
        """
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.cache_file = self.cache_dir / "learning_cache.json"
        self.boost_increment = boost_increment
        self.persistent = persistent
        self._lock = Lock()
        
        # Core data structures
        self.keyword_boosts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.query_mappings: Dict[str, str] = {}  # exact query → collection
        self.failure_log: List[dict] = []
        self.success_log: List[dict] = []
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache only if persistent mode
        if self.persistent:
            self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    
                self.query_mappings = data.get('query_mappings', {})
                self.failure_log = data.get('failure_log', [])
                self.success_log = data.get('success_log', [])
                
                # Rebuild keyword_boosts from stored data
                boosts = data.get('keyword_boosts', {})
                for kw, collections in boosts.items():
                    for coll, score in collections.items():
                        self.keyword_boosts[kw][coll] = score
                        
            except (json.JSONDecodeError, IOError) as e:
                print(f"[LearningCache] Warning: Could not load cache: {e}")
    
    def _save_cache(self) -> None:
        """Save cache to file (only in persistent mode)."""
        if not self.persistent:
            return  # In-memory only for fresh sessions
        try:
            data = {
                'query_mappings': self.query_mappings,
                'keyword_boosts': {k: dict(v) for k, v in self.keyword_boosts.items()},
                'failure_log': self.failure_log[-1000:],  # Keep last 1000
                'success_log': self.success_log[-1000:],
                'timestamp': time.time()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except IOError as e:
            print(f"[LearningCache] Warning: Could not save cache: {e}")
    
    def record_failure(
        self,
        query: str,
        expected_collection: str,
        got_collection: Optional[str],
        candidates: Optional[List[str]] = None
    ) -> None:
        """
        Record a discovery failure and learn from it.
        
        Args:
            query: The natural language query
            expected_collection: The correct collection (ground truth)
            got_collection: What the discovery returned (or None)
            candidates: List of candidate collections that were considered
        """
        with self._lock:
            # Extract keywords from query
            query_keywords = extract_keywords(query)
            
            # Also extract keywords from expected collection name
            coll_keywords = extract_collection_keywords(expected_collection)
            
            # Boost the expected collection for all query keywords
            for keyword in query_keywords:
                self.keyword_boosts[keyword][expected_collection] += self.boost_increment
                
                # If wrong collection was returned, penalize it slightly
                if got_collection and got_collection != expected_collection:
                    self.keyword_boosts[keyword][got_collection] -= self.boost_increment * 0.3
            
            # Map collection keywords to collection (important for cryptic names)
            for kw in coll_keywords:
                self.keyword_boosts[kw][expected_collection] += self.boost_increment * 2
            
            # Store exact query mapping for future direct matching
            query_normalized = query.lower().strip()
            self.query_mappings[query_normalized] = expected_collection
            
            # Log the failure
            self.failure_log.append({
                'query': query,
                'expected': expected_collection,
                'got': got_collection,
                'candidates': candidates,
                'keywords': list(query_keywords),
                'timestamp': time.time()
            })
            
            self._save_cache()
    
    def record_success(
        self,
        query: str,
        collection: str
    ) -> None:
        """
        Record a successful discovery (optional reinforcement learning).
        
        Args:
            query: The natural language query
            collection: The correctly matched collection
        """
        with self._lock:
            query_keywords = extract_keywords(query)
            
            # Small boost for successful matches (reinforcement)
            for keyword in query_keywords:
                self.keyword_boosts[keyword][collection] += self.boost_increment * 0.2
            
            self.success_log.append({
                'query': query,
                'collection': collection,
                'timestamp': time.time()
            })
            
            # Save less frequently for successes
            if len(self.success_log) % 10 == 0:
                self._save_cache()
    
    def get_boost(self, query: str, collection: str) -> float:
        """
        Get the learned boost score for a collection based on query.
        
        Args:
            query: The natural language query
            collection: The collection to get boost for
        
        Returns:
            Boost score (positive means boosted, negative means penalized)
        """
        with self._lock:
            # Check for exact query match first
            query_normalized = query.lower().strip()
            if query_normalized in self.query_mappings:
                if self.query_mappings[query_normalized] == collection:
                    return 1000.0  # Very high boost for exact match
            
            # Calculate keyword-based boost
            keywords = extract_keywords(query)
            boost = 0.0
            
            for kw in keywords:
                boost += self.keyword_boosts.get(kw, {}).get(collection, 0.0)
            
            return boost
    
    def get_boosted_rankings(
        self,
        query: str,
        candidates: List[str],
        base_scores: Optional[Dict[str, float]] = None
    ) -> List[tuple]:
        """
        Re-rank candidates using learned boosts.
        
        Args:
            query: The natural language query
            candidates: List of candidate collection names
            base_scores: Optional dict of base scores from discovery
        
        Returns:
            List of (collection, combined_score) sorted by score descending
        """
        scored = []
        
        for coll in candidates:
            base = base_scores.get(coll, 0.0) if base_scores else 0.0
            boost = self.get_boost(query, coll)
            combined = base + boost
            scored.append((coll, combined, base, boost))
        
        # Sort by combined score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [(coll, score) for coll, score, _, _ in scored]
    
    def get_direct_match(self, query: str) -> Optional[str]:
        """
        Check if we have a direct query → collection mapping.
        
        Returns:
            Collection name if exact match exists, None otherwise
        """
        query_normalized = query.lower().strip()
        return self.query_mappings.get(query_normalized)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'total_failures_recorded': len(self.failure_log),
            'total_successes_recorded': len(self.success_log),
            'unique_keywords': len(self.keyword_boosts),
            'query_mappings': len(self.query_mappings),
            'cache_file': str(self.cache_file)
        }
    
    def get_top_keywords_for_collection(self, collection: str, top_n: int = 10) -> List[tuple]:
        """
        Get top keywords that boost a collection.
        
        Returns:
            List of (keyword, boost_score) sorted by score
        """
        scores = []
        for kw, collections in self.keyword_boosts.items():
            if collection in collections:
                scores.append((kw, collections[collection]))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    
    def get_collection_hint(self, collection: str, max_terms: int = 5) -> str:
        """
        Get a learned context hint for a collection to help the LLM understand it.
        
        This is the key innovation: after learning from failures, we can tell the LLM
        what cryptic collection names actually mean based on past query associations.
        
        Args:
            collection: The collection name
            max_terms: Maximum number of terms to include in hint
        
        Returns:
            Hint string like "[HINT: Relevant for: FBI, crime codes, classification]"
            or empty string if no learned keywords
        
        Example:
            >>> cache.record_failure("Find FBI crime codes", "FbiCode", "Crime")
            >>> hint = cache.get_collection_hint("FbiCode")
            >>> print(hint)
            "[HINT: Relevant for queries about: fbi, crime, codes]"
        """
        keywords = self.get_top_keywords_for_collection(collection, max_terms)
        
        if not keywords:
            return ""
        
        # Filter to only significant keywords (score > threshold)
        significant = [(kw, score) for kw, score in keywords if score >= self.boost_increment]
        
        if not significant:
            return ""
        
        terms = [kw for kw, score in significant[:max_terms]]
        return f"[HINT: Relevant for queries about: {', '.join(terms)}]"
    
    def get_all_hints(self) -> Dict[str, str]:
        """
        Get hints for all collections that have learned keywords.
        
        Returns:
            Dict mapping collection names to their hints
        """
        hints = {}
        
        # Find all collections that have keywords
        all_collections = set()
        for kw, colls in self.keyword_boosts.items():
            all_collections.update(colls.keys())
        
        for coll in all_collections:
            hint = self.get_collection_hint(coll)
            if hint:
                hints[coll] = hint
        
        return hints
    
    def clear(self) -> None:
        """Clear all learned data."""
        with self._lock:
            self.keyword_boosts = defaultdict(lambda: defaultdict(float))
            self.query_mappings = {}
            self.failure_log = []
            self.success_log = []
            
            if self.cache_file.exists():
                self.cache_file.unlink()


# Test function
if __name__ == "__main__":
    print("Testing LearningCache...")
    
    cache = LearningCache(cache_dir=".test_learning_cache")
    cache.clear()
    
    # Simulate failures
    cache.record_failure("Find FBI crime codes", "FbiCode", "Crime", ["Crime", "Ward", "District"])
    cache.record_failure("Show FBI classification", "FbiCode", "Crime", ["Crime", "Iucr"])
    cache.record_failure("What is the FBI code for robbery", "FbiCode", "Crime", ["Crime", "Ward"])
    
    # Test boost
    boost1 = cache.get_boost("Find FBI codes", "FbiCode")
    boost2 = cache.get_boost("Find FBI codes", "Crime")
    
    print(f"\nAfter 3 failures:")
    print(f"  FbiCode boost for 'FBI' query: {boost1}")
    print(f"  Crime boost for 'FBI' query: {boost2}")
    
    # Test re-ranking
    candidates = ["Crime", "Ward", "FbiCode", "District"]
    ranked = cache.get_boosted_rankings("Find FBI codes", candidates)
    
    print(f"\nRe-ranked candidates for 'Find FBI codes':")
    for coll, score in ranked:
        print(f"  {coll}: {score}")
    
    # Stats
    print(f"\nStats: {cache.get_stats()}")
    
    # Top keywords
    print(f"\nTop keywords for FbiCode: {cache.get_top_keywords_for_collection('FbiCode')}")
    
    # Cleanup
    cache.clear()
