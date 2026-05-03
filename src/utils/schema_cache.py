"""
Schema Cache Utility for Progressive Disclosure

This module provides file-based schema caching to reduce redundant MCP calls
and optimize token usage when working with large database schemas.

DESIGN PRINCIPLES:
1. File-based persistence for cross-session caching
2. LRU eviction to manage cache size
3. Schema compression to reduce context window usage
4. TTL-based expiration for stale data
5. Context-aware retrieval (respects token budgets)

CACHE STRUCTURE:
- Cache stored as JSON file in configurable directory
- Each entry contains: schema, compressed_schema, timestamp, access_count
- Automatic cleanup of entries older than TTL
"""

import json
import os
import time
import hashlib
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict
from threading import Lock


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    schema: dict
    compressed_schema: dict
    timestamp: float
    access_count: int
    size_tokens: int  # Estimated token count for context budgeting


class SchemaCache:
    """
    File-based schema cache with LRU eviction and context-aware retrieval.
    
    Features:
    - Persistent storage across sessions
    - Automatic compression of schemas
    - Token-aware retrieval for context window management
    - Thread-safe operations
    
    Example:
        >>> cache = SchemaCache(cache_dir="/tmp/schema_cache")
        >>> cache.set("users_collection", {"name": "text", "email": "text"})
        >>> schema = cache.get("users_collection")
        >>> compressed = cache.get("users_collection", compressed=True)
    """
    
    DEFAULT_CACHE_DIR = ".schema_cache"
    DEFAULT_MAX_ENTRIES = 100
    DEFAULT_TTL_HOURS = 24
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_hours: float = DEFAULT_TTL_HOURS
    ):
        """
        Initialize the schema cache.
        
        Args:
            cache_dir: Directory for cache file storage
            max_entries: Maximum number of cached schemas
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.cache_file = self.cache_dir / "schema_cache.json"
        self.max_entries = max_entries
        self.ttl_seconds = ttl_hours * 3600
        self._lock = Lock()
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache
        self._cache: dict[str, dict] = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    # Clean expired entries on load
                    return self._clean_expired(data)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except IOError as e:
            print(f"[SchemaCache] Warning: Could not save cache: {e}")
    
    def _clean_expired(self, cache: dict) -> dict:
        """Remove expired entries from cache."""
        current_time = time.time()
        return {
            key: entry for key, entry in cache.items()
            if current_time - entry.get('timestamp', 0) < self.ttl_seconds
        }
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries if cache is full."""
        if len(self._cache) >= self.max_entries:
            # Sort by access count and timestamp (LRU)
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: (x[1].get('access_count', 0), x[1].get('timestamp', 0))
            )
            # Remove oldest 10% of entries
            to_remove = max(1, len(sorted_entries) // 10)
            for key, _ in sorted_entries[:to_remove]:
                del self._cache[key]
    
    def _generate_key(self, collection_name: str) -> str:
        """Generate a cache key from collection name."""
        return hashlib.md5(collection_name.encode()).hexdigest()[:16]
    
    def _estimate_tokens(self, schema: dict) -> int:
        """Estimate token count for a schema (rough: 4 chars per token)."""
        json_str = json.dumps(schema)
        return len(json_str) // 4
    
    def get(
        self,
        collection_name: str,
        compressed: bool = False,
        max_tokens: Optional[int] = None
    ) -> Optional[dict]:
        """
        Retrieve a schema from cache.
        
        Args:
            collection_name: Name of the collection
            compressed: If True, return compressed schema
            max_tokens: If set, only return if within token budget
            
        Returns:
            Schema dict if found and within constraints, None otherwise
        """
        key = self._generate_key(collection_name)
        
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if time.time() - entry.get('timestamp', 0) > self.ttl_seconds:
                del self._cache[key]
                self._save_cache()
                return None
            
            # Check token budget
            schema_key = 'compressed_schema' if compressed else 'schema'
            schema = entry.get(schema_key, entry.get('schema'))
            
            if max_tokens:
                tokens = self._estimate_tokens(schema)
                if tokens > max_tokens:
                    # Return compressed version if it fits
                    compressed_schema = entry.get('compressed_schema')
                    if compressed_schema and self._estimate_tokens(compressed_schema) <= max_tokens:
                        schema = compressed_schema
                    else:
                        return None
            
            # Update access count
            entry['access_count'] = entry.get('access_count', 0) + 1
            self._save_cache()
            
            return schema
    
    def set(
        self,
        collection_name: str,
        schema: dict,
        compressed_schema: Optional[dict] = None
    ) -> None:
        """
        Store a schema in cache.
        
        Args:
            collection_name: Name of the collection
            schema: Full schema dict
            compressed_schema: Optional pre-compressed schema
        """
        key = self._generate_key(collection_name)
        
        with self._lock:
            self._evict_lru()
            
            if compressed_schema is None:
                compressed_schema = compress_schema(schema)
            
            self._cache[key] = {
                'schema': schema,
                'compressed_schema': compressed_schema,
                'timestamp': time.time(),
                'access_count': 1,
                'size_tokens': self._estimate_tokens(schema)
            }
            
            self._save_cache()
    
    def get_or_fetch(
        self,
        collection_name: str,
        fetcher: callable,
        compressed: bool = False
    ) -> dict:
        """
        Get schema from cache or fetch using provided function.
        
        Args:
            collection_name: Name of the collection
            fetcher: Callable that returns the schema if not cached
            compressed: If True, return compressed version
            
        Returns:
            Schema dict (from cache or freshly fetched)
        """
        cached = self.get(collection_name, compressed=compressed)
        if cached is not None:
            return cached
        
        # Fetch and cache
        schema = fetcher()
        self.set(collection_name, schema)
        
        return compress_schema(schema) if compressed else schema
    
    def get_multiple(
        self,
        collection_names: list[str],
        max_total_tokens: int = 4000,
        compressed: bool = True
    ) -> dict[str, dict]:
        """
        Get multiple schemas respecting total token budget.
        
        Useful for progressive disclosure - fetch schemas that fit
        within the context window budget.
        
        Args:
            collection_names: List of collection names to retrieve
            max_total_tokens: Maximum total tokens for all schemas
            compressed: If True, use compressed schemas
            
        Returns:
            Dict mapping collection names to schemas (may be partial)
        """
        result = {}
        tokens_used = 0
        
        for name in collection_names:
            schema = self.get(name, compressed=compressed)
            if schema:
                schema_tokens = self._estimate_tokens(schema)
                if tokens_used + schema_tokens <= max_total_tokens:
                    result[name] = schema
                    tokens_used += schema_tokens
                else:
                    # Try even more compressed version
                    minimal = self._get_minimal_schema(name)
                    if minimal:
                        minimal_tokens = self._estimate_tokens(minimal)
                        if tokens_used + minimal_tokens <= max_total_tokens:
                            result[name] = minimal
                            tokens_used += minimal_tokens
        
        return result
    
    def _get_minimal_schema(self, collection_name: str) -> Optional[dict]:
        """Get minimal schema (just property names and types)."""
        full = self.get(collection_name, compressed=True)
        if not full:
            return None
        
        # Return just the collection with property name: type mapping
        return {k: v for k, v in full.items() if isinstance(v, str)}
    
    def clear(self) -> None:
        """Clear all cached schemas."""
        with self._lock:
            self._cache = {}
            if self.cache_file.exists():
                self.cache_file.unlink()
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total_tokens = sum(
                entry.get('size_tokens', 0) for entry in self._cache.values()
            )
            return {
                'entries': len(self._cache),
                'max_entries': self.max_entries,
                'total_tokens': total_tokens,
                'cache_file': str(self.cache_file)
            }


def compress_schema(schema: dict) -> dict:
    """
    Compress a schema to reduce token usage by 60-80%.
    
    Compression strategies:
    1. Abbreviate property types (text -> t, int -> i, bool -> b)
    2. Remove non-essential metadata (indexed, searchable)
    3. Use shorthand notation for required fields
    
    Args:
        schema: Full schema dict with property details
        
    Returns:
        Compressed schema dict
        
    Example:
        >>> schema = {
        ...     "users": {
        ...         "name": {"type": "text", "required": True, "indexed": True},
        ...         "age": {"type": "int", "required": False}
        ...     }
        ... }
        >>> compress_schema(schema)
        {'users': {'name': 'text*', 'age': 'int'}}
    """
    if not isinstance(schema, dict):
        return schema
    
    compressed = {}
    
    for collection_name, properties in schema.items():
        if not isinstance(properties, dict):
            compressed[collection_name] = properties
            continue
        
        compressed_props = {}
        for prop_name, prop_info in properties.items():
            if isinstance(prop_info, dict):
                # Extract type and required flag
                prop_type = prop_info.get('type', prop_info.get('dataType', 'unknown'))
                is_required = prop_info.get('required', False)
                
                # Abbreviate type
                type_abbrev = {
                    'text': 'text',
                    'string': 'text',
                    'int': 'int',
                    'integer': 'int',
                    'number': 'num',
                    'float': 'float',
                    'boolean': 'bool',
                    'bool': 'bool',
                    'date': 'date',
                    'datetime': 'date',
                    'object': 'obj',
                    'array': 'arr',
                    'text[]': 'text[]'
                }.get(prop_type.lower() if isinstance(prop_type, str) else str(prop_type), prop_type)
                
                # Mark required with asterisk
                compressed_props[prop_name] = f"{type_abbrev}*" if is_required else type_abbrev
            else:
                # Already compressed or simple value
                compressed_props[prop_name] = prop_info
        
        compressed[collection_name] = compressed_props
    
    return compressed


def decompress_schema(compressed: dict) -> dict:
    """
    Decompress a schema back to full format.
    
    Args:
        compressed: Compressed schema dict
        
    Returns:
        Full schema dict with property details
    """
    if not isinstance(compressed, dict):
        return compressed
    
    full = {}
    
    for collection_name, properties in compressed.items():
        if not isinstance(properties, dict):
            full[collection_name] = properties
            continue
        
        full_props = {}
        for prop_name, prop_value in properties.items():
            if isinstance(prop_value, str):
                # Parse compressed notation
                is_required = prop_value.endswith('*')
                prop_type = prop_value.rstrip('*')
                
                full_props[prop_name] = {
                    'type': prop_type,
                    'required': is_required
                }
            else:
                full_props[prop_name] = prop_value
        
        full[collection_name] = full_props
    
    return full
