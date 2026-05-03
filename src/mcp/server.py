"""Schema registry and collection discovery for 143+ Weaviate collections.

Implements progressive disclosure: only the schemas relevant to a query are
loaded into the LLM context, saving ~90% of tokens vs. sending everything.
"""

from typing import Literal, Optional
from pathlib import Path
import json

from src.utils.schema_cache import SchemaCache, compress_schema

# Optional: semantic embeddings for improved discovery
try:
    from src.utils.embeddings import (
        embed_query, cosine_similarity, _simple_embedding
    )
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class MCPServer:
    """
    MCP Server with schema registry and discovery tools.
    
    Implements progressive disclosure to reduce context window usage:
    - search_collections: Find relevant collections by name/description
    - get_collection_schema: Load full schema for a specific collection
    - get_property_details: Get detailed info about a property
    - validate_aggregation: Check if an aggregation is valid
    
    Example:
        >>> server = MCPServer.from_queries_file("data/weaviate-gorilla.json")
        >>> server.search_collections("restaurant", detail_level="name")
        ['Restaurants', 'Menus', 'Reservations']
    """
    
    def __init__(self, schema_registry: dict[str, dict], enable_semantic: bool = False):
        """
        Initialize MCP server with schema registry.
        
        Args:
            schema_registry: Dict mapping collection names to their schemas
                Example: {"Restaurants": {"description": "...", "properties": {...}}}
            enable_semantic: If True, use semantic embeddings for improved discovery
        """
        self.schemas = schema_registry
        self.cache = SchemaCache()
        self.enable_semantic = enable_semantic and EMBEDDINGS_AVAILABLE
        
        # Index for faster searching
        self._collection_names = list(schema_registry.keys())
        
        # Embeddings cache for semantic search
        self._embeddings = {}
        if self.enable_semantic:
            self._compute_embeddings()
    
    @classmethod
    def from_queries_file(cls, queries_file: str) -> "MCPServer":
        """
        Create MCPServer by loading schemas from a queries JSON file.
        
        Args:
            queries_file: Path to weaviate-gorilla.json or similar
        """
        from src.utils.load_queries import load_queries
        
        queries = load_queries(queries_file)
        registry = {}
        
        for query in queries:
            for coll in query.database_schema.weaviate_collections:
                if coll.name not in registry:
                    registry[coll.name] = {
                        "description": getattr(coll, 'envisioned_use_case_overview', 'No description'),
                        "properties": {}
                    }
                    for prop in coll.properties:
                        # Handle data_type which can be a list
                        dtype = prop.data_type
                        if isinstance(dtype, list):
                            dtype = dtype[0] if dtype else "unknown"
                        
                        registry[coll.name]["properties"][prop.name] = {
                            "type": dtype,
                            "description": getattr(prop, 'description', 'No description')
                        }
        
        return cls(registry)
    
    @classmethod
    def from_multiple_sources(cls, *sources: str) -> "MCPServer":
        """Merge schemas from Weaviate Gorilla, BIRD, and extra JSON files into one registry.

        Args:
            *sources: Paths to schema files (weaviate-gorilla.json, bird-to-weaviate.json, etc.)

        Example:
            server = MCPServer.from_multiple_sources(
                "data/weaviate-gorilla.json",
                "data/bird-processor/bird-to-weaviate.json"
            )
        """
        combined_registry = {}
        
        for source in sources:
            if source.endswith('.json'):
                import json
                with open(source) as f:
                    data = json.load(f)
                
                # Check if it's a queries file or direct schema file
                if 'weaviate_collections' in data:
                    # Direct schema file (like bird-to-weaviate.json)
                    for coll in data['weaviate_collections']:
                        combined_registry[coll['name']] = {
                            'description': coll.get('envisioned_use_case_overview', ''),
                            'properties': {
                                p['name']: {
                                    'type': p['data_type'][0] if isinstance(p['data_type'], list) else p['data_type'],
                                    'description': p.get('description', '')
                                }
                                for p in coll['properties']
                            }
                        }
                else:
                    # Queries file - use existing loader
                    temp_server = cls.from_queries_file(source)
                    combined_registry.update(temp_server.schemas)
        
        return cls(combined_registry)
    
    def register_collection(self, name: str, schema: dict) -> None:
        """
        Register a new collection at runtime.
        
        Args:
            name: Collection name
            schema: Dict with 'description' and 'properties' keys
        
        Example:
            server.register_collection("Products", {
                "description": "E-commerce products",
                "properties": {
                    "productName": {"type": "text", "description": "Product name"},
                    "price": {"type": "number", "description": "Price in USD"}
                }
            })
        """
        self.schemas[name] = schema
        self._collection_names = list(self.schemas.keys())
        # Recompute embeddings for new collection
        if self.enable_semantic:
            self._compute_embedding_for_collection(name, schema)
    
    def _compute_embeddings(self):
        """Pre-compute embeddings for all collections."""
        for name, schema in self.schemas.items():
            self._compute_embedding_for_collection(name, schema)
    
    def _compute_embedding_for_collection(self, name: str, schema: dict):
        """Compute embeddings for a single collection."""
        # Create rich text representation of collection
        desc = schema.get('description', '') or ''
        props = schema.get('properties', {})
        
        # Combine collection name, description, and property names
        prop_text = ' '.join([
            f"{p.replace('_', ' ')} {info.get('type', '')}" 
            for p, info in props.items()
        ])
        full_text = f"{name.replace('_', ' ')}: {desc}. Properties: {prop_text}"
        
        # Use simple embedding (fast, no API call)
        self._embeddings[name] = _simple_embedding(full_text)
    
    def search_collections(
        self,
        query: str,
        detail_level: Literal["name", "summary", "full"] = "name"
    ) -> list | dict:
        """
        Search for collections matching a query.
        
        Uses entity-priority schema linking (inspired by BIRD/SPIDER2.0):
        1. Exact collection name matches ranked first
        2. Word-level matches ranked second
        3. Description matches ranked third
        
        Args:
            query: Search term (case-insensitive), can be multiple words
            detail_level: 
                - "name": Return just names (~20 tokens)
                - "summary": Return names + descriptions (~100 tokens)
                - "full": Return full schemas (~500 tokens)
        
        Returns:
            List of names or dict with details depending on detail_level
        """
        query_lower = query.lower()
        
        # Tokenize query into individual words for better matching
        query_words = [w.strip() for w in query_lower.split() if len(w.strip()) > 2]
        
        # Score-based matching for better ranking
        scored_matches = []
        
        for name, schema in self.schemas.items():
            name_lower = name.lower()
            desc_lower = schema.get("description", "").lower()
            score = 0
            
            # Priority 1: Exact collection name appears in query (highest priority)
            # e.g., "find doctors" → "Doctors" gets high score
            if name_lower in query_lower:
                score += 100
            
            # Priority 2: Collection name word matches query word
            # e.g., "restaurant" matches "Restaurants"  
            for word in query_words:
                if word in name_lower or name_lower.startswith(word):
                    score += 50
                # Plural/singular handling
                if word.rstrip('s') in name_lower or name_lower.rstrip('s') == word:
                    score += 40
            
            # Priority 2.5: Property name matches query word (NEW - from BIRD research)
            # e.g., "sales" matches "num_sales" property → boost RegionSales collection
            for prop_name in schema.get("properties", {}).keys():
                prop_lower = prop_name.lower().replace("_", "")  # "num_sales" -> "numsales"
                prop_words = prop_name.lower().replace("_", " ").split()  # ["num", "sales"]
                for word in query_words:
                    if word in prop_lower:
                        score += 30
                    # Check individual property words
                    for prop_word in prop_words:
                        if word == prop_word or (len(word) > 3 and word in prop_word):
                            score += 25
            
            # Priority 3: Query words match description
            for word in query_words:
                if word in desc_lower:
                    score += 10
            
            if score > 0:
                scored_matches.append((name, score))
        
        # Priority 4: Semantic embedding similarity (NEW - QueReyDB-inspired)
        if self.enable_semantic and self._embeddings:
            query_embedding = _simple_embedding(query)
            
            for name in self.schemas.keys():
                if name in self._embeddings:
                    similarity = cosine_similarity(query_embedding, self._embeddings[name])
                    # Scale similarity (0-1) to score points (0-80)
                    semantic_score = int(similarity * 80)
                    
                    # Find existing score for this collection
                    found = False
                    for i, (n, s) in enumerate(scored_matches):
                        if n == name:
                            scored_matches[i] = (n, s + semantic_score)
                            found = True
                            break
                    
                    # Add if not already in list (no keyword matches but semantic match)
                    if not found and semantic_score > 10:  # threshold
                        scored_matches.append((name, semantic_score))
        
        # Sort by score (highest first) and extract names
        scored_matches.sort(key=lambda x: x[1], reverse=True)
        matches = [name for name, score in scored_matches]
        
        if detail_level == "name":
            return matches
        elif detail_level == "summary":
            return {
                name: self.schemas[name].get("description", "")[:100]
                for name in matches
            }
        else:  # full
            return {
                name: compress_schema({name: self.schemas[name]})
                for name in matches
            }
    
    def get_collection_schema(
        self,
        collection_name: str,
        compressed: bool = True
    ) -> Optional[dict]:
        """
        Get the full schema for a specific collection.
        
        Args:
            collection_name: Name of the collection
            compressed: If True, return compressed schema (~60% smaller)
        
        Returns:
            Schema dict or None if not found
        """
        if collection_name not in self.schemas:
            return None
        
        schema = self.schemas[collection_name]
        
        if compressed:
            return compress_schema({collection_name: schema})[collection_name]
        return schema
    
    def get_property_details(
        self,
        collection_name: str,
        property_name: str
    ) -> Optional[dict]:
        """
        Get detailed information about a specific property.
        
        Args:
            collection_name: Name of the collection
            property_name: Name of the property
        
        Returns:
            Property details including type, valid operators, valid aggregations
        """
        if collection_name not in self.schemas:
            return None
        
        props = self.schemas[collection_name].get("properties", {})
        if property_name not in props:
            return None
        
        prop_info = props[property_name]
        prop_type = prop_info.get("type", "unknown")
        
        # Determine valid operations based on type
        if prop_type in ["int", "integer", "number", "float"]:
            return {
                "name": property_name,
                "type": prop_type,
                "valid_operators": ["=", "<", ">", "<=", ">="],
                "valid_aggregations": ["MIN", "MAX", "MEAN", "MEDIAN", "MODE", "SUM"]
            }
        elif prop_type in ["text", "string"]:
            return {
                "name": property_name,
                "type": prop_type,
                "valid_operators": ["=", "LIKE"],
                "valid_aggregations": ["TOP_OCCURRENCES"]
            }
        elif prop_type in ["bool", "boolean"]:
            return {
                "name": property_name,
                "type": prop_type,
                "valid_operators": ["=", "!="],
                "valid_aggregations": ["TOTAL_TRUE", "TOTAL_FALSE", "PERCENTAGE_TRUE", "PERCENTAGE_FALSE"]
            }
        else:
            return {
                "name": property_name,
                "type": prop_type,
                "valid_operators": [],
                "valid_aggregations": []
            }
    
    def validate_aggregation(
        self,
        collection_name: str,
        property_name: str,
        metric: str
    ) -> dict:
        """
        Validate if an aggregation metric is valid for a property.
        
        Args:
            collection_name: Name of the collection
            property_name: Name of the property
            metric: Aggregation metric to validate
        
        Returns:
            {"valid": bool, "reason": str | None, "suggestion": str | None}
        """
        prop_details = self.get_property_details(collection_name, property_name)
        
        if prop_details is None:
            return {
                "valid": False,
                "reason": f"Property '{property_name}' not found in '{collection_name}'",
                "suggestion": None
            }
        
        valid_aggs = prop_details.get("valid_aggregations", [])
        
        if metric.upper() in valid_aggs:
            return {"valid": True, "reason": None, "suggestion": None}
        
        # Common mistakes
        if metric.upper() == "COUNT":
            return {
                "valid": False,
                "reason": "COUNT is not a valid aggregation metric. Use total_count parameter instead.",
                "suggestion": "total_count=True"
            }
        
        return {
            "valid": False,
            "reason": f"'{metric}' is not valid for {prop_details['type']} properties",
            "suggestion": valid_aggs[0] if valid_aggs else None
        }
    
    def list_collections(self) -> list[str]:
        """Return all collection names."""
        return self._collection_names
    
    def stats(self) -> dict:
        """Return server statistics."""
        total_props = sum(
            len(s.get("properties", {})) for s in self.schemas.values()
        )
        return {
            "total_collections": len(self.schemas),
            "total_properties": total_props,
            "collection_names": self._collection_names
        }
