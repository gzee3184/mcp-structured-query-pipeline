"""Local sentence-transformer embeddings for collection discovery.

Preferred model: all-MiniLM-L6-v2 (80MB, runs locally without API).
Falls back to OpenAI API, then to a character-frequency pseudo-embedding.
"""

import os
import json
import numpy as np
from typing import Optional
from functools import lru_cache

# Try to import sentence-transformers (preferred)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Fallback to openai for embeddings
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Global model instance (lazy loaded)
_embedding_model = None


def get_local_model():
    """Get or create the local embedding model."""
    global _embedding_model
    if _embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        # all-MiniLM-L6-v2 is fast and good quality (80MB)
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def get_embedding_client():
    """Get OpenAI client configured for embeddings."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return openai.OpenAI(api_key=api_key, base_url=base_url)


@lru_cache(maxsize=2000)
def get_embedding(text: str, model: str = "local") -> tuple:
    """
    Get embedding for a text string.
    
    Args:
        text: Text to embed
        model: "local" for sentence-transformers, "openai" for API
    
    Returns:
        Tuple of floats (hashable for caching)
    """
    # Try local model first (preferred)
    if SENTENCE_TRANSFORMERS_AVAILABLE and model == "local":
        try:
            local_model = get_local_model()
            embedding = local_model.encode(text, convert_to_numpy=True)
            return tuple(embedding.tolist())
        except Exception as e:
            print(f"Local embedding error: {e}")
    
    # Fallback to OpenAI API
    if OPENAI_AVAILABLE and model != "local":
        try:
            client = get_embedding_client()
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return tuple(response.data[0].embedding)
        except Exception as e:
            print(f"API embedding error: {e}")
    
    # Final fallback: simple hash-based embedding
    return tuple(_simple_embedding(text))


def _simple_embedding(text: str, dim: int = 256) -> list[float]:
    """
    Fallback: Create a simple pseudo-embedding based on character frequencies.
    Not as good as real embeddings but works without API.
    """
    text = text.lower()
    # Create embedding based on character n-grams
    embedding = [0.0] * dim
    for i, char in enumerate(text):
        idx = hash(char + str(i % 10)) % dim
        embedding[idx] += 1.0
    
    # Normalize
    norm = sum(x*x for x in embedding) ** 0.5
    if norm > 0:
        embedding = [x / norm for x in embedding]
    return embedding


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def embed_collection_schema(collection: dict) -> dict:
    """
    Add embeddings to a collection schema.
    
    Embeds:
    1. Collection description
    2. Each property (name + type + description)
    
    Returns collection with 'embeddings' field added.
    """
    embeddings = {}
    
    # Embed collection description
    description = collection.get('envisioned_use_case_overview', '') or collection.get('description', '')
    if description:
        embeddings['description'] = get_embedding(description)
    
    # Embed each property
    property_embeddings = {}
    for prop in collection.get('properties', []):
        prop_name = prop.get('name', '')
        prop_type = prop.get('data_type', ['unknown'])[0] if isinstance(prop.get('data_type'), list) else prop.get('data_type', 'unknown')
        prop_desc = prop.get('description', '')
        
        # Create rich text for embedding
        prop_text = f"{prop_name.replace('_', ' ')}: {prop_type} property. {prop_desc}"
        property_embeddings[prop_name] = get_embedding(prop_text)
    
    embeddings['properties'] = property_embeddings
    
    return embeddings


def embed_query(query: str) -> list[float]:
    """Embed a natural language query."""
    return get_embedding(query)


def score_collection_by_embedding(
    query_embedding: list[float],
    collection_embeddings: dict,
    alpha: float = 0.3
) -> float:
    """
    Score a collection based on embedding similarity.
    
    Args:
        query_embedding: Embedding of the user query
        collection_embeddings: Dict with 'description' and 'properties' embeddings
        alpha: Weight for description vs properties (0=properties only, 1=description only)
    
    Returns:
        Similarity score (0-1)
    """
    scores = []
    
    # Score against description
    if 'description' in collection_embeddings:
        desc_sim = cosine_similarity(query_embedding, collection_embeddings['description'])
        scores.append(('description', desc_sim, alpha))
    
    # Score against each property (take max)
    if 'properties' in collection_embeddings:
        prop_scores = []
        for prop_name, prop_embedding in collection_embeddings['properties'].items():
            sim = cosine_similarity(query_embedding, prop_embedding)
            prop_scores.append(sim)
        
        if prop_scores:
            max_prop_sim = max(prop_scores)
            scores.append(('properties', max_prop_sim, 1 - alpha))
    
    # Weighted combination
    if not scores:
        return 0.0
    
    total_weight = sum(w for _, _, w in scores)
    if total_weight == 0:
        return 0.0
    
    return sum(sim * w for _, sim, w in scores) / total_weight


# Test function
if __name__ == "__main__":
    # Test with simple embedding
    print("Testing embedding utilities...")
    
    text1 = "Find games with sales over 10000"
    text2 = "num_sales: number property. Number of sales in region"
    text3 = "game_name: text property. Name of the game"
    
    emb1 = _simple_embedding(text1)
    emb2 = _simple_embedding(text2)
    emb3 = _simple_embedding(text3)
    
    print(f"Query: '{text1}'")
    print(f"Similarity to 'num_sales': {cosine_similarity(emb1, emb2):.4f}")
    print(f"Similarity to 'game_name': {cosine_similarity(emb1, emb3):.4f}")
