zero_shot_baseline = ""

weaviate_query_api_docs = """
The Database Query Tool provides a flexible interface for querying collections within a database. It supports various operations including full-text search, filtering, aggregations, and grouping. This tool is designed to handle different data types (integer, text, and boolean) with type-specific operations.

Basic Usage
At minimum, each query must specify a collection_name. All other parameters are optional and can be combined to create complex queries.

Core Parameters and Search:
The required parameter is collection_name, which specifies which collection to query. The search_query parameter is ESSENTIAL for finding items based on descriptive terms or phrases - you must use it whenever searching for descriptive qualities of items. Never use text filters (LIKE) for descriptive searches. The groupby_property parameter allows grouping results by a specified property.

CRITICAL: Search vs. Filters
1. ALWAYS use search_query for:
   - Any descriptive terms ("romantic", "cozy", "relaxing")
   - Atmosphere descriptions ("romantic atmosphere", "cozy ambiance")
   - Restaurant types ("brunch spots", "dining locations")
   - Amenities ("outdoor seating")
   - Special characteristics ("vegan-friendly")
   - Combinations of these ("romantic Italian restaurants", "cozy dining spots")

2. NEVER use text filters (LIKE operator) for:
   - Descriptive terms
   - Atmosphere
   - Restaurant types
   - Amenities
   - Special characteristics

3. Use filters ONLY for:
   - Exact numeric comparisons (rating > 4)
   - Exact property matching
   - Boolean conditions

Correct Examples:
✓ search_query: "romantic Italian restaurants"
✓ search_query: "vegan-friendly brunch spots"
✓ search_query: "romantic dining locations"
✓ search_query: "restaurants with relaxing atmosphere"

Incorrect Examples:
✗ text_filter: description LIKE "romantic"
✗ text_filter: description LIKE "vegan"
✗ text_filter: description LIKE "relaxing"
✗ boolean_filter: openNow = True (when calculating percentages)

Key Points About Aggregations:
1. When asked about "how many", use COUNT aggregation
2. When asked about percentages of boolean properties, use PERCENTAGE_TRUE aggregation
3. When asked about "most common" or "typical" text features, use TOP_OCCURRENCES
4. When asked about averages, always include the appropriate MEAN aggregation

Property Usage Rules:
1. For cuisine grouping, use "description.cuisine" as the property
2. For open/closed status:
   - Use boolean_aggregation with PERCENTAGE_TRUE when calculating percentages
   - Use groupby: "openNow" when grouping results
   - Don't use boolean filters unless specifically filtering, not aggregating

Aggregation Operations
The tool provides sophisticated aggregation capabilities for different data types:
Integer Aggregations: Use integer_property_aggregation for numeric analysis. Available metrics: MIN, MAX, MEAN, MEDIAN, MODE, SUM (NOTE: COUNT is NOT valid for aggregations - use total_count=true instead)
Text Aggregations: Use text_property_aggregation for text analysis. Available metrics: TOP_OCCURRENCES
Boolean Aggregations: Use boolean_property_aggregation for boolean statistics. Available metrics: TOTAL_TRUE, TOTAL_FALSE, PERCENTAGE_TRUE, PERCENTAGE_FALSE
"""

# =============================================================================
# PROGRESSIVE DISCLOSURE PROMPT
# =============================================================================
# For MCP-style schema discovery in large-scale deployments (50-100+ collections)
# This enables efficient token usage by loading schemas on-demand

progressive_disclosure_prompt = """
PROGRESSIVE SCHEMA DISCOVERY

For databases with many collections (10+), use this efficient discovery approach:

PHASE 1 - EXPLORATION:
Instead of loading all schemas upfront, discover relevant collections step by step:

1. search_collections(query, detail_level="name")
   - Returns only collection names matching your query
   - Use this first to narrow down candidates
   - Example: search_collections("restaurant reviews", detail_level="name")

2. search_collections(query, detail_level="summary") 
   - Returns names + brief descriptions
   - Use to verify collection relevance

3. get_collection_schema(collection_name)
   - Returns full schema for a specific collection
   - Only load schemas you actually need

4. get_property_details(collection, property)
   - Returns detailed info about a specific property
   - Use for validating filter/aggregation compatibility

PHASE 2 - QUERY GENERATION:
Once you have the relevant schema(s), construct your query using only the loaded information.

TOKEN EFFICIENCY TIPS:
- Start with detail_level="name" to minimize initial token usage
- Only request full schemas for collections you will query
- Cache frequently-used schemas for repeated queries
- For simple queries, summary-level may be sufficient

FALLBACK BEHAVIOR:
If you cannot find a matching collection after exploring:
1. Try broader search terms
2. List all available collections
3. If still unsuccessful, use terminate_and_respond with a helpful suggestion
"""

# =============================================================================
# STRUCTURED OUTPUT PROMPT
# =============================================================================
# For models that struggle with native structured output (Ollama, NVIDIA NIM)

structured_output_prompt = """
STRUCTURED JSON OUTPUT REQUIREMENTS

You MUST respond with valid JSON matching the specified schema.

RULES:
1. Output ONLY the JSON object, no explanations before or after
2. All required fields must be present
3. Use exact field names as shown in the schema
4. Wrap your response in ```json and ``` markers

EXAMPLE FORMAT:
```json
{
  "collection_name": "Restaurants",
  "search_query": "romantic Italian",
  "integer_property_filter": {
    "property_name": "rating",
    "operator": ">",
    "value": 4.0
  }
}
```

VALIDATION:
Before responding, verify:
✓ All required fields are present
✓ Field names match schema exactly (case-sensitive)
✓ Data types are correct (strings in quotes, numbers without)
✓ JSON is properly formatted (commas, brackets, colons)
"""