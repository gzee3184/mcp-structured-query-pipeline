"""Tool schema builders for LLM function-calling across providers.

V1 schema: single filter per type, no output_properties scope, no ORDER BY.
V2 schema (--tool-schema v2): decouples output_properties (SELECT) from the
primary collection, supports multi-predicate filters[], order_by[], DISTINCT,
LIMIT, and HAVING. V2 enables the LLM to express richer SQL-equivalent queries.

Each provider (OpenAI, Anthropic, Ollama, Cohere, Together) has its own tool
format, plus circuit-breaker tools to prevent infinite retry loops.
"""

import weaviate
from src.models import (
    IntPropertyFilter,
    TextPropertyFilter,
    BooleanPropertyFilter,
    IntAggregation,
    TextAggregation,
    BooleanAggregation,
    GroupBy
)
from src.models import (
    OpenAIParameters,
    OpenAIFunction,
    OpenAITool,
    AnthropicTool,
    AnthropicToolInputSchema,
    OllamaFunctionParameters,
    OllamaFunction,
    OllamaTool,
    CohereFunctionParameters,
    CohereFunction,
    CohereTool,
    TogetherAITool,
    TogetherAIFunction,
    TogetherAIParameters
)
import re
from typing import Tuple, Union, Any, Dict, List
from pydantic import BaseModel
from typing import Literal, Optional

# def get_collections_info(client):
#     """Get information about collections for building tools."""
#     try:
#         # Get the schema using schema.get() instead of collections.get()
#         schema = client.schema.get()
        
#         # Extract collection names
#         collections_enum = []
#         collections_description = "Available collections:\n"
        
#         if 'classes' in schema:
#             for cls in schema['classes']:
#                 collection_name = cls['class']
#                 collections_enum.append(collection_name)
                
#                 # Add collection info to description
#                 collections_description += f"- {collection_name}: {cls.get('description', 'No description')}\n"
#                 collections_description += "  Properties:\n"
                
#                 # Add property info to description
#                 for prop in cls.get('properties', []):
#                     prop_name = prop['name']
#                     prop_type = prop['dataType']
#                     prop_desc = prop.get('description', 'No description')
#                     collections_description += f"  - {prop_name} ({prop_type}): {prop_desc}\n"
                
#                 collections_description += "\n"
        
#         return collections_description, collections_enum
    
#     except Exception as e:
#         print(f"Error getting collections info: {str(e)}")
#         return "Error retrieving schema", []

def get_collections_info(client):
    """Get information about collections for building tools (Weaviate v4 Client compatible)."""
    try:
        # v4 Client uses client.collections.list_all() instead of client.schema.get()
        collections = client.collections.list_all()
        
        collections_enum = []
        collections_description = "Available collections:\n"
        
        for collection_name, config in collections.items():
            collections_enum.append(collection_name)
            description = config.description if config.description else "No description"
            collections_description += f"- {collection_name}: {description}\n"
            collections_description += "  Properties:\n"
            
            # v4 Access to properties
            for prop in config.properties:
                prop_name = prop.name
                prop_type = str(prop.data_type) # Simplification for v4 types
                collections_description += f"  - {prop_name} ({prop_type})\n"
            
            collections_description += "\n"
        
        return collections_description, collections_enum
    
    except Exception as e:
        print(f"Error getting collections info: {str(e)}")
        # Fallback for v3 client if needed, or return empty
        return "Error retrieving schema", []

def build_weaviate_query_tool_for_openai(collections_description: str, collections_list: list[str]) -> OpenAITool:
    properties = {
        "collection_name": {
            "type": "string",
            "description": "The name of the collection to query. Must be EXACT CASE-SENSITIVE, matching one of the enum values.",
            "enum": collections_list
        },
        "search_query": {
            "type": "string",
            "description": "A search query to return objects from a search index."
        },
        "integer_property_filter": {
            "type": "object",
            "description": "Filter numeric properties. Must include property_name, operator, and value.",
            "properties": {
                "property_name": {"type": "string"},
                "operator": {"type": "string", "enum": ["=", "<", ">", "<=", ">="]},
                "value": {"type": "number"}
            }
        },
        "text_property_filter": {
            "type": "object", 
            "description": "Filter text properties. Must include property_name, operator, and value.",
            "properties": {
                "property_name": {"type": "string"},
                "operator": {"type": "string", "enum": ["=", "LIKE"]},
                "value": {"type": "string"}
            }
        },
        "boolean_property_filter": {
            "type": "object",
            "description": "Filter boolean properties.",
            "properties": {
                "property_name": {"type": "string"},
                "operator": {"type": "string", "enum": ["=", "!="]},
                "value": {"type": "boolean"}
            }
        },
        "integer_property_aggregation": {
            "type": "object",
            "description": "Aggregate numeric properties. DO NOT USE 'COUNT' here.",
            "properties": {
                "property_name": {"type": "string"},
                "metrics": {"type": "string", "enum": ["MIN", "MAX", "MEAN", "MEDIAN", "MODE", "SUM"]}
            }
        },
        "text_property_aggregation": {
            "type": "object",
            "description": "Aggregate text properties.",
            "properties": {
                "property_name": {"type": "string"},
                "metrics": {"type": "string", "enum": ["TOP_OCCURRENCES"]},
                "top_occurrences_limit": {"type": "integer"}
            }
        },
        "boolean_property_aggregation": {
            "type": "object",
            "description": "Aggregate boolean properties.",
            "properties": {
                "property_name": {"type": "string"},
                "metrics": {"type": "string", "enum": ["TOTAL_TRUE", "TOTAL_FALSE", "PERCENTAGE_TRUE", "PERCENTAGE_FALSE"]}
            }
        },
        "groupby_property": {
            "type": "string",
            "description": "Group the results by a property."
        },
        "additional_collections": {
            "type": "array",
            "description": "For queries requiring data from MULTIPLE collections (e.g., JOIN-like queries spanning several tables), list ALL additional collections needed beyond the primary collection_name. Each entry specifies a collection and optionally its role. IMPORTANT: Include this whenever the question references data that spans multiple tables/collections.",
            "items": {
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "The EXACT CASE-SENSITIVE name of an additional collection needed.",
                        "enum": collections_list
                    },
                    "role": {
                        "type": "string",
                        "description": "Brief description of why this collection is needed (e.g., 'contains student enrollment records', 'has course details')."
                    }
                },
                "required": ["collection_name"]
            }
        },
        "join_keys": {
            "type": "array",
            "description": "Foreign key relationships connecting the collections. Include when multiple collections are used to show how they relate.",
            "items": {
                "type": "object",
                "properties": {
                    "left_collection": {"type": "string", "description": "First collection name"},
                    "left_property": {"type": "string", "description": "Property in the first collection (the FK or shared key)"},
                    "right_collection": {"type": "string", "description": "Second collection name"},
                    "right_property": {"type": "string", "description": "Property in the second collection"}
                },
                "required": ["left_collection", "left_property", "right_collection", "right_property"]
            }
        }
    }

    return OpenAITool(
        type="function",
        function=OpenAIFunction(
            name="query_database",
            description=f"Query a database. Available collections:\n{collections_description}",
            parameters=OpenAIParameters(
                type="object",
                properties=properties,
                required=["collection_name"]
            )
        )
    )


def build_weaviate_query_tool_for_openai_v2(collections_description: str, collections_list: list[str]) -> OpenAITool:
    """V2 tool schema: decouples output scope from primary collection, adds
    multi-predicate filters, ORDER BY, DISTINCT, and explicit LIMIT.

    Shipped behind --tool-schema=v2. Returns an OpenAITool compatible with the
    same LMService.one_step_function_selection_test() path as V1.
    """
    agg_enum = [
        "NONE", "COUNT", "SUM", "MIN", "MAX", "MEAN", "MEDIAN", "MODE",
        "TOP_OCCURRENCES", "PERCENTAGE_TRUE", "PERCENTAGE_FALSE",
        "TOTAL_TRUE", "TOTAL_FALSE",
    ]
    filter_ops = [
        "=", "!=", "<", ">", "<=", ">=", "LIKE", "IN", "BETWEEN",
        "IS_NULL", "IS_NOT_NULL",
    ]
    property_types = ["integer", "text", "boolean", "date"]

    properties = {
        "collection_name": {
            "type": "string",
            "description": (
                "The PRIMARY collection the query iterates over. This is the "
                "collection that acts as the base set for WHERE and GROUP BY. "
                "It is NOT necessarily the same collection whose properties "
                "you RETURN — for that, use output_properties. "
                "Must be EXACT CASE-SENSITIVE, matching one of the enum values."
            ),
            "enum": collections_list,
        },
        "additional_collections": {
            "type": "array",
            "description": (
                "For JOIN-like queries, list other collections needed beyond "
                "collection_name. Each entry names a collection and its role."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "EXACT CASE-SENSITIVE collection name.",
                        "enum": collections_list,
                    },
                    "role": {
                        "type": "string",
                        "description": "Why this collection is needed (e.g., 'holds filter values', 'provides output columns').",
                    },
                },
                "required": ["collection_name"],
            },
        },
        "join_keys": {
            "type": "array",
            "description": "Foreign-key relationships between collections used in the query.",
            "items": {
                "type": "object",
                "properties": {
                    "left_collection": {"type": "string"},
                    "left_property": {"type": "string"},
                    "right_collection": {"type": "string"},
                    "right_property": {"type": "string"},
                },
                "required": ["left_collection", "left_property", "right_collection", "right_property"],
            },
        },
        "output_properties": {
            "type": "array",
            "description": (
                "The properties the query RETURNS — the SELECT clause. "
                "List every column the question asks for, and the collection it belongs to. "
                "For JOIN queries, output properties may come from multiple collections. "
                "If a property is on the primary collection_name you may omit the collection field. "
                "For aggregated outputs (COUNT, SUM, etc.), set the aggregation field. "
                "Examples:\n"
                "  - 'list patients' IDs and diagnosis' -> "
                "[{collection:'Patient', property_name:'id'}, {collection:'Patient', property_name:'diagnosis'}]\n"
                "  - 'count customers' -> "
                "[{collection:'Customer', property_name:'id', aggregation:'COUNT'}]\n"
                "  - 'sum of points by constructor' -> "
                "[{collection:'Constructor', property_name:'name'}, {collection:'ConstructorResults', property_name:'points', aggregation:'SUM'}]"
            ),
            "items": {
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Collection owning this output property. Omit if same as collection_name.",
                    },
                    "property_name": {
                        "type": "string",
                        "description": "Exact property name as defined in the schema.",
                    },
                    "aggregation": {
                        "type": "string",
                        "enum": agg_enum,
                        "description": "If the column is aggregated in the question, specify the function; otherwise NONE.",
                    },
                },
                "required": ["property_name"],
            },
        },
        "filters": {
            "type": "array",
            "description": (
                "WHERE-clause predicates. All entries are combined using filter_boolean_op "
                "(default AND). Each filter is scoped to a specific collection — this matters "
                "for JOIN queries where a filter may apply to a non-primary collection. "
                "Use one filter per predicate; do NOT combine multiple predicates into one entry. "
                "Examples:\n"
                "  - 'WHERE albumin<3.5' -> [{collection:'Laboratory', property_name:'albumin', operator:'<', value:3.5, property_type:'integer'}]\n"
                "  - 'WHERE SEX=\"F\" AND ALB<3.5' -> two filter entries with filter_boolean_op='AND'\n"
                "  - 'WHERE id IN [1,2,3]' -> operator='IN', value=[1,2,3]\n"
                "  - 'WHERE date BETWEEN a AND b' -> operator='BETWEEN', value=[a,b]"
            ),
            "items": {
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Collection owning the filtered property. Omit if same as collection_name.",
                    },
                    "property_name": {"type": "string"},
                    "operator": {"type": "string", "enum": filter_ops},
                    "value": {
                        "description": (
                            "The comparison value. Scalar for =/!=/</>/<=/>=/LIKE, "
                            "array [a,b,...] for IN, array [lo,hi] for BETWEEN, "
                            "omitted for IS_NULL/IS_NOT_NULL."
                        ),
                    },
                    "property_type": {"type": "string", "enum": property_types},
                },
                "required": ["property_name", "operator"],
            },
        },
        "filter_boolean_op": {
            "type": "string",
            "enum": ["AND", "OR"],
            "description": "How to combine entries in filters. Default AND.",
        },
        "group_by_properties": {
            "type": "array",
            "description": "GROUP BY clause — typically needed when output_properties contains aggregated entries alongside non-aggregated ones.",
            "items": {"type": "string"},
        },
        "having_filters": {
            "type": "array",
            "description": (
                "HAVING clause — post-aggregation filters. Parallel in shape to `filters` "
                "but applied AFTER the GROUP BY. Use for predicates on aggregated "
                "values (e.g., 'HAVING COUNT(*) > 5'). "
                "Examples: [{property_name:'id', operator:'>', value:5, aggregation:'COUNT'}]"
            ),
            "items": {
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Collection the aggregated property belongs to.",
                    },
                    "property_name": {"type": "string"},
                    "operator": {"type": "string", "enum": filter_ops},
                    "value": {"description": "Comparison value (same semantics as filters)."},
                    "aggregation": {
                        "type": "string",
                        "enum": agg_enum,
                        "description": "If the HAVING predicate is over an aggregate (e.g. HAVING COUNT(id) > 5), specify it here.",
                    },
                },
                "required": ["property_name", "operator"],
            },
        },
        "order_by": {
            "type": "array",
            "description": (
                "ORDER BY clause with explicit direction. Supports multi-column ordering. "
                "For ORDER BY on an aggregate (e.g., ORDER BY COUNT(*) DESC), set aggregation."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "property_name": {"type": "string"},
                    "aggregation": {"type": "string", "enum": agg_enum},
                    "direction": {"type": "string", "enum": ["ASC", "DESC"]},
                },
                "required": ["property_name", "direction"],
            },
        },
        "limit": {
            "type": "integer",
            "description": "LIMIT clause — number of rows to return. Use together with order_by for top-N queries.",
        },
        "distinct": {
            "type": "boolean",
            "description": "SELECT DISTINCT — deduplicate rows by output_properties.",
        },
        "search_query": {
            "type": "string",
            "description": "Optional semantic search query. Use only when the question asks for similarity/semantic matching.",
        },
    }

    return OpenAITool(
        type="function",
        function=OpenAIFunction(
            name="query_database",
            description=f"Query a database. Available collections:\n{collections_description}",
            parameters=OpenAIParameters(
                type="object",
                properties=properties,
                required=["collection_name", "output_properties"],
            ),
        ),
    )


def build_weaviate_query_tool_for_ollama(collections_description: str, collections_list: list[str]) -> OllamaTool:
    # Similar enhancements for Ollama
    query_parameters = {
        "type": "object",
        "properties": {
            "collection_name": {
                "type": "string",
                "description": "The EXACT CASE-SENSITIVE name of the collection.",
                "enum": collections_list
            },
            # ... (Include the rest of the schema with the same strict descriptions as above)
            # Keeping this brief, but you should copy the logic from the openai tool above
            "integer_property_aggregation": {
                "type": "object",
                "properties": {
                    "property_name": {"type": "string"},
                    "metrics": {"type": "string", "enum": ["MIN", "MAX", "MEAN", "MEDIAN", "MODE", "SUM"]}
                },
                "required": ["property_name", "metrics"]
            },
            # ...
        },
        "required": ["collection_name"]
    }

    query_function = OllamaFunction(
        name="query_database",
        description=f"Query a database. Available collections:\n{collections_description}",
        parameters=query_parameters
    )
    return OllamaTool(function=query_function)

# def build_weaviate_query_tool_for_openai(collections_description: str, collections_list: list[str]) -> OpenAITool:
#     properties = {
#         "collection_name": {
#             "type": "string",
#             "description": "The collection to query.",
#             "enum": collections_list
#         },
#         "search_query": {
#             "type": "string",
#             "description": "A search query to return objects from a search index."
#         },
#         "integer_property_filter": {
#             "type": "object",
#             "description": "Filter numeric properties using comparison operators",
#             "properties": {
#                 "property_name": {"type": "string"},
#                 "operator": {"type": "string", "enum": ["=", "<", ">", "<=", ">="]},
#                 "value": {"type": "number"}
#             }
#         },
#         "text_property_filter": {
#             "type": "object", 
#             "description": "Filter text properties using equality or LIKE operators",
#             "properties": {
#                 "property_name": {"type": "string"},
#                 "operator": {"type": "string", "enum": ["=", "LIKE"]},
#                 "value": {"type": "string"}
#             }
#         },
#         "boolean_property_filter": {
#             "type": "object",
#             "description": "Filter boolean properties using equality operators",
#             "properties": {
#                 "property_name": {"type": "string"},
#                 "operator": {"type": "string", "enum": ["=", "!="]},
#                 "value": {"type": "boolean"}
#             }
#         },
#         "integer_property_aggregation": {
#             "type": "object",
#             "description": "Aggregate numeric properties using statistical functions",
#             "properties": {
#                 "property_name": {"type": "string"},
#                 "metrics": {"type": "string", "enum": ["MIN", "MAX", "MEAN", "MEDIAN", "MODE", "SUM"]}
#             }
#         },
#         "text_property_aggregation": {
#             "type": "object",
#             "description": "Aggregate text properties using frequency analysis",
#             "properties": {
#                 "property_name": {"type": "string"},
#                 "metrics": {"type": "string", "enum": ["TOP_OCCURRENCES"]},
#                 "top_occurrences_limit": {"type": "integer"}
#             }
#         },
#         "boolean_property_aggregation": {
#             "type": "object",
#             "description": "Aggregate boolean properties using statistical functions",
#             "properties": {
#                 "property_name": {"type": "string"},
#                 "metrics": {"type": "string", "enum": ["TOTAL_TRUE", "TOTAL_FALSE", "PERCENTAGE_TRUE", "PERCENTAGE_FALSE"]}
#             }
#         },
#         "groupby_property": {
#             "type": "string",
#             "description": "Group the results by a property."
#         }
#     }

#     return OpenAITool(
#         type="function",
#         function=OpenAIFunction(
#             name="query_database",
#             description=f"""Query a database.

#             Available collections in this database:
#             {collections_description}""",
#             parameters=OpenAIParameters(
#                 type="object",
#                 properties=properties,
#                 required=["collection_name"]
#             )
#         )
#     )

def build_weaviate_query_tool_for_anthropic(collections_description: str, collections_list: list[str]) -> AnthropicTool:
    properties = {
        "collection_name": {
            "type": "string",
            "description": "The collection to query",
            "enum": collections_list
        },
        "search_query": {
            "type": "string",
            "description": "A search query to return objects from a search index."
        },
        "integer_property_filter": {
            "type": "object",
            "description": "Filter numeric properties using comparison operators",
            "properties": {
                "property_name": {"type": "string"},
                "operator": {"type": "string", "enum": ["=", "<", ">", "<=", ">="]},
                "value": {"type": "number"}
            }
        },
        "text_property_filter": {
            "type": "object",
            "description": "Filter text properties using equality or LIKE operators",
            "properties": {
                "property_name": {"type": "string"},
                "operator": {"type": "string", "enum": ["=", "LIKE"]},
                "value": {"type": "string"}
            }
        },
        "boolean_property_filter": {
            "type": "object",
            "description": "Filter boolean properties using equality operators",
            "properties": {
                "property_name": {"type": "string"},
                "operator": {"type": "string", "enum": ["=", "!="]},
                "value": {"type": "boolean"}
            }
        },
        "integer_property_aggregation": {
            "type": "object",
            "description": "Aggregate numeric properties using statistical functions",
            "properties": {
                "property_name": {"type": "string"},
                "metrics": {"type": "string", "enum": ["MIN", "MAX", "MEAN", "MEDIAN", "MODE", "SUM"]}
            }
        },
        "text_property_aggregation": {
            "type": "object",
            "description": "Aggregate text properties using frequency analysis",
            "properties": {
                "property_name": {"type": "string"},
                "metrics": {"type": "string", "enum": ["TOP_OCCURRENCES"]},
                "top_occurrences_limit": {"type": "integer"}
            }
        },
        "boolean_property_aggregation": {
            "type": "object",
            "description": "Aggregate boolean properties using statistical functions",
            "properties": {
                "property_name": {"type": "string"},
                "metrics": {"type": "string", "enum": ["TOTAL_TRUE", "TOTAL_FALSE", "PERCENTAGE_TRUE", "PERCENTAGE_FALSE"]}
            }
        },
        "groupby_property": {
            "type": "string",
            "description": "Group the results by a property."
        }
    }

    return AnthropicTool(
        name="query_database",
        description=f"""Query a database.

        Available collections in this database:
        {collections_description}""",
        input_schema=AnthropicToolInputSchema(
            type="object",
            properties=properties,
            required=["collection_name"]
        )
    )

# def build_weaviate_query_tool_for_ollama(collections_description: str, collections_list: list[str]) -> OllamaTool:
#     query_parameters = {
#         "type": "object",
#         "properties": {
#             "collection_name": {
#                 "type": "string",
#                 "enum": collections_list
#             },
#             "search_query": {
#                 "type": "string"
#             },
#             "integer_property_filter": {
#                 "type": "object",
#                 "properties": {
#                     "property_name": {"type": "string"},
#                     "operator": {"type": "string", "enum": ["=", "<", ">", "<=", ">="]},
#                     "value": {"type": "integer"}
#                 },
#                 "required": ["property_name", "operator", "value"]
#             },
#             "text_property_filter": {
#                 "type": "object", 
#                 "properties": {
#                     "property_name": {"type": "string"},
#                     "operator": {"type": "string", "enum": ["=", "LIKE"]},
#                     "value": {"type": "string"}
#                 },
#                 "required": ["property_name", "operator", "value"]
#             },
#             "boolean_property_filter": {
#                 "type": "object",
#                 "properties": {
#                     "property_name": {"type": "string"},
#                     "operator": {"type": "string", "enum": ["="]},
#                     "value": {"type": "boolean"}
#                 },
#                 "required": ["property_name", "operator", "value"]
#             },
#             "integer_property_aggregation": {
#                 "type": "object",
#                 "properties": {
#                     "property_name": {"type": "string"},
#                     "metrics": {"type": "string", "enum": ["MIN", "MAX", "MEAN", "MEDIAN", "MODE", "SUM"]}
#                 },
#                 "required": ["property_name", "metrics"]
#             },
#             "text_property_aggregation": {
#                 "type": "object",
#                 "properties": {
#                     "property_name": {"type": "string"},
#                     "metrics": {"type": "string", "enum": ["TOP_OCCURRENCES"]},
#                     "top_occurrences_limit": {"type": "integer"}
#                 },
#                 "required": ["property_name", "metrics"]
#             },
#             "boolean_property_aggregation": {
#                 "type": "object",
#                 "properties": {
#                     "property_name": {"type": "string"},
#                     "metrics": {"type": "string", "enum": ["TOTAL_TRUE", "TOTAL_FALSE", "PERCENTAGE_TRUE", "PERCENTAGE_FALSE"]}
#                 },
#                 "required": ["property_name", "metrics"]
#             },
#             "groupby_property": {
#                 "type": "string"
#             }
#         },
#         "required": ["collection_name"]
#     }

#     query_function = OllamaFunction(
#         name="query_database",
#         description=f"""Query a database.

#         Available collections in this database:
#         {collections_description}""",
#         parameters=query_parameters
#     )
#     return OllamaTool(
#         function=query_function
#     )

def build_weaviate_query_tool_for_cohere(collections_description: str, collections_list: list[str]) -> CohereTool:
    properties = {
        "collection_name": {
            "type": "string",
            "description": "The collection to query.",
            "enum": collections_list
        },
        "search_query": {
            "type": "string",
            "description": "A search query to return objects from a search index."
        },
        "integer_property_filter": {
            "type": "object",
            "description": "Filter numeric properties using comparison operators",
            "properties": {
                "property_name": {"type": "string"},
                "operator": {"type": "string", "enum": ["=", "<", ">", "<=", ">="]},
                "value": {"type": "number"}
            }
        },
        "text_property_filter": {
            "type": "object",
            "description": "Filter text properties using equality or LIKE operators",
            "properties": {
                "property_name": {"type": "string"},
                "operator": {"type": "string", "enum": ["=", "LIKE"]},
                "value": {"type": "string"}
            }
        },
        "boolean_property_filter": {
            "type": "object",
            "description": "Filter boolean properties using equality operators",
            "properties": {
                "property_name": {"type": "string"},
                "operator": {"type": "string", "enum": ["=", "!="]},
                "value": {"type": "boolean"}
            }
        },
        "integer_property_aggregation": {
            "type": "object",
            "description": "Aggregate numeric properties using statistical functions",
            "properties": {
                "property_name": {"type": "string"},
                "metrics": {"type": "string", "enum": ["MIN", "MAX", "MEAN", "MEDIAN", "MODE", "SUM"]}
            }
        },
        "text_property_aggregation": {
            "type": "object",
            "description": "Aggregate text properties using frequency analysis",
            "properties": {
                "property_name": {"type": "string"},
                "metrics": {"type": "string", "enum": ["TOP_OCCURRENCES"]},
                "top_occurrences_limit": {"type": "integer"}
            }
        },
        "boolean_property_aggregation": {
            "type": "object",
            "description": "Aggregate boolean properties using statistical functions",
            "properties": {
                "property_name": {"type": "string"},
                "metrics": {"type": "string", "enum": ["TOTAL_TRUE", "TOTAL_FALSE", "PERCENTAGE_TRUE", "PERCENTAGE_FALSE"]}
            }
        },
        "groupby_property": {
            "type": "string",
            "description": "Group the results by a property."
        }
    }

    return CohereTool(
        type="function",
        function=CohereFunction(
            name="query_database",
            description=f"""Query a database.

            Available collections in this database:
            {collections_description}""",
            parameters=CohereFunctionParameters(
                type="object",
                properties=properties,
                required=["collection_name"]
            )
        )
    )

def build_weaviate_query_tool_for_together(collections_description: str, collections_list: list[str]) -> TogetherAITool:
    properties = {
        "collection_name": {
            "type": "string",
            "description": "The collection to query.",
            "enum": collections_list
        },
        "search_query": {
            "type": "string",
            "description": "A search query to return objects from a search index."
        },
        "integer_property_filter": {
            "type": "object",
            "description": "Filter numeric properties using comparison operators",
            "properties": {
                "property_name": {"type": "string"},
                "operator": {"type": "string", "enum": ["=", "<", ">", "<=", ">="]},
                "value": {"type": "number"}
            }
        },
        "text_property_filter": {
            "type": "object",
            "description": "Filter text properties using equality or LIKE operators",
            "properties": {
                "property_name": {"type": "string"},
                "operator": {"type": "string", "enum": ["=", "LIKE"]},
                "value": {"type": "string"}
            }
        },
        "boolean_property_filter": {
            "type": "object",
            "description": "Filter boolean properties using equality operators",
            "properties": {
                "property_name": {"type": "string"},
                "operator": {"type": "string", "enum": ["=", "!="]},
                "value": {"type": "boolean"}
            }
        },
        "integer_property_aggregation": {
            "type": "object",
            "description": "Aggregate numeric properties using statistical functions",
            "properties": {
                "property_name": {"type": "string"},
                "metrics": {"type": "string", "enum": ["MIN", "MAX", "MEAN", "MEDIAN", "MODE", "SUM"]}
            }
        },
        "text_property_aggregation": {
            "type": "object",
            "description": "Aggregate text properties using frequency analysis",
            "properties": {
                "property_name": {"type": "string"},
                "metrics": {"type": "string", "enum": ["TOP_OCCURRENCES"]},
                "top_occurrences_limit": {"type": "integer"}
            }
        },
        "boolean_property_aggregation": {
            "type": "object",
            "description": "Aggregate boolean properties using statistical functions",
            "properties": {
                "property_name": {"type": "string"},
                "metrics": {"type": "string", "enum": ["TOTAL_TRUE", "TOTAL_FALSE", "PERCENTAGE_TRUE", "PERCENTAGE_FALSE"]}
            }
        },
        "groupby_property": {
            "type": "string",
            "description": "Group the results by a property."
        }
    }

    return TogetherAITool(
        type="function",
        function=TogetherAIFunction(
            name="query_database",
            description=f"""Query a database.

            Available collections in this database:
            {collections_description}""",
            parameters=TogetherAIParameters(
                type="object",
                properties=properties,
                required=["collection_name"]
            )
        )
    )


# =============================================================================
# CIRCUIT BREAKER TOOLS
# =============================================================================
# These tools should be added LAST to the tool list, giving the model an escape
# hatch when it cannot determine the correct function call after multiple attempts.
#
# WHEN TO USE (model should call this tool):
# 1. After 3+ failed attempts to format a valid query
# 2. When the required information is not available in the schema
# 3. When the user's query is ambiguous and cannot be resolved
# 4. When stuck in a loop re-trying the same invalid parameters
#
# This prevents infinite tool calling loops and allows graceful degradation.
# =============================================================================

CIRCUIT_BREAKER_DESCRIPTION = """
EMERGENCY EXIT TOOL - Call this ONLY as a LAST RESORT when you are STUCK.

You MUST call this tool if ANY of the following are true:
1. You have already tried calling other tools 3 or more times and keep getting errors
2. You cannot find a matching collection or property in the schema
3. The user's query is too ambiguous to construct a valid database query
4. You are repeating the same tool call with the same parameters

DO NOT call this tool on your first attempt. Always try the query_database tool first.

When called, this will terminate the tool calling loop and return a text response explaining why the query could not be completed.
"""


def build_circuit_breaker_tool_for_openai() -> OpenAITool:
    """
    Build a circuit breaker tool for OpenAI-compatible APIs.
    
    This tool should be added LAST to the tools list.
    """
    properties = {
        "reason": {
            "type": "string",
            "description": "Explain why you cannot complete the database query. Be specific about what went wrong (e.g., 'collection not found', 'ambiguous property name', 'invalid filter format')."
        },
        "attempted_collection": {
            "type": "string",
            "description": "The collection name you tried to query, if any."
        },
        "attempted_operation": {
            "type": "string",
            "description": "Description of the operation you attempted (e.g., 'filter by rating > 4', 'aggregate average price')."
        },
        "suggestion": {
            "type": "string",
            "description": "A suggestion for the user on how to rephrase their query or what information is needed."
        }
    }

    return OpenAITool(
        type="function",
        function=OpenAIFunction(
            name="terminate_and_respond",
            description=CIRCUIT_BREAKER_DESCRIPTION,
            parameters=OpenAIParameters(
                type="object",
                properties=properties,
                required=["reason"]
            )
        )
    )


def build_circuit_breaker_tool_for_anthropic() -> AnthropicTool:
    """Build a circuit breaker tool for Anthropic API."""
    properties = {
        "reason": {
            "type": "string",
            "description": "Explain why you cannot complete the database query."
        },
        "attempted_collection": {
            "type": "string",
            "description": "The collection name you tried to query, if any."
        },
        "attempted_operation": {
            "type": "string",
            "description": "Description of the operation you attempted."
        },
        "suggestion": {
            "type": "string",
            "description": "A suggestion for the user."
        }
    }

    return AnthropicTool(
        name="terminate_and_respond",
        description=CIRCUIT_BREAKER_DESCRIPTION,
        input_schema=AnthropicToolInputSchema(
            type="object",
            properties=properties,
            required=["reason"]
        )
    )


def build_circuit_breaker_tool_for_ollama() -> dict:
    """Build a circuit breaker tool for Ollama (returns dict for compatibility)."""
    return {
        "type": "function",
        "function": {
            "name": "terminate_and_respond",
            "description": CIRCUIT_BREAKER_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Explain why you cannot complete the database query."
                    },
                    "attempted_collection": {
                        "type": "string",
                        "description": "The collection name you tried to query, if any."
                    },
                    "attempted_operation": {
                        "type": "string",
                        "description": "Description of the operation you attempted."
                    },
                    "suggestion": {
                        "type": "string",
                        "description": "A suggestion for the user."
                    }
                },
                "required": ["reason"]
            }
        }
    }


def build_circuit_breaker_tool_for_cohere() -> CohereTool:
    """Build a circuit breaker tool for Cohere API."""
    properties = {
        "reason": {
            "type": "string",
            "description": "Explain why you cannot complete the database query."
        },
        "attempted_collection": {
            "type": "string",
            "description": "The collection name you tried to query, if any."
        },
        "attempted_operation": {
            "type": "string",
            "description": "Description of the operation you attempted."
        },
        "suggestion": {
            "type": "string",
            "description": "A suggestion for the user."
        }
    }

    return CohereTool(
        type="function",
        function=CohereFunction(
            name="terminate_and_respond",
            description=CIRCUIT_BREAKER_DESCRIPTION,
            parameters=CohereFunctionParameters(
                type="object",
                properties=properties,
                required=["reason"]
            )
        )
    )


def build_circuit_breaker_tool_for_together() -> TogetherAITool:
    """Build a circuit breaker tool for Together AI API."""
    properties = {
        "reason": {
            "type": "string",
            "description": "Explain why you cannot complete the database query."
        },
        "attempted_collection": {
            "type": "string",
            "description": "The collection name you tried to query, if any."
        },
        "attempted_operation": {
            "type": "string",
            "description": "Description of the operation you attempted."
        },
        "suggestion": {
            "type": "string",
            "description": "A suggestion for the user."
        }
    }

    return TogetherAITool(
        type="function",
        function=TogetherAIFunction(
            name="terminate_and_respond",
            description=CIRCUIT_BREAKER_DESCRIPTION,
            parameters=TogetherAIParameters(
                type="object",
                properties=properties,
                required=["reason"]
            )
        )
    )


def get_circuit_breaker_for_provider(provider: str):
    """
    Get the appropriate circuit breaker tool for a given provider.
    
    Args:
        provider: One of 'openai', 'anthropic', 'ollama', 'cohere', 'together'
        
    Returns:
        Circuit breaker tool in the appropriate format for the provider
    """
    builders = {
        "openai": build_circuit_breaker_tool_for_openai,
        "anthropic": build_circuit_breaker_tool_for_anthropic,
        "ollama": build_circuit_breaker_tool_for_ollama,
        "cohere": build_circuit_breaker_tool_for_cohere,
        "together": build_circuit_breaker_tool_for_together,
    }
    
    builder = builders.get(provider.lower())
    if builder:
        return builder()
    
    # Default to OpenAI format
    return build_circuit_breaker_tool_for_openai()


def append_circuit_breaker(tools: list, provider: str) -> list:
    """
    Append the circuit breaker tool to a list of tools.
    
    The circuit breaker is always added LAST to ensure the model
    tries all other tools first before falling back.
    
    Args:
        tools: List of existing tools
        provider: The LLM provider name
        
    Returns:
        New list with circuit breaker appended
    """
    circuit_breaker = get_circuit_breaker_for_provider(provider)
    return tools + [circuit_breaker]