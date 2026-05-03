"""Validates LLM-generated queries against schemas and drives self-correction.

No code execution -- only structural checks (property exists, type matches,
operator legal). When validation fails, builds a correction prompt with fix
suggestions and feeds it back to the LLM for a retry.
"""

from typing import Optional, Tuple
from pydantic import BaseModel
from collections import defaultdict

from src.models import (
    WeaviateQuery,
    IntPropertyFilter,
    TextPropertyFilter,
    BooleanPropertyFilter,
    IntAggregation,
    TextAggregation,
    BooleanAggregation
)


class ValidationResult(BaseModel):
    """Result of query validation."""
    valid: bool
    errors: list[str] = []
    warnings: list[str] = []
    fix_suggestions: list[str] = []


class StructuredSandbox:
    """
    Validates WeaviateQuery objects against schema.
    
    This is the "code execution" sandbox that:
    1. Receives structured query from LLM
    2. Validates against schema rules
    3. Returns validated query or errors
    
    No arbitrary code execution - only structured validation.
    Safe by design: only read operations, no modifications possible.
    
    Example:
        >>> sandbox = StructuredSandbox(mcp_server)
        >>> result = sandbox.validate(query)
        >>> if result.valid:
        ...     # Safe to execute
        ... else:
        ...     print(result.errors)
    """
    
    # Valid aggregation metrics by property type
    INT_AGGREGATIONS = ["MIN", "MAX", "MEAN", "MEDIAN", "MODE", "SUM"]
    TEXT_AGGREGATIONS = ["TOP_OCCURRENCES"]
    BOOL_AGGREGATIONS = ["TOTAL_TRUE", "TOTAL_FALSE", "PERCENTAGE_TRUE", "PERCENTAGE_FALSE"]
    
    # Valid filter operators by property type
    INT_OPERATORS = ["=", "!=", "<", ">", "<=", ">="]
    TEXT_OPERATORS = ["=", "!=", "LIKE"]
    BOOL_OPERATORS = ["=", "!="]
    
    def __init__(self, mcp_server):
        """
        Initialize sandbox with MCP server for schema access.
        
        Args:
            mcp_server: MCPServer instance with schema registry
        """
        self.server = mcp_server
    
    def validate(self, query: WeaviateQuery) -> ValidationResult:
        """
        Validate a WeaviateQuery against the schema.
        
        Args:
            query: WeaviateQuery object to validate
        
        Returns:
            ValidationResult with valid flag, errors, and warnings
        """
        errors = []
        warnings = []
        
        # 1. Validate collection exists
        collection = query.target_collection
        if not collection:
            errors.append("target_collection is required")
            return ValidationResult(valid=False, errors=errors)
        
        schema = self.server.get_collection_schema(collection, compressed=False)
        if schema is None:
            errors.append(f"Collection '{collection}' not found")
            return ValidationResult(valid=False, errors=errors)
        
        properties = schema.get("properties", {})
        
        # 2. Validate integer filter
        if query.integer_property_filter:
            filter_errors = self._validate_int_filter(
                query.integer_property_filter, properties
            )
            errors.extend(filter_errors)
        
        # 3. Validate text filter
        if query.text_property_filter:
            filter_errors = self._validate_text_filter(
                query.text_property_filter, properties
            )
            errors.extend(filter_errors)
        
        # 4. Validate boolean filter
        if query.boolean_property_filter:
            filter_errors = self._validate_bool_filter(
                query.boolean_property_filter, properties
            )
            errors.extend(filter_errors)
        
        # 5. Validate integer aggregation
        if query.integer_property_aggregation:
            agg_errors = self._validate_int_aggregation(
                query.integer_property_aggregation, properties
            )
            errors.extend(agg_errors)
        
        # 6. Validate text aggregation
        if query.text_property_aggregation:
            agg_errors = self._validate_text_aggregation(
                query.text_property_aggregation, properties
            )
            errors.extend(agg_errors)
        
        # 7. Validate boolean aggregation
        if query.boolean_property_aggregation:
            agg_errors = self._validate_bool_aggregation(
                query.boolean_property_aggregation, properties
            )
            errors.extend(agg_errors)
        
        # 8. Validate groupby property
        if query.groupby_property:
            if query.groupby_property not in properties:
                errors.append(
                    f"groupby_property '{query.groupby_property}' not found in {collection}"
                )
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_and_return(self, query: WeaviateQuery) -> Tuple[bool, WeaviateQuery | str]:
        """
        Validate query and return it if valid.
        
        Args:
            query: WeaviateQuery to validate
        
        Returns:
            (True, query) if valid, (False, error_message) if invalid
        """
        result = self.validate(query)
        if result.valid:
            return True, query
        return False, "; ".join(result.errors)
    
    def _validate_int_filter(
        self, 
        filter: IntPropertyFilter, 
        properties: dict
    ) -> list[str]:
        """Validate integer property filter."""
        errors = []
        
        if filter.property_name not in properties:
            errors.append(f"Integer filter property '{filter.property_name}' not found")
            return errors
        
        prop_type = properties[filter.property_name].get("type", "")
        if prop_type not in ["int", "integer", "number", "float"]:
            errors.append(
                f"Property '{filter.property_name}' is {prop_type}, not numeric"
            )
        
        if filter.operator not in self.INT_OPERATORS:
            errors.append(
                f"Invalid operator '{filter.operator}' for integer filter. "
                f"Valid: {self.INT_OPERATORS}"
            )
        
        return errors
    
    def _validate_text_filter(
        self, 
        filter: TextPropertyFilter, 
        properties: dict
    ) -> list[str]:
        """Validate text property filter."""
        errors = []
        
        if filter.property_name not in properties:
            errors.append(f"Text filter property '{filter.property_name}' not found")
            return errors
        
        prop_type = properties[filter.property_name].get("type", "")
        if prop_type not in ["text", "string", "text[]"]:
            errors.append(
                f"Property '{filter.property_name}' is {prop_type}, not text"
            )
        
        if filter.operator not in self.TEXT_OPERATORS:
            errors.append(
                f"Invalid operator '{filter.operator}' for text filter. "
                f"Valid: {self.TEXT_OPERATORS}"
            )
        
        return errors
    
    def _validate_bool_filter(
        self, 
        filter: BooleanPropertyFilter, 
        properties: dict
    ) -> list[str]:
        """Validate boolean property filter."""
        errors = []
        
        if filter.property_name not in properties:
            errors.append(f"Boolean filter property '{filter.property_name}' not found")
            return errors
        
        prop_type = properties[filter.property_name].get("type", "")
        if prop_type not in ["bool", "boolean"]:
            errors.append(
                f"Property '{filter.property_name}' is {prop_type}, not boolean"
            )
        
        if filter.operator not in self.BOOL_OPERATORS:
            errors.append(
                f"Invalid operator '{filter.operator}' for boolean filter. "
                f"Valid: {self.BOOL_OPERATORS}"
            )
        
        return errors
    
    def _validate_int_aggregation(
        self, 
        agg: IntAggregation, 
        properties: dict
    ) -> list[str]:
        """Validate integer aggregation."""
        errors = []
        
        if agg.property_name not in properties:
            errors.append(f"Aggregation property '{agg.property_name}' not found")
            return errors
        
        # Handle metrics as string or list
        metrics = agg.metrics if isinstance(agg.metrics, list) else [agg.metrics]
        
        for metric in metrics:
            if metric.upper() not in self.INT_AGGREGATIONS:
                if metric.upper() == "COUNT":
                    errors.append(
                        "COUNT is not a valid metric. Use total_count=True instead."
                    )
                else:
                    errors.append(
                        f"Invalid metric '{metric}' for integer aggregation. "
                        f"Valid: {self.INT_AGGREGATIONS}"
                    )
        
        return errors
    
    def _validate_text_aggregation(
        self, 
        agg: TextAggregation, 
        properties: dict
    ) -> list[str]:
        """Validate text aggregation."""
        errors = []
        
        if agg.property_name not in properties:
            errors.append(f"Aggregation property '{agg.property_name}' not found")
            return errors
        
        metrics = agg.metrics if isinstance(agg.metrics, list) else [agg.metrics]
        
        for metric in metrics:
            if metric.upper() not in self.TEXT_AGGREGATIONS:
                errors.append(
                    f"Invalid metric '{metric}' for text aggregation. "
                    f"Valid: {self.TEXT_AGGREGATIONS}"
                )
        
        return errors
    
    def _validate_bool_aggregation(
        self,
        agg: BooleanAggregation,
        properties: dict
    ) -> list[str]:
        """Validate boolean aggregation."""
        errors = []

        if agg.property_name not in properties:
            errors.append(f"Aggregation property '{agg.property_name}' not found")
            return errors

        metrics = agg.metrics if isinstance(agg.metrics, list) else [agg.metrics]

        for metric in metrics:
            if metric.upper() not in self.BOOL_AGGREGATIONS:
                errors.append(
                    f"Invalid metric '{metric}' for boolean aggregation. "
                    f"Valid: {self.BOOL_AGGREGATIONS}"
                )

        return errors

    # =========================================================================
    # SELF-CORRECTION LOOP
    # =========================================================================

    def validate_with_suggestions(self, query: WeaviateQuery) -> ValidationResult:
        """
        Validate a query and generate actionable fix suggestions for each error.

        Returns ValidationResult with fix_suggestions populated.
        """
        result = self.validate(query)
        if result.valid:
            return result

        suggestions = []
        collection = query.target_collection
        schema = self.server.get_collection_schema(collection, compressed=False) if collection else None
        properties = schema.get("properties", {}) if schema else {}

        for error in result.errors:
            suggestion = self._generate_fix_suggestion(error, collection, properties)
            if suggestion:
                suggestions.append(suggestion)

        result.fix_suggestions = suggestions
        return result

    # =========================================================================
    # V2 Tool-Schema Validation
    # =========================================================================
    # V2 queries use lists (filters[], output_properties[], order_by[]) scoped
    # per collection. V1's singular-filter WeaviateQuery model can't express
    # this, so V2 validates on the raw dict directly.

    V2_INT_OPS = ["=", "!=", "<", ">", "<=", ">="]
    V2_TEXT_OPS = ["=", "!=", "LIKE"]
    V2_BOOL_OPS = ["=", "!="]
    V2_SET_OPS = ["IN", "BETWEEN"]
    V2_NULL_OPS = ["IS_NULL", "IS_NOT_NULL"]

    _SCHEMA_TYPE_ALIASES = {
        "integer": {"int", "integer", "number", "float"},
        "text": {"text", "string", "text[]"},
        "boolean": {"bool", "boolean"},
        "date": {"date", "datetime", "text", "string"},  # dates often stored as text in Weaviate
    }

    def _v2_collection_props(self, collection_name: str) -> Optional[dict]:
        """Fetch the properties dict for a collection, or None if missing."""
        if not collection_name:
            return None
        schema = self.server.get_collection_schema(collection_name, compressed=False)
        if schema is None:
            return None
        return schema.get("properties", {}) or {}

    def validate_v2(self, query_args: dict) -> ValidationResult:
        """Validate a V2 tool-call dict against the schema.

        Checks:
        1. collection_name exists in schema registry
        2. every additional_collections entry resolves to a known collection
        3. every filter[i]: scoped collection exists; property exists on it;
           property_type declaration matches the schema property type if given;
           operator is legal for that type
        4. every output_property[j]: scoped collection exists; property exists
        5. every order_by[i]: scoped collection exists; property exists

        Missing `collection` fields default to `collection_name`.
        """
        errors: list[str] = []
        warnings: list[str] = []
        suggestions: list[str] = []

        primary = query_args.get("collection_name")
        if not primary:
            errors.append("collection_name is required")
            return ValidationResult(valid=False, errors=errors)

        primary_props = self._v2_collection_props(primary)
        if primary_props is None:
            errors.append(f"Collection '{primary}' not found in schema registry")
            all_colls = self.server.list_collections()
            closest = self._find_closest_collection(primary, all_colls)
            if closest:
                suggestions.append(f"Replace collection_name '{primary}' with '{closest}'")

        # Build a {collection_name -> properties_dict} lookup for everything the
        # query declares it will touch. We treat unknown collections as soft
        # warnings rather than hard errors — PR3 may widen this.
        scoped_colls: dict[str, dict] = {}
        if primary_props is not None:
            scoped_colls[primary] = primary_props

        for ac in query_args.get("additional_collections") or []:
            if not isinstance(ac, dict):
                continue
            ac_name = ac.get("collection_name")
            if not ac_name or ac_name in scoped_colls:
                continue
            ac_props = self._v2_collection_props(ac_name)
            if ac_props is None:
                errors.append(f"additional_collections: '{ac_name}' not found in schema registry")
                closest = self._find_closest_collection(ac_name, self.server.list_collections())
                if closest:
                    suggestions.append(f"Replace additional collection '{ac_name}' with '{closest}'")
            else:
                scoped_colls[ac_name] = ac_props

        def _resolve(entry_coll: Optional[str]) -> tuple[str, Optional[dict]]:
            coll = entry_coll or primary
            return coll, scoped_colls.get(coll)

        # ---- Filters
        for i, f in enumerate(query_args.get("filters") or []):
            if not isinstance(f, dict):
                continue
            coll, props = _resolve(f.get("collection"))
            pname = f.get("property_name", "")
            op = f.get("operator", "")
            ptype = (f.get("property_type") or "").lower()

            if props is None:
                errors.append(f"filters[{i}]: collection '{coll}' not in declared query set")
                continue

            if pname not in props:
                errors.append(f"filters[{i}]: property '{pname}' not found in {coll}")
                closest = self._find_closest_property(pname, props)
                if closest:
                    real_type = props[closest].get("type", "unknown") if isinstance(props[closest], dict) else "unknown"
                    suggestions.append(f"filters[{i}]: replace '{pname}' with '{closest}' (type: {real_type})")
                continue

            # Property type vs schema
            schema_type = (props[pname].get("type", "") if isinstance(props[pname], dict) else "").lower()
            if ptype and ptype not in self._SCHEMA_TYPE_ALIASES and ptype != schema_type:
                # Declared type is unknown — soft warning
                warnings.append(f"filters[{i}]: declared property_type='{ptype}' not recognized")
            elif ptype and ptype in self._SCHEMA_TYPE_ALIASES:
                if schema_type and schema_type not in self._SCHEMA_TYPE_ALIASES[ptype]:
                    errors.append(f"filters[{i}]: property '{pname}' is schema-type '{schema_type}', not '{ptype}'")

            # Operator legality
            if op in self.V2_NULL_OPS:
                pass  # legal for any property type
            elif op in self.V2_SET_OPS:
                if not isinstance(f.get("value"), list):
                    warnings.append(f"filters[{i}]: operator '{op}' expects a list value")
            else:
                # Match operator set to property type
                effective_type = ptype or schema_type
                if effective_type in self._SCHEMA_TYPE_ALIASES["integer"]:
                    if op not in self.V2_INT_OPS:
                        errors.append(f"filters[{i}]: operator '{op}' invalid for integer; valid: {self.V2_INT_OPS + self.V2_SET_OPS + self.V2_NULL_OPS}")
                elif effective_type in self._SCHEMA_TYPE_ALIASES["text"]:
                    if op not in self.V2_TEXT_OPS:
                        errors.append(f"filters[{i}]: operator '{op}' invalid for text; valid: {self.V2_TEXT_OPS + self.V2_SET_OPS + self.V2_NULL_OPS}")
                elif effective_type in self._SCHEMA_TYPE_ALIASES["boolean"]:
                    if op not in self.V2_BOOL_OPS:
                        errors.append(f"filters[{i}]: operator '{op}' invalid for boolean; valid: {self.V2_BOOL_OPS + self.V2_NULL_OPS}")

        # ---- Output properties (required in V2)
        outputs = query_args.get("output_properties") or []
        if not outputs:
            errors.append("output_properties is required and must contain at least one entry")
        for j, o in enumerate(outputs):
            if not isinstance(o, dict):
                continue
            coll, props = _resolve(o.get("collection"))
            pname = o.get("property_name", "")
            if props is None:
                errors.append(f"output_properties[{j}]: collection '{coll}' not in declared query set")
                continue
            if pname and pname not in props:
                errors.append(f"output_properties[{j}]: property '{pname}' not found in {coll}")
                closest = self._find_closest_property(pname, props)
                if closest:
                    suggestions.append(f"output_properties[{j}]: replace '{pname}' with '{closest}'")

        # ---- Order by
        for k, ob in enumerate(query_args.get("order_by") or []):
            if not isinstance(ob, dict):
                continue
            coll, props = _resolve(ob.get("collection"))
            pname = ob.get("property_name", "")
            if props is None:
                errors.append(f"order_by[{k}]: collection '{coll}' not in declared query set")
                continue
            if pname and pname not in props:
                errors.append(f"order_by[{k}]: property '{pname}' not found in {coll}")
                closest = self._find_closest_property(pname, props)
                if closest:
                    suggestions.append(f"order_by[{k}]: replace '{pname}' with '{closest}'")

        # ---- Having filters (post-aggregation)
        for m_idx, hf in enumerate(query_args.get("having_filters") or []):
            if not isinstance(hf, dict):
                continue
            coll, props = _resolve(hf.get("collection"))
            pname = hf.get("property_name", "")
            if props is None:
                errors.append(f"having_filters[{m_idx}]: collection '{coll}' not in declared query set")
                continue
            if pname and pname not in props:
                errors.append(f"having_filters[{m_idx}]: property '{pname}' not found in {coll}")
                closest = self._find_closest_property(pname, props)
                if closest:
                    suggestions.append(f"having_filters[{m_idx}]: replace '{pname}' with '{closest}'")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            fix_suggestions=suggestions,
        )

    def build_v2_correction_prompt(self, original_query: str, query_args: dict,
                                   validation_result: ValidationResult,
                                   schema_desc: str) -> str:
        """V2 correction prompt — mirrors build_correction_prompt but for V2 args."""
        error_lines = "\n".join(f"  - ERROR: {e}" for e in validation_result.errors)
        suggestion_lines = "\n".join(f"  - FIX: {s}" for s in validation_result.fix_suggestions if s)

        args_preview = _format_query_args(query_args)

        return f"""Your previous query had validation errors. Please fix them and re-emit the tool call with corrected parameters.

Original question: {original_query}

Your previous attempt (V2 tool schema):
{args_preview}

Validation errors:
{error_lines}

Suggested fixes:
{suggestion_lines}

Schema for reference:
{schema_desc}

Generate a CORRECTED query that fixes all the errors above. Keep output_properties accurate — only change what's needed to fix the errors. Use exact property names from the schema."""

    def _generate_fix_suggestion(self, error: str, collection: str, properties: dict) -> str:
        """Generate a specific fix suggestion for a validation error."""
        # Property not found -> suggest closest match
        if "not found" in error and "property" in error.lower():
            import re
            match = re.search(r"'(\w+)'", error)
            if match:
                wrong_prop = match.group(1)
                closest = self._find_closest_property(wrong_prop, properties)
                if closest:
                    prop_type = properties[closest].get("type", "unknown") if isinstance(properties[closest], dict) else "unknown"
                    return f"Replace '{wrong_prop}' with '{closest}' (type: {prop_type})"

        # Invalid metric
        if "invalid metric" in error.lower() or "COUNT is not a valid" in error:
            if "COUNT" in error.upper():
                return "Remove the aggregation metric and set total_count=True instead"
            if "integer" in error.lower():
                return f"Use one of: {', '.join(self.INT_AGGREGATIONS)}"
            if "text" in error.lower():
                return f"Use: TOP_OCCURRENCES"
            if "boolean" in error.lower():
                return f"Use one of: {', '.join(self.BOOL_AGGREGATIONS)}"

        # Invalid operator
        if "invalid operator" in error.lower():
            if "integer" in error.lower():
                return f"Use one of: {', '.join(self.INT_OPERATORS)}"
            if "text" in error.lower():
                return f"Use one of: {', '.join(self.TEXT_OPERATORS)}"

        # Type mismatch
        if "not numeric" in error or "not text" in error or "not boolean" in error:
            import re
            match = re.search(r"'(\w+)' is (\w+)", error)
            if match:
                prop_name, actual_type = match.group(1), match.group(2)
                return f"Property '{prop_name}' is {actual_type}. Move it to the appropriate filter/aggregation type."

        # Collection not found
        if "Collection" in error and "not found" in error:
            all_collections = self.server.list_collections()
            import re
            match = re.search(r"'(\w+)'", error)
            if match:
                wrong_name = match.group(1)
                closest = self._find_closest_collection(wrong_name, all_collections)
                if closest:
                    return f"Replace '{wrong_name}' with '{closest}'"

        return ""

    def _find_closest_property(self, wrong_name: str, properties: dict) -> Optional[str]:
        """Find the closest matching property name using edit distance."""
        if not properties:
            return None

        wrong_lower = wrong_name.lower().replace("_", "")
        best_match = None
        best_score = 0

        for prop_name in properties:
            prop_lower = prop_name.lower().replace("_", "")

            # Substring containment
            if wrong_lower in prop_lower or prop_lower in wrong_lower:
                score = len(wrong_lower) / max(len(prop_lower), 1)
                if score > best_score:
                    best_score = score
                    best_match = prop_name

            # Common prefix
            common = 0
            for a, b in zip(wrong_lower, prop_lower):
                if a == b:
                    common += 1
                else:
                    break
            prefix_score = common / max(len(wrong_lower), len(prop_lower), 1)
            if prefix_score > best_score:
                best_score = prefix_score
                best_match = prop_name

        return best_match if best_score > 0.3 else None

    def _find_closest_collection(self, wrong_name: str, collections: list) -> Optional[str]:
        """Find the closest matching collection name."""
        wrong_lower = wrong_name.lower()
        best_match = None
        best_score = 0

        for coll in collections:
            coll_lower = coll.lower()
            if wrong_lower in coll_lower or coll_lower in wrong_lower:
                score = len(wrong_lower) / max(len(coll_lower), 1)
                if score > best_score:
                    best_score = score
                    best_match = coll

        return best_match if best_score > 0.3 else None

    def build_correction_prompt(self, original_query: str, query_args: dict,
                                validation_result: ValidationResult,
                                schema_desc: str) -> str:
        """
        Build a correction prompt that feeds validation errors back to the LLM.

        This is the core of the self-correction loop: instead of discarding a
        failed query, we tell the LLM what went wrong and how to fix it.

        Args:
            original_query: The user's natural language query
            query_args: The LLM's original (invalid) query arguments
            validation_result: The validation result with errors and suggestions
            schema_desc: The collection schema description

        Returns:
            A correction prompt string for the LLM
        """
        error_lines = "\n".join(f"  - ERROR: {e}" for e in validation_result.errors)
        suggestion_lines = "\n".join(f"  - FIX: {s}" for s in validation_result.fix_suggestions if s)

        return f"""Your previous query had validation errors. Please fix them.

Original question: {original_query}

Your previous attempt:
{_format_query_args(query_args)}

Validation errors:
{error_lines}

Suggested fixes:
{suggestion_lines}

Schema for reference:
{schema_desc}

Generate a CORRECTED query that fixes all the errors above. Use exact property names from the schema."""

    # Error pattern tracking for preventive injection
    _error_patterns = defaultdict(int)

    def track_error_pattern(self, errors: list[str]):
        """Track common error patterns to inject preventive instructions."""
        for error in errors:
            if "COUNT" in error:
                self._error_patterns["count_as_metric"] += 1
            elif "not found" in error and "property" in error.lower():
                self._error_patterns["wrong_property_name"] += 1
            elif "not numeric" in error or "not text" in error:
                self._error_patterns["type_mismatch"] += 1
            elif "Invalid operator" in error:
                self._error_patterns["invalid_operator"] += 1

    def get_preventive_instructions(self) -> str:
        """
        Based on tracked error patterns, generate preventive instructions
        to inject into future prompts.
        """
        instructions = []

        if self._error_patterns.get("count_as_metric", 0) >= 2:
            instructions.append(
                "IMPORTANT: Never use COUNT as an aggregation metric. "
                "Use total_count=True for counting."
            )
        if self._error_patterns.get("wrong_property_name", 0) >= 2:
            instructions.append(
                "IMPORTANT: Property names are case-sensitive and must match "
                "the schema exactly. Check property names carefully."
            )
        if self._error_patterns.get("type_mismatch", 0) >= 2:
            instructions.append(
                "IMPORTANT: Check property types before creating filters/aggregations. "
                "Use integer filters for numeric properties, text filters for text properties."
            )

        return "\n".join(instructions)


class SemanticValidator:
    """
    LLM-based semantic validation: verifies that a generated query actually
    answers the user's original question.

    This addresses the gap where structural validation passes but the query
    is semantically wrong (e.g., wrong collection, wrong aggregation type,
    filters that don't match the question's intent).

    Inspired by DIN-SQL's self-correction (+2-5%) and CHESS's unit testing.
    """

    SEMANTIC_CHECK_PROMPT = """You are a database query validator. Given a natural language question and a generated Weaviate query, determine if the query correctly answers the question.

Original question: {question}

Generated query:
- Collection: {collection}
- Search query: {search_query}
- Filters: {filters}
- Aggregations: {aggregations}
- Group by: {groupby}
- Total count: {total_count}

Available collections in the schema: {available_collections}

Analyze:
1. Is the correct collection selected? Does "{collection}" match what the question is asking about?
2. Are the filters appropriate for the question?
3. Are the aggregations correct (if any)?
4. Is anything missing that the question requires?

Respond with EXACTLY one of:
- CORRECT - if the query properly answers the question
- WRONG_COLLECTION: <correct_collection> - if the wrong collection was selected
- WRONG_FILTER: <explanation> - if filters are incorrect
- WRONG_AGGREGATION: <explanation> - if aggregation is wrong
- MISSING_COMPONENT: <explanation> - if something required is missing"""

    def __init__(self, lm_service):
        """
        Args:
            lm_service: LMService instance for making LLM calls
        """
        self.lm = lm_service

    def validate(self, question: str, query_args: dict, candidate_collections: list) -> dict:
        """
        Semantically validate a generated query against the original question.

        Returns:
            dict with 'valid' (bool), 'issue_type' (str or None), 'feedback' (str or None)
        """
        filters = []
        for ftype in ['integer_property_filter', 'text_property_filter', 'boolean_property_filter']:
            f = query_args.get(ftype)
            if f:
                filters.append(f"{ftype}: {f}")

        aggregations = []
        for atype in ['integer_property_aggregation', 'text_property_aggregation', 'boolean_property_aggregation']:
            a = query_args.get(atype)
            if a:
                aggregations.append(f"{atype}: {a}")

        prompt = self.SEMANTIC_CHECK_PROMPT.format(
            question=question,
            collection=query_args.get('collection_name', 'N/A'),
            search_query=query_args.get('search_query', 'None'),
            filters='; '.join(filters) if filters else 'None',
            aggregations='; '.join(aggregations) if aggregations else 'None',
            groupby=query_args.get('groupby_property', 'None'),
            total_count=query_args.get('total_count', False),
            available_collections=', '.join(candidate_collections[:10]),
        )

        try:
            import time
            for retry in range(3):
                try:
                    response = self.lm.generate(prompt)
                    break
                except Exception as e:
                    if '429' in str(e) or 'Too Many' in str(e):
                        time.sleep(2 * (2 ** retry))
                    else:
                        raise

            if not isinstance(response, str):
                response = str(response)

            response = response.strip()

            if response.startswith('CORRECT'):
                return {'valid': True, 'issue_type': None, 'feedback': None}

            # Parse the issue
            for prefix in ['WRONG_COLLECTION:', 'WRONG_FILTER:', 'WRONG_AGGREGATION:', 'MISSING_COMPONENT:']:
                if response.startswith(prefix):
                    issue_type = prefix.rstrip(':')
                    feedback = response[len(prefix):].strip()
                    return {'valid': False, 'issue_type': issue_type, 'feedback': feedback}

            # If response doesn't match expected format, treat as valid (conservative)
            return {'valid': True, 'issue_type': None, 'feedback': None}

        except Exception:
            # On error, don't block the pipeline
            return {'valid': True, 'issue_type': None, 'feedback': None}

    def build_semantic_correction_prompt(self, question: str, query_args: dict,
                                          semantic_result: dict, schema_desc: str) -> str:
        """Build a correction prompt incorporating semantic feedback."""
        import json
        return f"""Your previous query has a semantic issue. Please fix it.

Original question: {question}

Your previous query:
{json.dumps(query_args, indent=2)}

Issue: {semantic_result['issue_type']}
Feedback: {semantic_result['feedback']}

Schema for reference:
{schema_desc}

Generate a CORRECTED query that addresses the semantic issue above. Make sure the query actually answers the original question."""


def _format_query_args(args: dict) -> str:
    """Format query args dict for display in correction prompt."""
    import json
    try:
        return json.dumps(args, indent=2)
    except (TypeError, ValueError):
        return str(args)
