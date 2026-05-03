"""Component-level AST scoring for Weaviate query evaluation.

Scores collection match, filters (property + operator + value), aggregations,
search_query, groupby, and total_count. Auto-detects V2 predictions and
converts them to V1 format before scoring.
"""

import re
from typing import Optional


def v2_to_v1_format(v2_args: dict) -> dict:
    """Adapter: convert V2 tool-schema args to V1 for scoring.

    Needed because ground-truth is stored in V1 format but the V2 schema
    uses filters[], output_properties[], etc. Safe no-op on V1 args.
    """
    if not v2_args:
        return {}

    # If it already has V1-style fields, return as-is
    if any(v2_args.get(f) for f in ["integer_property_filter", "text_property_filter", "boolean_property_filter"]):
        return v2_args

    # If it doesn't have V2-style fields either, return as-is
    if "filters" not in v2_args and "output_properties" not in v2_args:
        return v2_args

    v1 = {}
    v1["collection_name"] = v2_args.get("collection_name", "")

    for f in v2_args.get("filters", []):
        prop_type = f.get("property_type", "")
        if not prop_type:
            val = f.get("value")
            op = f.get("operator", "")
            if isinstance(val, bool):
                prop_type = "boolean"
            elif isinstance(val, (int, float)):
                prop_type = "integer"
            elif op in (">", "<", ">=", "<=") and val is not None:
                prop_type = "integer"
            else:
                prop_type = "text"

        key = {"integer": "integer_property_filter",
               "boolean": "boolean_property_filter"}.get(prop_type, "text_property_filter")
        if key not in v1:
            v1[key] = {"property_name": f.get("property_name", ""),
                       "operator": f.get("operator", ""),
                       "value": f.get("value")}

    if v2_args.get("search_query"):
        v1["search_query"] = v2_args["search_query"]

    gbp = v2_args.get("group_by_properties") or v2_args.get("groupby_property")
    if gbp:
        v1["groupby_property"] = gbp[0] if isinstance(gbp, list) else gbp

    if v2_args.get("total_count") is not None:
        v1["total_count"] = v2_args["total_count"]

    for op in v2_args.get("output_properties", []):
        agg = op.get("aggregation", "NONE")
        if agg and agg != "NONE":
            prop_name = op.get("property_name", "")
            if agg in ("SUM", "MIN", "MAX", "MEAN", "MEDIAN", "MODE", "COUNT"):
                key = "integer_property_aggregation"
                if key not in v1:
                    v1[key] = {"property_name": prop_name, "metrics": [agg]}
                else:
                    v1[key]["metrics"].append(agg)
            elif agg == "TOP_OCCURRENCES":
                v1.setdefault("text_property_aggregation",
                              {"property_name": prop_name, "metrics": [agg]})
            elif agg in ("TOTAL_TRUE", "TOTAL_FALSE", "PERCENTAGE_TRUE", "PERCENTAGE_FALSE"):
                v1.setdefault("boolean_property_aggregation",
                              {"property_name": prop_name, "metrics": [agg]})

    return v1


def calculate_ast_score(predicted: dict, expected: dict) -> float:
    """
    Calculate comprehensive AST match score.

    Components and weights:
    - Collection name: 1.0 (must match, else score=0)
    - Each filter type: 1.0 (property: 0.5, operator: 0.25, value: 0.25)
    - Each aggregation type: 1.0 (property: 0.5, metrics: 0.5)
    - search_query: 0.5 (semantic similarity)
    - groupby_property: 0.5
    - total_count: 0.25

    Returns: Normalized score in [0, 1]
    """
    if not predicted or not expected:
        return 0.0

    # Auto-detect V2 shape and convert, so callers don't need to know
    is_v2 = ("output_properties" in predicted or
             (isinstance(predicted.get("filters"), list) and predicted["filters"]) or
             "group_by_properties" in predicted)
    is_v1 = any(predicted.get(f) for f in [
        "integer_property_filter", "text_property_filter", "boolean_property_filter",
        "integer_property_aggregation", "text_property_aggregation", "boolean_property_aggregation"])
    if is_v2 and not is_v1:
        predicted = v2_to_v1_format(predicted)

    score = 0.0
    total = 0.0

    # 1. Collection Name (Critical - wrong collection = 0)
    total += 1.0
    pred_coll = predicted.get("collection_name") or predicted.get("target_collection", "")
    exp_coll = expected.get("collection_name") or expected.get("target_collection", "")
    if pred_coll == exp_coll:
        score += 1.0
    else:
        return 0.0

    # 2. Filters (int, text, boolean)
    for filter_type in ["integer_property_filter", "text_property_filter", "boolean_property_filter"]:
        if expected.get(filter_type):
            total += 1.0
            pred_f = predicted.get(filter_type) or {}
            exp_f = expected[filter_type]
            if not pred_f:
                continue

            sub_score = 0.0

            # Property name match
            if _normalize_prop(pred_f.get("property_name")) == _normalize_prop(exp_f.get("property_name")):
                sub_score += 0.5

            # Operator match (with normalization)
            p_op = _normalize_operator(pred_f.get("operator", ""))
            e_op = _normalize_operator(exp_f.get("operator", ""))
            if p_op == e_op:
                sub_score += 0.25

            # Value match (with type normalization)
            if _values_match(pred_f.get("value"), exp_f.get("value"), filter_type):
                sub_score += 0.25

            score += sub_score

    # 3. Aggregations (int, text, boolean)
    for agg_type in ["integer_property_aggregation", "text_property_aggregation", "boolean_property_aggregation"]:
        if expected.get(agg_type):
            total += 1.0
            pred_a = predicted.get(agg_type) or {}
            exp_a = expected[agg_type]
            if not pred_a:
                continue

            sub_score = 0.0

            # Property name match
            if _normalize_prop(pred_a.get("property_name")) == _normalize_prop(exp_a.get("property_name")):
                sub_score += 0.5

            # Metrics match
            pred_metrics = _normalize_metrics(pred_a.get("metrics"))
            exp_metrics = _normalize_metrics(exp_a.get("metrics"))

            if pred_metrics and exp_metrics:
                # Calculate overlap ratio
                overlap = len(pred_metrics & exp_metrics)
                union = len(pred_metrics | exp_metrics)
                if union > 0:
                    sub_score += 0.5 * (overlap / union)

            score += sub_score

    # 4. search_query relevance
    if expected.get("search_query"):
        total += 0.5
        pred_sq = predicted.get("search_query", "")
        exp_sq = expected["search_query"]

        if pred_sq and exp_sq:
            sim = _text_similarity(str(pred_sq), str(exp_sq))
            score += 0.5 * sim
        elif pred_sq is None and exp_sq is None:
            score += 0.5

    # 5. groupby_property
    if expected.get("groupby_property"):
        total += 0.5
        if _normalize_prop(predicted.get("groupby_property")) == _normalize_prop(expected["groupby_property"]):
            score += 0.5

    # 6. total_count
    if expected.get("total_count") is not None:
        total += 0.25
        if predicted.get("total_count") == expected["total_count"]:
            score += 0.25

    return score / total if total > 0 else 0.0


def calculate_ast_score_detailed(predicted: dict, expected: dict) -> dict:
    """
    Calculate detailed per-component AST scores for analysis.

    Returns a dict with individual component scores and the overall score.
    """
    details = {
        "collection_match": False,
        "overall_score": 0.0,
        "components": {}
    }

    if not predicted or not expected:
        return details

    # Collection
    pred_coll = predicted.get("collection_name") or predicted.get("target_collection", "")
    exp_coll = expected.get("collection_name") or expected.get("target_collection", "")
    details["collection_match"] = pred_coll == exp_coll
    details["components"]["collection"] = 1.0 if pred_coll == exp_coll else 0.0

    if not details["collection_match"]:
        details["overall_score"] = 0.0
        return details

    # Filters
    for filter_type in ["integer_property_filter", "text_property_filter", "boolean_property_filter"]:
        if expected.get(filter_type):
            pred_f = predicted.get(filter_type) or {}
            exp_f = expected[filter_type]

            prop_match = _normalize_prop(pred_f.get("property_name")) == _normalize_prop(exp_f.get("property_name"))
            op_match = _normalize_operator(pred_f.get("operator", "")) == _normalize_operator(exp_f.get("operator", ""))
            val_match = _values_match(pred_f.get("value"), exp_f.get("value"), filter_type)

            component_score = (0.5 if prop_match else 0) + (0.25 if op_match else 0) + (0.25 if val_match else 0)
            details["components"][filter_type] = {
                "score": component_score,
                "property_match": prop_match,
                "operator_match": op_match,
                "value_match": val_match,
            }

    # Aggregations
    for agg_type in ["integer_property_aggregation", "text_property_aggregation", "boolean_property_aggregation"]:
        if expected.get(agg_type):
            pred_a = predicted.get(agg_type) or {}
            exp_a = expected[agg_type]

            prop_match = _normalize_prop(pred_a.get("property_name")) == _normalize_prop(exp_a.get("property_name"))
            pred_metrics = _normalize_metrics(pred_a.get("metrics"))
            exp_metrics = _normalize_metrics(exp_a.get("metrics"))

            metrics_iou = 0.0
            if pred_metrics and exp_metrics:
                overlap = len(pred_metrics & exp_metrics)
                union = len(pred_metrics | exp_metrics)
                metrics_iou = overlap / union if union > 0 else 0

            component_score = (0.5 if prop_match else 0) + 0.5 * metrics_iou
            details["components"][agg_type] = {
                "score": component_score,
                "property_match": prop_match,
                "metrics_iou": metrics_iou,
            }

    # search_query
    if expected.get("search_query"):
        pred_sq = predicted.get("search_query", "")
        exp_sq = expected["search_query"]
        sim = _text_similarity(str(pred_sq or ""), str(exp_sq)) if pred_sq and exp_sq else 0.0
        details["components"]["search_query"] = {"score": sim, "similarity": sim}

    # groupby_property
    if expected.get("groupby_property"):
        match = _normalize_prop(predicted.get("groupby_property")) == _normalize_prop(expected["groupby_property"])
        details["components"]["groupby_property"] = {"score": 1.0 if match else 0.0, "match": match}

    # total_count
    if expected.get("total_count") is not None:
        match = predicted.get("total_count") == expected["total_count"]
        details["components"]["total_count"] = {"score": 1.0 if match else 0.0, "match": match}

    details["overall_score"] = calculate_ast_score(predicted, expected)
    return details


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def _normalize_prop(name) -> str:
    """Normalize a property name for comparison."""
    if name is None:
        return ""
    return str(name).lower().strip().replace("_", "").replace("-", "")


def _normalize_operator(op) -> str:
    """Normalize an operator string."""
    if op is None:
        return ""
    op = str(op).strip()
    # Common aliases
    aliases = {
        "==": "=",
        "eq": "=",
        "ne": "!=",
        "lt": "<",
        "gt": ">",
        "le": "<=",
        "lte": "<=",
        "ge": ">=",
        "gte": ">=",
    }
    return aliases.get(op.lower(), op)


def _normalize_metrics(metrics) -> set:
    """Normalize metrics to a set of uppercase strings."""
    if metrics is None:
        return set()
    if isinstance(metrics, str):
        return {metrics.upper().strip()}
    if isinstance(metrics, list):
        return {str(m).upper().strip() for m in metrics}
    return set()


def _values_match(pred_val, exp_val, filter_type: str = "") -> bool:
    """Check if two filter values match with type-aware normalization."""
    if pred_val is None and exp_val is None:
        return True
    if pred_val is None or exp_val is None:
        return False

    # Boolean comparison
    if "boolean" in filter_type:
        return bool(pred_val) == bool(exp_val)

    # Numeric comparison (handle int/float/str)
    try:
        return float(str(pred_val)) == float(str(exp_val))
    except (ValueError, TypeError):
        pass

    # String comparison (case-insensitive)
    return str(pred_val).lower().strip() == str(exp_val).lower().strip()


def _text_similarity(text1: str, text2: str) -> float:
    """
    Compute simple text similarity between two strings.
    Uses word overlap (Jaccard-like) as a lightweight alternative to embeddings.
    """
    if not text1 or not text2:
        return 0.0

    # Exact match
    if text1.lower().strip() == text2.lower().strip():
        return 1.0

    # Word overlap
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))

    if not words1 or not words2:
        return 0.0

    overlap = len(words1 & words2)
    union = len(words1 | words2)

    return overlap / union if union > 0 else 0.0
