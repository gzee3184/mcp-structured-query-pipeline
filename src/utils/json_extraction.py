"""
JSON Extraction Utility for LLM Responses

This module provides robust JSON extraction from LLM responses, handling common
edge cases encountered with Ollama and NVIDIA NIM models.

EXTRACTION STRATEGY (in order of priority):
1. Try to find JSON in ```json code blocks (most reliable)
2. Try to find JSON in generic ``` code blocks  
3. Look for outermost {} or [] brackets
4. Try parsing the entire response as JSON
5. Return None if all strategies fail

ROBUSTNESS FEATURES:
- Handles nested code blocks and escaped characters
- Strips markdown formatting and explanatory text
- Validates extracted JSON is parseable before returning
- Provides detailed error information for debugging
"""

import json
import re
from typing import Optional, Any, Tuple, TypeVar, Type
from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)


class JSONExtractionError(Exception):
    """Raised when JSON extraction fails."""
    def __init__(self, message: str, raw_response: str, attempted_strategies: list[str]):
        super().__init__(message)
        self.raw_response = raw_response
        self.attempted_strategies = attempted_strategies


def extract_json_from_response(response: str) -> Optional[str]:
    """
    Extract JSON string from an LLM response using multiple strategies.
    
    This function is designed to handle common LLM output patterns:
    - JSON wrapped in ```json ... ``` code blocks
    - JSON wrapped in generic ``` ... ``` code blocks
    - JSON embedded within explanatory text
    - Raw JSON responses
    
    Args:
        response: The raw LLM response string
        
    Returns:
        Extracted JSON string if found and valid, None otherwise
        
    Example:
        >>> extract_json_from_response('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
        
        >>> extract_json_from_response('Here is the result: {"key": "value"}')
        '{"key": "value"}'
    """
    if not response or not isinstance(response, str):
        return None
    
    response = response.strip()
    strategies_tried = []
    
    # Strategy 1: Extract from ```json ... ``` code blocks
    strategies_tried.append("json_code_block")
    json_block_pattern = r'```json\s*([\s\S]*?)\s*```'
    json_matches = re.findall(json_block_pattern, response, re.IGNORECASE)
    if json_matches:
        # Take the last match (often the final/refined answer)
        for match in reversed(json_matches):
            extracted = match.strip()
            if _is_valid_json(extracted):
                return extracted
    
    # Strategy 2: Extract from generic ``` ... ``` code blocks
    strategies_tried.append("generic_code_block")
    generic_block_pattern = r'```\s*([\s\S]*?)\s*```'
    generic_matches = re.findall(generic_block_pattern, response)
    if generic_matches:
        for match in reversed(generic_matches):
            extracted = match.strip()
            # Skip if it looks like a language identifier line
            if extracted and not extracted.startswith(('python', 'javascript', 'typescript', 'bash', 'sh')):
                if _is_valid_json(extracted):
                    return extracted
    
    # Strategy 3: Find outermost JSON object {} or array []
    strategies_tried.append("bracket_matching")
    extracted = _extract_by_bracket_matching(response)
    if extracted and _is_valid_json(extracted):
        return extracted
    
    # Strategy 4: Try parsing entire response as JSON
    strategies_tried.append("full_response")
    if _is_valid_json(response):
        return response
    
    # Strategy 5: Try with common prefixes/suffixes stripped
    strategies_tried.append("stripped_response")
    cleaned = _strip_common_wrappers(response)
    if cleaned != response and _is_valid_json(cleaned):
        return cleaned
    
    return None


def _extract_by_bracket_matching(text: str) -> Optional[str]:
    """
    Extract JSON by finding matching brackets.
    
    Finds the first { or [ and its matching closing bracket,
    handling nested structures correctly.
    """
    # Find first opening bracket
    first_brace = text.find('{')
    first_bracket = text.find('[')
    
    if first_brace == -1 and first_bracket == -1:
        return None
    
    # Determine which comes first
    if first_brace == -1:
        start = first_bracket
        open_char, close_char = '[', ']'
    elif first_bracket == -1:
        start = first_brace
        open_char, close_char = '{', '}'
    else:
        if first_brace < first_bracket:
            start = first_brace
            open_char, close_char = '{', '}'
        else:
            start = first_bracket
            open_char, close_char = '[', ']'
    
    # Find matching closing bracket
    depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    
    return None


def _strip_common_wrappers(text: str) -> str:
    """Strip common LLM response wrappers."""
    # Remove common prefixes
    prefixes = [
        "Here is the JSON:",
        "Here's the JSON:",
        "The JSON is:",
        "Result:",
        "Output:",
        "Response:",
    ]
    
    result = text.strip()
    for prefix in prefixes:
        if result.lower().startswith(prefix.lower()):
            result = result[len(prefix):].strip()
            break
    
    # Remove trailing explanations after JSON
    # Find the last } or ] and truncate there
    last_brace = result.rfind('}')
    last_bracket = result.rfind(']')
    last_close = max(last_brace, last_bracket)
    
    if last_close != -1 and last_close < len(result) - 1:
        result = result[:last_close + 1]
    
    return result


def _is_valid_json(text: str) -> bool:
    """Check if text is valid JSON."""
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def parse_json(text: str) -> Optional[Any]:
    """
    Parse a JSON string into a Python object.
    
    Args:
        text: JSON string to parse
        
    Returns:
        Parsed Python object, or None if parsing fails
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def validate_and_parse(
    response: str, 
    schema: Type[T],
    raise_on_error: bool = False
) -> Tuple[Optional[T], Optional[str]]:
    """
    Extract JSON from response and validate against a Pydantic schema.
    
    This is the main entry point for validated JSON extraction.
    
    Args:
        response: Raw LLM response string
        schema: Pydantic model class to validate against
        raise_on_error: If True, raise exceptions instead of returning None
        
    Returns:
        Tuple of (validated_model, error_message)
        - On success: (model_instance, None)
        - On failure: (None, error_description)
        
    Example:
        >>> class MyModel(BaseModel):
        ...     name: str
        ...     value: int
        >>> result, error = validate_and_parse('{"name": "test", "value": 42}', MyModel)
        >>> result.name
        'test'
    """
    # Step 1: Extract JSON
    json_str = extract_json_from_response(response)
    if json_str is None:
        error_msg = f"Could not extract JSON from response. First 100 chars: {response[:100]}..."
        if raise_on_error:
            raise JSONExtractionError(error_msg, response, ["all strategies failed"])
        return None, error_msg
    
    # Step 2: Parse JSON
    try:
        parsed_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing failed: {e}. Extracted: {json_str[:100]}..."
        if raise_on_error:
            raise JSONExtractionError(error_msg, response, ["parse_json"])
        return None, error_msg
    
    # Step 3: Validate against schema
    try:
        validated = schema.model_validate(parsed_data)
        return validated, None
    except ValidationError as e:
        error_msg = f"Schema validation failed: {e}"
        if raise_on_error:
            raise e
        return None, error_msg


def extract_multiple_json_objects(response: str) -> list[str]:
    """
    Extract all JSON objects/arrays from a response.
    
    Useful when the LLM returns multiple JSON structures.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        List of valid JSON strings found in the response
    """
    results = []
    
    # Find all code blocks first
    json_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(json_block_pattern, response, re.IGNORECASE)
    
    for match in matches:
        if _is_valid_json(match.strip()):
            results.append(match.strip())
    
    # If no code blocks, try bracket matching for multiple objects
    if not results:
        remaining = response
        while remaining:
            extracted = _extract_by_bracket_matching(remaining)
            if extracted and _is_valid_json(extracted):
                results.append(extracted)
                # Find where this JSON ends and continue
                idx = remaining.find(extracted) + len(extracted)
                remaining = remaining[idx:]
            else:
                break
    
    return results
