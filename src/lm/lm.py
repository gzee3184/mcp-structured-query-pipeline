"""LLM service abstraction layer for multi-provider tool-calling.

Wraps OpenAI, Bedrock, Anthropic, Cohere, Together, and Ollama behind a
uniform interface. The pipeline's only LLM interaction point.
"""

import os

try:
    import ollama
except ImportError:
    ollama = None

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:
    boto3 = None

import openai
import anthropic
import cohere
from typing import Literal
from pydantic import BaseModel
from src.models import TestLMConnectionModel, ResponseOrToolCalls
from src.utils.weaviate_fc_utils import (
    OpenAITool,
    AnthropicTool,
    OllamaTool,
    CohereTool,
    TogetherAITool
)
from src.lm.db_gorilla_prompts import (
    zero_shot_baseline,
    weaviate_query_api_docs
)
from src.utils.json_extraction import extract_json_from_response, validate_and_parse
import json
import time
import asyncio
from functools import wraps

# google models are accessed through the openai SDK with a check on `model_name`
# together models are accessed through the openai SDK with a different base URL
# grok models are accessed through the openai SDK with a different base URL
LMModelProvider = Literal["ollama", "openai", "anthropic", "cohere", "together", "bedrock"]


# Bedrock returns toolUse dicts; the pipeline expects .function.name/.arguments (OpenAI format).
class BedrockToolCall:
    """Shim adapting Bedrock Converse toolUse dicts to OpenAI tool_call shape."""
    def __init__(self, name: str, arguments: dict):
        self.function = type('Function', (), {
            'name': name,
            'arguments': json.dumps(arguments) if isinstance(arguments, dict) else arguments
        })()

# need to add models to this...!
# fixes test_lm.py


def resilient_api_call(max_retries: int = 5, max_delay: float = 5.0, base_delay: float = 1.0):
    """
    Decorator for resilient API calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 5)
        max_delay: Maximum delay between retries in seconds (default: 5.0)
        base_delay: Initial delay in seconds (default: 1.0)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()
                    
                    # Check if it's a retryable error
                    retryable_errors = [
                        'rate limit', 'ratelimit', '429', 'too many requests',
                        'timeout', 'connection', 'temporarily unavailable',
                        '503', '502', '500'
                    ]
                    
                    is_retryable = any(err in error_str for err in retryable_errors)
                    
                    if not is_retryable and attempt == 0:
                        # Not a retryable error, raise immediately
                        raise e
                    
                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff, capped at max_delay
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        print(f"\033[93m[Retry {attempt + 1}/{max_retries}] {type(e).__name__}: {str(e)[:100]}... Retrying in {delay:.1f}s\033[0m")
                        time.sleep(delay)
                    else:
                        print(f"\033[91m[Failed] Max retries ({max_retries}) exceeded\033[0m")
            
            # All retries exhausted
            raise last_exception
        return wrapper
    return decorator


class LMService():
    def __init__(
            self,
            model_provider: LMModelProvider,
            model_name: str,
            api_key: str | None = None
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        # Per-call metadata for evaluation: latency + token counts, updated after every API call
        self.last_call_meta = {"latency_s": 0.0, "input_tokens": 0, "output_tokens": 0}
        match self.model_provider:
            case "ollama":
                ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                self.lm_client = ollama.Client(host=ollama_host)
            case "openai":
                # Auto-detect: models with "/" are NVIDIA NIM, others are native OpenAI
                if "/" in self.model_name:
                    base_url = "https://integrate.api.nvidia.com/v1"
                else:
                    base_url = None  # Uses default OpenAI endpoint
                self.lm_client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=120.0  # 2-minute timeout to prevent socket hangs
            )
            case "bedrock":
                if boto3 is None:
                    raise ImportError("boto3 is required for Bedrock provider. Install with: pip install boto3")
                bedrock_config = BotoConfig(
                    read_timeout=120,
                    connect_timeout=10,
                    retries={"max_attempts": 3, "mode": "adaptive"}
                )
                self.lm_client = boto3.client(
                    "bedrock-runtime",
                    region_name=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")),
                    config=bedrock_config
                )
                # Auto-resolve inference profile IDs: Anthropic/Meta models
                # need "us." prefix for Bedrock on-demand; Qwen uses bare IDs.
                if not self.model_name.startswith(("us.", "global.")):
                    if self.model_name.startswith(("anthropic.", "meta.")):
                        self.model_name = f"us.{self.model_name}"
                        print(f"\033[96m[Bedrock] Auto-resolved to inference profile: {self.model_name}\033[0m")
                # if self.model_name in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"]:
                #     print("\033[96mUsing Gemini through the OpenAI SDK.\033[0m")
                #     self.lm_client = openai.OpenAI(
                #         api_key=api_key,
                #         base_url="https://generativelanguage.googleapis.com/v1beta/"
                #     )
                # elif self.model_name.startswith("grok-"):
                #     print("\033[96mUsing Grok through the OpenAI SDK.\033[0m")
                #     self.lm_client = openai.OpenAI(
                #         api_key=api_key,
                #         base_url="https://api.x.ai/v1"
                #     )
                # elif "/" in self.model_name and any(prefix in self.model_name for prefix in ["meta/", "qwen/", "nvidia/", "mistralai/"]):
                #     print("\033[96mUsing NVIDIA NIM through the OpenAI SDK.\033[0m")
                #     self.lm_client = openai.OpenAI(
                #         api_key=api_key,
                #         base_url="https://integrate.api.nvidia.com/v1"
                #     )
                # else:
                #     self.lm_client = openai.OpenAI(
                #         api_key=api_key
                #     )
            case "anthropic":
                self.lm_client = anthropic.Anthropic(
                    api_key=api_key
                )
            case "cohere":
                self.lm_client = cohere.ClientV2(
                    api_key=api_key
                )
            case "together":
                print("\033[96mUsing Together.ai through the OpenAI SDK.\033[0m")
                self.lm_client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.together.xyz/v1"
                )
            case _:
                raise ValueError(f"Unsupported model provider: {self.model_provider}") 
        
        print("Running connection test:")
        self.connection_test()
    def generate(
        self, 
        prompt: str,
        system_prompt: str = None, 
        output_model: BaseModel | None = None
        ) -> str | dict:
        
        # Fix: Define default prompt inside function to avoid SyntaxError in signature
        if system_prompt is None:
            system_prompt = """You are a database query expert. Use the supplied tools to assist the user.

            CRITICAL TOOL CALLING RULES:
            1. Collection names are CASE-SENSITIVE - use exact names from the tool schema
            2. Property names must match the schema EXACTLY (check descriptions carefully)
            3. Valid aggregation metrics:
            - Integer properties: MIN, MAX, MEAN, MEDIAN, MODE, SUM
            - Text properties: TOP_OCCURRENCES
            - Boolean properties: PERCENTAGE_TRUE
            - NEVER use COUNT as a metric
            4. Filter structure must include: property_name, operator, and value
            5. Valid operators: =, !=, <, >, <=, >=, LIKE

            Always call the tool with complete, correctly formatted parameters."""
        match self.model_provider:
            case "ollama":
                messages = [{"role": "user", "content": prompt}]
                # Note, this isn't implemented
                if output_model:
                    # Create an instance with default values
                    model_instance = output_model(generic_response="Hello! This is a test response.")
                    # Append output format instructions if model provided
                    messages[0]["content"] += f"\nRespond with the following JSON: {model_instance.model_dump_json()}"
                
                response = self.lm_client.chat(
                    model="qwen2.5:72b",
                    messages=messages,
                    format="json" if output_model else None
                )
                return response["message"]["content"]
                
            case "openai":
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Follow the response format instructions."},
                    {"role": "user", "content": prompt}
                ]
                
                # Detect NVIDIA NIM
                is_nvidia_nim = (hasattr(self.lm_client, '_base_url') and 
                                'nvidia.com' in str(self.lm_client._base_url))
                
                if output_model:
                    if is_nvidia_nim:
                        # NVIDIA NIM: Use simple JSON mode (no schema enforcement)
                        print(f"\033[96m[NVIDIA NIM] Using JSON mode\033[0m")
                        
                        # Add JSON schema to prompt
                        json_schema = output_model.model_json_schema()
                        json_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{json.dumps(json_schema, indent=2)}"
                        
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON."},
                            {"role": "user", "content": json_prompt}
                        ]
                        
                        # Use standard chat completion
                        response = self.lm_client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=0.1
                        )
                        
                        # Parse JSON response using robust extraction
                        content = response.choices[0].message.content
                        
                        # Use the new robust JSON extraction utility
                        validated_model, error = validate_and_parse(content, output_model)
                        
                        if validated_model is not None:
                            return validated_model
                        else:
                            # Extraction or validation failed, try simpler fallback
                            print(f"\033[93m[WARN] JSON validation failed: {error}\033[0m")
                            
                            # Try direct JSON extraction as fallback
                            json_str = extract_json_from_response(content)
                            if json_str:
                                try:
                                    parsed_data = json.loads(json_str)
                                    return output_model(**parsed_data)
                                except Exception as e2:
                                    print(f"\033[91m[ERROR] Fallback parsing also failed: {e2}\033[0m")
                            
                            print(f"\033[91m[ERROR] Could not parse response. First 200 chars: {content[:200]}\033[0m")
                            return content
                    else:
                        # Regular OpenAI: Use beta.parse
                        response = self.lm_client.beta.chat.completions.parse(
                            model=self.model_name,
                            messages=messages,
                            response_format=output_model
                        )
                        return response.choices[0].message.parsed
                else:
                    print(self.model_name)
                    response = self.lm_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages
                    )
                    return response.choices[0].message.content
                    
            case "anthropic":
                messages = [{"role": "user", "content": prompt}]
                response = self.lm_client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    messages=messages,
                )
                return response.content[0].text

            case "cohere":
                messages = [{"role": "user", "content": prompt}]
                if output_model:
                    raise NotImplementedError("Not implemented.")
                    '''
                    # Create an instance with default values
                    model_instance = output_model(generic_response="Hello! This is a test response.")
                    # Append output format instructions if model provided
                    messages[0]["content"] += f"\nRespond with the following JSON format: {model_instance.model_dump_json()}"
                    '''
                response = self.lm_client.chat(
                    model=self.model_name,
                    messages=messages
                )
                return response

            case "together":
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Follow the response format instructions."},
                    {"role": "user", "content": prompt}
                ]
                response = self.lm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                return response.choices[0].message.content

            case "bedrock":
                # Bedrock Converse API — universal across all model families
                # Anthropic models support a system parameter; for others embed in user message
                is_anthropic_model = "anthropic." in self.model_name
                converse_kwargs = {
                    "modelId": self.model_name,
                    "inferenceConfig": {"temperature": 0.1, "maxTokens": 1024},
                }
                if is_anthropic_model:
                    converse_kwargs["system"] = [{"text": "You are a helpful assistant. Follow the response format instructions."}]
                    converse_kwargs["messages"] = [{"role": "user", "content": [{"text": prompt}]}]
                else:
                    # Embed system prompt in user message for non-Anthropic models
                    combined = "You are a helpful assistant. Follow the response format instructions.\n\n" + prompt
                    converse_kwargs["messages"] = [{"role": "user", "content": [{"text": combined}]}]

                if output_model:
                    # Add JSON schema to prompt for structured output
                    json_schema = output_model.model_json_schema()
                    json_instruction = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(json_schema, indent=2)}"
                    # Append to the last user message
                    last_msg = converse_kwargs["messages"][-1]["content"][-1]
                    last_msg["text"] += json_instruction

                _t0 = time.time()
                response = self.lm_client.converse(**converse_kwargs)
                _elapsed = time.time() - _t0
                _usage = response.get("usage", {})
                self.last_call_meta = {
                    "latency_s": round(_elapsed, 3),
                    "input_tokens": _usage.get("inputTokens", 0),
                    "output_tokens": _usage.get("outputTokens", 0),
                }
                content = response["output"]["message"]["content"][0]["text"]

                if output_model:
                    validated_model, error = validate_and_parse(content, output_model)
                    if validated_model is not None:
                        return validated_model
                    # Fallback: try direct extraction
                    json_str = extract_json_from_response(content)
                    if json_str:
                        try:
                            return output_model(**json.loads(json_str))
                        except Exception as e2:
                            print(f"\033[91m[ERROR] Bedrock JSON fallback failed: {e2}\033[0m")
                    print(f"\033[91m[ERROR] Bedrock: Could not parse response. First 200 chars: {content[:200]}\033[0m")
                    return content
                return content

            case _:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def connection_test(self) -> None:
        prompt = "Say hello"
        
        # Skip structured output test for NVIDIA NIM
        is_nvidia_nim = (self.model_provider == "openai" and 
                        hasattr(self.lm_client, '_base_url') and 
                        'nvidia.com' in str(self.lm_client._base_url))
        
        # Only use structured output for regular OpenAI
        output_model = TestLMConnectionModel if (self.model_provider == "openai" and not is_nvidia_nim) else None
        
        print(f"\033[96mPrinting prompt: {prompt}\nwith output model: {output_model}\033[0m")
        response = self.generate(prompt, output_model)
        print("\033[92mLM Connection test result:\033[0m")
        print(response)

    # def connection_test(self) -> None:
    #     prompt = "Say hello"
    #     output_model = TestLMConnectionModel if self.model_provider == "openai" else None
    #     print(f"\033[96mPrinting prompt: {prompt}\nwith output model: {output_model}\033[0m")
    #     response = self.generate(prompt, output_model)
    #     print("\033[92mLM Connection test result:\033[0m")
    #     print(response)
    
    def one_step_function_selection_test(
            self,
            prompt: str,
            tools: list[OpenAITool] | list[AnthropicTool] | list[OllamaTool] | list[CohereTool] | list[TogetherAITool],
            parallel_tool_calls: bool = False,
            primary_fix: str = "none",
            tool_schema: str = "v1"
        ) -> dict | None:
        # --- ENHANCED SYSTEM PROMPT ---
        # This explicitly addresses the semantic errors (case sensitivity, COUNT vs SUM, etc.)
        # preventing the model from achieving high AST scores.
        #
        # tool_schema="v2" swaps to the extended schema prompt (output_properties,
        # filters list, order_by, distinct). See weaviate_fc_utils.py for the
        # matching tool definition. V1 is the default — unchanged behavior.

        # Multi-collection section varies based on primary_fix option
        if primary_fix == "prompt":
            mc_section = """MULTI-COLLECTION QUERIES:
When a question requires data from MULTIPLE collections, you MUST:
- Identify ALL relevant collections, then determine which is PRIMARY using the rule below.
- List ALL other needed collections in additional_collections with their roles.
- Specify join_keys showing how the collections connect via shared properties (foreign keys).
- Look at the [Database group: ...] annotations to identify related collections.
If the question only involves ONE collection, leave additional_collections and join_keys empty.

PRIMARY COLLECTION SELECTION — OUTPUT-AS-PRIMARY RULE:
The PRIMARY collection (collection_name) must be the OUTPUT TABLE: the collection whose
PROPERTIES contain the data the user wants to SEE or RETRIEVE. Collections used only to
filter or narrow results go in additional_collections.

Think step by step: "What specific data does the user want returned? Which collection has those properties?"
- "List names of patients with low albumin" → Output = names → names are in Patient → PRIMARY = Patient
- "What was the overall rating for Aaron Mooy?" → Output = rating → rating is in PlayerAttributes → PRIMARY = PlayerAttributes
- "How many customers had monthly consumption > 1000?" → Output = consumption count → consumption is in Yearmonth → PRIMARY = Yearmonth
- "What is the sprint speed and agility of player X?" → Output = speed, agility → these are in PlayerAttributes → PRIMARY = PlayerAttributes

The primary is the collection whose properties appear in the OUTPUT — not the collection the question is topically "about"."""
        else:
            mc_section = """MULTI-COLLECTION QUERIES:
When a question requires data from MULTIPLE collections, you MUST:
- Set collection_name to the PRIMARY collection (the main one the question is about).
- List ALL other needed collections in additional_collections with their roles.
- Specify join_keys showing how the collections connect via shared properties (foreign keys).
- Look at the [Database group: ...] annotations to identify related collections.
If the question only involves ONE collection, leave additional_collections and join_keys empty."""

        if tool_schema == "v2":
            # V2 schema has output_properties (SELECT equivalent), a filters list,
            # order_by, distinct, limit. The primary collection (collection_name)
            # is decoupled from what the query returns.
            system_prompt = """You are a database query expert. Use the supplied tool `query_database` to answer questions.

MANDATORY: You MUST ALWAYS respond with a tool call. NEVER respond with plain text. If anything is uncertain, call the tool with your best guess.

CRITICAL RULES:
1. Collection names are CASE-SENSITIVE — use the EXACT strings from the enum.
2. Property names must match the schema EXACTLY (check descriptions carefully).
3. `output_properties` is REQUIRED. It is the SELECT equivalent — list EXACTLY the properties the question asks to RETURN/LIST/COUNT/SUM, with the collection that owns it. Do NOT add extra "helpful" properties the question didn't ask for. Do NOT leave it empty.
4. `collection_name` (the PRIMARY collection) is the one the query iterates over — typically the collection whose properties appear in a WHERE predicate or that forms the entity being queried. It is NOT necessarily the same collection as the output.
5. Use `filters` (an array) for WHERE predicates. Each entry has property_name, operator, value, and collection. Combine multiple with `filter_boolean_op` (default AND).
6. Use `order_by` for ORDER BY clauses (with direction). Use `limit` for LIMIT. Use `distinct: true` for SELECT DISTINCT. Use `having_filters` for post-aggregation predicates (HAVING clauses).
7. Valid filter operators: =, !=, <, >, <=, >=, LIKE, IN, BETWEEN, IS_NULL, IS_NOT_NULL.
8. Valid aggregation functions (for output_properties or order_by): NONE, COUNT, SUM, MIN, MAX, MEAN, MEDIAN, MODE, TOP_OCCURRENCES, PERCENTAGE_TRUE, PERCENTAGE_FALSE, TOTAL_TRUE, TOTAL_FALSE.

MULTI-COLLECTION QUERIES:
- Set `collection_name` to the PRIMARY collection. The PRIMARY is typically the collection that OWNS THE OUTPUT COLUMNS the question asks to return (the SELECT-target table). Filter-only collections go in `additional_collections`.
- Use this tie-breaker: if output spans multiple collections, pick the one that contains the SUBJECT ENTITY of the question (who/what the question is asking about).
- If the question asks for an aggregation like COUNT on column X, the PRIMARY is the collection owning X.
- List all other needed collections in `additional_collections`.
- Specify `join_keys` connecting them via foreign keys.
- `output_properties` may span multiple collections — list each with its owning `collection`.

Concrete pattern:
- "List patients' IDs where lab WBC<3.5" → output columns are Patient.ID → PRIMARY = Patient, filter is on Laboratory.
- "Count customers with consumption > 1000" → output is COUNT(customer.id) → PRIMARY = Customers, filter/measurement is on Yearmonth.
- "List the top constructor names by points" → output is constructor.name + aggregated points → PRIMARY = Constructors (the subject), points come from ConstructorResults.

FEW-SHOT EXAMPLES:

Example 1 — "For patients with albumin < 3.5, list their ID, sex, diagnosis."
{
  "collection_name": "ThrombosisPredictionPatient",
  "additional_collections": [{"collection_name": "ThrombosisPredictionLaboratory", "role": "holds albumin values for filtering"}],
  "join_keys": [{"left_collection": "ThrombosisPredictionPatient", "left_property": "ID", "right_collection": "ThrombosisPredictionLaboratory", "right_property": "ID"}],
  "output_properties": [
    {"collection": "ThrombosisPredictionPatient", "property_name": "ID"},
    {"collection": "ThrombosisPredictionPatient", "property_name": "SEX"},
    {"collection": "ThrombosisPredictionPatient", "property_name": "Diagnosis"}
  ],
  "filters": [
    {"collection": "ThrombosisPredictionLaboratory", "property_name": "ALB", "operator": "<", "value": 3.5, "property_type": "integer"}
  ]
}

Example 2 — "How many customers paid in Euro and had consumption over 1000?"
{
  "collection_name": "DebitCardSpecializingCustomers",
  "additional_collections": [{"collection_name": "DebitCardSpecializingYearmonth", "role": "holds consumption"}],
  "join_keys": [{"left_collection": "DebitCardSpecializingCustomers", "left_property": "CustomerID", "right_collection": "DebitCardSpecializingYearmonth", "right_property": "Customer ID"}],
  "output_properties": [
    {"collection": "DebitCardSpecializingCustomers", "property_name": "CustomerID", "aggregation": "COUNT"}
  ],
  "filters": [
    {"collection": "DebitCardSpecializingCustomers", "property_name": "Currency", "operator": "=", "value": "EUR", "property_type": "text"},
    {"collection": "DebitCardSpecializingYearmonth", "property_name": "Consumption", "operator": ">", "value": 1000, "property_type": "integer"}
  ],
  "filter_boolean_op": "AND"
}

Example 3 — "Top 5 constructors by total points in the 2010 season."
{
  "collection_name": "Formula1Constructorresults",
  "additional_collections": [
    {"collection_name": "Formula1Constructors", "role": "provides constructor names"},
    {"collection_name": "Formula1Races", "role": "scopes by season"}
  ],
  "join_keys": [
    {"left_collection": "Formula1Constructorresults", "left_property": "constructor Id", "right_collection": "Formula1Constructors", "right_property": "constructor Id"},
    {"left_collection": "Formula1Constructorresults", "left_property": "race Id", "right_collection": "Formula1Races", "right_property": "race ID"}
  ],
  "output_properties": [
    {"collection": "Formula1Constructors", "property_name": "name"},
    {"collection": "Formula1Constructorresults", "property_name": "points", "aggregation": "SUM"}
  ],
  "filters": [
    {"collection": "Formula1Races", "property_name": "year", "operator": "=", "value": 2010, "property_type": "integer"}
  ],
  "group_by_properties": ["name"],
  "order_by": [{"collection": "Formula1Constructorresults", "property_name": "points", "aggregation": "SUM", "direction": "DESC"}],
  "limit": 5
}

Example 4 — "List all female patients whose white blood cell count is below 3.5 and total cholesterol is above 250."
Hint: female refers to SEX = 'F'; white blood cell count below 3.5 refers to WBC < 3.5; total cholesterol above 250 refers to T-CHO > 250
{
  "collection_name": "ThrombosisPredictionPatient",
  "additional_collections": [{"collection_name": "ThrombosisPredictionLaboratory", "role": "holds lab measurements"}],
  "join_keys": [{"left_collection": "ThrombosisPredictionPatient", "left_property": "ID", "right_collection": "ThrombosisPredictionLaboratory", "right_property": "ID"}],
  "output_properties": [
    {"collection": "ThrombosisPredictionPatient", "property_name": "ID"},
    {"collection": "ThrombosisPredictionPatient", "property_name": "SEX"}
  ],
  "filters": [
    {"collection": "ThrombosisPredictionPatient", "property_name": "SEX", "operator": "=", "value": "F", "property_type": "text"},
    {"collection": "ThrombosisPredictionLaboratory", "property_name": "WBC", "operator": "<", "value": 3.5, "property_type": "integer"},
    {"collection": "ThrombosisPredictionLaboratory", "property_name": "T-CHO", "operator": ">", "value": 250, "property_type": "integer"}
  ],
  "filter_boolean_op": "AND"
}

Example 5 — "What are the names and nationalities of circuits in countries that have more than 3 races, ordered by name?"
{
  "collection_name": "Formula1Circuits",
  "output_properties": [
    {"property_name": "name"},
    {"property_name": "country"}
  ],
  "filters": [
    {"property_name": "country", "operator": "IN", "value": ["Italy", "USA", "UK", "Germany", "Spain"], "property_type": "text"}
  ],
  "order_by": [{"property_name": "name", "direction": "ASC"}]
}

NEVER reply with plain text. Always call the tool with complete, correctly formatted parameters."""
        else:
            system_prompt = f"""You are a database query expert. Use the supplied tools to assist the user.

MANDATORY: You MUST ALWAYS respond with a tool call. NEVER respond with text. Even if the schema seems incomplete or you are unsure which collection is best, you MUST call the tool with your best guess. Pick the most relevant collection from the available options. Do NOT explain why you cannot answer — just call the tool.

CRITICAL RULES:
1. Collection names are CASE-SENSITIVE. You must use the EXACT name from the 'enum' list (e.g., use 'Restaurants', not 'restaurants').
2. Property names must match the schema EXACTLY (check descriptions carefully).
3. Valid aggregation metrics are STRICTLY limited to the enums provided:
   - Integer properties: MIN, MAX, MEAN, MEDIAN, MODE, SUM
   - Text properties: TOP_OCCURRENCES
   - Boolean properties: PERCENTAGE_TRUE, TOTAL_TRUE, TOTAL_FALSE
   - NEVER use 'COUNT' as a metric in an aggregation object. 'COUNT' is not a valid enum value for metrics.
4. Filter structure must include: property_name, operator, and value.
5. Valid operators: =, !=, <, >, <=, >=, LIKE.

{mc_section}

Always call the tool with complete, correctly formatted parameters matching the schema. NEVER reply with plain text."""

        # Combine with API docs
        enhanced_prompt = weaviate_query_api_docs + "\n" + prompt
        # if self.model_provider == "openai":
        #     messages = [
        #         {
        #             "role": "system",
        #             "content": system_prompt
        #         },
        #         {
        #             "role": "user",
        #             "content": enhanced_prompt
        #         }
        #     ]
        #     if self.model_name in ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]:
        #         response = self.lm_client.chat.completions.create(
        #             model=self.model_name,
        #             messages=messages,
        #             tools=tools
        #         )
        #     else:
        #         response = self.lm_client.chat.completions.create(
        #             model=self.model_name,
        #             messages=messages,
        #             tools=tools,
        #             parallel_tool_calls=parallel_tool_calls
        #         )

        #     if self.model_name in ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]:
        #         tool_calls = response.choices[0].message.tool_calls
        #         for tool_call in tool_calls:
        #             tool_call.function.arguments = tool_call.function.arguments.replace('\\u003e', '>')
        #             tool_call.function.arguments = tool_call.function.arguments.replace('\\u003c', '<')
        #     else:
        #         tool_calls = response.choices[0].message.tool_calls
            
        #     if tool_calls:
        #         return tool_calls
        #     return None
        
        # if self.model_provider == "ollama":
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": prompt
        #         }
        #     ]
        #     response = self.lm_client.chat(
        #         model=self.model_name,
        #         messages=messages,
        #         tools=tools
        #     )
        #     if not response["message"].get("tool_calls"):
        #         return None
        #     else:
        #         # maybe also worth looking into how parallel fc is interfaced with ollama for this
        #         tool = response["message"]["tool_calls"][0]
        #         return tool["function"]["arguments"]
        if self.model_provider == "openai":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_prompt}
            ]
            
            try:
                _t0 = time.time()
                response = self.lm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="required",
                    parallel_tool_calls=parallel_tool_calls,
                    temperature=0.1  # Low temperature for consistency
                )
                _elapsed = time.time() - _t0
                _usage = response.usage
                self.last_call_meta = {
                    "latency_s": round(_elapsed, 3),
                    "input_tokens": _usage.prompt_tokens if _usage else 0,
                    "output_tokens": _usage.completion_tokens if _usage else 0,
                }

                tool_calls = response.choices[0].message.tool_calls
                if tool_calls:
                    return tool_calls
                return None
            except Exception as e:
                print(f"Error in OpenAI/NIM generation: {e}")
                return None
        
        if self.model_provider == "ollama":
            messages=[
                {"role": "system", "content": system_prompt}, # Added System Prompt for Ollama
                {"role": "user", "content": enhanced_prompt}
            ]
            response = self.lm_client.chat(
                model=self.model_name,
                messages=messages,
                tools=tools
            )
            if not response["message"].get("tool_calls"):
                return None
            else:
                tool = response["message"]["tool_calls"][0]
                return tool["function"]["arguments"]

        if self.model_provider == "anthropic":
            max_retries = 5
            base_delay = 15
            
            for attempt in range(max_retries):
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    response = self.lm_client.messages.create(
                        model=self.model_name,
                        max_tokens=4096,
                        tools=[tool.model_dump() for tool in tools],
                        messages=messages
                    )
                    if response.stop_reason == "tool_use":
                        tool_use = next(block for block in response.content if block.type == "tool_use")
                        tool_name, tool_input = tool_use.name, tool_use.input
                        # future work will need to parse the name as well
                        return tool_input
                    else:
                        return None
                        
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        raise e
                    
                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** attempt)  # 10, 20, 40, 80, 160 seconds
                    print(f"Anthropic API call failed, retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)

        if self.model_provider == "bedrock":
            # Convert OpenAI tool format → Bedrock Converse toolSpec format
            bedrock_tools = []
            for tool in tools:
                tool_dict = tool.model_dump() if hasattr(tool, 'model_dump') else tool
                func = tool_dict.get("function", tool_dict)
                input_schema = func.get("parameters", {})
                # Bedrock Converse requires inputSchema.json to be a valid JSON Schema object
                # Remove any keys that might cause validation issues
                if "additionalProperties" not in input_schema:
                    input_schema["additionalProperties"] = False
                bedrock_tools.append({
                    "toolSpec": {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "inputSchema": {"json": input_schema}
                    }
                })

            # Build messages — Anthropic models support system param, others don't
            is_anthropic_model = "anthropic." in self.model_name
            # toolChoice "any" forces tool use (like tool_choice="required")
            # Meta/Llama models don't support it — fall back to "auto"
            is_meta_model = "meta." in self.model_name
            if is_meta_model:
                tool_choice = {"auto": {}}
            else:
                tool_choice = {"any": {}}
            converse_kwargs = {
                "modelId": self.model_name,
                "toolConfig": {
                    "tools": bedrock_tools,
                    "toolChoice": tool_choice,
                },
                "inferenceConfig": {"temperature": 0.1},
            }
            if is_anthropic_model:
                converse_kwargs["system"] = [{"text": system_prompt}]
                converse_kwargs["messages"] = [
                    {"role": "user", "content": [{"text": enhanced_prompt}]}
                ]
            else:
                # Embed system prompt in user message for Qwen/Llama/Meta models
                combined_prompt = system_prompt + "\n\n" + enhanced_prompt
                converse_kwargs["messages"] = [
                    {"role": "user", "content": [{"text": combined_prompt}]}
                ]

            max_retries = 5
            base_delay = 5
            for attempt in range(max_retries):
                try:
                    _t0 = time.time()
                    response = self.lm_client.converse(**converse_kwargs)
                    _elapsed = time.time() - _t0
                    _usage = response.get("usage", {})
                    self.last_call_meta = {
                        "latency_s": round(_elapsed, 3),
                        "input_tokens": _usage.get("inputTokens", 0),
                        "output_tokens": _usage.get("outputTokens", 0),
                    }

                    # Parse response — look for toolUse blocks
                    output_message = response.get("output", {}).get("message", {})
                    content_blocks = output_message.get("content", [])

                    tool_calls = []
                    for block in content_blocks:
                        if "toolUse" in block:
                            tu = block["toolUse"]
                            tool_calls.append(BedrockToolCall(
                                name=tu["name"],
                                arguments=tu["input"]
                            ))

                    if tool_calls:
                        return tool_calls

                    # Some models (Llama, Qwen) may return JSON text instead of tool calls
                    # when toolChoice is "auto". Try to parse it as a tool call.
                    for block in content_blocks:
                        if "text" in block:
                            text_content = block["text"]
                            print(f"\033[93m[WARN] Bedrock returned text instead of tool call: {text_content[:200]}\033[0m")
                            # Attempt to extract JSON and create a synthetic tool call
                            try:
                                json_str = extract_json_from_response(text_content)
                                if json_str:
                                    parsed = json.loads(json_str)
                                    # If parsed JSON has collection_name, it's likely a query tool call
                                    if isinstance(parsed, dict) and "collection_name" in parsed:
                                        tool_name = bedrock_tools[0]["toolSpec"]["name"] if bedrock_tools else "query_database"
                                        return [BedrockToolCall(name=tool_name, arguments=parsed)]
                            except (json.JSONDecodeError, Exception):
                                pass
                    return None

                except Exception as e:
                    error_str = str(e)
                    if attempt == max_retries - 1:
                        print(f"\033[91m[ERROR] Bedrock tool call failed after {max_retries} attempts: {e}\033[0m")
                        return None
                    # Exponential backoff for throttling
                    if "ThrottlingException" in error_str or "Too many requests" in error_str.lower():
                        delay = base_delay * (2 ** attempt)
                        print(f"\033[93m[WARN] Bedrock throttled, retrying in {delay}s (attempt {attempt + 1}/{max_retries})\033[0m")
                        time.sleep(delay)
                    else:
                        print(f"\033[91m[ERROR] Bedrock API error: {e}\033[0m")
                        return None

        if self.model_provider == "cohere":
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the supplied tools to assist the user."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            response = self.lm_client.chat(
                model=self.model_name,
                messages=messages,
                tools=[tool.model_dump() for tool in tools]
            )
            
            if response.message.tool_calls:
                # Return first tool call arguments for consistency with other providers
                return json.loads(response.message.tool_calls[0].function.arguments)
            return None

        if self.model_provider == "together":
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the supplied tools to assist the user."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            response = self.lm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=[tool.model_dump() for tool in tools],
                tool_choice="auto"
            )
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                return json.loads(tool_calls[0].function.arguments)
            return None

        else:
            raise ValueError(f"Function calling not yet supported for the LMService with {self.model_provider}")

    def call_tools_with_structured_outputs(
            self,
            prompt: str,
            tools: list[OpenAITool]
    ):
        # This will just assume the 1 Tool harcoded into the `ResponseOrToolCalls` model
        if self.model_provider != "openai":
            raise ValueError("call_tools_with_structured_outputs is only supported for openai.")

        tools_description = tools[0].model_dump() # Note, this will cause problems when extending to multiple tools -- shouldn't matter for now.

        # Construct messages guiding the model to return JSON in `output_model` format
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. Follow the response format instructions and use the tools if needed. Here is a description of the available tools {tools_description}"},
            {"role": "user", "content": prompt}
        ]

        # Use the beta parse endpoint to get a structured output
        response = self.lm_client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=ResponseOrToolCalls
        )

        # Extract the parsed structured response
        parsed_response = response.choices[0].message.parsed
        if parsed_response.use_tools == True:
            return parsed_response.tool_calls # returns the arguments to be parsed in the run script
        else:
            return None

    # =========================================================================
    # MCP PROGRESSIVE QUERY (Two-Phase Pipeline)
    # =========================================================================

    # V1: simple discovery -> generation. V2 adds a self-correction loop.
    def progressive_query(
        self,
        prompt: str,
        mcp_server,  # MCPServer instance
        sandbox=None,  # Optional StructuredSandbox for validation
        system_prompt: str = None
    ) -> dict:
        """
        Two-phase query using MCP progressive disclosure.
        
        Phase 1 (Discovery): Use MCP tools to find relevant collection
        Phase 2 (Generation): Build query with just that collection's schema
        
        This reduces context window usage by ~85% compared to loading all schemas upfront.
        
        Args:
            prompt: Natural language query from user
            mcp_server: MCPServer instance with schema registry
            sandbox: Optional StructuredSandbox for query validation
            system_prompt: Optional custom system prompt
        
        Returns:
            Dict with discovery_result, query_args, and validation (if sandbox provided)
        """
        from src.mcp.discovery_tools import get_discovery_tools_for_provider
        from src.utils.weaviate_fc_utils import build_weaviate_query_tool_for_openai
        
        if system_prompt is None:
            system_prompt = """You are a database query expert using progressive schema discovery.

WORKFLOW:
1. First, search for relevant collections using search_collections
2. Then, get the schema for the matched collection using get_collection_schema
3. Finally, use the schema to construct your query

Be precise with collection and property names - they are case-sensitive."""
        
        result = {
            "discovery_result": None,
            "discovered_collection": None,
            "query_args": None,
            "validation": None,
            "error": None
        }
        
        # Phase 1: Discovery - find relevant collection
        discovery_tools = get_discovery_tools_for_provider(self.model_provider)
        
        discovery_prompt = f"""Search for the most relevant database collection to answer this query:

User Query: {prompt}

First, call search_collections to find matching collections.
Then, call get_collection_schema to get the schema for the best match."""
        
        try:
            discovery_response = self.one_step_function_selection_test(
                prompt=discovery_prompt,
                tools=discovery_tools,
                parallel_tool_calls=False
            )
            
            if discovery_response is None:
                result["error"] = "Discovery phase returned no tool calls"
                return result
            
            # Parse discovery response
            tool_call = discovery_response[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            # Execute the discovery tool
            if tool_name == "search_collections":
                matches = mcp_server.search_collections(
                    query=tool_args.get("query", ""),
                    detail_level=tool_args.get("detail_level", "name")
                )
                result["discovery_result"] = matches
                
                # Get first match as target collection
                if isinstance(matches, list) and len(matches) > 0:
                    result["discovered_collection"] = matches[0]
                elif isinstance(matches, dict) and len(matches) > 0:
                    result["discovered_collection"] = list(matches.keys())[0]
            
            elif tool_name == "get_collection_schema":
                collection_name = tool_args.get("collection_name", "")
                schema = mcp_server.get_collection_schema(collection_name)
                result["discovery_result"] = schema
                result["discovered_collection"] = collection_name
            
            else:
                result["error"] = f"Unexpected discovery tool: {tool_name}"
                return result
            
        except Exception as e:
            result["error"] = f"Discovery phase error: {str(e)}"
            return result
        
        # Phase 2: Generation - try top N collection matches (Selection technique from BIRD)
        # Instead of just taking matches[0], try up to 5 and stop on first success
        discovery_matches = result["discovery_result"]
        if isinstance(discovery_matches, dict):
            discovery_matches = list(discovery_matches.keys())
        
        if not discovery_matches:
            result["error"] = "No collections discovered"
            return result
        
        MAX_COLLECTION_ATTEMPTS = 5
        attempted_collections = []
        
        for collection in discovery_matches[:MAX_COLLECTION_ATTEMPTS]:
            attempted_collections.append(collection)
            
            # Get full schema for the collection
            schema = mcp_server.get_collection_schema(collection, compressed=False)
            if schema is None:
                continue
            
            # Build tool for just this collection
            schema_desc = f"Collection: {collection}\nProperties:\n"
            for prop_name, prop_info in schema.get("properties", {}).items():
                prop_type = prop_info.get("type", "unknown")
                schema_desc += f"  - {prop_name} ({prop_type})\n"
            
            query_tool = build_weaviate_query_tool_for_openai(
                collections_description=schema_desc,
                collections_list=[collection]
            )
            
            try:
                query_response = self.one_step_function_selection_test(
                    prompt=prompt,
                    tools=[query_tool],
                    parallel_tool_calls=False
                )
                
                if query_response is not None:
                    # LLM called the tool
                    query_args = json.loads(query_response[0].function.arguments)
                    result["discovered_collection"] = collection
                    result["query_args"] = query_args
                    
                    # Phase 3: Validation (if sandbox provided)
                    if sandbox is not None:
                        from src.models import WeaviateQuery
                        from src.models import IntPropertyFilter, TextPropertyFilter, BooleanPropertyFilter
                        from src.models import IntAggregation, TextAggregation, BooleanAggregation
                        
                        # Build WeaviateQuery from args
                        query_obj = WeaviateQuery(
                            target_collection=query_args.get("collection_name", ""),
                            search_query=query_args.get("search_query"),
                            groupby_property=query_args.get("groupby_property")
                        )
                        
                        # Add filters
                        if query_args.get("integer_property_filter"):
                            f = query_args["integer_property_filter"]
                            query_obj.integer_property_filter = IntPropertyFilter(
                                property_name=f.get("property_name", ""),
                                operator=f.get("operator", "="),
                                value=f.get("value", 0)
                            )
                        
                        # Validate
                        validation_result = sandbox.validate(query_obj)
                        result["validation"] = {
                            "valid": validation_result.valid,
                            "errors": validation_result.errors
                        }
                        
                        # If validation failed, try next collection (MULTI-MATCH FIX)
                        if not validation_result.valid:
                            continue  # Try next collection
                    
                    # Success - return (either no sandbox, or validation passed)
                    return result
                else:
                    # LLM didn't call tool, try next collection
                    continue
                    
            except Exception as e:
                # Continue to next collection
                continue
        
        # All attempts failed
        result["error"] = f"Query generation failed for all {len(attempted_collections)} attempted collections: {attempted_collections}"
        return result

    def progressive_query_v2(
        self,
        prompt: str,
        mcp_server,
        sandbox=None,
        system_prompt: str = None,
        max_corrections: int = 2
    ) -> dict:
        """
        Enhanced two-phase query with self-correction loop.

        Improvements over progressive_query:
        1. Self-correction: feeds validation errors back to the LLM instead of
           just moving to the next collection
        2. Preventive instructions: injects common error warnings based on
           tracked error patterns
        3. Better error context: includes fix suggestions in correction prompts

        Args:
            prompt: Natural language query from user
            mcp_server: MCPServer instance with schema registry
            sandbox: Optional StructuredSandbox for validation + self-correction
            system_prompt: Optional custom system prompt
            max_corrections: Max self-correction attempts per collection (default: 2)

        Returns:
            Dict with discovery_result, query_args, validation, and correction_attempts
        """
        from src.mcp.discovery_tools import get_discovery_tools_for_provider
        from src.utils.weaviate_fc_utils import build_weaviate_query_tool_for_openai

        if system_prompt is None:
            system_prompt = """You are a database query expert using progressive schema discovery.

WORKFLOW:
1. First, search for relevant collections using search_collections
2. Then, get the schema for the matched collection using get_collection_schema
3. Finally, use the schema to construct your query

Be precise with collection and property names - they are case-sensitive."""

        # Add preventive instructions from sandbox error tracking
        if sandbox is not None and hasattr(sandbox, 'get_preventive_instructions'):
            preventive = sandbox.get_preventive_instructions()
            if preventive:
                system_prompt += f"\n\n{preventive}"

        result = {
            "discovery_result": None,
            "discovered_collection": None,
            "query_args": None,
            "validation": None,
            "error": None,
            "correction_attempts": 0,
            "difficulty": None,
        }

        # Phase 1: Discovery (same as v1)
        discovery_tools = get_discovery_tools_for_provider(self.model_provider)

        discovery_prompt = f"""Search for the most relevant database collection to answer this query:

User Query: {prompt}

First, call search_collections to find matching collections.
Then, call get_collection_schema to get the schema for the best match."""

        try:
            discovery_response = self.one_step_function_selection_test(
                prompt=discovery_prompt,
                tools=discovery_tools,
                parallel_tool_calls=False
            )

            if discovery_response is None:
                result["error"] = "Discovery phase returned no tool calls"
                return result

            tool_call = discovery_response[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            if tool_name == "search_collections":
                matches = mcp_server.search_collections(
                    query=tool_args.get("query", ""),
                    detail_level=tool_args.get("detail_level", "name")
                )
                result["discovery_result"] = matches
                if isinstance(matches, list) and len(matches) > 0:
                    result["discovered_collection"] = matches[0]
                elif isinstance(matches, dict) and len(matches) > 0:
                    result["discovered_collection"] = list(matches.keys())[0]
            elif tool_name == "get_collection_schema":
                collection_name = tool_args.get("collection_name", "")
                schema = mcp_server.get_collection_schema(collection_name)
                result["discovery_result"] = schema
                result["discovered_collection"] = collection_name
            else:
                result["error"] = f"Unexpected discovery tool: {tool_name}"
                return result

        except Exception as e:
            result["error"] = f"Discovery phase error: {str(e)}"
            return result

        # Phase 2: Generation with self-correction
        discovery_matches = result["discovery_result"]
        if isinstance(discovery_matches, dict):
            discovery_matches = list(discovery_matches.keys())

        if not discovery_matches:
            result["error"] = "No collections discovered"
            return result

        MAX_COLLECTION_ATTEMPTS = 5

        for collection in discovery_matches[:MAX_COLLECTION_ATTEMPTS]:
            schema = mcp_server.get_collection_schema(collection, compressed=False)
            if schema is None:
                continue

            schema_desc = f"Collection: {collection}\nProperties:\n"
            for prop_name, prop_info in schema.get("properties", {}).items():
                prop_type = prop_info.get("type", "unknown")
                schema_desc += f"  - {prop_name} ({prop_type})\n"

            query_tool = build_weaviate_query_tool_for_openai(
                collections_description=schema_desc,
                collections_list=[collection]
            )

            # Try generation + self-correction loop
            current_prompt = prompt
            for correction_attempt in range(max_corrections + 1):
                try:
                    query_response = self.one_step_function_selection_test(
                        prompt=current_prompt,
                        tools=[query_tool],
                        parallel_tool_calls=False
                    )

                    if query_response is None:
                        break  # LLM didn't call tool, try next collection

                    query_args = json.loads(query_response[0].function.arguments)
                    result["discovered_collection"] = collection
                    result["query_args"] = query_args

                    # Validation + self-correction
                    if sandbox is not None:
                        from src.models import WeaviateQuery, IntPropertyFilter, TextPropertyFilter, BooleanPropertyFilter

                        query_obj = WeaviateQuery(
                            target_collection=query_args.get("collection_name", ""),
                            search_query=query_args.get("search_query"),
                            groupby_property=query_args.get("groupby_property")
                        )

                        if query_args.get("integer_property_filter"):
                            f = query_args["integer_property_filter"]
                            query_obj.integer_property_filter = IntPropertyFilter(
                                property_name=f.get("property_name", ""),
                                operator=f.get("operator", "="),
                                value=f.get("value", 0)
                            )
                        if query_args.get("text_property_filter"):
                            f = query_args["text_property_filter"]
                            query_obj.text_property_filter = TextPropertyFilter(
                                property_name=f.get("property_name", ""),
                                operator=f.get("operator", "="),
                                value=f.get("value", "")
                            )
                        if query_args.get("boolean_property_filter"):
                            f = query_args["boolean_property_filter"]
                            query_obj.boolean_property_filter = BooleanPropertyFilter(
                                property_name=f.get("property_name", ""),
                                operator=f.get("operator", "="),
                                value=f.get("value", True)
                            )

                        # Use enhanced validation with suggestions
                        if hasattr(sandbox, 'validate_with_suggestions'):
                            validation_result = sandbox.validate_with_suggestions(query_obj)
                        else:
                            validation_result = sandbox.validate(query_obj)

                        result["validation"] = {
                            "valid": validation_result.valid,
                            "errors": validation_result.errors
                        }

                        if validation_result.valid:
                            result["correction_attempts"] = correction_attempt
                            return result

                        # Self-correction: feed errors back to LLM
                        if correction_attempt < max_corrections and hasattr(sandbox, 'build_correction_prompt'):
                            sandbox.track_error_pattern(validation_result.errors)
                            current_prompt = sandbox.build_correction_prompt(
                                original_query=prompt,
                                query_args=query_args,
                                validation_result=validation_result,
                                schema_desc=schema_desc
                            )
                            result["correction_attempts"] = correction_attempt + 1
                            continue
                        else:
                            break  # No more corrections, try next collection
                    else:
                        return result  # No sandbox, accept as-is

                except Exception:
                    break

        result["error"] = f"Query generation failed for all attempted collections"
        return result

'''
Note, vLLM function call snippet:

https://docs.vllm.ai/en/latest/getting_started/examples/offline_chat_with_tools.html
'''