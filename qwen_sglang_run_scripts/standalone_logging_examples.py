import time
from dataclasses import dataclass, field
from typing import Any

# -----------------------------------------------------------------------------
# Bedrock CallStats Tracker
# -----------------------------------------------------------------------------

@dataclass
class CallStats:
    """
    Tracker for Bedrock API calls to aggregate token usage and wall time.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_s: float = 0.0
    n_calls: int = 0
    errors: list[str] = field(default_factory=list)

    def record(self, usage: dict[str, Any] | None, elapsed: float, error: str | None = None) -> None:
        self.n_calls += 1
        self.wall_time_s += elapsed
        if usage:
            self.input_tokens += int(usage.get("inputTokens") or 0)
            self.output_tokens += int(usage.get("outputTokens") or 0)
        if error:
            self.errors.append(error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "wall_time_s": round(self.wall_time_s, 3),
            "errors": self.errors[-5:],  # last 5 for debugging
        }

def bedrock_converse_example(client, kwargs: dict, stats: CallStats):
    """
    Example of how we wrap the Bedrock client converse API to log tokens and latency.
    """
    start = time.perf_counter()
    try:
        resp = client.converse(**kwargs)
        elapsed = time.perf_counter() - start
        
        # Bedrock puts usage inside a 'usage' dictionary
        usage = resp.get("usage", {})
        stats.record(usage, elapsed)
        
        print(f">>> [Bedrock] Call: {elapsed:.3f}s | In: {usage.get('inputTokens')} | Out: {usage.get('outputTokens')} tokens")
        return resp
    except Exception as e:
        elapsed = time.perf_counter() - start
        stats.record(None, elapsed, error=str(e))
        raise

# -----------------------------------------------------------------------------
# SGLang (OpenAI API) Tracker
# -----------------------------------------------------------------------------

def sglang_chat_completion_example(client, kwargs: dict):
    """
    Example of how we wrap the local SGLang OpenAI API adapter call to log tokens and latency.
    """
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - start
        
        # OpenAI API puts usage in an object attribute
        usage = getattr(response, "usage", None)
        in_toks = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_toks = getattr(usage, "completion_tokens", 0) if usage else 0
        
        print(f">>> [SGLang/OpenAIAdapter] Qwen Call: {elapsed:.3f}s | In: {in_toks} | Out: {out_toks} tokens")
        return response
    except Exception as e:
        print(f">>> [SGLang/OpenAIAdapter] Error: {e}")
        raise
