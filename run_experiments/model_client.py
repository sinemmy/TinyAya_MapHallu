"""
model_client.py — Unified Cohere v2 chat client with logprobs.

Always requests logprobs. Returns the full API response as a JSON-serialisable dict.
"""

import math
import os
import time

import cohere
from dotenv import load_dotenv

load_dotenv()

_client: cohere.ClientV2 | None = None


def _get_client() -> cohere.ClientV2:
    global _client
    if _client is None:
        api_key = os.getenv("COHERE_API") or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("Set COHERE_API or COHERE_API_KEY in your environment / .env file")
        _client = cohere.ClientV2(api_key=api_key)
    return _client


_MAX_RETRIES = 5
_BASE_DELAY = 2  # seconds


def query_model(
    prompt: str,
    *,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 512,
    response_format: dict | None = None,
) -> dict:
    """
    Send a single user-turn chat request and return the full response as a dict.

    Always sets ``logprobs=True``.
    Retries with exponential backoff on rate-limit (429) and transient (403, 5xx) errors.

    Returns a dict with keys:
        text, finish_reason, usage, logprobs (raw list), response_json (full)
    """
    co = _get_client()

    kwargs: dict = dict(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=True,
    )
    if response_format is not None:
        kwargs["response_format"] = response_format

    for attempt in range(_MAX_RETRIES):
        try:
            resp = co.chat(**kwargs)
            break
        except Exception as e:
            status = getattr(e, "status_code", None)
            # 5xx are server crashes — don't retry them, fail fast
            retryable = status in (403, 429)
            if not retryable or attempt == _MAX_RETRIES - 1:
                raise
            delay = _BASE_DELAY * (2 ** attempt)
            print(f"  [retry] attempt {attempt + 1}/{_MAX_RETRIES}, status {status}, "
                  f"waiting {delay}s...")
            time.sleep(delay)

    # Extract text
    text = ""
    if resp.message and resp.message.content:
        text = resp.message.content[0].text or ""

    # Extract raw logprobs list
    raw_logprobs = None
    if hasattr(resp, "logprobs") and resp.logprobs:
        raw_logprobs = _logprobs_to_serialisable(resp.logprobs)

    # Usage
    usage = None
    if resp.usage:
        usage = {
            "input_tokens": getattr(resp.usage.tokens, "input_tokens", None) if resp.usage.tokens else None,
            "output_tokens": getattr(resp.usage.tokens, "output_tokens", None) if resp.usage.tokens else None,
        }

    return {
        "text": text,
        "finish_reason": resp.finish_reason,
        "usage": usage,
        "logprobs": raw_logprobs,
    }


def _logprobs_to_serialisable(logprobs_data) -> list:
    """Convert Cohere LogprobItem objects into plain dicts for JSON storage."""
    out = []
    for token in logprobs_data:
        entry: dict = {}
        if hasattr(token, "text"):
            entry["text"] = token.text
        # Per-chunk logprobs list
        if hasattr(token, "logprobs") and token.logprobs:
            entry["logprobs"] = list(token.logprobs)
        elif hasattr(token, "logprob"):
            entry["logprob"] = token.logprob
        elif hasattr(token, "log_probability"):
            entry["logprob"] = token.log_probability
        out.append(entry)
    return out


def calculate_sequence_probability(logprobs_data: list | None) -> float:
    """
    Collapse raw logprobs (serialised list of dicts) into a single sequence probability.

    Returns exp(mean(log_probs)).  Returns 0.0 if no data.
    """
    if not logprobs_data:
        return 0.0

    log_probs: list[float] = []
    for token in logprobs_data:
        if isinstance(token, dict):
            if "logprobs" in token and token["logprobs"]:
                log_probs.extend(token["logprobs"])
            elif "logprob" in token:
                log_probs.append(token["logprob"])
        else:
            # Legacy Cohere objects (fallback)
            if hasattr(token, "logprobs") and token.logprobs:
                log_probs.extend(token.logprobs)
            elif hasattr(token, "logprob"):
                log_probs.append(token.logprob)
            elif hasattr(token, "log_probability"):
                log_probs.append(token.log_probability)

    if not log_probs:
        return 0.0

    mean_lp = sum(log_probs) / len(log_probs)
    return math.exp(mean_lp)
