"""
generate/groq_client.py — Groq API client with precise rate-limit handling.

Model tier strategy:
  Primary:   meta-llama/llama-4-scout-17b-16e-instruct  (MoE, highest TPM)
  Fallback:  llama-3.3-70b-versatile                    (dense, high quality)
  Meta-task: llama-3.1-8b-instant                       (claim extraction only)

Retry strategy:
  On 429: parse x-ratelimit-reset-tokens header for exact wait duration.
          This is more precise than exponential backoff — we wait exactly
          as long as Groq tells us to, no more, no less.
  On 5xx: retry once immediately, then fall back.
  After 3 failed attempts on primary: fall back to secondary model.
  After 3 failed attempts on secondary: raise RuntimeError.
"""

import re
import time
from typing import Optional, Tuple

import groq

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)

_client: Optional[groq.Groq] = None


def _get_client() -> groq.Groq:
    global _client
    if _client is None:
        if not Config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to your .env file."
            )
        _client = groq.Groq(api_key=Config.GROQ_API_KEY)
    return _client


def _parse_reset_seconds(header_value: str) -> float:
    """
    Parse x-ratelimit-reset-tokens header value into seconds.

    Groq header format examples:
      "2s"       → 2.0
      "2.5s"     → 2.5
      "500ms"    → 0.5
      "1m30s"    → 90.0
      "1m30.5s"  → 90.5

    Falls back to 3.0 seconds if parsing fails.
    """
    if not header_value:
        return 3.0

    total = 0.0

    minutes = re.search(r"(\d+)m", header_value)
    if minutes:
        total += int(minutes.group(1)) * 60

    seconds = re.search(r"(\d+(?:\.\d+)?)s", header_value)
    if seconds:
        total += float(seconds.group(1))

    millis = re.search(r"(\d+)ms", header_value)
    if millis and not seconds:  # avoid double-counting if "s" already matched
        total += int(millis.group(1)) / 1000.0

    return total if total > 0 else 3.0


def _try_model(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Attempt generation with a specific model.
    Returns the response text, or None if all retries are exhausted.
    """
    client = _get_client()

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content
            log.info(f"Generated with {model} ({len(text)} chars)")
            return text

        except groq.RateLimitError as e:
            # Parse exact wait time from Groq's rate-limit header
            reset_header = None
            if hasattr(e, "response") and e.response is not None:
                reset_header = e.response.headers.get("x-ratelimit-reset-tokens")

            wait = _parse_reset_seconds(reset_header)
            log.warning(
                f"429 on {model} (attempt {attempt}/{max_retries}): "
                f"sleeping {wait:.1f}s (header: '{reset_header}')"
            )

            if attempt < max_retries:
                time.sleep(wait)
            else:
                log.warning(f"All retries exhausted for {model}")
                return None

        except groq.InternalServerError as e:
            log.warning(f"5xx on {model} (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(1.0)  # Brief pause before retry on server errors
            else:
                return None

        except groq.APIConnectionError as e:
            log.error(f"Connection error on {model}: {e}")
            return None

        except Exception as e:
            log.error(f"Unexpected error on {model}: {e}")
            return None

    return None


def generate(
    prompt: str,
    system_prompt: str = "",
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
) -> Tuple[str, str]:
    """
    Generate a completion using Groq with automatic fallback.

    Args:
        prompt: The user message content.
        system_prompt: Optional system-level instructions.
        model: Override primary model (defaults to Config.PRIMARY_MODEL).
        temperature: Defaults to Config.TEMPERATURE (0.1).
        max_tokens: Defaults to Config.MAX_TOKENS (1024).

    Returns:
        Tuple of (answer_text, model_used).

    Raises:
        RuntimeError: If all models are exhausted.
    """
    if temperature is None:
        temperature = Config.TEMPERATURE
    if max_tokens is None:
        max_tokens = Config.MAX_TOKENS

    primary = model or Config.PRIMARY_MODEL

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Try primary model
    result = _try_model(primary, messages, temperature, max_tokens)
    if result is not None:
        return result, primary

    # Fall back to secondary model
    if primary != Config.FALLBACK_MODEL:
        log.warning(f"Primary model exhausted. Falling back to {Config.FALLBACK_MODEL}")
        result = _try_model(Config.FALLBACK_MODEL, messages, temperature, max_tokens)
        if result is not None:
            return result, Config.FALLBACK_MODEL

    raise RuntimeError(
        f"All models exhausted. Could not generate a response. "
        f"Check your GROQ_API_KEY and rate limits."
    )


def generate_meta(prompt: str, system_prompt: str = "") -> str:
    """
    Generate using the meta-task model (llama-3.1-8b-instant).
    Used for claim extraction — cheaper, higher RPD budget.

    Args:
        prompt: Task prompt (e.g., claim extraction instruction).
        system_prompt: Optional system message.

    Returns:
        Response text string.
    """
    text, model = generate(
        prompt=prompt,
        system_prompt=system_prompt,
        model=Config.META_TASK_MODEL,
    )
    log.debug(f"Meta-task generation with {model}: {len(text)} chars")
    return text