"""Shared utilities — .env loading, JSON parsing, Gemini client caching, retry."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# .env loader (no python-dotenv dependency)
# ---------------------------------------------------------------------------

def load_dotenv(env_path: str | Path | None = None) -> None:
    """Load KEY=VALUE pairs from a .env file into ``os.environ``.

    - Lines starting with ``#`` are treated as comments.
    - Values are stripped of surrounding quotes (``"`` and ``'``).
    - Existing environment variables are **not** overwritten.
    """
    path = Path(env_path) if env_path else PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# JSON extraction from LLM responses
# ---------------------------------------------------------------------------

def parse_llm_json(raw: str, fallback_keys: Optional[dict] = None) -> dict:
    """Extract a JSON object from a potentially messy LLM response.

    Tries three strategies in order:
    1. Direct ``json.loads`` on the stripped input.
    2. Strip markdown code fences and retry.
    3. Extract the first ``{...}`` block via regex.

    If all fail, returns *fallback_keys* merged with the raw text under
    a ``"_raw"`` key.  When *fallback_keys* is ``None`` the default
    fallback is ``{"nudge_text": <raw>, "explanation": ""}``.
    """
    # Strategy 1: direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown fences
    stripped = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 3: first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback
    if fallback_keys is not None:
        return {**fallback_keys, "_raw": raw.strip()}
    return {"nudge_text": raw.strip(), "explanation": ""}


# ---------------------------------------------------------------------------
# Gemini client cache (module-level singleton)
# ---------------------------------------------------------------------------

_gemini_clients: dict[str, object] = {}


def get_gemini_client(api_key: Optional[str] = None) -> object:
    """Return a cached ``genai.Client`` for the given API key.

    Creates a new client only on the first call per key.
    """
    from google import genai

    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if key not in _gemini_clients:
        _gemini_clients[key] = genai.Client(api_key=key)
    return _gemini_clients[key]


# ---------------------------------------------------------------------------
# Gemini API call with 429 retry (shared by all agents)
# ---------------------------------------------------------------------------

def call_gemini_with_retry(
    client: object,
    model: str,
    contents: str,
    max_retries: int = 5,
) -> str:
    """Call Gemini ``generate_content`` with automatic 429 rate-limit retry.

    Re-uses the same backoff pattern proven in ``src/embed.py``.

    Args:
        client: ``genai.Client`` instance (from ``get_gemini_client``).
        model: Model name, e.g. ``"gemini-2.5-flash"``.
        contents: Prompt text.
        max_retries: Maximum number of retry attempts on 429.

    Returns:
        The model's text response.

    Raises:
        RuntimeError: If max retries are exhausted.
        Exception: Any non-429 error is re-raised immediately.
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
            return response.text
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                match = re.search(r"retry in (\d+)", msg, re.IGNORECASE)
                wait = int(match.group(1)) + 5 if match else 65
                print(f"\n    ⏳ Rate limit (429) — warte {wait}s ... ",
                      end="", flush=True)
                time.sleep(wait)
                print("weiter.")
            else:
                raise
    raise RuntimeError(
        f"Max retries ({max_retries}) für Gemini API erreicht (model={model})"
    )
