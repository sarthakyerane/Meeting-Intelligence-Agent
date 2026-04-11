"""
LLM Chain: Groq (llama-3.3-70b) → Ollama (llama3, offline fallback)

Single call_llm() interface — callers don't care which provider fired.
Response includes provider name for the X-LLM-Provider header.
"""

import json
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


def call_llm(prompt: str, expect_json: bool = True) -> tuple[str, str]:
    """
    Try Groq first, fall back to Ollama if Groq fails or key is missing.
    Returns (response_text, provider_name).
    Raises RuntimeError only if both fail.
    """
    from app.config import get_settings
    settings = get_settings()

    providers = [
        ("groq",   _call_groq,   settings.groq_api_key),
        ("ollama", _call_ollama, settings.ollama_base_url),
    ]

    last_error = None
    for name, fn, credential in providers:
        if not credential:
            logger.info(f"[LLM] Skipping {name} — no credential set")
            continue
        try:
            t0 = time.time()
            result = fn(prompt, credential)
            elapsed = (time.time() - t0) * 1000
            logger.info(f"[LLM] {name} responded in {elapsed:.0f}ms")
            return result, name
        except Exception as e:
            logger.warning(f"[LLM] {name} failed: {e}")
            last_error = e
            continue

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def _call_groq(prompt: str, api_key: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return chat.choices[0].message.content


def _call_ollama(prompt: str, base_url: str) -> str:
    import httpx
    resp = httpx.post(
        f"{base_url}/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def parse_json_response(raw: str) -> Any:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        raise
