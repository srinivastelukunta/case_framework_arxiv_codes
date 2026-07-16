"""Shared structured-output LLM caller (SDK with a stdlib HTTP fallback).

Study 1's coder (src/coding/llm_precoder.py) keeps its own call path for
stability; new code (Study 2 tool coding) uses this shared helper. Same
contract: Messages API with output_config.format json_schema, no beta header.

The anthropic SDK depends on the `jiter` native extension, which a Windows
Application Control policy on this dev box intermittently blocks (it loads
lazily inside client.messages, so the ImportError surfaces at call time). On
any ImportError we fall back to a urllib + stdlib-json call hitting the same
endpoint, so results are byte-identical and cacheable.
"""

from __future__ import annotations

import json
import os

MODEL = "claude-opus-4-8"


class StructuredRefused(RuntimeError):
    """The model declined the request (stop_reason == 'refusal')."""


def call_structured(prompt: str, schema: dict, max_tokens: int = 1200) -> str:
    """Return the model's JSON text response, validated against `schema`."""
    try:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
    except ImportError:
        return _http_call(prompt, schema, max_tokens)
    if response.stop_reason == "refusal":
        raise StructuredRefused("model refused the request")
    return next(b.text for b in response.content if b.type == "text")


def _http_call(prompt: str, schema: dict, max_tokens: int) -> str:
    import urllib.request

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
        "ANTHROPIC_AUTH_TOKEN"
    )
    if not api_key:
        raise RuntimeError("no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN set")
    body = json.dumps(
        {
            "model": MODEL,
            "max_tokens": max_tokens,
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("stop_reason") == "refusal":
        raise StructuredRefused("model refused the request")
    return next(b["text"] for b in payload["content"] if b["type"] == "text")
