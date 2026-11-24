"""LLM provider switch: DeepSeek (OpenAI-compatible chat API) or Anthropic Claude.

config.json -> "llm": {"provider": "deepseek" | "anthropic" | "auto"}.
"auto" picks DeepSeek when DEEPSEEK_API_KEY is set, else Claude when Anthropic
credentials exist, else none (the keyword lexicon scores headlines and the
analyst panel is skipped).

DeepSeek settings come from the environment (PointStock/.env):
    DEEPSEEK_API_KEY=...
    DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
    DEEPSEEK_MODEL=...            # headline scoring
    DEEPSEEK_PANEL_MODEL=...      # optional; analyst panel (defaults to DEEPSEEK_MODEL)

DeepSeek's JSON mode guarantees valid JSON but not a schema, so the schema is
put in the system prompt and every caller validates/clips what comes back.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

DEEPSEEK_DEFAULT_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


def anthropic_available(env=None) -> bool:
    env = env if env is not None else os.environ
    return bool(env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")
                or env.get("ANTHROPIC_PROFILE") or (Path.home() / ".config" / "anthropic").exists())


def active_provider(configured: str = "auto", env=None) -> str | None:
    env = env if env is not None else os.environ
    has_ds = bool(env.get("DEEPSEEK_API_KEY"))
    if configured == "deepseek":
        return "deepseek" if has_ds else None
    if configured == "anthropic":
        return "anthropic" if anthropic_available(env) else None
    if has_ds:
        return "deepseek"
    return "anthropic" if anthropic_available(env) else None


def deepseek_model(role: str = "news", env=None) -> str:
    env = env if env is not None else os.environ
    base = env.get("DEEPSEEK_MODEL") or DEEPSEEK_DEFAULT_MODEL
    return (env.get("DEEPSEEK_PANEL_MODEL") or base) if role == "panel" else base


def _post(url: str, payload: dict, api_key: str, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {body}") from None


def deepseek_json(system: str, user: str, schema: dict, role: str = "news",
                  max_tokens: int = 8000, timeout: int = 180, env=None) -> tuple[dict, str]:
    """One JSON-mode chat completion. Returns (parsed object, model id that answered)."""
    env = env if env is not None else os.environ
    key = env.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    base = (env.get("DEEPSEEK_BASE_URL") or DEEPSEEK_DEFAULT_BASE).rstrip("/")
    model = deepseek_model(role, env)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"{system}\n\nReply with one JSON object only, matching this "
                                          f"JSON schema exactly:\n{json.dumps(schema)}"},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    resp = _post(f"{base}/chat/completions", payload, key, timeout)
    choice = (resp.get("choices") or [{}])[0]
    if choice.get("finish_reason") not in (None, "stop"):
        raise RuntimeError(f"DeepSeek stopped with {choice.get('finish_reason')}")
    content = (choice.get("message") or {}).get("content") or ""
    if not content.strip():
        raise RuntimeError("DeepSeek returned empty content")
    return json.loads(content), resp.get("model", model)
