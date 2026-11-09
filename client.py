"""Small OpenAI-compatible chat completions client (stdlib only)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class OpenAICompatibleClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 60.0):
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout

    def chat(self, messages: list[dict[str, Any]], *, model: str,
             temperature: float = 0.0, max_tokens: int = 512) -> str:
        if not self.api_key: raise RuntimeError("OPENAI_API_KEY is required for online agent calls")
        body = json.dumps({"model": model, "messages": messages,
                           "temperature": temperature, "max_tokens": max_tokens}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                return "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return str(content)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"OpenAI-compatible chat request failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAI-compatible chat request failed: {exc}") from exc
