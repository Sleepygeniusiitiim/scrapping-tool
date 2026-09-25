"""
OpenRouter client with the same interface as gemini_client.Gemini.

* One OpenAI-compatible endpoint for many models; OpenRouter itself falls
  back through `models` when the first choice is down or rate-limited.
* Structured output: the Pydantic schema is sent as a JSON schema
  (`response_format`), with plain JSON mode as a fallback for models or
  providers that reject schemas. Replies are validated locally either way.
* Errors map onto the Gemini exception types the pipeline already handles:
  a bad key → GeminiError("… API key …") (stops the run), no credits /
  exhausted limits → GeminiQuotaError (stops cleanly, pages are retried).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import List, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from gemini_client import GeminiError, GeminiQuotaError

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.8-flash"
# Tried in order by OpenRouter when the chosen model fails or is rate-limited.
FALLBACK_OPENROUTER_MODELS = ["openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash"]
APP_URL = "https://scrapping-tool-theta.vercel.app"
APP_TITLE = "Candidate Sourcing Agent"

_RETRYABLE = {408, 429, 500, 502, 503, 504}
_MAX_RATE_WAIT_S = 45.0
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

T = TypeVar("T", bound=BaseModel)


def _error_text(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error") or {}
        meta = err.get("metadata") or {}
        return f"{err.get('message') or resp.text[:200]} {meta.get('raw') or ''}".strip()[:400]
    except ValueError:
        return resp.text[:300]


def _parse_json(text: str) -> dict:
    """The reply as JSON — tolerating ```json fences or chatter around the object."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        m = _JSON_BLOCK.search(text)
        if not m:
            raise
        return json.loads(m.group())


class OpenRouter:
    def __init__(self, api_key: str, model: str = DEFAULT_OPENROUTER_MODEL,
                 fallbacks: Optional[List[str]] = None, max_concurrency: int = 2):
        if not api_key:
            raise GeminiError("OPENROUTER_API_KEY is required.")
        self.api_key = api_key.strip()
        self.model = (model or DEFAULT_OPENROUTER_MODEL).strip()
        self.fallbacks = [m for m in (fallbacks if fallbacks is not None else FALLBACK_OPENROUTER_MODELS)
                          if m != self.model]
        self.mode = "openrouter"
        self._sem = asyncio.Semaphore(max_concurrency)
        self._schema_ok = True      # flips off if the provider rejects json_schema

    def _payload(self, prompt: str, schema: Type[T], system_instruction: Optional[str],
                 temperature: float, thinking_budget: int) -> dict:
        schema_json = schema.model_json_schema()
        system = (system_instruction or "").strip()
        system += ("\n\nReply with ONE JSON object only — no prose, no code fences — matching this JSON schema:\n"
                   + json.dumps(schema_json, ensure_ascii=False))
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system.strip()}, {"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 8000,
            "reasoning": {"effort": "low" if thinking_budget <= 0 else "medium", "exclude": True},
        }
        if self.fallbacks:
            body["models"] = [self.model, *self.fallbacks]
        if self._schema_ok:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "strict": False, "schema": schema_json}}
        else:
            body["response_format"] = {"type": "json_object"}
        return body

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        thinking_budget: int = 0,
        max_retries: int = 4,
    ) -> T:
        headers = {"Authorization": f"Bearer {self.api_key}", "HTTP-Referer": APP_URL,
                   "X-Title": APP_TITLE, "Content-Type": "application/json"}
        last: Optional[str] = None
        async with self._sem:
            async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0)) as client:
                attempt = 0
                while attempt < max_retries:
                    attempt += 1
                    body = self._payload(prompt, schema, system_instruction, temperature, thinking_budget)
                    try:
                        resp = await client.post(OPENROUTER_URL, headers=headers, json=body)
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        last = f"{type(exc).__name__}: {exc}"
                        await asyncio.sleep(min(20.0, 2 ** attempt))
                        continue

                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                        except ValueError:
                            last = "non-JSON response"
                            continue
                        if data.get("error"):          # errors can arrive with HTTP 200 mid-route
                            last = str(data["error"])[:300]
                            await asyncio.sleep(min(20.0, 2 ** attempt))
                            continue
                        choice = (data.get("choices") or [{}])[0]
                        text = ((choice.get("message") or {}).get("content") or "").strip()
                        if data.get("model"):
                            self.model_used = data["model"]
                        if not text:
                            return schema()            # moderation / empty reply → nothing found
                        try:
                            return schema.model_validate(_parse_json(text))
                        except (ValueError, ValidationError) as exc:
                            last = f"invalid JSON from model: {str(exc)[:160]}"
                            await asyncio.sleep(1.0)
                            continue

                    msg = _error_text(resp)
                    code = resp.status_code
                    last = f"{code}: {msg}"
                    if code in (401, 403) and ("key" in msg.lower() or code == 401):
                        raise GeminiError(f"OpenRouter rejected the API key ({code}): {msg}")
                    if code == 402:
                        raise GeminiQuotaError(
                            f"OpenRouter credits are used up ({msg}). Add credits at openrouter.ai/settings/credits "
                            "or pick a ':free' model.")
                    if code == 400 and self._schema_ok and ("response_format" in msg or "schema" in msg.lower()):
                        self._schema_ok = False          # retry with plain JSON mode
                        attempt -= 1
                        continue
                    if code == 404 and self.fallbacks:
                        # Unknown model id → drop it and use the next one.
                        log.warning("OpenRouter model %s not found; using %s", self.model, self.fallbacks[0])
                        self.model = self.fallbacks.pop(0)
                        attempt -= 1
                        continue
                    if code == 429:
                        wait = float(resp.headers.get("retry-after") or 2 ** attempt)
                        if "per-day" in msg.lower() or "free-models-per-day" in msg.lower() or \
                                wait > _MAX_RATE_WAIT_S or attempt >= max_retries:
                            raise GeminiQuotaError(
                                f"OpenRouter rate limit reached ({msg}). Wait a minute, lower 'Batch size', "
                                "or add credits — ':free' models have small daily limits.")
                        await asyncio.sleep(wait + random.uniform(0.3, 1.5))
                        continue
                    if code in _RETRYABLE:
                        await asyncio.sleep(min(30.0, 2 ** attempt + random.uniform(0, 1)))
                        continue
                    raise GeminiError(f"OpenRouter error {msg[:300]}")
        raise GeminiError(f"OpenRouter call failed after {max_retries} attempts: {last}")

    async def ping(self) -> str:
        class _Pong(BaseModel):
            ok: bool = True

        await self.generate_structured('Return {"ok": true}.', _Pong, max_retries=2)
        used = getattr(self, "model_used", self.model)
        return f"OpenRouter OK ({used})"
