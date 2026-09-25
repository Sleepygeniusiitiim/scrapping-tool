"""
Thin wrapper around the google-genai SDK for Gemini 2.5 Flash.

* Structured output: responses are constrained to a Pydantic schema and
  validated again locally.
* Retries 429 / 5xx with exponential backoff (+ jitter).
* Key mode: Google issues two kinds of keys that both work with google-genai:
    - Gemini API (AI Studio) keys, usually starting with "AIza"
    - Vertex AI express-mode keys, e.g. starting with "AQ."
  mode="auto" tries the Gemini API first and transparently switches to
  Vertex AI express mode if the key is rejected there.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional, Type, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

# gemini-2.5-* is closed to new projects; 3.x Flash is the current generation.
DEFAULT_MODEL = "gemini-3.8-flash"
T = TypeVar("T", bound=BaseModel)

_RETRYABLE = {408, 429, 500, 502, 503, 504}
# Tried in order when a model is retired (404), overloaded (503) or out of
# quota (429). Free-tier quotas and load are per model, so the next one
# usually answers straight away.
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
_SWITCH_CODES = {403, 404, 429, 503}   # 403 here = model not available to this project

# Last model that answered, reused by later requests on the same warm
# serverless instance so they don't hit an overloaded model first.
_last_good_model: Optional[str] = None


class GeminiError(RuntimeError):
    pass


class Gemini:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, mode: str = "auto", max_concurrency: int = 4):
        if not api_key:
            raise GeminiError("GEMINI_API_KEY is required.")
        self.api_key = api_key.strip()
        self.model = _last_good_model if (_last_good_model and model == DEFAULT_MODEL) else model
        self._fallbacks = [m for m in FALLBACK_MODELS if m != self.model]
        # auto → start with the Gemini API (AI Studio). Newer AI Studio keys can also
        # start with "AQ.", so the key prefix doesn't tell us the endpoint.
        self.mode = mode if mode in ("gemini", "vertex") else "gemini"
        self._auto = mode == "auto"
        self._first_error: Optional[Exception] = None   # error that triggered an endpoint switch
        self._left_vertex = False                         # already fell back from a disabled Vertex API
        self._client = self._make_client(self.mode)
        self._sem = asyncio.Semaphore(max_concurrency)

    # -- client ------------------------------------------------------------
    def _make_client(self, mode: str) -> genai.Client:
        if mode == "vertex":
            return genai.Client(vertexai=True, api_key=self.api_key)
        return genai.Client(api_key=self.api_key)

    def _switch_mode(self) -> bool:
        """In auto mode, flip Gemini API <-> Vertex express once."""
        if not self._auto:
            return False
        self._auto = False  # only flip once
        self.mode = "vertex" if self.mode == "gemini" else "gemini"
        self._client = self._make_client(self.mode)
        log.info("Gemini key rejected on first endpoint; switched to %s mode", self.mode)
        return True

    def _thinking_config(self, thinking_budget: int) -> types.ThinkingConfig:
        """Gemini 2.x takes a token budget; Gemini 3.x takes a thinking level."""
        if self.model.startswith("gemini-2"):
            return types.ThinkingConfig(thinking_budget=thinking_budget)
        return types.ThinkingConfig(thinking_level="low" if thinking_budget <= 0 else "medium")

    # -- core call ---------------------------------------------------------
    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        thinking_budget: int = 0,
        max_retries: int = 5,
    ) -> T:
        """Call Gemini with a JSON schema and return a validated model instance."""
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=self._thinking_config(thinking_budget),
        )
        last_exc: Optional[Exception] = None
        async with self._sem:
            attempt = 0
            while attempt < max_retries:
                attempt += 1
                try:
                    resp = await self._client.aio.models.generate_content(
                        model=self.model, contents=prompt, config=config
                    )
                    global _last_good_model
                    _last_good_model = self.model
                    parsed = getattr(resp, "parsed", None)
                    if isinstance(parsed, schema):
                        return parsed
                    text = (resp.text or "").strip()
                    if not text:
                        # Safety block or empty candidate — treat as "nothing found".
                        return schema()
                    return schema.model_validate_json(text)
                except genai_errors.APIError as exc:
                    last_exc = exc
                    code = getattr(exc, "code", None)
                    msg = str(exc)
                    low = msg.lower()
                    wrong_endpoint = code == 401 or "api key not valid" in low or "api_key_invalid" in low or (
                        "api key" in low and "not supported" in low)
                    if code in (400, 401, 403) and wrong_endpoint:
                        # Key belongs to the other endpoint (AI Studio vs Vertex express).
                        if self._switch_mode():
                            self._first_error = exc
                            attempt -= 1  # the switch doesn't count as a retry
                            continue
                        raise GeminiError(f"Gemini rejected the API key ({code}): {msg[:300]}") from exc
                    if self._first_error is not None and code == 403 and (
                            "service_disabled" in low or "has not been used" in low or "aiplatform" in low):
                        # We switched to Vertex, but Vertex isn't enabled for this project, so the
                        # original endpoint's error is the real one.
                        first = self._first_error
                        raise GeminiError(f"Gemini rejected the API key ({first.code}): {str(first)[:300]}") from first
                    vertex_disabled = code == 403 and (
                        "service_disabled" in low or "has not been used" in low or "aiplatform" in low)
                    if vertex_disabled and self.mode == "vertex" and not self._left_vertex:
                        # Vertex AI isn't enabled for this Google project → use the Gemini API instead.
                        log.warning("Vertex AI API disabled for this key's project; switching to the Gemini API")
                        self._left_vertex = True
                        self._auto = False
                        self.mode = "gemini"
                        self._client = self._make_client("gemini")
                        attempt -= 1
                        continue
                    if code == 400 and "thinking" in low and config.thinking_config is not None:
                        # This model doesn't accept the thinking setting → retry without it.
                        config.thinking_config = None
                        attempt -= 1
                        continue
                    if code in _SWITCH_CODES and self._fallbacks:
                        # Retired / overloaded / out of quota → try the next model now.
                        nxt = self._fallbacks.pop(0)
                        log.warning("Gemini model %s returned %s; switching to %s", self.model, code, nxt)
                        self.model = nxt
                        config.thinking_config = self._thinking_config(thinking_budget)
                        attempt -= 1  # a model switch doesn't count as a retry
                        continue
                    if code in _RETRYABLE:
                        await asyncio.sleep(min(60.0, (2 ** attempt) + random.uniform(0, 1.5)))
                        continue
                    raise GeminiError(f"Gemini error {code}: {msg[:300]}") from exc
                except (ValidationError, ValueError) as exc:
                    # Malformed / truncated JSON — retry once or twice.
                    last_exc = exc
                    await asyncio.sleep(1.0)
                    continue
                except (asyncio.TimeoutError, ConnectionError, OSError) as exc:
                    last_exc = exc
                    await asyncio.sleep(min(30.0, 2 ** attempt))
                    continue
        raise GeminiError(f"Gemini call failed after {max_retries} attempts: {last_exc}")

    async def ping(self) -> str:
        """Tiny call used by the UI 'Test connection' button."""
        class _Pong(BaseModel):
            ok: bool = True

        await self.generate_structured('Return {"ok": true}.', _Pong, max_retries=2)
        return f"Gemini OK ({self.model}, {self.mode} endpoint)"
