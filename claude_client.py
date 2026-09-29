"""
Claude (Anthropic) for the *thinking* steps only: understanding the command and planning the searches.
Page reading, classification and fit checks never use it — they stay on the cheaper chain.

Same interface as the other AI clients (generate_structured / ping), so it sits at the front of the
reasoning chain (ai_router.reasoning); when it fails or runs out of credit, the usual chain takes over.

Key: ANTHROPIC_API_KEY (Vercel) or the Claude key saved on the page. Model: CLAUDE_MODEL (default
claude-opus-5-5). Effort: CLAUDE_EFFORT (low | medium | high; default medium).
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Type, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from gemini_client import GeminiError, GeminiQuotaError

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

T = TypeVar("T", bound=BaseModel)


class Claude:
    label = "Claude"

    def __init__(self, api_key: str, model: str = "", effort: str = ""):
        if not api_key:
            raise GeminiError("ANTHROPIC_API_KEY is required.")
        self.model = (model or os.getenv("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL).strip()
        self.effort = (effort or os.getenv("CLAUDE_EFFORT") or "medium").strip()
        self.client = anthropic.AsyncAnthropic(api_key=api_key.strip(), timeout=120.0, max_retries=2)
        self._schema_ok = True      # off if the API rejects this schema for structured output

    def _args(self, prompt: str, system: str) -> dict:
        return dict(model=self.model, max_tokens=16000, system=system,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"effort": self.effort},
                    betas=[FALLBACK_BETA], fallbacks="default")   # a declined request is re-run on another model

    async def generate_structured(self, prompt: str, schema: Type[T], system_instruction: Optional[str] = None,
                                  temperature: float = 0.2, thinking_budget: int = 0, max_retries: int = 2) -> T:
        # temperature / thinking_budget are ignored: current Claude models choose their own thinking depth
        # (effort), and sampling parameters are not accepted.
        system = (system_instruction or "").strip()
        try:
            if self._schema_ok:
                try:
                    resp = await self.client.beta.messages.parse(output_format=schema,
                                                                 **self._args(prompt, system))
                    self._check(resp)
                    if resp.parsed_output is not None:
                        return resp.parsed_output
                except anthropic.BadRequestError as exc:
                    if _out_of_credit(exc):
                        raise
                    self._schema_ok = False          # schema not supported → plain JSON below
            resp = await self.client.beta.messages.create(**self._args(
                prompt, system + "\n\nReply with ONE JSON object only — no prose, no code fences — matching this "
                                 "JSON schema:\n" + json.dumps(schema.model_json_schema(), ensure_ascii=False)))
            self._check(resp)
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            try:
                return schema.model_validate(json.loads(text))
            except ValueError:
                m = _JSON_BLOCK.search(text)
                if not m:
                    raise GeminiError(f"Claude returned no JSON: {text[:160]}")
                return schema.model_validate(json.loads(m.group()))
        except ValidationError as exc:
            raise GeminiError(f"Claude reply did not match the schema: {str(exc)[:200]}")
        except anthropic.AuthenticationError as exc:
            raise GeminiError(f"Claude rejected the API key: {exc.message}")
        except anthropic.PermissionDeniedError as exc:
            raise GeminiError(f"Claude API key lacks permission: {exc.message}")
        except anthropic.NotFoundError as exc:
            raise GeminiError(f"Claude model {self.model} not found: {exc.message}")
        except anthropic.RateLimitError as exc:
            raise GeminiQuotaError(f"Claude rate limit reached: {exc.message}")
        except anthropic.BadRequestError as exc:
            if _out_of_credit(exc):
                raise GeminiQuotaError(f"Claude credits are used up ({exc.message}). Add credits at "
                                       "platform.claude.com → Billing, or switch provider.")
            raise GeminiError(f"Claude bad request: {exc.message}")
        except anthropic.APIStatusError as exc:
            raise GeminiError(f"Claude error {exc.status_code}: {exc.message}")
        except anthropic.APIConnectionError as exc:
            raise GeminiError(f"Claude unreachable: {exc}")

    @staticmethod
    def _check(resp) -> None:
        if resp.stop_reason == "refusal":
            raise GeminiError("Claude declined this request")
        if resp.stop_reason == "max_tokens":
            raise GeminiError("Claude reply was cut off (max_tokens)")

    async def ping(self) -> str:
        class _Pong(BaseModel):
            ok: bool = True

        await self.generate_structured('Return {"ok": true}.', _Pong)
        return f"Claude OK ({self.model}, planning only)"


def _out_of_credit(exc: anthropic.APIStatusError) -> bool:
    return bool(re.search(r"credit balance|billing|insufficient", str(exc.message), re.I))
