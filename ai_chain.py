"""
Fallback chain over several AI providers with the Gemini client's interface.

The provider chosen on the page is tried first. When it runs out of credits,
hits a daily / rate limit or rejects its key, the chain switches to the next
provider that has a key (free tiers first) and stays there for the rest of the
request. Any other error only skips to the next provider for that one call.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

from gemini_client import GeminiError, GeminiQuotaError

log = logging.getLogger(__name__)

# After the chosen provider: free tiers first, then paid ones.
FALLBACK_ORDER = ["gemini", "groq", "cerebras", "mistral", "openrouter", "deepseek", "kimi"]


def _exhausted(exc: GeminiError) -> bool:
    """Out of credits / limits or a bad key: this provider won't work again in this request."""
    return isinstance(exc, GeminiQuotaError) or "API key" in str(exc)


class AIChain:
    def __init__(self, entries: List[Tuple[str, Callable[[], object]]]):
        """entries: (label, factory) in the order to try. Factories build clients lazily."""
        if not entries:
            raise GeminiError("No AI provider has a key.")
        self._entries = entries
        self._clients: dict[int, object] = {}
        self._idx = 0
        self.notices: List[str] = []

    @property
    def labels(self) -> List[str]:
        return [label for label, _ in self._entries]

    def _client(self, i: int):
        if i not in self._clients:
            self._clients[i] = self._entries[i][1]()
        return self._clients[i]

    def _switch(self, i: int, exc: GeminiError) -> None:
        if self._idx != i:           # a concurrent call already moved on
            return
        self._idx = i + 1
        if self._idx < len(self._entries):
            note = (f"{self._entries[i][0]} unavailable ({str(exc)[:140]}) — switched to "
                    f"{self._entries[self._idx][0]}.")
            log.warning(note)
            self.notices.append(note)

    async def generate_structured(self, *args, **kwargs):
        errors: List[GeminiError] = []
        i = self._idx
        while i < len(self._entries):
            try:
                return await self._client(i).generate_structured(*args, **kwargs)
            except GeminiError as exc:
                errors.append(exc)
                if _exhausted(exc):
                    self._switch(i, exc)
                    i = max(i + 1, self._idx)
                else:
                    i += 1
        if not errors:
            raise GeminiQuotaError("Every AI provider is out of credits or limits: " + "; ".join(self.notices))
        if any(isinstance(e, GeminiQuotaError) for e in errors):
            raise GeminiQuotaError("Every AI provider is out of credits or limits — last error: "
                                   f"{str(errors[-1])[:300]}")
        raise errors[-1]

    async def ping(self) -> str:
        """Checks every provider in the chain, so the page shows which fallbacks work."""
        parts, first_ok = [], None
        for i, (label, _) in enumerate(self._entries):
            try:
                msg = await self._client(i).ping()
                parts.append(f"✓ {msg}")
                first_ok = first_ok or msg
            except Exception as exc:
                parts.append(f"✗ {label}: {str(exc)[:120]}")
        if not first_ok:
            raise GeminiError("No AI provider works — " + " · ".join(parts))
        return " · ".join(parts) if len(parts) > 1 else first_ok
