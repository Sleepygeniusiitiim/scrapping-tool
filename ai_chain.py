"""
Fallback chain over several AI providers with the Gemini client's interface.

The provider chosen on the page is tried first. When it runs out of credits,
hits a daily / rate limit or rejects its key, the chain switches to the next
provider that has a key (free tiers first) and stays there for the rest of the
request. Any other error only skips to the next provider for that one call.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

from gemini_client import GeminiError, GeminiQuotaError

log = logging.getLogger(__name__)

# After the chosen provider: free tiers first, then paid ones.
FALLBACK_ORDER = ["gemini", "groq", "cerebras", "mistral", "sambanova", "nvidia", "github", "openrouter",
                  "deepseek", "kimi"]

# Providers that ran out of credits / limits recently, remembered by this server process so the next batches
# don't spend a call (and a log line) finding out again. label → (until, reason).
_DOWN: Dict[str, Tuple[float, str]] = {}
DOWN_MINUTES = 30


def _mark_down(label: str, exc: Exception) -> None:
    msg = str(exc)
    minutes = 180 if re.search(r"daily|per-day|per day|midnight", msg, re.I) else DOWN_MINUTES
    _DOWN[label] = (time.time() + minutes * 60, msg[:200])


def down_now() -> Dict[str, str]:
    now = time.time()
    return {k: why for k, (until, why) in _DOWN.items() if until > now}


def _exhausted(exc: GeminiError) -> bool:
    """Out of credits / limits or a bad key: this provider won't work again in this request."""
    return isinstance(exc, GeminiQuotaError) or "API key" in str(exc)


class AIChain:
    def __init__(self, entries: List[Tuple[str, Callable[[], object]]]):
        """entries: (label, factory) in the order to try. Factories build clients lazily."""
        if not entries:
            raise GeminiError("No AI provider has a key.")
        down = down_now()
        up = [e for e in entries if e[0] not in down]
        # Skip providers known to be out of credits; if every one is, try them all anyway (limits reset).
        self.skipped = [e[0] for e in entries if e[0] in down] if up else []
        self._entries = up or entries
        self._all = entries
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
        if isinstance(exc, GeminiQuotaError):
            _mark_down(self._entries[i][0], exc)
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
        self._entries, self._clients, self._idx = self._all, {}, 0       # test every provider, skipped or not
        for i, (label, _) in enumerate(self._entries):
            try:
                msg = await self._client(i).ping()
                _DOWN.pop(label, None)                   # works again (credits added / limit reset)
                parts.append(f"✓ {msg}")
                first_ok = first_ok or msg
            except Exception as exc:
                parts.append(f"✗ {label}: {str(exc)[:120]}")
        if not first_ok:
            raise GeminiError("No AI provider works — " + " · ".join(parts))
        return " · ".join(parts) if len(parts) > 1 else first_ok
