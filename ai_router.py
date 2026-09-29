"""
Hybrid AI routing: a small local model does the bulk work, the hosted models do the reasoning.

    bulk      — reading pages / classifying posts and comments (thousands of calls per run): the local model
                (Ollama / vLLM / llama.cpp, OpenAI-compatible, LOCAL_LLM_URL) first — no quota, no cost — then
                the hosted chain if it is down or returns invalid JSON.
    reasoning — understanding the command, planning sources, deciding record matches: Gemini and the other
                hosted models (unchanged chain); the local model is not used.

Without LOCAL_LLM_URL nothing changes: both tasks use the hosted chain.
"""

from __future__ import annotations

import os
from typing import List, Optional

from ai_chain import AIChain
from openrouter_client import OpenRouter


def local_available() -> bool:
    return bool(os.getenv("LOCAL_LLM_URL", "").strip())


def local_label() -> str:
    return f"Local model ({os.getenv('LOCAL_LLM_MODEL', 'qwen2.5:3b-instruct')})"


def _local_factory():
    key = os.getenv("LOCAL_LLM_KEY", "").strip() or "local"
    concurrency = int(os.getenv("LOCAL_LLM_CONCURRENCY", "4") or 4)
    return lambda: OpenRouter(key, provider="local", max_concurrency=concurrency, fallbacks=[])


def bulk(chain: Optional[AIChain]) -> Optional[AIChain]:
    """The chain for bulk parsing: local model first when configured."""
    if not local_available():
        return chain
    entries: List = [(local_label(), _local_factory())]
    if chain is not None:
        entries += list(chain._entries)
    routed = AIChain(entries)
    if chain is not None:
        routed.notices = chain.notices          # one list of "switched to …" notes for the whole request
    return routed


def reasoning(chain: Optional[AIChain], claude_key: str = "", claude_model: str = "") -> Optional[AIChain]:
    """The chain for the thinking steps (understanding the command, planning searches): Claude first when a
    key is set (ANTHROPIC_API_KEY or the page), then the usual chain. Page reading never goes through here."""
    key = (claude_key or os.getenv("ANTHROPIC_API_KEY", "")).strip()
    if not key:
        return chain
    from claude_client import Claude
    entries: List = [("Claude", lambda: Claude(key, model=claude_model))]
    if chain is not None:
        entries += list(chain._all)
    routed = AIChain(entries)
    if chain is not None:
        routed.notices = chain.notices
    return routed


def embed_endpoint() -> Optional[str]:
    """OpenAI-compatible /v1/embeddings next to the local chat endpoint."""
    url = os.getenv("LOCAL_EMBED_URL", "").strip()
    if url:
        return url
    chat = os.getenv("LOCAL_LLM_URL", "").strip()
    return chat.replace("/chat/completions", "/embeddings") if chat else None


def hosted_chain(llm_keys: Optional[dict] = None, provider: str = "") -> Optional[AIChain]:
    """The hosted fallback chain from env vars (+ optional per-job keys), for code running outside a request
    (background workers). Same order as the web app: the chosen provider first, free tiers next."""
    from ai_chain import FALLBACK_ORDER
    from gemini_client import DEFAULT_MODEL, Gemini
    from openrouter_client import PROVIDERS
    llm_keys = {k: v for k, v in (llm_keys or {}).items() if v}
    provider = (provider or os.getenv("LLM_PROVIDER") or "").strip().lower()

    def key_for(p: str) -> str:
        env = "GEMINI_API_KEY" if p == "gemini" else PROVIDERS[p]["env"]
        return (llm_keys.get(p) or os.getenv(env, "")).strip()

    def factory(p: str, key: str):
        model = os.getenv(p.upper() + "_MODEL", "").strip()
        if p == "gemini":
            return lambda: Gemini(key, model=model or DEFAULT_MODEL, mode=os.getenv("GEMINI_MODE", "auto"))
        return lambda: OpenRouter(key, model=model, provider=p)

    order = ([provider] if provider in FALLBACK_ORDER else []) + [p for p in FALLBACK_ORDER if p != provider]
    entries = [("Gemini" if p == "gemini" else PROVIDERS[p]["label"], factory(p, k)) for p in order if (k := key_for(p))]
    return AIChain(entries) if entries else None
