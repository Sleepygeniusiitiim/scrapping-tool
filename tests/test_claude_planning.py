import asyncio
import json

import anthropic
import httpx2
from pydantic import BaseModel

import ai_chain
import ai_router
from claude_client import Claude
from gemini_client import GeminiQuotaError


class _Plan(BaseModel):
    queries: list[str] = []


def _claude_with(handler) -> Claude:
    c = Claude("sk-ant-test")
    c.client = anthropic.AsyncAnthropic(api_key="sk-ant-test", max_retries=0,
                                        http_client=anthropic.DefaultAsyncHttpxClient(
                                            transport=httpx2.MockTransport(handler)))
    return c


def _msg(text, stop="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}}


def test_claude_plans_with_fallbacks_and_structured_output():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((body, request.headers.get("anthropic-beta", "")))
        return httpx2.Response(200, json=_msg('{"queries": ["welder India"]}'))

    out = asyncio.run(_claude_with(handler).generate_structured("plan", _Plan, "system"))
    assert out.queries == ["welder India"]
    body, beta = seen[0]
    assert body["model"] == "claude-opus-5-5" and body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in beta
    assert "temperature" not in body


def test_claude_out_of_credit_falls_back_to_the_usual_chain():
    ai_chain._DOWN.clear()

    def handler(request):
        return httpx2.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error", "message": "Your credit balance is too low to access the API."}})

    class Other:
        async def generate_structured(self, *a, **k):
            return _Plan(queries=["from other"])

    claude = _claude_with(handler)
    chain = ai_router.reasoning(ai_chain.AIChain([("Other", Other)]), "sk-ant-test")
    chain._clients[0] = claude
    assert asyncio.run(chain.generate_structured("plan", _Plan)).queries == ["from other"]
    try:
        asyncio.run(claude.generate_structured("plan", _Plan))
        assert False
    except GeminiQuotaError:
        pass
    ai_chain._DOWN.clear()


def test_no_claude_key_leaves_the_chain_alone(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    chain = ai_chain.AIChain([("Other", object)])
    assert ai_router.reasoning(chain, "") is chain


def test_claude_reads_pages_only_when_ticked(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("idx", Path(__file__).resolve().parent.parent / "api" / "index.py")
    idx = importlib.util.module_from_spec(spec)
    sys.modules["idx"] = idx
    spec.loader.exec_module(idx)
    base = ai_chain.AIChain([("Other", object)])
    assert idx._reader(base, None, "sk-ant-test", None) is base                 # unticked: as before
    ticked = idx._reader(base, "1", "sk-ant-test", None)
    assert ticked.labels == ["Claude", "Other"]
