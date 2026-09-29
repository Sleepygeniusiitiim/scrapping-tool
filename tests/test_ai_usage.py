import asyncio

import httpx
from pydantic import BaseModel

import ai_chain
import openrouter_client
import pipeline
from gemini_client import GeminiQuotaError


class _Out(BaseModel):
    ok: bool = False


def test_focus_keeps_contact_lines_within_budget():
    filler = "\n".join(f"menu item {i} lorem ipsum dolor sit amet" for i in range(3000))
    page = "Welders wanted in Dubai\n" + filler + "\nRamesh: interested, my whatsapp +91 98765 43210\n" + filler
    out = pipeline._focus(page, 6000)
    assert len(out) <= 6000
    assert out.startswith("Welders wanted in Dubai")
    assert "+91 98765 43210" in out
    assert pipeline._focus("short page", 6000) == "short page"


def test_chain_skips_provider_that_ran_out():
    ai_chain._DOWN.clear()
    calls = []

    class Dead:
        async def generate_structured(self, *a, **k):
            calls.append("dead")
            raise GeminiQuotaError("credits are used up")

    class Live:
        async def generate_structured(self, *a, **k):
            calls.append("live")
            return _Out(ok=True)

    chain = ai_chain.AIChain([("Dead", Dead), ("Live", Live)])
    assert asyncio.run(chain.generate_structured("p", _Out)).ok
    assert calls == ["dead", "live"]
    calls.clear()
    chain2 = ai_chain.AIChain([("Dead", Dead), ("Live", Live)])          # next batch / request
    assert chain2.skipped == ["Dead"]
    assert asyncio.run(chain2.generate_structured("p", _Out)).ok
    assert calls == ["live"]
    ai_chain._DOWN.clear()


def test_output_cap_and_retry_when_cut_off(monkeypatch):
    sent = []

    def handler(request):
        import json
        body = json.loads(request.content)
        cap = body.get("max_tokens") or body.get("max_completion_tokens")
        sent.append(cap)
        if cap < 8000:
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": fa'},
                                                          "finish_reason": "length"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'},
                                                      "finish_reason": "stop"}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(openrouter_client.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler)))
    c = openrouter_client.OpenRouter("k", provider="groq")
    assert asyncio.run(c.generate_structured("p", _Out)).ok
    assert sent == [openrouter_client.MAX_OUTPUT_TOKENS, 8000]


def test_openrouter_uses_leftover_credit(monkeypatch):
    sent = []

    def handler(request):
        import json
        cap = json.loads(request.content)["max_tokens"]
        sent.append(cap)
        if cap > 2000:
            return httpx.Response(402, json={"error": {"message": "You requested up to 3000 tokens, but can only "
                                                                  "afford 2200"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(openrouter_client.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler)))
    c = openrouter_client.OpenRouter("k", provider="openrouter")
    assert asyncio.run(c.generate_structured("p", _Out)).ok
    assert sent == [3000, 2000]
