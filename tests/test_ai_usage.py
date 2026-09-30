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


def test_indian_intent_drops_pakistani_profiles():
    import rule_extractor as rx
    assert rx.origin_of("Find Indian candidates interested in abroad jobs as welders") == "India"
    assert rx.other_origin("TIG Welder · Lahore, Punjab, Pakistan · 6 years at Descon") == "Pakistan"
    assert rx.other_origin("Welder, whatsapp +92 301 2345678") == "Pakistan"
    assert rx.other_origin("Pipe welder from Ludhiana, Punjab, 5 years at L&T") is None
    assert rx.other_origin("Indian welder, 3 yrs in Hyderabad") is None
    assert not rx.mentions_any("Lahore, Punjab, Pakistan", ["India"])
    assert rx.mentions_any("Ludhiana, Punjab", ["India"])


def test_rules_fallback_when_every_ai_is_out(monkeypatch):
    class OutAI:
        async def generate_structured(self, *a, **k):
            raise GeminiQuotaError("Every AI provider is out of credits")

    page = ("POST by Hiring Agency: Welders needed for Saudi, send CV\n"
            "COMMENT by Ravi Kumar: Interested sir, TIG welder 6 years experience at L&T, from Chennai. "
            "whatsapp 9876543210\n")
    recs, dropped, used_ai, off = asyncio.run(pipeline._extract_page(
        OutAI(), "Indian welders interested in abroad jobs", "https://example.com/post", page, False, "rules",
        ["welder"]))
    assert not used_ai
    try:
        asyncio.run(pipeline._extract_page(OutAI(), "Indian welders", "https://example.com/post", page, False, "ai",
                                           ["welder"]))
        raised = False
    except GeminiQuotaError:
        raised = True
    assert raised          # process_batch catches this and re-reads the page with the rules


def test_contact_rich_page_that_did_not_match_is_suggested():
    import suggestions
    page = ("Free recruitment Kuwait — walk-in interviews. Send CV: hr@gulfjobs-kw.com, jobs@alnasr-kw.com, "
            "recruit@kwt-manpower.com. Call +965 5555 1234, +965 6666 2345, +91 98765 43210")
    rows = suggestions.pick([{"url": "https://jobsatgulf.org/free-recruitment-kuwait/", "title": "Kuwait walk-ins",
                              "text": page, "matched": 0},
                             {"url": "https://example.com/thread", "title": "t", "text": page, "matched": 5},
                             {"url": "https://example.com/few", "title": "f", "text": "mail a@b.co", "matched": 0}])
    assert [r["url"] for r in rows] == ["https://jobsatgulf.org/free-recruitment-kuwait/"]
    assert rows[0]["domain"] == "jobsatgulf.org" and rows[0]["n_contacts"] >= 5


def test_recruiter_posts_are_not_candidates():
    page = ("POST by Gulf Manpower Consultants: Urgent requirement — 20 TIG welders for Saudi. Interested candidates "
            "send CV to hr@gulfmanpower.in or whatsapp 9811122233. Salary 1800 SAR, food and accommodation free.\n"
            "COMMENT by Priya Sharma: We are hiring welders too, share your CV at priya@abcrecruit.com\n"
            "COMMENT by Ravi Kumar: Interested sir, TIG welder 6 years at L&T, from Chennai. whatsapp 9876543210\n")
    counts = {}
    recs, dropped, used_ai, off = asyncio.run(pipeline._extract_page(
        None, "Indian welders interested in abroad jobs", "https://example.com/post", page, False, "rules",
        ["welder"], None, None, counts))
    names = [r.name for r in recs]
    assert "Ravi Kumar" in names
    assert not any(n and ("Manpower" in n or "Priya" in n) for n in names)
    assert all(r.phone != "+919811122233" for r in recs)


def test_ai_marked_recruiter_is_dropped():
    class AI:
        async def generate_structured(self, *a, **k):
            from schema import PageExtraction
            return PageExtraction.model_validate({"candidates": [
                {"name": "Neha HR", "evidence_snippet": "Hiring welders for Qatar", "person_type": "recruiter",
                 "shows_interest": True},
                {"name": "Amit", "evidence_snippet": "I am a welder looking for a job in Qatar", "shows_interest": True}]})

    page = "POST by Neha HR: Hiring welders for Qatar\nCOMMENT by Amit: I am a welder looking for a job in Qatar\n"
    counts = {}
    recs, *_ = asyncio.run(pipeline._extract_page(AI(), "welders", "https://example.com/p", page, False, "ai",
                                                  ["welder"], None, None, counts))
    assert [r.name for r in recs] == ["Amit"] and counts["recruiters"] == 1
