"""Source planning, region expansion and fan-out search merging."""
import pipeline
import integrations
from intent_miner import planner
from intent_miner.models import QuerySpec


def test_business_command_plans_maps_and_cities():
    spec = QuerySpec(target="organizations", professions=["truck driving school"], origin=["North India"])
    spec = planner.finalize(spec, "truck driving schools in North India", True, [], 16)
    top = [c.source for c in spec.source_plan[:3]]
    assert top[0] == "maps" and "directories" in top
    assert "Ludhiana" in spec.places and len(spec.places) > 20
    assert any(q.source == "maps" and q.query.endswith("in Ludhiana") for q in spec.queries)


def test_fanout_merges_engines(monkeypatch):
    class R:
        def __init__(self, hits):
            self.hits, self.error = hits, None
    from search_module import SearchHit
    monkeypatch.setattr(pipeline, "search_query", lambda q, **k: R([SearchHit(url="https://a.com/1", title="A", snippet=""),
                                                                   SearchHit(url="https://b.com/2", title="B", snippet="")]))
    monkeypatch.setattr(integrations, "search_available", lambda keys: ["serper"])
    monkeypatch.setattr(integrations, "web_search", lambda *a, **k: ([{"url": "https://b.com/2", "title": "B", "snippet": ""}],
                                                                     None, False))
    r = pipeline.run_query("q", 10, "wt-wt", "all", {})
    assert r["hits"][0]["url"] == "https://b.com/2" and set(r["hits"][0]["engines"]) == {"ddg", "serper"}


def test_out_of_credit_search_api_is_skipped(monkeypatch):
    import httpx

    class Resp:
        status_code = 429
        text = '{"error": "Your account has run out of searches."}'

    class Client:
        def __init__(self, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, *a, **k): return Resp()

    monkeypatch.setattr(integrations.httpx, "Client", Client)
    integrations._EXHAUSTED.clear()
    keys = {"serpapi": "k", "serper": "k2"}
    assert integrations.search_available(keys)[0] == "serper"
    hits, err, _ = integrations.web_search("serpapi", keys, "q", 10, "in-en")
    assert not hits and "run out" in err
    assert "serpapi" not in integrations.search_available(keys) and "serper" in integrations.search_available(keys)
    integrations._EXHAUSTED.clear()


def test_scrapedo_search_and_fallback(monkeypatch):
    import search_module
    from search_module import QueryOutcome, SearchHit

    def fake(query, token, max_results=10, region="in-en", timeout=30):
        o = QueryOutcome(query=query)
        if token == "dead":
            o.error, o.rate_limited = "Scrape.do rate limit / out of credits", True
        else:
            o.hits = [SearchHit(url="https://gillinternational.in/", title="Gill International", snippet="", query=query)]
        return o

    monkeypatch.setattr(search_module, "google_search_scrapedo", fake)
    integrations._EXHAUSTED.clear()
    hits = integrations.search_first({"scrapedo": "ok"}, '"Gill International" contact')
    assert hits[0]["url"] == "https://gillinternational.in/"
    hits, err, _ = integrations.web_search("scrapedo", {"scrapedo": "dead"}, "q", 10, "in-en")
    assert not hits and "scrapedo" in integrations.exhausted()
    integrations._EXHAUSTED.clear()


def test_hiring_side_commands_are_business_searches():
    from intent_miner.understand import hiring_side
    assert hiring_side("profile of Top Management & HRs from outside India & foreign companies, where people "
                       "are interested for hiring of Indian Candidates")
    assert hiring_side("Find foreign companies whose HR managers are hiring Indian welders")
    assert not hiring_side("Find Indian candidates who are interested for abroad opportunities as welders")
    assert not hiring_side("Truck or Bus Driving Centers in North India")


def test_freelance_bid_pages_are_skipped():
    from search_module import is_useful_url
    assert not is_useful_url("https://www.freelancer.in/projects/internet-marketing/website-optimization-seo-fix-county")
    assert not is_useful_url("https://www.upwork.com/freelance-jobs/apply/SEO_123")
    assert is_useful_url("https://www.linkedin.com/posts/acme_hiring-from-india-activity-1")


def test_page_choice_of_target_wins(monkeypatch):
    import asyncio
    from intent_miner import understand as u
    from intent_miner.models import QuerySpec

    class AI:
        async def generate_structured(self, prompt, schema, **k):
            AI.prompt = prompt
            return QuerySpec.model_validate({"summary": "s", "target": "people", "queries": []})

    monkeypatch.setattr(u.planner, "finalize", lambda spec, *a, **k: spec)
    spec = asyncio.run(u.understand(AI(), "Find welders in Dubai", [], None, 4, target="organizations"))
    assert spec.target == "organizations" and "LOOKING FOR ORGANIZATIONS" in AI.prompt
    spec = asyncio.run(u.understand(AI(), "HR managers of companies hiring Indian welders", [], None, 4))
    assert spec.target == "organizations"            # no choice made: the hiring-side check corrects the AI
    spec = asyncio.run(u.understand(AI(), "HR managers hiring welders", [], None, 4, target="people"))
    assert spec.target == "people"                   # the page's choice wins
