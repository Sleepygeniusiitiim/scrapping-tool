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
