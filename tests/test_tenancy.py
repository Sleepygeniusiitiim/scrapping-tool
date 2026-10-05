"""Organizations: who may do what, which keys a request uses, and (with TEST_DATABASE_URL set) that one
organization's data is invisible to another."""

import importlib.util
import os
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

import accounts  # noqa: E402
import supabase_db as db  # noqa: E402


def _idx():
    idx = sys.modules.get("idx")
    if idx is None:
        spec = importlib.util.spec_from_file_location("idx", ROOT / "api" / "index.py")
        idx = importlib.util.module_from_spec(spec)
        sys.modules["idx"] = idx
        spec.loader.exec_module(idx)
    return idx


ORGS = {"a1": {"id": "a1", "name": "Acme", "schema_name": "org_a1", "active": True},
        "b2": {"id": "b2", "name": "Beta", "schema_name": "org_b2", "active": True}}
SESSIONS = {
    "tok-member": {"id": "u1", "org_id": "a1", "role": "member", "email": "m@acme.test", "org_name": "Acme",
                   "schema_name": "org_a1"},
    "tok-admin": {"id": "u2", "org_id": "a1", "role": "org_admin", "email": "a@acme.test", "org_name": "Acme",
                  "schema_name": "org_a1"},
}
STORED = {"master": {"integrations": {"serper": "MASTER-SERPER", "brave": "MASTER-BRAVE"}},
          "a1": {"integrations": {"serper": "ACME-SERPER"}}}


@pytest.fixture
def client(monkeypatch):
    for env in ("SERPER_API_KEY", "BRAVE_API_KEY", "SERPAPI_API_KEY", "GOOGLE_CSE_KEY", "SEARXNG_URL",
                "SCRAPEDO_API_KEY", "SCRAPEDO_TOKEN"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("APP_PASSWORD", "owner-pw")
    monkeypatch.setenv("DATABASE_URL", "postgresql://mocked/db")      # stored keys are looked up (mocked below)
    monkeypatch.setattr(db, "init_supabase", lambda: None)
    monkeypatch.setattr(accounts, "session_user", lambda t: SESSIONS.get(t))
    monkeypatch.setattr(accounts, "get_org", lambda oid: ORGS.get(oid))
    monkeypatch.setattr(accounts, "get_settings", lambda oid, fresh=False: STORED.get(oid, {}))
    return TestClient(_idx().app)


def test_no_credentials_is_refused(client):
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/auth/me", headers={"X-App-Password": "wrong"}).status_code == 401


def test_member_works_in_their_own_organization(client):
    me = client.get("/api/auth/me", headers={"Authorization": "Bearer tok-member"}).json()["me"]
    assert (me["role"], me["org_id"], me["can_edit_keys"], me["can_manage_users"]) == ("member", "a1", False, False)


def test_owner_switches_organization(client):
    h = {"X-App-Password": "owner-pw"}
    assert client.get("/api/auth/me", headers=h).json()["me"]["org_id"] is None          # master workspace
    me = client.get("/api/auth/me", headers={**h, "X-Org": "b2"}).json()["me"]
    assert (me["role"], me["org_id"], me["can_edit_keys"]) == ("super", "b2", True)
    assert client.get("/api/auth/me", headers={**h, "X-Org": "zz"}).status_code == 404


def test_schema_reaches_sync_endpoints(client, monkeypatch):
    """The async auth dependency sets the schema; sync endpoints (thread pool) must see it."""
    seen = []
    idx = _idx()
    monkeypatch.setattr(idx.integrations, "summary", lambda keys: seen.append(db.current_schema()) or {})
    client.get("/api/readiness", headers={"Authorization": "Bearer tok-member"})
    client.get("/api/readiness", headers={"X-App-Password": "owner-pw", "X-Org": "b2"})
    client.get("/api/readiness", headers={"X-App-Password": "owner-pw"})
    assert seen == ["org_a1", "org_b2", "public"]
    assert db.current_schema() == "public"                                   # nothing leaks out of a request


def test_keys_org_then_master_then_page_only_for_owner(client):
    page = '{"brave": "PAGE-BRAVE", "serper": "PAGE-SERPER"}'
    idx = _idx()
    got = {}
    orig = idx.integrations.summary

    def spy(keys):
        got.update(keys)
        return orig(keys)
    idx.integrations.summary = spy
    try:
        client.get("/api/readiness", headers={"Authorization": "Bearer tok-member", "X-Integrations": page})
        assert got["serper"] == "ACME-SERPER" and got["brave"] == "MASTER-BRAVE"   # page keys ignored
        got.clear()
        client.get("/api/readiness", headers={"X-App-Password": "owner-pw", "X-Org": "b2"})
        assert got["serper"] == "MASTER-SERPER"                                     # Beta has none: master
        got.clear()
        client.get("/api/readiness", headers={"X-App-Password": "owner-pw", "X-Integrations": page})
        assert got["serper"] == "PAGE-SERPER"                                       # the owner's page wins
    finally:
        idx.integrations.summary = orig


def test_only_the_owner_sees_or_changes_keys(client):
    for tok in ("tok-member", "tok-admin"):
        h = {"Authorization": f"Bearer {tok}"}
        assert client.get("/api/admin/orgs/a1/keys", headers=h).status_code == 403
        assert client.put("/api/admin/orgs/a1/keys", headers=h, json={"integrations": {"serper": "x"}}).status_code == 403
        assert client.post("/api/admin/orgs", headers=h, json={"name": "Mine"}).status_code == 403


def test_org_admin_manages_only_their_own_users(client, monkeypatch):
    monkeypatch.setattr(accounts, "list_users", lambda oid: [{"id": "u1", "org_id": oid}])
    monkeypatch.setattr(db, "init_supabase", lambda: None)
    admin = {"Authorization": "Bearer tok-admin"}
    assert client.get("/api/admin/orgs/a1/users", headers=admin).status_code == 200
    assert client.get("/api/admin/orgs/b2/users", headers=admin).status_code == 403
    assert client.get("/api/admin/orgs/a1/users", headers={"Authorization": "Bearer tok-member"}).status_code == 403
    assert client.post("/api/privacy/purge", headers={"Authorization": "Bearer tok-member"}, json={}).status_code == 403


def test_member_cannot_use_page_ai_keys(client, monkeypatch):  # noqa: ARG001 (client: the mocks)
    """An organization user's AI chain is built from the saved keys only."""
    for env in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY", "MISTRAL_API_KEY",
                "SAMBANOVA_API_KEY", "NVIDIA_API_KEY", "GITHUB_MODELS_TOKEN", "DEEPSEEK_API_KEY", "MOONSHOT_API_KEY",
                "LLM_PROVIDER"):
        monkeypatch.delenv(env, raising=False)
    idx = _idx()
    token = accounts.set_principal({"role": "member", "org_id": "a1"})
    try:
        with pytest.raises(idx.HTTPException):                 # no saved AI key and the page key is ignored
            idx._gemini(x_llm_provider="groq", x_llm_keys='{"groq": "PAGE"}')
        STORED["a1"]["llm_keys"] = {"groq": "ACME-GROQ"}
        chain = idx._gemini(x_llm_keys='{"cerebras": "PAGE"}')
        assert len(chain.labels) == 1
    finally:
        STORED["a1"].pop("llm_keys", None)
        accounts._principal.reset(token)


# ---------------------------------------------------------------------------
# Real database (optional): TEST_DATABASE_URL=postgresql://… pytest tests/test_tenancy.py
# ---------------------------------------------------------------------------
TEST_DB = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run against Postgres")
def test_organizations_end_to_end(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    monkeypatch.setenv("APP_PASSWORD", "owner-pw")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    accounts._cache.clear()
    c = TestClient(_idx().app)
    owner = {"X-App-Password": "owner-pw"}
    tag = uuid.uuid4().hex[:6]

    r = c.post("/api/admin/orgs", headers=owner,
               json={"name": f"Acme {tag}", "admin_email": f"boss{tag}@acme.test", "admin_password": "boss-pass-1"})
    assert r.status_code == 200, r.text
    acme = r.json()["org"]
    beta = c.post("/api/admin/orgs", headers=owner, json={"name": f"Beta {tag}"}).json()["org"]

    login = c.post("/api/auth/login", json={"email": f"boss{tag}@acme.test", "password": "boss-pass-1"})
    assert login.status_code == 200, login.text
    boss = {"Authorization": "Bearer " + login.json()["token"]}
    assert c.post("/api/auth/login", json={"email": f"boss{tag}@acme.test", "password": "nope-nope"}).status_code == 401

    # the org admin enrols a member, who can sign in
    r = c.post(f"/api/admin/orgs/{acme['id']}/users", headers=boss,
               json={"email": f"m{tag}@acme.test", "password": "member-pw-1"})
    assert r.status_code == 200, r.text
    assert c.post(f"/api/admin/orgs/{beta['id']}/users", headers=boss,
                  json={"email": f"x{tag}@acme.test", "password": "member-pw-1"}).status_code == 403
    member = {"Authorization": "Bearer " + c.post("/api/auth/login", json={
        "email": f"m{tag}@acme.test", "password": "member-pw-1"}).json()["token"]}

    # keys: master, then Acme's own; stored encrypted
    assert c.put("/api/admin/orgs/master/keys", headers=owner,
                 json={"integrations": {"serper": "MASTER-SERPER-KEY"}}).status_code == 200
    assert c.put(f"/api/admin/orgs/{acme['id']}/keys", headers=owner,
                 json={"integrations": {"brave": "ACME-BRAVE-KEY-1"}}).status_code == 200
    with db.use_schema("public"):
        raw = accounts._q("SELECT data FROM org_settings WHERE org_id = %s", (acme["id"],), "one")["data"]
    assert "ACME-BRAVE" not in raw
    eff = accounts.effective_settings(acme["id"])["integrations"]
    assert eff == {"serper": "MASTER-SERPER-KEY", "brave": "ACME-BRAVE-KEY-1"}
    shown = c.get(f"/api/admin/orgs/{acme['id']}/keys", headers=owner).json()
    assert shown["own"]["integrations"]["brave"].endswith("EY-1") and "ACME" not in str(shown)

    # data isolation: a candidate saved in Acme is not visible in Beta or the master workspace
    rec = {"name": "Ravi Kumar", "phone": "+919812345670", "source_url": f"https://example.test/{tag}",
           "wave": "W1"}
    r = c.post("/api/candidates", headers=member, json={"records": [rec]})
    assert r.status_code == 200, r.text

    def urls(h):
        out = c.get("/api/candidates", headers=h)
        assert out.status_code == 200, out.text
        return [x.get("source_url") for x in out.json().get("candidates", out.json().get("records", []))]
    assert rec["source_url"] in urls(member)
    assert rec["source_url"] in urls({**owner, "X-Org": acme["id"]})      # the owner can look inside
    assert rec["source_url"] not in urls({**owner, "X-Org": beta["id"]})
    assert rec["source_url"] not in urls(owner)

    # disabling a user ends their sessions
    uid = c.get("/api/auth/me", headers=member).json()["me"]["user_id"]
    assert c.patch(f"/api/admin/users/{uid}", headers=boss, json={"active": False}).status_code == 200
    assert c.get("/api/auth/me", headers=member).status_code == 401


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run against Postgres")
def test_worker_runs_a_job_inside_its_organization(monkeypatch):
    import asyncio
    import jobs
    monkeypatch.setenv("DATABASE_URL", TEST_DB)
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("GROQ_API_KEY", "env-groq")
    accounts._cache.clear()
    org = accounts.create_org(f"Worker Co {uuid.uuid4().hex[:6]}")
    accounts.save_settings(org["id"], {"integrations": {"serper": "ORG-SERPER-KEY"}})
    with db.use_schema("public"):
        jobs._q("UPDATE im_jobs SET status = 'cancelled' WHERE status = 'queued'")
    job = jobs.enqueue("run", "find welders", {"rounds": 1}, org_id=org["id"])

    spec = importlib.util.spec_from_file_location("worker_mod", ROOT / "worker" / "worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    from intent_miner import runner
    seen = {}

    async def fake_rounds(ai, command, opts, keys, emit, stop, progress):
        seen.update(schema=db.current_schema(), serper=keys.get("serper"), command=command)
        worker._stop["flag"] = True
        return {"run_id": "r1"}
    monkeypatch.setattr(runner, "run_rounds", fake_rounds)
    monkeypatch.setattr(worker, "POLL_S", 0.01)
    asyncio.run(worker.main())
    assert seen == {"schema": org["schema_name"], "serper": "ORG-SERPER-KEY", "command": "find welders"}
    assert jobs.get(str(job["id"]), 0, org["id"])["status"] == "done"
    assert db.current_schema() == "public"
