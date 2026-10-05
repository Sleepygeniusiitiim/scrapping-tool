"""
Organizations, users and their API keys.

Roles
-----
* super admin — signs in with APP_PASSWORD (the owner). Sees every organization, switches into any of them
  (their data), creates organizations and their admins, and sets each organization's API keys and the master
  keys every organization falls back to.
* org admin   — signs in with email + password. Works in their own organization only; enrols and manages its
  users. Cannot see or change API keys.
* member      — signs in with email + password; runs searches and works with the organization's data.

Storage (always in the master "public" schema): organizations, app_users, user_sessions, org_settings.
Each organization's data lives in its own schema (supabase_db: "org_<id>"), so one organization can never
read another's tables.

API keys: org_settings.org_id = '<organization id>' or 'master'. Values are encrypted (Fernet) with
SECRET_KEY (or, if unset, a key derived from APP_PASSWORD). Effective keys = the organization's own, then the
master's, then the server environment variables.
"""

from __future__ import annotations

import base64
import contextvars
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import uuid
from typing import Dict, List, Optional

import psycopg2.extras

import supabase_db as db

ROLES = ("org_admin", "member")
SESSION_DAYS = 30
MASTER = "master"

SCHEMA = """
CREATE TABLE IF NOT EXISTS organizations (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, schema_name TEXT NOT NULL UNIQUE, active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS app_users (
    id TEXT PRIMARY KEY, org_id TEXT NOT NULL REFERENCES organizations(id), email TEXT NOT NULL UNIQUE,
    name TEXT, role TEXT NOT NULL, password_hash TEXT NOT NULL, active BOOLEAN DEFAULT TRUE,
    created_by TEXT, created_at TIMESTAMPTZ DEFAULT NOW(), last_login TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS user_sessions (
    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(), expires_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS org_settings (
    org_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TIMESTAMPTZ DEFAULT NOW(), updated_by TEXT
);
"""
_ready = False
_lock = threading.Lock()
SETTINGS_TTL_S = 30          # how long a server instance reuses keys it read (saves re-reading per call)
_cache: Dict[str, tuple] = {}

# Who is making this request: {"role": "super"|"org_admin"|"member", "org_id", "org_name", "user_id", "email",
# "schema"}. Set by the API's auth dependency; None outside a request.
_principal = contextvars.ContextVar("principal", default=None)


def set_principal(p: Optional[dict]):
    return _principal.set(p)


def principal() -> Optional[dict]:
    return _principal.get()


class AccessError(Exception):
    """Not allowed (403) or not signed in (401)."""

    def __init__(self, msg: str, status: int = 403):
        super().__init__(msg)
        self.status = status


def _q(sql: str, params=None, fetch: str = ""):
    global _ready
    with db.use_schema("public"):
        if not _ready:
            with _lock:
                if not _ready:
                    db._ensure_schema()

                    def mk():
                        with db._connect() as conn:
                            with conn.cursor() as cur:
                                cur.execute(SCHEMA)
                            conn.commit()
                    db._with_retry(mk, "Creating account tables")
                    _ready = True

        def run():
            with db._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, params)
                    out = cur.fetchall() if fetch == "all" else cur.fetchone() if fetch == "one" else None
                conn.commit()
                return out
        return db._with_retry(run, "Accounts")


# ---------------------------------------------------------------------------
# Passwords & tokens
# ---------------------------------------------------------------------------
def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 240_000)
    return f"pbkdf2$240000${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, n, salt, dk = stored.split("$")
        got = hashlib.pbkdf2_hmac("sha256", pw.encode(), base64.b64decode(salt), int(n))
        return hmac.compare_digest(got, base64.b64decode(dk))
    except Exception:
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _check_password_rules(pw: str) -> None:
    if len(pw or "") < 8:
        raise ValueError("Passwords need at least 8 characters.")


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------
def schema_of(org_id: str) -> str:
    return "org_" + re.sub(r"[^a-z0-9]", "", org_id.lower())[:40]


def provision(schema: str) -> None:
    """Create every table of the app inside an organization's schema."""
    import categories
    import gov_registry
    import messaging
    import privacy
    import suggestions
    import vectors
    from intent_miner import store
    with db.use_schema(schema):
        db._ensure_schema()
        store._ensure()
        gov_registry._ensure()
        for mod in (privacy, categories, messaging, suggestions):
            mod._q("SELECT 1")
        vectors.ensure()


def create_org(name: str) -> dict:
    name = re.sub(r"\s+", " ", name or "").strip()[:120]
    if not name:
        raise ValueError("Give the organization a name.")
    if _q("SELECT 1 FROM organizations WHERE lower(name) = lower(%s)", (name,), "one"):
        raise ValueError(f"An organization called “{name}” already exists.")
    oid = uuid.uuid4().hex[:12]
    sch = schema_of(oid)
    provision(sch)
    row = _q("INSERT INTO organizations (id, name, schema_name) VALUES (%s, %s, %s) RETURNING *", (oid, name, sch), "one")
    return db._serialize_row(dict(row))


def list_orgs() -> List[dict]:
    rows = _q("""SELECT o.*, (SELECT COUNT(*) FROM app_users u WHERE u.org_id = o.id AND u.active) AS users,
                        (SELECT COUNT(*) > 0 FROM org_settings s WHERE s.org_id = o.id) AS has_keys
                 FROM organizations o ORDER BY o.created_at""", fetch="all")
    return [db._serialize_row(dict(r)) for r in rows]


def get_org(org_id: str) -> Optional[dict]:
    r = _q("SELECT * FROM organizations WHERE id = %s", (org_id,), "one")
    return db._serialize_row(dict(r)) if r else None


def rename_org(org_id: str, name: str = "", active: Optional[bool] = None) -> dict:
    org = get_org(org_id)
    if not org:
        raise ValueError("No such organization.")
    _q("UPDATE organizations SET name = COALESCE(NULLIF(%s, ''), name), active = COALESCE(%s, active) WHERE id = %s",
       ((name or "").strip()[:120], active, org_id))
    return get_org(org_id)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
def _public_user(r: dict) -> dict:
    d = db._serialize_row(dict(r))
    d.pop("password_hash", None)
    return d


def create_user(org_id: str, email: str, password: str, role: str = "member", name: str = "",
                created_by: str = "") -> dict:
    email = (email or "").strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", email):
        raise ValueError("Enter a valid email address.")
    if role not in ROLES:
        raise ValueError("Role must be org_admin or member.")
    _check_password_rules(password)
    if not get_org(org_id):
        raise ValueError("No such organization.")
    if _q("SELECT 1 FROM app_users WHERE email = %s", (email,), "one"):
        raise ValueError("A user with that email already exists.")
    row = _q("""INSERT INTO app_users (id, org_id, email, name, role, password_hash, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *""",
             (uuid.uuid4().hex, org_id, email, (name or "").strip()[:120], role, hash_password(password), created_by),
             "one")
    return _public_user(row)


def list_users(org_id: str) -> List[dict]:
    rows = _q("SELECT * FROM app_users WHERE org_id = %s ORDER BY created_at", (org_id,), "all")
    return [_public_user(r) for r in rows]


def get_user(user_id: str) -> Optional[dict]:
    r = _q("SELECT * FROM app_users WHERE id = %s", (user_id,), "one")
    return dict(r) if r else None


def update_user(user_id: str, *, name: Optional[str] = None, role: Optional[str] = None,
                active: Optional[bool] = None, password: Optional[str] = None) -> dict:
    if role is not None and role not in ROLES:
        raise ValueError("Role must be org_admin or member.")
    if password:
        _check_password_rules(password)
    _q("""UPDATE app_users SET name = COALESCE(%s, name), role = COALESCE(%s, role), active = COALESCE(%s, active),
                password_hash = COALESCE(%s, password_hash) WHERE id = %s""",
       (name, role, active, hash_password(password) if password else None, user_id))
    if active is False or password:
        _q("DELETE FROM user_sessions WHERE user_id = %s", (user_id,))          # signed out everywhere
    u = get_user(user_id)
    if not u:
        raise ValueError("No such user.")
    return _public_user(u)


# ---------------------------------------------------------------------------
# Sign-in
# ---------------------------------------------------------------------------
def login(email: str, password: str) -> dict:
    u = _q("""SELECT u.*, o.active AS org_active, o.name AS org_name FROM app_users u
              JOIN organizations o ON o.id = u.org_id WHERE u.email = %s""", ((email or "").strip().lower(),), "one")
    if not u or not verify_password(password or "", u["password_hash"]):
        raise AccessError("Wrong email or password.", 401)
    if not u["active"] or not u["org_active"]:
        raise AccessError("This account is disabled — ask your administrator.", 401)
    token = secrets.token_urlsafe(32)
    _q("""INSERT INTO user_sessions (token_hash, user_id, expires_at)
          VALUES (%s, %s, NOW() + make_interval(days => %s))""", (_token_hash(token), u["id"], SESSION_DAYS))
    _q("UPDATE app_users SET last_login = NOW() WHERE id = %s", (u["id"],))
    return {"token": token, "user": _public_user(u)}


def session_user(token: str) -> Optional[dict]:
    if not token:
        return None
    r = _q("""SELECT u.*, o.name AS org_name, o.schema_name, o.active AS org_active FROM user_sessions s
              JOIN app_users u ON u.id = s.user_id JOIN organizations o ON o.id = u.org_id
              WHERE s.token_hash = %s AND s.expires_at > NOW()""", (_token_hash(token),), "one")
    if not r or not r["active"] or not r["org_active"]:
        return None
    return dict(r)


def logout(token: str) -> None:
    _q("DELETE FROM user_sessions WHERE token_hash = %s", (_token_hash(token or ""),))


# ---------------------------------------------------------------------------
# API keys (encrypted)
# ---------------------------------------------------------------------------
def _fernet():
    from cryptography.fernet import Fernet
    secret = os.getenv("SECRET_KEY") or os.getenv("APP_PASSWORD") or ""
    if not secret:
        raise RuntimeError("Set SECRET_KEY (or APP_PASSWORD) on the server to store API keys.")
    key = base64.urlsafe_b64encode(hashlib.sha256(("org-keys:" + secret).encode()).digest())
    return Fernet(key)


def get_settings(org_id: str, fresh: bool = False) -> dict:
    """{"integrations": {...}, "llm_keys": {...}, "llm_models": {...}, "llm_provider", "claude_key",
    "claude_model", "gemini_key"} — decrypted."""
    hit = _cache.get(org_id)
    if hit and not fresh and time.time() - hit[0] < SETTINGS_TTL_S:
        return json.loads(hit[1])
    out = _read_settings(org_id)
    _cache[org_id] = (time.time(), json.dumps(out))
    return out


def _read_settings(org_id: str) -> dict:
    r = _q("SELECT data FROM org_settings WHERE org_id = %s", (org_id,), "one")
    if not r:
        return {}
    try:
        return json.loads(_fernet().decrypt(r["data"].encode()).decode())
    except Exception:
        return {}                                   # unreadable (SECRET_KEY changed): treated as not set


def save_settings(org_id: str, data: dict, by: str = "") -> dict:
    """Merge: a field sent as "" clears it, a field left out keeps the stored value."""
    cur = get_settings(org_id, fresh=True)
    for section in ("integrations", "llm_keys", "llm_models"):
        merged = dict(cur.get(section) or {})
        for k, v in (data.get(section) or {}).items():
            v = str(v).strip()
            if v:
                merged[str(k)] = v
            else:
                merged.pop(str(k), None)
        cur[section] = merged
    for k in ("llm_provider", "claude_key", "claude_model", "gemini_key"):
        if k in data:
            v = str(data[k] or "").strip()
            if v:
                cur[k] = v
            else:
                cur.pop(k, None)
    blob = _fernet().encrypt(json.dumps(cur).encode()).decode()
    _q("""INSERT INTO org_settings (org_id, data, updated_by) VALUES (%s, %s, %s)
          ON CONFLICT (org_id) DO UPDATE SET data = EXCLUDED.data, updated_at = NOW(), updated_by = EXCLUDED.updated_by""",
       (org_id, blob, by))
    _cache.pop(org_id, None)
    return masked(cur)


def masked(settings: dict) -> dict:
    """What an admin screen may show: which keys are set (last 4 characters), never the keys."""
    def m(v):
        v = str(v or "")
        return ("•••• " + v[-4:]) if len(v) > 8 else ("set" if v else "")
    return {"integrations": {k: m(v) for k, v in (settings.get("integrations") or {}).items()},
            "llm_keys": {k: m(v) for k, v in (settings.get("llm_keys") or {}).items()},
            "llm_models": dict(settings.get("llm_models") or {}),
            "llm_provider": settings.get("llm_provider", ""), "claude_model": settings.get("claude_model", ""),
            "claude_key": m(settings.get("claude_key")), "gemini_key": m(settings.get("gemini_key"))}


def effective_settings(org_id: Optional[str]) -> dict:
    """The organization's settings over the master's (environment variables are the last fallback, applied by
    the callers)."""
    master = get_settings(MASTER)
    own = get_settings(org_id) if org_id and org_id != MASTER else {}
    out = {"integrations": {**(master.get("integrations") or {}), **(own.get("integrations") or {})},
           "llm_keys": {**(master.get("llm_keys") or {}), **(own.get("llm_keys") or {})},
           "llm_models": {**(master.get("llm_models") or {}), **(own.get("llm_models") or {})}}
    for k in ("llm_provider", "claude_key", "claude_model", "gemini_key"):
        v = own.get(k) or master.get(k)
        if v:
            out[k] = v
    return out
