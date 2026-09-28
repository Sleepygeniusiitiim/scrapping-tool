"""
Auto-reply to "interested" comments on YOUR OWN Instagram / Facebook posts, through Meta's official APIs.

    Meta webhook (comment on your post) ─► interest? ─► public reply on the comment
                                                      ─► private reply (DM / Messenger) asking for contact
    Meta webhook (their DM reply)        ─► phone / email / name extracted ─► saved as a candidate
                                                      ─► thank-you message

Setup (see README): a Meta developer app with the Instagram and Messenger products, your Instagram
Business / Creator account linked to your Facebook Page, a Page access token, and this webhook URL.
Only your own posts are handled: Meta does not allow replying to comments on other people's posts, and
automated comments elsewhere are treated as spam.

Environment:
    META_VERIFY_TOKEN   any secret string; typed into Meta's webhook settings too
    META_APP_SECRET     app secret (verifies that webhook calls really come from Meta)
    META_PAGE_TOKEN     long-lived Page access token (with Instagram permissions)
    META_PAGE_ID        your Facebook Page id
    META_IG_USER_ID     your Instagram Business account id (to ignore your own comments)
    META_GRAPH_VERSION  optional, default v23.0
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from typing import Dict, List, Optional

import httpx
import psycopg2.extras

import rule_extractor
import supabase_db as db
from schema import CandidateRecord, clean_email, clean_phone

DEFAULTS = {
    "enabled": "0",
    "public_reply": "Thank you for your interest! We have sent you a message with the next steps. 🙏",
    "dm_text": ("Hi {name}, thanks for your interest in our opening! To take this forward, please reply with:\n"
                "1. Your full name\n2. WhatsApp number\n3. Email\n4. Years of experience and current city\n"
                "Our recruiter will contact you. Reply STOP if you are not interested."),
    "thanks_text": "Thank you {name}! We have received your details — our recruiter will contact you soon.",
    "extra_keywords": "interested, yes, me, dm, details, apply, i want",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta_settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS meta_threads (
    platform TEXT NOT NULL, user_id TEXT NOT NULL,
    username TEXT, name TEXT, comment_id TEXT, comment_text TEXT, post_id TEXT,
    dm_recipient_id TEXT, public_replied BOOLEAN DEFAULT FALSE, dm_sent BOOLEAN DEFAULT FALSE, error TEXT,
    last_message TEXT, phone TEXT, email TEXT, status TEXT DEFAULT 'commented',
    created_at TIMESTAMPTZ DEFAULT NOW(), updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (platform, user_id)
);
CREATE INDEX IF NOT EXISTS idx_meta_threads_recipient ON meta_threads (dm_recipient_id);
"""
_ready = False


def _q(sql: str, params=None, fetch: str = ""):
    global _ready
    db._ensure_schema()

    def run():
        global _ready
        with db._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if not _ready:
                    cur.execute(SCHEMA)
                    _ready = True
                cur.execute(sql, params)
                out = cur.fetchall() if fetch == "all" else cur.fetchone() if fetch == "one" else None
            conn.commit()
            return out
    return db._with_retry(run, "Meta auto-reply storage")


# ---------------------------------------------------------------------------
# Settings (editable on the page)
# ---------------------------------------------------------------------------
def get_settings() -> Dict[str, str]:
    rows = _q("SELECT key, value FROM meta_settings", fetch="all") or []
    return {**DEFAULTS, **{r["key"]: r["value"] for r in rows}}


def save_settings(values: Dict[str, str]) -> Dict[str, str]:
    for k, v in values.items():
        if k in DEFAULTS:
            _q("""INSERT INTO meta_settings (key, value) VALUES (%s, %s)
                  ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""", (k, str(v)[:2000]))
    return get_settings()


def env_status() -> Dict[str, bool]:
    return {k: bool(os.getenv(k, "").strip()) for k in
            ("META_VERIFY_TOKEN", "META_APP_SECRET", "META_PAGE_TOKEN", "META_PAGE_ID", "META_IG_USER_ID")}


def threads(limit: int = 200) -> List[dict]:
    rows = _q("SELECT * FROM meta_threads ORDER BY updated_at DESC LIMIT %s", (limit,), "all") or []
    return [db._serialize_row(dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------
def verify_signature(raw: bytes, header: Optional[str]) -> bool:
    secret = os.getenv("META_APP_SECRET", "").strip()
    if not secret or not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.split("=", 1)[1])


def _graph(path: str) -> str:
    return f"https://graph.facebook.com/{os.getenv('META_GRAPH_VERSION', 'v23.0').strip()}/{path.lstrip('/')}"


def _is_interest(text: str, settings: Dict[str, str]) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if rule_extractor.shows_interest(t):
        return True
    extra = [k.strip().lower() for k in settings.get("extra_keywords", "").split(",") if k.strip()]
    return any(re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", t) for k in extra) and len(t) <= 120


_MY_NAME = re.compile(r"(?i)\b(?:my name is|name\s*[:\-]|i am|i'm|this is)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})")
_NOT_NAME = {"a", "an", "the", "interested", "from", "nurse", "driver", "and", "with", "looking", "working", "whatsapp",
             "mobile", "phone", "email", "here", "ready", "available"}


def _name_in(text: str) -> Optional[str]:
    """'My name is Priya Nair, WhatsApp …' → 'Priya Nair' (stops at non-name words)."""
    m = _MY_NAME.search(text or "")
    if not m:
        return None
    words = []
    for w in m.group(1).split():
        if w.lower() in _NOT_NAME:
            break
        words.append(w.capitalize())
    return " ".join(words) or None


def _fmt(template: str, name: Optional[str]) -> str:
    first = (name or "").split()[0] if name else "there"
    return template.replace("{name}", first)


async def _post(c: httpx.AsyncClient, path: str, payload: dict) -> dict:
    token = os.getenv("META_PAGE_TOKEN", "").strip()
    r = await c.post(_graph(path), params={"access_token": token}, json=payload)
    try:
        data = r.json()
    except ValueError:
        data = {"error": {"message": r.text[:200]}}
    if r.status_code >= 400 or data.get("error"):
        raise RuntimeError(str((data.get("error") or {}).get("message") or data)[:300])
    return data


async def _profile_name(c: httpx.AsyncClient, platform: str, user_id: str) -> Optional[str]:
    token = os.getenv("META_PAGE_TOKEN", "").strip()
    fields = "name,username" if platform == "instagram" else "first_name,last_name,name"
    try:
        r = await c.get(_graph(user_id), params={"fields": fields, "access_token": token})
        d = r.json() if r.status_code == 200 else {}
    except Exception:
        return None
    return d.get("name") or " ".join(x for x in (d.get("first_name"), d.get("last_name")) if x) or None


def _source_url(platform: str, username: Optional[str], user_id: str) -> str:
    if platform == "instagram" and username:
        return f"https://www.instagram.com/{username}/"
    page = os.getenv("META_PAGE_ID", "page")
    return f"https://www.facebook.com/{page}#commenter-{user_id}"


def _save_candidate(platform: str, user_id: str, username: Optional[str], name: Optional[str], text: str,
                    phone: Optional[str] = None, email: Optional[str] = None, replied: bool = False) -> None:
    rec = CandidateRecord(
        name=name or username, evidence_snippet=text[:300], phone=phone, email=email,
        source_url=_source_url(platform, username, user_id), platform=f"{platform}_own_post",
        profile_url=f"https://www.instagram.com/{username}/" if platform == "instagram" and username else None,
        shows_interest=True, contact_source="shared_in_reply" if replied and (phone or email) else None)
    db.save_candidates([rec])


async def handle_comment(platform: str, comment_id: str, text: str, user_id: str, username: Optional[str],
                         name: Optional[str], post_id: Optional[str]) -> dict:
    """An interested comment on our own post → public reply + private reply asking for contact details."""
    settings = get_settings()
    own = {os.getenv("META_PAGE_ID", ""), os.getenv("META_IG_USER_ID", "")}
    if not user_id or user_id in own:
        return {"skipped": "own comment"}
    if not _is_interest(text, settings):
        return {"skipped": "no interest shown"}
    existing = _q("SELECT dm_sent FROM meta_threads WHERE platform = %s AND user_id = %s",
                  (platform, user_id), "one")
    _q("""INSERT INTO meta_threads (platform, user_id, username, name, comment_id, comment_text, post_id)
          VALUES (%s, %s, %s, %s, %s, %s, %s)
          ON CONFLICT (platform, user_id) DO UPDATE SET comment_id = EXCLUDED.comment_id,
              comment_text = EXCLUDED.comment_text, post_id = EXCLUDED.post_id, updated_at = NOW(),
              username = COALESCE(EXCLUDED.username, meta_threads.username),
              name = COALESCE(EXCLUDED.name, meta_threads.name)""",
       (platform, user_id, username, name, comment_id, text[:1000], post_id))
    _save_candidate(platform, user_id, username, name, text)
    if settings.get("enabled") != "1" or not os.getenv("META_PAGE_TOKEN"):
        return {"saved": True, "replied": False, "reason": "auto-reply is off or META_PAGE_TOKEN missing"}
    if existing and existing.get("dm_sent"):
        return {"saved": True, "replied": False, "reason": "already messaged this person"}
    result, error = {"saved": True}, None
    async with httpx.AsyncClient(timeout=15) as c:
        if not name:                           # greet them by name when their profile shares it
            name = await _profile_name(c, platform, user_id)
        try:
            if settings.get("public_reply"):
                path = f"{comment_id}/replies" if platform == "instagram" else f"{comment_id}/comments"
                await _post(c, path, {"message": _fmt(settings["public_reply"], name)})
                result["public_reply"] = True
        except RuntimeError as exc:
            error = f"public reply: {exc}"
        try:
            sent = await _post(c, f"{os.getenv('META_PAGE_ID', '').strip()}/messages",
                               {"recipient": {"comment_id": comment_id},
                                "message": {"text": _fmt(settings["dm_text"], name)}})
            result["dm"] = True
            _q("""UPDATE meta_threads SET dm_sent = TRUE, dm_recipient_id = %s, status = 'messaged',
                      public_replied = %s, error = %s, updated_at = NOW()
                  WHERE platform = %s AND user_id = %s""",
               (sent.get("recipient_id") or user_id, bool(result.get("public_reply")), error, platform, user_id))
        except RuntimeError as exc:
            error = (error + "; " if error else "") + f"private reply: {exc}"
            _q("UPDATE meta_threads SET error = %s, updated_at = NOW() WHERE platform = %s AND user_id = %s",
               (error, platform, user_id))
    result["error"] = error
    return result


async def handle_message(platform: str, sender_id: str, text: str) -> dict:
    """Their reply to our DM → contact details saved; STOP → marked not interested."""
    row = _q("""SELECT * FROM meta_threads WHERE dm_recipient_id = %s OR (platform = %s AND user_id = %s)
                ORDER BY updated_at DESC LIMIT 1""", (sender_id, platform, sender_id), "one")
    if row is None:
        return {"skipped": "message from someone who did not comment on our posts"}
    row = dict(row)
    if re.fullmatch(r"\s*(stop|no|not interested|unsubscribe)\s*[.!]*\s*", text or "", re.IGNORECASE):
        _q("""UPDATE meta_threads SET status = 'not_interested', last_message = %s, updated_at = NOW()
              WHERE platform = %s AND user_id = %s""", (text[:1000], row["platform"], row["user_id"]))
        return {"status": "not_interested"}
    emails, phones = rule_extractor.emails_in(text), rule_extractor.phones_in(text)
    name = _name_in(text) or row.get("name")
    async with httpx.AsyncClient(timeout=15) as c:
        if not name and os.getenv("META_PAGE_TOKEN"):
            name = await _profile_name(c, platform, sender_id)
        phone = clean_phone(phones[0]) if phones else row.get("phone")
        email = clean_email(emails[0]) if emails else row.get("email")
        _q("""UPDATE meta_threads SET last_message = %s, phone = %s, email = %s, name = COALESCE(%s, name),
                  status = %s, updated_at = NOW() WHERE platform = %s AND user_id = %s""",
           (text[:1000], phone, email, name, "contact_shared" if (phone or email) else "replied",
            row["platform"], row["user_id"]))
        _save_candidate(row["platform"], row["user_id"], row.get("username"), name,
                        f"{row.get('comment_text') or ''} | reply: {text}", phone, email, replied=True)
        settings = get_settings()
        if (phone or email) and settings.get("enabled") == "1" and settings.get("thanks_text") and \
                os.getenv("META_PAGE_TOKEN") and not row.get("phone") and not row.get("email"):
            try:
                await _post(c, f"{os.getenv('META_PAGE_ID', '').strip()}/messages",
                            {"recipient": {"id": sender_id}, "message": {"text": _fmt(settings["thanks_text"], name)}})
            except RuntimeError:
                pass
    return {"status": "contact_shared" if (phone or email) else "replied", "phone": bool(phone), "email": bool(email)}


async def process_event(payload: dict) -> List[dict]:
    """Route one webhook delivery (Instagram or Page object) to the handlers."""
    out = []
    obj = payload.get("object")
    for entry in payload.get("entry", []) or []:
        for ch in entry.get("changes", []) or []:
            v = ch.get("value") or {}
            if obj == "instagram" and ch.get("field") == "comments":
                frm = v.get("from") or {}
                out.append(await handle_comment("instagram", v.get("id", ""), v.get("text", ""), frm.get("id", ""),
                                                frm.get("username"), None, (v.get("media") or {}).get("id")))
            elif obj == "page" and ch.get("field") == "feed" and v.get("item") == "comment" and v.get("verb") == "add":
                frm = v.get("from") or {}
                out.append(await handle_comment("facebook", v.get("comment_id", ""), v.get("message", ""),
                                                frm.get("id", ""), None, frm.get("name"), v.get("post_id")))
        for m in entry.get("messaging", []) or []:
            msg = m.get("message") or {}
            if msg.get("is_echo") or not msg.get("text"):
                continue
            platform = "instagram" if obj == "instagram" else "facebook"
            out.append(await handle_message(platform, (m.get("sender") or {}).get("id", ""), msg["text"]))
    return out
