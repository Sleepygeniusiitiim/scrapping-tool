"""
Messaging the leads of a run / batch / category: WhatsApp through Pinnacle (WhatsApp Business API, approved
templates) and email through Brevo (transactional email API).

Safeguards, always on:
* the do-not-contact list is checked for every recipient (privacy.py) — opted-out people are never messaged;
* every send is logged in `outreach_log`; anyone messaged on the same channel within `skip_days` is skipped,
  so re-running a send never spams the same person;
* emails carry an opt-out line; WhatsApp business-initiated messages must use a template Meta approved
  (Pinnacle console), which is what this sends.

Personalisation tokens in subject / body / template placeholders: {name} {first_name} {role} {location}
{platform} {source_url}.
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
import time
from typing import Dict, Iterable, List, Optional

import httpx
import psycopg2.extras

import supabase_db as db

PINNACLE_URL = os.getenv("PINNACLE_API_URL", "https://lsq.pinnacle.in/api/v1/sendmessage").strip()
BREVO_URL = "https://api.brevo.com/v3/smtp/email"
MAX_BATCH = 25            # recipients per request (keeps a request inside the serverless time limit)
SEND_GAP_S = 0.35         # pause between messages (provider rate limits)

SCHEMA = """
CREATE TABLE IF NOT EXISTS outreach_log (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, channel TEXT NOT NULL, to_addr TEXT NOT NULL,
    name TEXT, source_url TEXT, campaign TEXT, status TEXT NOT NULL, provider_id TEXT, error TEXT,
    sent_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_outreach_log_addr ON outreach_log (channel, to_addr, sent_at DESC);
"""
_ready = False
_lock = threading.Lock()


def _q(sql: str, params=None, fetch: str = ""):
    global _ready
    if not _ready:
        with _lock:
            if not _ready:
                db._ensure_schema()

                def mk():
                    with db._connect() as conn:
                        with conn.cursor() as cur:
                            cur.execute(SCHEMA)
                        conn.commit()
                db._with_retry(mk, "Creating outreach log")
                _ready = True

    def run():
        with db._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                out = cur.fetchall() if fetch == "all" else cur.fetchone() if fetch == "one" else None
            conn.commit()
            return out
    return db._with_retry(run, "Outreach log")


# ---------------------------------------------------------------------------
# Recipients
# ---------------------------------------------------------------------------
def wa_number(phone: Optional[str]) -> str:
    """International number, digits only (919876543210); '' when it can't be one."""
    d = re.sub(r"\D", "", phone or "")
    if d.startswith("00"):
        d = d[2:]
    if len(d) == 10 and d[0] in "6789":         # a bare Indian mobile
        d = "91" + d
    return d if 10 <= len(d) <= 15 else ""


def _email(e: Optional[str]) -> str:
    e = (e or "").strip().lower()
    return e if re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", e) else ""


def recently_messaged(channel: str, addrs: List[str], days: int) -> set:
    if not addrs or days <= 0:
        return set()
    rows = _q("""SELECT DISTINCT to_addr FROM outreach_log WHERE channel = %s AND status = 'sent'
                 AND to_addr = ANY(%s) AND sent_at > NOW() - make_interval(days => %s)""",
              (channel, addrs, int(days)), "all")
    return {r["to_addr"] for r in rows}


def plan(records: Iterable[dict], channels: List[str], only_interested: bool = True, skip_days: int = 30) -> dict:
    """Who would be messaged on which channel, and why the others are left out. records: contacts / leads
    with name, phone, email, shows_interest, source_url, current_role, current_location, platform."""
    import privacy
    try:
        lists = privacy._load()
    except Exception:
        lists = {"phones": set(), "emails": set(), "profiles": set()}
    out: Dict[str, list] = {"whatsapp": [], "email": []}
    skipped = {"not_interested": 0, "opted_out": 0, "no_number": 0, "no_email": 0, "duplicate": 0,
               "recently_messaged": 0}
    seen = {"whatsapp": set(), "email": set()}
    for r in records:
        if only_interested and r.get("shows_interest") is False:
            skipped["not_interested"] += 1
            continue
        if privacy.suppressed(r.get("phone"), r.get("email"), r.get("profile_url"), lists):
            skipped["opted_out"] += 1
            continue
        if "whatsapp" in channels:
            n = wa_number(r.get("phone"))
            if not n:
                skipped["no_number"] += 1
            elif n in seen["whatsapp"]:
                skipped["duplicate"] += 1
            else:
                seen["whatsapp"].add(n)
                out["whatsapp"].append({**r, "to": n})
        if "email" in channels:
            e = _email(r.get("email"))
            if not e:
                skipped["no_email"] += 1
            elif e in seen["email"]:
                skipped["duplicate"] += 1
            else:
                seen["email"].add(e)
                out["email"].append({**r, "to": e})
    for ch in ("whatsapp", "email"):
        done = recently_messaged(ch, [x["to"] for x in out[ch]], skip_days)
        if done:
            skipped["recently_messaged"] += sum(1 for x in out[ch] if x["to"] in done)
            out[ch] = [x for x in out[ch] if x["to"] not in done]
    return {"whatsapp": out["whatsapp"], "email": out["email"], "skipped": skipped}


# ---------------------------------------------------------------------------
# Personalisation
# ---------------------------------------------------------------------------
def fill(template: str, r: dict, defaults: Optional[dict] = None) -> str:
    d = defaults or {}
    name = (r.get("name") or r.get("display_name") or "").strip()
    first = name.split()[0] if name else ""
    values = {"name": name or d.get("name", "there"), "first_name": first or d.get("name", "there"),
              "role": r.get("current_role") or r.get("profession") or d.get("role", ""),
              "location": r.get("current_location") or r.get("origin") or "",
              "platform": r.get("platform") or "", "source_url": (r.get("source_url") or "").split("#")[0]}
    return re.sub(r"\{(\w+)\}", lambda m: str(values.get(m.group(1), m.group(0))), template or "")


def email_html(body: str, opt_out: str) -> str:
    paras = "".join(f"<p>{html.escape(p).replace(chr(10), '<br>')}</p>" for p in body.split("\n\n") if p.strip())
    return (f'<div style="font-family:Arial,Helvetica,sans-serif;font-size:15px;line-height:1.55;color:#111">{paras}'
            f'<p style="font-size:12px;color:#777;margin-top:24px">{html.escape(opt_out)}</p></div>')


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
def send_whatsapp(keys: dict, to: str, template_id: str, placeholders: List[str],
                  client: Optional[httpx.Client] = None) -> dict:
    """Pinnacle WhatsApp Business API — an approved template message."""
    api_key, waba = keys.get("pinnacle_api_key", ""), re.sub(r"\D", "", keys.get("pinnacle_waba_number", ""))
    if not (api_key and waba):
        return {"ok": False, "error": "Pinnacle API key and WABA number are not set (⚙️ Settings → Messaging)"}
    body = {"from": waba, "to": to, "type": "template",
            "message": {"templateid": template_id, "placeholders": placeholders}}
    c = client or httpx.Client(timeout=20)
    try:
        r = c.post(PINNACLE_URL, headers={"apikey": api_key, "wabaNumber": "+" + waba,
                                          "Content-Type": "application/json"}, json=body)
    except Exception as exc:
        return {"ok": False, "error": f"Pinnacle unreachable: {type(exc).__name__}"}
    finally:
        if client is None:
            c.close()
    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text[:300]}
    code = str(data.get("code") or data.get("status") or r.status_code)
    ok = r.status_code < 300 and code in ("200", "201", "202", "success", "Success", "true", "True")
    msg_id = data.get("messageId") or data.get("message_id") or (data.get("data") or {}).get("messageid") \
        if isinstance(data, dict) else None
    if ok:
        return {"ok": True, "id": str(msg_id or ""), "raw": data}
    return {"ok": False, "error": f"Pinnacle HTTP {r.status_code}: {json.dumps(data)[:300]}"}


def send_email(keys: dict, to: str, to_name: str, subject: str, body: str, opt_out: str,
               client: Optional[httpx.Client] = None) -> dict:
    """Brevo transactional email."""
    api_key, sender = keys.get("brevo_api_key", ""), _email(keys.get("brevo_sender_email"))
    if not (api_key and sender):
        return {"ok": False, "error": "Brevo API key and a verified sender email are not set (⚙️ Settings → Messaging)"}
    payload = {"sender": {"email": sender, "name": keys.get("brevo_sender_name") or "Recruitment Team"},
               "to": [{"email": to, **({"name": to_name} if to_name else {})}],
               "subject": subject, "textContent": f"{body}\n\n{opt_out}", "htmlContent": email_html(body, opt_out),
               "replyTo": {"email": sender}, "tags": ["sourcing-agent"]}
    c = client or httpx.Client(timeout=20)
    try:
        r = c.post(BREVO_URL, headers={"api-key": api_key, "accept": "application/json",
                                       "content-type": "application/json"}, json=payload)
    except Exception as exc:
        return {"ok": False, "error": f"Brevo unreachable: {type(exc).__name__}"}
    finally:
        if client is None:
            c.close()
    if r.status_code in (200, 201, 202):
        try:
            return {"ok": True, "id": str(r.json().get("messageId") or "")}
        except ValueError:
            return {"ok": True, "id": ""}
    return {"ok": False, "error": f"Brevo HTTP {r.status_code}: {r.text[:300]}",
            "stop": r.status_code in (401, 403)}


# ---------------------------------------------------------------------------
# A batch
# ---------------------------------------------------------------------------
def send_batch(keys: dict, channel: str, recipients: List[dict], campaign: str, *, template_id: str = "",
               placeholders: Optional[List[str]] = None, subject: str = "", body: str = "", opt_out: str = "",
               defaults: Optional[dict] = None, gap_s: float = SEND_GAP_S) -> dict:
    """Send to up to MAX_BATCH recipients (already planned); log every attempt; mark sent candidates."""
    results, sent, failed = [], 0, 0
    with httpx.Client(timeout=20) as c:
        for i, r in enumerate(recipients[:MAX_BATCH]):
            if channel == "whatsapp":
                res = send_whatsapp(keys, r["to"], template_id, [fill(p, r, defaults) for p in placeholders or []],
                                    client=c)
            else:
                res = send_email(keys, r["to"], (r.get("name") or "").strip(), fill(subject, r, defaults),
                                 fill(body, r, defaults), fill(opt_out, r, defaults), client=c)
            status = "sent" if res["ok"] else "failed"
            sent, failed = sent + res["ok"], failed + (not res["ok"])
            _q("""INSERT INTO outreach_log (channel, to_addr, name, source_url, campaign, status, provider_id, error)
                  VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
               (channel, r["to"], r.get("name"), r.get("source_url"), (campaign or "")[:200], status,
                res.get("id"), res.get("error")))
            results.append({"to": r["to"], "name": r.get("name"), "status": status, "error": res.get("error")})
            if res.get("stop"):
                break                                    # bad key: don't try the rest
            if gap_s and i < len(recipients) - 1:
                time.sleep(gap_s)
    urls = [r.get("source_url") for r, x in zip(recipients, results) if x["status"] == "sent" and r.get("source_url")]
    if urls:
        try:
            _q("""UPDATE candidates SET outreach_status = 'sent', outreach_sent_at = NOW()
                  WHERE source_url = ANY(%s) AND COALESCE(outreach_status, 'new') IN ('new', 'drafted')""", (urls,))
        except Exception:
            pass
    return {"sent": sent, "failed": failed, "results": results}


def log(limit: int = 200, campaign: str = "") -> List[dict]:
    rows = _q(f"""SELECT id, channel, to_addr, name, source_url, campaign, status, provider_id, error, sent_at
                  FROM outreach_log {"WHERE campaign = %s" if campaign else ""} ORDER BY sent_at DESC LIMIT %s""",
              (campaign, limit) if campaign else (limit,), "all")
    return [db._serialize_row(dict(r)) for r in rows]
