"""
Email verification without sending anything.

    1. syntax           — a well-formed address
    2. disposable       — throw-away inbox services (mailinator, 10minutemail, …) are flagged
    3. DNS              — the domain exists and accepts mail: MX records (or an A record = implicit MX);
                          a "null MX" (RFC 7505) or no DNS at all means it cannot receive mail
    4. SMTP handshake   — connect to the domain's mail server on port 25, EHLO / HELO, MAIL FROM, RCPT TO:<address>,
                          then QUIT. The server's answer to RCPT says whether the mailbox exists (250 = yes,
                          550 / 5.1.1 = no, 4xx = try later). No DATA command is sent, so no email is delivered.
    5. catch-all probe  — in the same session, RCPT TO a random address that cannot exist. If the server accepts it
                          too, the domain accepts everything and a "250" proves nothing: the result is
                          "catch_all" (inconclusive), never "valid".

Status per address: valid | invalid | catch_all | unknown | disposable | no_mail (domain cannot receive mail).

Port 25 is blocked on most cloud platforms, including Vercel's serverless functions (they run on AWS Lambda).
Where it is blocked, verification falls back to, in order:
    * SMTP_VERIFY_URL — this same file run on any small server with port 25 open (a VPS):
          SMTP_VERIFY_TOKEN=<secret> SMTP_VERIFY_FROM=verify@yourcompany.com python email_verify.py serve 8025
      and on Vercel: SMTP_VERIFY_URL=http://<server>:8025/verify, SMTP_VERIFY_TOKEN=<secret>
    * a verification API (Hunter, ZeroBounce, NeverBounce) when its key is set
    * DNS-only: the domain is checked, the mailbox is "unknown".

SMTP_VERIFY_FROM should be an address on a domain you own (e.g. verify@magicbillion.in) and SMTP_HELO_DOMAIN that
server's host name; mail servers distrust made-up senders. Probe sparingly — many RCPT checks from one IP get it
rate-limited or listed.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import socket
import string
import sys
import time
from typing import Dict, List, Optional, Tuple

import httpx

SMTP_TIMEOUT_S = 10
MAX_RCPT_PER_SESSION = 8
SMTP_PORT = int(os.getenv("SMTP_PORT", "25") or 25)          # only changed for tests

_SYNTAX = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
                     r"[A-Za-z]{2,24}$")

DISPOSABLE = {
    "mailinator.com", "10minutemail.com", "10minutemail.net", "guerrillamail.com", "guerrillamail.net",
    "guerrillamail.org", "sharklasers.com", "grr.la", "tempmail.com", "temp-mail.org", "temp-mail.io", "tempmail.net",
    "tempmailo.com", "throwawaymail.com", "yopmail.com", "yopmail.net", "yopmail.fr", "getnada.com", "nada.email",
    "trashmail.com", "trashmail.de", "dispostable.com", "maildrop.cc", "mailnesia.com", "mintemail.com",
    "fakeinbox.com", "mohmal.com", "emailondeck.com", "spamgourmet.com", "mytemp.email", "tempinbox.com",
    "burnermail.io", "33mail.com", "mailcatch.com", "moakt.com", "tmail.ws", "tmpmail.org", "tmpmail.net",
    "discard.email", "spambox.us", "mail.tm", "emailfake.com", "fakemail.net", "inboxkitten.com", "harakirimail.com",
    "mailpoof.com", "linshiyouxiang.net", "1secmail.com", "1secmail.org", "1secmail.net", "kzccv.com", "qiott.com",
    "wuuvo.com", "icznn.com", "ezztt.com", "vjuum.com", "laafd.com", "txcct.com", "anonaddy.me", "tempr.email",
    "dropmail.me", "10mail.org", "emltmp.com", "spam4.me", "mailsac.com", "tempail.com", "etempmail.com",
    "minuteinbox.com", "correotemporal.org", "wegwerfmail.de", "einrot.com", "cuvox.de", "dayrep.com",
    "armyspy.com", "gustr.com", "jourrapide.com", "rhyta.com", "superrito.com", "teleworm.us",
}
# Big providers known to answer RCPT with 250 for any address (the probe usually shows it, this is a backstop).
ACCEPT_ALL_PROVIDERS = {"yahoo.com", "yahoo.co.in", "yahoo.in", "ymail.com", "rocketmail.com", "aol.com"}

_cache: Dict[str, Tuple[float, object]] = {}
CACHE_S = 6 * 3600


def _cached(key: str):
    v = _cache.get(key)
    return v[1] if v and time.time() - v[0] < CACHE_S else None


def _store(key: str, value):
    _cache[key] = (time.time(), value)
    return value


def domain_of(email: str) -> str:
    return (email or "").rsplit("@", 1)[-1].strip().lower()


def is_disposable(email_or_domain: str) -> bool:
    d = domain_of(email_or_domain) if "@" in email_or_domain else email_or_domain.lower()
    return d in DISPOSABLE or any(d.endswith("." + x) for x in DISPOSABLE)


# ---------------------------------------------------------------------------
# DNS (over HTTPS — works on serverless platforms without a resolver library)
# ---------------------------------------------------------------------------
async def _doh(name: str, rtype: str) -> Optional[dict]:
    for url in ("https://dns.google/resolve", "https://cloudflare-dns.com/dns-query"):
        try:
            async with httpx.AsyncClient(timeout=8) as c:
                r = await c.get(url, params={"name": name, "type": rtype}, headers={"accept": "application/dns-json"})
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    return None


async def mail_hosts(domain: str) -> dict:
    """{status: ok|no_mail|no_domain|unknown, hosts: [mx hosts by priority], implicit: bool, reason}"""
    hit = _cached("mx:" + domain)
    if hit is not None:
        return hit
    mx = await _doh(domain, "MX")
    if mx is None:
        return {"status": "unknown", "hosts": [], "reason": "DNS lookup failed"}
    if mx.get("Status") == 3:
        return _store("mx:" + domain, {"status": "no_domain", "hosts": [], "reason": "domain does not exist"})
    records = []
    for a in mx.get("Answer") or []:
        if a.get("type") == 15:
            parts = str(a.get("data", "")).split()
            if len(parts) == 2 and parts[0].isdigit():
                records.append((int(parts[0]), parts[1].rstrip(".").lower()))
    if records:
        records.sort()
        if len(records) == 1 and records[0][1] in ("", "."):
            return _store("mx:" + domain, {"status": "no_mail", "hosts": [], "reason": "null MX: the domain accepts no mail"})
        return _store("mx:" + domain, {"status": "ok", "hosts": [h for _, h in records if h], "implicit": False,
                                       "reason": f"{len(records)} MX record(s)"})
    a = await _doh(domain, "A")
    if a and any(x.get("type") == 1 for x in a.get("Answer") or []):
        return _store("mx:" + domain, {"status": "ok", "hosts": [domain], "implicit": True,
                                       "reason": "no MX record; the domain's A record receives mail (implicit MX)"})
    return _store("mx:" + domain, {"status": "no_mail", "hosts": [],
                                   "reason": "no MX and no A record: the domain cannot receive mail"})


# ---------------------------------------------------------------------------
# SMTP handshake (zero-send)
# ---------------------------------------------------------------------------
class SmtpUnavailable(Exception):
    """Port 25 cannot be reached from here (blocked by the hosting platform or a firewall)."""


async def _reply(reader: asyncio.StreamReader) -> Tuple[int, str]:
    lines = []
    while True:
        line = await asyncio.wait_for(reader.readline(), SMTP_TIMEOUT_S)
        if not line:
            raise ConnectionError("connection closed")
        text = line.decode("utf-8", "replace").rstrip("\r\n")
        lines.append(text)
        if len(text) < 4 or text[3] != "-":
            break
    code = int(lines[-1][:3]) if lines[-1][:3].isdigit() else 0
    return code, " ".join(l[4:] for l in lines)[:300]


async def _cmd(reader, writer, line: str) -> Tuple[int, str]:
    writer.write((line + "\r\n").encode())
    await writer.drain()
    return await _reply(reader)


def _sender() -> Tuple[str, str]:
    helo = os.getenv("SMTP_HELO_DOMAIN", "").strip() or socket.getfqdn() or "localhost"
    frm = os.getenv("SMTP_VERIFY_FROM", "").strip() or f"verify@{helo if '.' in helo else 'example.com'}"
    return helo, frm


def _random_local() -> str:
    return "zz-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=14))


async def smtp_session(host: str, domain: str, addresses: List[str], port: int = 0) -> dict:
    """One connection: RCPT for each address plus one random address (catch-all probe).
    Returns {codes: {address: (code, text)}, catch_all: bool|None, host, error}."""
    helo, sender = _sender()
    port = port or SMTP_PORT
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), SMTP_TIMEOUT_S)
    except (OSError, asyncio.TimeoutError) as exc:
        raise SmtpUnavailable(f"{host}:{port} unreachable ({type(exc).__name__})")
    out = {"host": host, "codes": {}, "catch_all": None, "error": None}
    try:
        code, text = await _reply(reader)
        if code != 220:
            out["error"] = f"greeting {code} {text}"
            return out
        code, text = await _cmd(reader, writer, f"EHLO {helo}")
        if code != 250:
            code, text = await _cmd(reader, writer, f"HELO {helo}")
            if code != 250:
                out["error"] = f"HELO refused: {code} {text}"
                return out
        code, text = await _cmd(reader, writer, f"MAIL FROM:<{sender}>")
        if code != 250:
            out["error"] = f"MAIL FROM refused: {code} {text}"
            return out
        for a in addresses[:MAX_RCPT_PER_SESSION]:
            out["codes"][a] = await _cmd(reader, writer, f"RCPT TO:<{a}>")
        probe = f"{_random_local()}@{domain}"
        pc, _ = await _cmd(reader, writer, f"RCPT TO:<{probe}>")
        out["catch_all"] = True if 200 <= pc < 300 else False if 500 <= pc < 600 else None
        try:
            await _cmd(reader, writer, "QUIT")
        except Exception:
            pass
    except (OSError, asyncio.TimeoutError, ConnectionError, ValueError) as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    finally:
        writer.close()
    return out


def _classify(code: int, text: str, catch_all: Optional[bool]) -> Tuple[str, str]:
    if 200 <= code < 300:
        if catch_all:
            return "catch_all", f"server accepts every address ({code}) — mailbox existence cannot be proven"
        if catch_all is None:
            return "unknown", f"accepted ({code}) but the catch-all probe was inconclusive"
        return "valid", f"mailbox exists (RCPT {code})"
    if 500 <= code < 600:
        if re.search(r"spamhaus|blocked|block ?list|blacklist|policy|reputation|5\.7\.|not permitted|relay", text, re.I):
            return "unknown", f"server refused this checker, not the mailbox ({code} {text[:80]})"
        if code in (550, 551, 553) or re.search(r"5\.1\.[0-3]|user unknown|no such user|does not exist|not found|"
                                                 r"invalid (?:recipient|mailbox)|unknown (?:user|recipient)|"
                                                 r"mailbox unavailable|no mailbox", text, re.I):
            return "invalid", f"mailbox does not exist ({code} {text[:80]})"
        return "unknown", f"rejected ({code} {text[:80]})"
    return "unknown", f"temporary answer ({code} {text[:80]}) — greylisting or rate limit"


_port25 = {"blocked_until": 0.0, "fails": 0}


async def _smtp_verify_domain(domain: str, addresses: List[str], hosts: List[str]) -> Dict[str, dict]:
    if time.time() < _port25["blocked_until"]:
        raise SmtpUnavailable("port 25 blocked on this server (seen on earlier checks)")
    last_err = None
    for host in hosts[:2]:
        try:
            s = await smtp_session(host, domain, addresses)
            _port25["fails"] = 0
        except SmtpUnavailable as exc:
            last_err = exc
            _port25["fails"] += 1
            if _port25["fails"] >= 3:            # several different mail servers unreachable: it is us, not them
                _port25["blocked_until"] = time.time() + 600
                break
            continue
        if s["error"] and not s["codes"]:
            return {a: {"status": "unknown", "reason": f"{host}: {s['error']}", "method": "smtp", "mx": host}
                    for a in addresses}
        out = {}
        for a in addresses:
            code, text = s["codes"].get(a, (0, "not checked (per-session limit)"))
            st, why = _classify(code, text, s["catch_all"]) if code else ("unknown", text)
            if st == "valid" and domain in ACCEPT_ALL_PROVIDERS:
                st, why = "catch_all", "this provider accepts every address at SMTP time"
            out[a] = {"status": st, "reason": why, "method": "smtp", "mx": host, "catch_all": s["catch_all"]}
        return out
    raise SmtpUnavailable(str(last_err) if last_err else "no mail host")


# ---------------------------------------------------------------------------
# Fallbacks when port 25 is blocked here
# ---------------------------------------------------------------------------
async def _remote(addresses: List[str]) -> Optional[Dict[str, dict]]:
    url, token = os.getenv("SMTP_VERIFY_URL", "").strip(), os.getenv("SMTP_VERIFY_TOKEN", "").strip()
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(url, json={"emails": addresses}, headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            res = r.json().get("results") or {}
            for v in res.values():
                v["method"] = "smtp (remote verifier)"
            return res
    except Exception:
        pass
    return None


async def _api(keys: Dict[str, str], email: str) -> Optional[dict]:
    """Hunter → ZeroBounce → NeverBounce, first with a key."""
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            if keys.get("hunter"):
                r = await c.get("https://api.hunter.io/v2/email-verifier", params={"email": email, "api_key": keys["hunter"]})
                if r.status_code == 200:
                    d = (r.json() or {}).get("data") or {}
                    st = {"valid": "valid", "invalid": "invalid", "accept_all": "catch_all", "disposable": "disposable",
                          "webmail": "unknown", "unknown": "unknown"}.get(d.get("status"), "unknown")
                    if d.get("accept_all"):
                        st = "catch_all" if st == "valid" else st
                    return {"status": st, "reason": f"Hunter: {d.get('status')} (score {d.get('score')})", "method": "hunter"}
            if keys.get("zerobounce"):
                r = await c.get("https://api.zerobounce.net/v2/validate",
                                params={"api_key": keys["zerobounce"], "email": email, "ip_address": ""})
                if r.status_code == 200:
                    d = r.json() or {}
                    st = {"valid": "valid", "invalid": "invalid", "catch-all": "catch_all", "spamtrap": "invalid",
                          "abuse": "invalid", "do_not_mail": "invalid"}.get(d.get("status"), "unknown")
                    if d.get("sub_status") == "disposable":
                        st = "disposable"
                    return {"status": st, "reason": f"ZeroBounce: {d.get('status')} {d.get('sub_status') or ''}".strip(),
                            "method": "zerobounce"}
            if keys.get("neverbounce"):
                r = await c.get("https://api.neverbounce.com/v4/single/check", params={"key": keys["neverbounce"], "email": email})
                if r.status_code == 200:
                    d = r.json() or {}
                    st = {"valid": "valid", "invalid": "invalid", "catchall": "catch_all", "disposable": "disposable"
                          }.get(d.get("result"), "unknown")
                    return {"status": st, "reason": f"NeverBounce: {d.get('result')}", "method": "neverbounce"}
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def verify_many(addresses: List[str], keys: Optional[Dict[str, str]] = None, smtp: bool = True
                      ) -> Dict[str, dict]:
    """{address: {status, reason, method, mx?, catch_all?, domain_status}} — addresses on one domain share one
    SMTP session."""
    keys = keys or {}
    out: Dict[str, dict] = {}
    by_domain: Dict[str, List[str]] = {}
    for raw in dict.fromkeys(a.strip() for a in addresses if a and a.strip()):
        a = raw.lower()
        hit = _cached("v:" + a)
        if hit is not None:
            out[raw] = hit
        elif not _SYNTAX.match(a):
            out[raw] = {"status": "invalid", "reason": "not a well-formed address", "method": "syntax"}
        elif is_disposable(a):
            out[raw] = {"status": "disposable", "reason": "throw-away inbox service", "method": "list"}
        else:
            by_domain.setdefault(domain_of(a), []).append(raw)

    async def one_domain(domain: str, addrs: List[str]):
        mx = await mail_hosts(domain)
        if mx["status"] in ("no_mail", "no_domain"):
            for a in addrs:
                out[a] = {"status": "no_mail", "reason": mx["reason"], "method": "dns", "domain_status": mx["status"]}
            return
        res: Optional[Dict[str, dict]] = None
        note = ""
        if smtp and mx["status"] == "ok":
            try:
                res = await _smtp_verify_domain(domain, [a.lower() for a in addrs], mx["hosts"])
                res = {a: res[a.lower()] for a in addrs}
            except SmtpUnavailable as exc:
                note = f"SMTP port 25 unreachable from this server ({exc})"
        if res is None or all(v["status"] == "unknown" for v in res.values()):
            remote = await _remote(addrs)
            if remote:
                res = {a: remote.get(a) or remote.get(a.lower()) or {"status": "unknown", "reason": "not returned",
                                                                      "method": "remote"} for a in addrs}
        if res is None or all(v["status"] == "unknown" for v in res.values()):
            api_res = {}
            for a in addrs[:5]:
                v = await _api(keys, a)
                if v:
                    api_res[a] = v
            if api_res:
                res = {**(res or {}), **api_res}
        for a in addrs:
            v = (res or {}).get(a) or {"status": "unknown", "method": "dns",
                                       "reason": f"domain accepts mail ({mx['reason']}); mailbox not checked"
                                                 + (f" — {note}" if note else "")}
            v["domain_status"] = mx["status"]
            out[a] = v

    await asyncio.gather(*(one_domain(d, a) for d, a in by_domain.items()))
    for a, v in out.items():
        if v["status"] != "unknown":
            _store("v:" + a.lower(), v)
    return out


async def verify(address: str, keys: Optional[Dict[str, str]] = None) -> dict:
    return (await verify_many([address], keys)).get(address) or {"status": "unknown", "reason": "not checked"}


def label(v: Optional[dict]) -> str:
    """Short text for tables: 'valid (smtp)', 'catch-all — inconclusive', …"""
    if not v:
        return ""
    st = v.get("status", "unknown")
    return {"valid": "✓ valid", "invalid": "✗ invalid", "catch_all": "catch-all — inconclusive",
            "disposable": "disposable inbox", "no_mail": "✗ domain takes no mail",
            "unknown": "unverified"}.get(st, st) + f" ({v.get('method', '')})"


# ---------------------------------------------------------------------------
# Remote verifier mode: run on a server with port 25 open
#   SMTP_VERIFY_TOKEN=secret SMTP_VERIFY_FROM=verify@yourdomain SMTP_HELO_DOMAIN=host.yourdomain \
#   python email_verify.py serve 8025
# ---------------------------------------------------------------------------
def _serve(port: int) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    token = os.getenv("SMTP_VERIFY_TOKEN", "")
    if not token:
        sys.exit("Set SMTP_VERIFY_TOKEN first.")
    os.environ.pop("SMTP_VERIFY_URL", None)          # never forward to itself

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.headers.get("Authorization", "") != f"Bearer {token}":
                self.send_response(401)
                self.end_headers()
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            emails = [str(e) for e in (body.get("emails") or [])][:50]
            res = asyncio.run(verify_many(emails, {}, smtp=True))
            data = json.dumps({"results": res}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"email verifier listening on :{port} (POST /verify)")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        _serve(int(sys.argv[2]) if len(sys.argv) > 2 else 8025)
    elif len(sys.argv) >= 2:
        print(json.dumps(asyncio.run(verify_many(sys.argv[1:])), indent=2))
    else:
        print("usage: python email_verify.py <email> [...]   |   python email_verify.py serve [port]")
