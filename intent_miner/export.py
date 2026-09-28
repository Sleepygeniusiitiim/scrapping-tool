"""Outputs: CSV / Excel / JSON files and Salesforce Lead creation (qualified leads only)."""

from __future__ import annotations

import csv
import io
import json
from typing import Dict, List

import httpx

COLUMNS = ["lead_score", "tier", "intent_score", "display_name", "platform", "profession", "origin", "destination",
           "timeline", "intent_type", "email", "phone", "profile_url", "last_activity", "confidence", "freshness",
           "source_quality", "status", "evidence", "why", "sources"]


def _flat(lead: dict, col: str) -> str:
    v = lead.get(col)
    if col == "sources":
        return " | ".join(s.get("url", "") for s in (v or []))
    if isinstance(v, list):
        return " | ".join(str(x) for x in v)
    return "" if v is None else str(v)


def to_csv(leads: List[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(COLUMNS)
    for L in leads:
        w.writerow([_flat(L, c) for c in COLUMNS])
    return ("﻿" + buf.getvalue()).encode("utf-8")


def to_xlsx(leads: List[dict]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    ws = wb.active
    ws.title = "Leads"
    ws.append(COLUMNS)
    for c in ws[1]:
        c.font = Font(bold=True)
    for L in leads:
        ws.append([L.get(c) if isinstance(L.get(c), (int, float)) else _flat(L, c) for c in COLUMNS])
    ws.freeze_panes = "A2"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def to_json(leads: List[dict]) -> bytes:
    return json.dumps(leads, indent=2, ensure_ascii=False, default=str).encode("utf-8")


async def push_salesforce(leads: List[dict], instance_url: str, token: str, min_score: int = 80) -> Dict:
    """Create Salesforce Leads for qualified, not-yet-pushed leads (score ≥ min_score)."""
    base = instance_url.rstrip("/") + "/services/data/v60.0/sobjects/Lead"
    created, skipped, errors = [], 0, []
    async with httpx.AsyncClient(timeout=20) as c:
        for L in leads:
            if L.get("salesforce_id") or (L.get("lead_score") or 0) < min_score:
                skipped += 1
                continue
            name = (L.get("display_name") or "Unknown").strip()
            first, _, last = name.rpartition(" ")
            body = {
                "FirstName": first[:40] or None, "LastName": (last or name)[:80] or "Unknown",
                "Company": f"{L.get('platform') or 'Web'} lead"[:255], "LeadSource": "Intent Miner",
                "Title": (L.get("profession") or "")[:128] or None, "Email": L.get("email") or None,
                "Phone": L.get("phone") or None,
                "Description": (f"Intent: {L.get('intent_type')} · score {L.get('lead_score')} ({L.get('tier')})\n"
                                f"Destination: {L.get('destination') or '-'} · Timeline: {L.get('timeline') or '-'}\n"
                                f"Evidence: {' | '.join(L.get('evidence') or [])}\n"
                                f"Source: {' | '.join(s.get('url', '') for s in L.get('sources') or [])}")[:32000],
            }
            r = await c.post(base, json={k: v for k, v in body.items() if v},
                             headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
            if r.status_code in (200, 201):
                created.append({"id": L["id"], "salesforce_id": r.json().get("id")})
            else:
                errors.append(f"{name}: HTTP {r.status_code} {r.text[:160]}")
                if r.status_code in (401, 403):
                    break
    return {"created": created, "skipped": skipped, "errors": errors[:5]}
