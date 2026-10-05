"""
Government directories (India) and the matcher that links leads to them.

What goes in: lists that government bodies publish for anyone to download, e.g.
    * MCA company / LLP master data (name, CIN / LLPIN, registered office, e-mail where published)
    * Udyam / MSME lists, state transport-department lists of motor-driving schools
    * NCVT / Skill India ITI and training-provider lists, Indian Nursing Council recognised institutions
    * state council registers of professionals (for checking a person's registration)
    * any data.gov.in resource (API: api.data.gov.in, free key)
as CSV / Excel files or straight from the data.gov.in API. Columns are recognised by their names.

Not accepted: electoral rolls, Aadhaar, e-Shram / UAN or similar ID databases — they are not public, the law
restricts their use, and leaked copies are illegal to use. The importer refuses files that look like them.

The matcher ("find this lead in the government data"):
    1. blocking   — pull only records sharing a rare name token, or the same phone / email / website
    2. features   — fuzzy name similarity (spelling variants, initials, Pvt/Ltd/Institute noise removed),
                    place (city / district / state against the lead's location and evidence),
                    category (driving school, nursing, ITI … against the lead's profession / org type),
                    hard identifiers (registration no. / CIN in the lead's text, same phone / email / domain)
    3. score      — 0-100; identifiers decide on their own, otherwise name 60 + place 25 + category 15
    4. AI check   — borderline candidates go to the LLM with both records side by side; it picks one or none
    5. decision   — organisations: matched at 80+; a person needs a near-exact name AND a second signal
                    (place, category or identifier) AND the AI's confirmation unless an identifier matched.
                    Weaker candidates are kept as "possible" for a human to check, never applied.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import psycopg2.extras
from pydantic import BaseModel, Field

import supabase_db as db

SCHEMA = """
CREATE TABLE IF NOT EXISTS gov_records (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'org',
    name TEXT NOT NULL, name_key TEXT NOT NULL,
    reg_no TEXT, category TEXT, address TEXT, city TEXT, district TEXT, state TEXT, pincode TEXT,
    phone TEXT, email TEXT, website TEXT, extra JSONB, source_url TEXT,
    imported_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_gov_records_dataset ON gov_records (dataset);
CREATE INDEX IF NOT EXISTS idx_gov_records_kind ON gov_records (kind);
ALTER TABLE im_leads ADD COLUMN IF NOT EXISTS gov_match JSONB;
"""
TRGM = """
CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA public;
CREATE INDEX IF NOT EXISTS idx_gov_records_name_trgm ON gov_records USING gin (name_key gin_trgm_ops);
"""

_ready: set = set()
_lock = threading.Lock()


def _ensure() -> None:
    if db.current_schema() in _ready:
        return
    with _lock:
        if db.current_schema() in _ready:
            return
        db._ensure_schema()
        from intent_miner import store
        store._ensure()                       # im_leads must exist before the ALTER

        def run():
            with db._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(SCHEMA)
                conn.commit()
            try:                              # fast fuzzy search where the database allows the extension
                with db._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(TRGM)
                    conn.commit()
            except Exception:
                pass
        db._with_retry(run, "Creating government-directory table")
        _ready.add(db.current_schema())


def _q(sql: str, params=None, fetch: str = "", what: str = "Government-directory query"):
    _ensure()

    def run():
        with db._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if fetch == "many":
                    psycopg2.extras.execute_values(cur, sql, params, page_size=500)
                    out = None
                else:
                    cur.execute(sql, params)
                    out = cur.fetchall() if fetch == "all" else cur.fetchone() if fetch == "one" else None
            conn.commit()
            return out
    return db._with_retry(run, what)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
_ABBR = {"pvt": "private", "ltd": "limited", "govt": "government", "inst": "institute", "instt": "institute",
         "tech": "technical", "trg": "training", "edu": "education", "intl": "international", "co": "company",
         "centre": "center", "sr": "senior", "sec": "secondary", "hr": "higher", "mfg": "manufacturing",
         "&": "and", "st": "saint", "mt": "mount"}
_HONORIFIC = {"mr", "mrs", "ms", "miss", "dr", "shri", "sri", "smt", "kumari", "km", "prof", "er", "sir", "madam",
              "late", "master", "baby", "md", "mohd"}
_LEGAL = {"private", "limited", "llp", "the", "and", "of", "company", "opc", "pvt", "ltd", "m/s", "ms", "firm",
          "proprietor", "proprietorship", "partnership"}
# frequent words in institution names: they never decide a match on their own
_GENERIC = _LEGAL | {"driving", "school", "motor", "training", "institute", "center", "academy", "college", "nursing",
                     "india", "indian", "education", "educational", "society", "trust", "technical", "industrial",
                     "iti", "government", "public", "services", "service", "solutions", "international", "global",
                     "school", "university", "hospital", "health", "care", "medical", "sciences", "science",
                     "management", "skill", "skills", "development", "foundation", "enterprises", "group",
                     "consultants", "consultancy", "overseas", "manpower", "placement", "new", "shri", "sri"}


def name_tokens(name: str, org: bool = True) -> List[str]:
    t = (name or "").lower().replace("&", " and ")
    t = re.sub(r"[^a-z0-9ऀ-ॿ ]+", " ", t)
    out = []
    for w in t.split():
        w = _ABBR.get(w, w)
        if w in _HONORIFIC and not org:
            continue
        if org and w in _LEGAL:
            continue
        out.append(w)
    return out


def name_key(name: str, org: bool = True) -> str:
    return " ".join(name_tokens(name, org))


def _tok_eq(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if len(a) == 1 or len(b) == 1:                   # initial: "R" ~ "Rajesh"
        return 0.7 if a[0] == b[0] else 0.0
    r = SequenceMatcher(None, a, b).ratio()           # spelling variants: Gurpreet / Gurprit, Centre / Center
    return r if r >= (0.78 if min(len(a), len(b)) >= 5 else 0.86) else 0.0


def name_similarity(a: str, b: str, org: bool = True) -> float:
    ta, tb = name_tokens(a, org), name_tokens(b, org)
    if not ta or not tb:
        return 0.0
    if org:                                          # distinctive words weigh more than "driving school"
        wa = {w: (0.35 if w in _GENERIC else 1.0) for w in ta}
        wb = {w: (0.35 if w in _GENERIC else 1.0) for w in tb}
    else:
        wa, wb = {w: 1.0 for w in ta}, {w: 1.0 for w in tb}
    matched_a = sum(wa[x] * max(_tok_eq(x, y) for y in tb) for x in wa)
    matched_b = sum(wb[y] * max(_tok_eq(y, x) for x in ta) for y in wb)
    recall, precision = matched_a / sum(wa.values()), matched_b / sum(wb.values())
    if not recall or not precision:
        return 0.0
    f1 = 2 * recall * precision / (recall + precision)
    if not org:
        return round(f1, 3)
    da, db_ = [w for w in ta if w not in _GENERIC], [w for w in tb if w not in _GENERIC]
    if da and db_:
        # "Sharma Motor Driving School" vs "Verma Motor Driving School": the generic words agree, the
        # distinctive ones don't — a different institute.
        if not any(_tok_eq(x, y) for x in da for y in db_):
            return round(min(f1, 0.4), 3)
        seq = SequenceMatcher(None, " ".join(sorted(da)), " ".join(sorted(db_))).ratio()
        return round(max(f1, seq * 0.9), 3)
    return round(f1, 3)


def _rare_tokens(name: str, org: bool) -> List[str]:
    toks = [w for w in name_tokens(name, org) if len(w) >= 3 and w not in _GENERIC]
    return sorted(set(toks), key=len, reverse=True)[:3]


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
_FIELDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("email", ("e-mail", "email", "mail id", "mail")),
    ("website", ("website", "web site", "url", "web address")),
    ("phone", ("mobile", "phone", "telephone", "contact no", "contact number", "tel", "landline", "fax")),
    ("reg_no", ("registration no", "registration number", "reg no", "reg. no", "regn", "ra no", "rc no",
                "licence no.", "license no.", "cin", "llpin", "llp identification",
                "udyam", "licence no", "license no", "licence number", "license number", "iti code", "mis code",
                "institute code", "affiliation no", "registration", "code")),
    ("name", ("name of the ra", "name of ra", "ra name", "name of recruiting agent", "recruiting agent name",
              "name of the recruiting agent", "name of the agency", "agency name", "name of agency",
              "name of the institute", "name of institute", "institute name", "name of school", "school name",
              "name of the school", "company name", "name of company", "llp name", "enterprise name",
              "name of enterprise", "college name", "name of college", "institution name", "name of institution",
              "centre name", "center name", "establishment name", "firm name", "organisation name",
              "organization name", "name of the driving school", "tp name", "training partner", "iti name",
              "professional name", "nurse name", "full name", "name")),
    ("category", ("category", "type", "class of", "trade", "course", "activity", "sector", "nic", "qualification",
                  "speciality", "specialty", "discipline")),
    ("address", ("registered office address", "address", "location", "office")),
    ("pincode", ("pin code", "pincode", "pin")),
    ("district", ("district",)),
    ("city", ("city", "town", "village", "tehsil", "taluk", "block")),
    ("state", ("state", "ut name", "union territory")),
]
_EXCLUDE = {
    "name": ("father", "husband", "mother", "director", "owner", "principal", "contact person", "state name",
             "district name", "city name", "course name", "trade name", "user name"),
    "phone": ("fax",),
    "reg_no": ("pin", "state code", "district code", "nic code", "pincode"),
    "address": ("email", "e-mail", "web", "mail"),
    "category": ("email",),
}
# ID databases the law restricts; refuse them at import.
_FORBIDDEN = re.compile(r"\bepic\b|elector|voter|aadha+r|\buid\b|uidai|e-?shram|\buan\b|ration card|"
                        r"\bpan (?:no|number|card)|part no|relation name|relation type", re.IGNORECASE)


def map_columns(header: List[str]) -> Dict[str, int]:
    cols: Dict[str, int] = {}
    low = [re.sub(r"\s+", " ", (h or "").strip().lower().replace("_", " ")) for h in header]
    for field, keys in _FIELDS:
        for key in keys:
            idx = next((i for i, h in enumerate(low) if h and key in h and i not in cols.values()
                        and not any(x in h for x in _EXCLUDE.get(field, ()))), None)
            if idx is not None:
                cols[field] = idx
                break
    return cols


def forbidden(dataset: str, header: List[str]) -> Optional[str]:
    hits = [str(h) for h in [dataset] + list(header) if h and _FORBIDDEN.search(str(h))]
    if hits:
        return ("This looks like an ID / voter / Aadhaar / e-Shram style list (" + ", ".join(map(str, hits[:3])) +
                "). Those are not public data and using them for recruitment outreach is not allowed, so the file "
                "was not imported.")
    return None


def _clean(v) -> str:
    v = "" if v is None else str(v).strip()
    return "" if v.lower() in ("na", "n/a", "nil", "none", "-", "--", "null", "not available", "0") else v


def rows_to_records(dataset: str, kind: str, rows: List[List[str]], source_url: str = "") -> Tuple[List[dict], dict]:
    rows = [r for r in rows if any(_clean(c) for c in r)]
    if not rows:
        return [], {"rows": 0, "columns": {}}
    h = next((i for i, r in enumerate(rows[:15]) if "name" in map_columns(r)), 0)
    header, body = rows[h], rows[h + 1:]
    bad = forbidden(dataset, header)
    if bad:
        raise ValueError(bad)
    cols = map_columns(header)
    if "name" not in cols:
        return [], {"rows": len(body), "columns": {}, "error": "no name column found"}
    org = kind != "person"
    out, skipped = [], 0
    for row in body:
        get = lambda f: _clean(row[cols[f]]) if f in cols and cols[f] < len(row) else ""
        name = get("name")
        if not name or len(name) < 3:
            skipped += 1
            continue
        extra = {header[i]: _clean(row[i]) for i in range(min(len(header), len(row)))
                 if i not in cols.values() and _clean(row[i]) and header[i]}
        web = get("website")
        out.append({
            "dataset": dataset, "kind": kind, "name": name[:300], "name_key": name_key(name, org),
            "reg_no": get("reg_no")[:120] or None, "category": get("category")[:200] or None,
            "address": get("address")[:500] or None, "city": get("city")[:100] or None,
            "district": get("district")[:100] or None, "state": get("state")[:100] or None,
            "pincode": get("pincode")[:12] or None, "phone": get("phone")[:120] or None,
            "email": (get("email").lower()[:200] or None) if "@" in get("email") else None,
            "website": web[:300] or None, "extra": json.dumps(dict(list(extra.items())[:25])),
            "source_url": source_url or None,
        })
    return out, {"rows": len(body), "skipped": skipped, "columns": {f: header[i] for f, i in cols.items()},
                 "header_row": h + 1}


def save_records(records: List[dict], replace_dataset: bool = False) -> int:
    if not records:
        return 0
    if replace_dataset:
        _q("DELETE FROM gov_records WHERE dataset = %s", (records[0]["dataset"],), what="Replacing dataset")
    else:                                    # importing the same file twice adds nothing
        have = _q("SELECT name_key, COALESCE(reg_no, '') AS r, COALESCE(district, city, '') AS d FROM gov_records "
                  "WHERE dataset = %s", (records[0]["dataset"],), "all")
        known = {(h["name_key"], h["r"], h["d"]) for h in have}
        records = [r for r in records
                   if (r["name_key"], r["reg_no"] or "", r["district"] or r["city"] or "") not in known]
        if not records:
            return 0
    cols = ["dataset", "kind", "name", "name_key", "reg_no", "category", "address", "city", "district", "state",
            "pincode", "phone", "email", "website", "extra", "source_url"]
    _q(f"INSERT INTO gov_records ({', '.join(cols)}) VALUES %s", [tuple(r[c] for c in cols) for r in records],
       "many", "Saving government records")
    return len(records)


_RA_NO = re.compile(r"\bB\s?-?\s?\d{2,5}\s?/\s?[A-Z]{2,6}\s?/\s?[A-Z]{2,6}[^\s,;]{0,40}", re.IGNORECASE)
MAX_PDF_PAGES = int(os.getenv("GOV_PDF_MAX_PAGES", "400") or 400)


def pdf_rows(data: bytes) -> List[List[str]]:
    """Table rows from a PDF (official lists are usually tables). Header rows repeat on every page; only the
    first is kept. Falls back to one row per text line when the PDF has no ruled tables."""
    import io
    import pdfplumber
    rows: List[List[str]] = []
    header = None
    lines: List[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages[:MAX_PDF_PAGES]:
            tables = page.extract_tables() or []
            for t in tables:
                for r in t:
                    cells = [re.sub(r"\s+", " ", c or "").strip() for c in r]
                    if not any(cells):
                        continue
                    if header is None and "name" in map_columns(cells):
                        header = cells
                        rows.append(cells)
                    elif header is not None and cells == header:
                        continue
                    else:
                        rows.append(cells)
            if not tables:
                lines += (page.extract_text() or "").splitlines()
    if rows:
        return rows
    # no tables: lines with a registration number become "name | reg no | rest" rows
    out = [["Name", "Registration No", "Address"]]
    for ln in lines:
        m = _RA_NO.search(ln)
        if m:
            before, after = ln[:m.start()].strip(" ,.-|0123456789"), ln[m.end():].strip(" ,.-|")
            out.append([before or after[:80], m.group(0), after])
    return out if len(out) > 1 else []


def _html_rows(data: bytes) -> List[List[str]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(data, "html.parser")
    best: List[List[str]] = []
    for table in soup.find_all("table"):
        rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])] for tr in table.find_all("tr")]
        if len(rows) > len(best):
            best = rows
    return best


def read_rows(filename: str, data: bytes) -> List[List[str]]:
    import portal_import
    if (filename or "").lower().endswith(".pdf") or data[:5] == b"%PDF-":
        return pdf_rows(data)
    return portal_import._read_rows(filename, data)


async def import_url(dataset: str, kind: str, url: str, replace: bool = False) -> dict:
    """A list published on a website — PDF, Excel / CSV download or an HTML table — straight into gov_records."""
    import httpx
    async with httpx.AsyncClient(timeout=90, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0 (compatible; list-import)"}) as c:
        r = await c.get(url)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} downloading the list")
    ctype = r.headers.get("content-type", "")
    name = url.split("?")[0].rsplit("/", 1)[-1]
    if "html" in ctype and not name.lower().endswith((".pdf", ".xls", ".xlsx", ".csv")):
        rows = await asyncio.to_thread(_html_rows, r.content)
    else:
        rows = await asyncio.to_thread(read_rows, name, r.content)
    records, info = rows_to_records(dataset, kind, rows, url)
    info["saved"] = await asyncio.to_thread(save_records, records, replace)
    await asyncio.to_thread(_index_dataset, dataset, info)
    return info


def import_file(dataset: str, kind: str, filename: str, data: bytes, source_url: str = "",
                replace: bool = False) -> dict:
    rows = read_rows(filename, data)
    records, info = rows_to_records(dataset, kind, rows, source_url)
    info["saved"] = save_records(records, replace)
    _index_dataset(dataset, info)
    return info


def _index_dataset(dataset: str, info: dict) -> None:
    try:
        import vectors
        info["indexed"] = vectors.index_gov_dataset(dataset)
    except Exception as exc:
        info["index_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"


async def import_datagov(api_key: str, resource_id: str, dataset: str, kind: str, max_records: int = 5000,
                         replace: bool = False) -> dict:
    """Pull a data.gov.in resource through its API (https://api.data.gov.in/resource/<id>)."""
    rows: List[List[str]] = []
    header: List[str] = []
    ids: List[str] = []
    title = ""
    async with httpx.AsyncClient(timeout=30) as c:
        offset = 0
        while offset < max_records:
            r = await c.get(f"https://api.data.gov.in/resource/{resource_id}",
                            params={"api-key": api_key, "format": "json", "offset": offset,
                                    "limit": min(1000, max_records - offset)})
            if r.status_code != 200:
                raise RuntimeError(f"data.gov.in HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            if not header:
                fields = data.get("field") or []
                ids = [f.get("id") for f in fields if f.get("id")]
                header = [f.get("name") or f.get("id") for f in fields if f.get("id")]
                title = data.get("title") or ""
            recs = data.get("records") or []
            if not recs:
                break
            if not ids:
                ids = list(recs[0].keys())
                header = ids
            rows += [[str(x.get(i, "") or "") for i in ids] for x in recs]
            offset += len(recs)
            if len(recs) < 1000 or offset >= int(data.get("total") or 0):
                break
    records, info = rows_to_records(dataset or title or resource_id, kind, [header] + rows,
                                    f"https://data.gov.in/resource/{resource_id}")
    info["saved"] = await asyncio.to_thread(save_records, records, replace)
    await asyncio.to_thread(_index_dataset, dataset or title or resource_id, info)
    info["title"] = title
    return info


def datasets() -> List[dict]:
    rows = _q("""SELECT dataset, kind, COUNT(*) AS records, COUNT(phone) AS with_phone, COUNT(email) AS with_email,
                        MAX(imported_at) AS imported_at
                 FROM gov_records GROUP BY dataset, kind ORDER BY MAX(imported_at) DESC""", fetch="all")
    return [db._serialize_row(dict(r)) for r in rows]


def delete_dataset(name: str) -> int:
    row = _q("WITH d AS (DELETE FROM gov_records WHERE dataset = %s RETURNING 1) SELECT COUNT(*) AS n FROM d",
             (name,), "one", "Deleting dataset")
    return int(row["n"])


def count() -> int:
    try:
        return int(_q("SELECT COUNT(*) AS n FROM gov_records", fetch="one")["n"])
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _digits(p: Optional[str]) -> str:
    return re.sub(r"\D", "", p or "")[-10:]


def _domain(u: Optional[str]) -> str:
    if not u:
        return ""
    host = urlparse(u if "//" in u else "http://" + u).hostname or ""
    return host.lower().removeprefix("www.")


def _lead_text(L: dict) -> str:
    return " ".join(str(x) for x in [L.get("display_name"), L.get("origin"), L.get("profession"),
                                     " ".join(L.get("evidence") or []), " ".join(L.get("why") or [])] if x)


def _semantic_candidates(L: dict, kind: str) -> List[dict]:
    """Records the hybrid search index ranks closest to the lead (meaning + words + fuzzy name)."""
    import asyncio
    import vectors
    q = " ".join(str(x) for x in (L.get("display_name"), L.get("origin"), L.get("profession")) if x)
    try:
        hits = asyncio.run(vectors.search(q, ["gov"], None, 15))
    except Exception:
        return []
    ids = [int(h["ref"]) for h in hits if str(h["ref"]).isdigit()]
    if not ids:
        return []
    rows = _q("SELECT * FROM gov_records WHERE id = ANY(%s) AND kind = %s", (ids, kind), "all",
              "Reading matched records")
    return [dict(r) for r in rows]


def _blocking(L: dict, kind: str) -> List[dict]:
    org = kind != "person"
    toks = _rare_tokens(L.get("display_name") or "", org)
    conds, params = [], []
    for t in toks:
        conds.append("name_key LIKE %s")
        params.append(f"%{t}%")
    d = _digits(L.get("phone"))
    if len(d) == 10:
        conds.append("right(regexp_replace(COALESCE(phone,''), '\\D', '', 'g'), 10) = %s")
        params.append(d)
    if L.get("email"):
        conds.append("lower(email) = lower(%s)")
        params.append(L["email"])
    dom = _domain(L.get("website"))
    if dom:
        conds.append("lower(website) LIKE %s")
        params.append(f"%{dom}%")
    rows = [dict(r) for r in _q(f"SELECT * FROM gov_records WHERE kind = %s AND ({' OR '.join(conds)}) LIMIT 400",
                                [kind] + params, "all", "Searching government records")] if conds else []
    seen = {r["id"] for r in rows}
    return rows + [r for r in _semantic_candidates(L, kind) if r["id"] not in seen]


def _place_score(L: dict, rec: dict) -> Tuple[float, str]:
    """1.0 = city / district / pincode agrees, 0.5 = only the state, 0.35 = lead place unknown, 0 = differs."""
    text = " ".join(str(x) for x in (L.get("origin"), " ".join(L.get("evidence") or [])) if x).lower()
    if not text.strip():
        return 0.35, ""
    for f in ("pincode", "city", "district"):
        v = (rec.get(f) or "").lower().strip()
        if v and len(v) >= 3 and re.search(r"\b" + re.escape(v) + r"\b", text):
            return 1.0, f"{f} {rec[f]}"
    st = (rec.get("state") or "").lower().strip()
    if st and re.search(r"\b" + re.escape(st) + r"\b", text):
        return 0.5, f"state {rec['state']}"
    return (0.0, "") if L.get("origin") else (0.35, "")


def _category_score(L: dict, rec: dict) -> float:
    want = set(re.findall(r"[a-z]{4,}", " ".join(str(x) for x in (L.get("profession"), L.get("display_name")) if x).lower()))
    have = set(re.findall(r"[a-z]{4,}", " ".join(str(x) for x in (rec.get("category"), rec.get("name"), rec.get("dataset"))
                                                 if x).lower()))
    if not want or not have:
        return 0.4
    stem = lambda s: {w[:5] for w in s}
    return 1.0 if stem(want) & stem(have) else 0.0


def _identifier(L: dict, rec: dict) -> str:
    if rec.get("reg_no") and len(rec["reg_no"]) >= 5 and rec["reg_no"].lower() in _lead_text(L).lower():
        return f"registration no. {rec['reg_no']} appears in the lead's own text"
    if _digits(L.get("phone")) and len(_digits(L.get("phone"))) == 10 and \
            _digits(L.get("phone")) in re.sub(r"\D", "", rec.get("phone") or ""):
        return "same phone number"
    if L.get("email") and rec.get("email") and L["email"].lower() == rec["email"].lower():
        return "same email"
    if _domain(L.get("website")) and _domain(L.get("website")) == _domain(rec.get("website")):
        return "same website"
    return ""


def score_candidate(L: dict, rec: dict, kind: str) -> dict:
    org = kind != "person"
    ident = _identifier(L, rec)
    ns = name_similarity(L.get("display_name") or "", rec["name"], org)
    ps, pwhy = _place_score(L, rec)
    cs = _category_score(L, rec)
    score = round(ns * 60 + ps * 25 + cs * 15)
    if ident:
        score = max(score, 95)
    return {"record": rec, "score": score, "name_similarity": ns, "place": ps, "place_why": pwhy,
            "category": cs, "identifier": ident}


class _Choice(BaseModel):
    choice: int = Field(-1, description="1-based number of the matching record, or -1 if none is the same")
    confidence: float = Field(0.0, ge=0, le=1)
    reason: str = ""


AI_SYSTEM = ("You decide whether a lead found online and a record from an official Indian government list are the "
             "SAME organisation or person. Compare names (allow spelling variants, abbreviations, Pvt/Ltd, "
             "transliteration), places, type of business / profession and any identifiers. Different branches or "
             "similarly named but different entities are NOT the same. When unsure, answer -1.")


def _rec_line(i: int, rec: dict) -> str:
    parts = [rec["name"], rec.get("category"), rec.get("address"), rec.get("city"), rec.get("district"),
             rec.get("state"), f"reg {rec['reg_no']}" if rec.get("reg_no") else None, f"list: {rec['dataset']}"]
    return f"{i}. " + " | ".join(str(p) for p in parts if p)


async def _ai_pick(ai, L: dict, cands: List[dict]) -> Optional[Tuple[dict, float, str]]:
    prompt = ("LEAD\n" + " | ".join(str(x) for x in (
        L.get("display_name"), L.get("profession"), L.get("origin"), L.get("website"),
        "; ".join((L.get("evidence") or [])[:2])) if x) +
        "\n\nGOVERNMENT RECORDS\n" + "\n".join(_rec_line(i + 1, c["record"]) for i, c in enumerate(cands)))
    res: _Choice = await ai.generate_structured(prompt, _Choice, system_instruction=AI_SYSTEM, temperature=0.0,
                                                max_retries=2)
    if 1 <= res.choice <= len(cands) and res.confidence >= 0.7:
        return cands[res.choice - 1], res.confidence, res.reason[:300]
    return None


def _result(c: dict, status: str, method: str, reason: str = "") -> dict:
    r = c["record"]
    return {"status": status, "method": method, "score": c["score"], "reason": reason,
            "dataset": r["dataset"], "record_id": r["id"], "name": r["name"], "reg_no": r.get("reg_no"),
            "category": r.get("category"), "address": r.get("address"), "city": r.get("city"),
            "district": r.get("district"), "state": r.get("state"), "phone": r.get("phone"),
            "email": r.get("email"), "website": r.get("website"),
            "signals": [s for s in (f"name {round(c['name_similarity'] * 100)}%", c["place_why"],
                                    "type matches" if c["category"] >= 1 else "", c["identifier"]) if s]}


async def match_lead(L: dict, ai=None, use_ai: bool = True) -> Optional[dict]:
    """Best government record for one lead: {status: matched|possible, …} or None."""
    kind = "org" if (L.get("lead_key") or "").startswith("org:") else "person"
    if not L.get("display_name") or len(name_tokens(L["display_name"], kind != "person")) < (1 if kind == "org" else 2):
        return None
    recs = await asyncio.to_thread(_blocking, L, kind)
    if not recs:
        return None
    seen, cands = set(), []
    for c in sorted((score_candidate(L, r, kind) for r in recs), key=lambda c: -c["score"]):
        r = c["record"]
        k = (r["name_key"], (r.get("reg_no") or "").lower(), (r.get("district") or r.get("city") or "").lower())
        if k not in seen:                    # the same record in two lists / imported twice counts once
            seen.add(k)
            cands.append(c)
        if len(cands) == 3:
            break
    best = cands[0]
    if best["identifier"]:
        return _result(best, "matched", "identifier", best["identifier"])
    if kind == "org":
        if best["score"] >= 80 and best["name_similarity"] >= 0.8 and \
                (len(cands) == 1 or cands[1]["score"] < best["score"] - 8):
            return _result(best, "matched", "rules")
        border = [c for c in cands if c["score"] >= 55 and c["name_similarity"] >= 0.6]
    else:
        border = [c for c in cands if c["name_similarity"] >= 0.85 and (c["place"] >= 1 or c["category"] >= 1)]
    if border and ai is not None and use_ai:
        try:
            picked = await _ai_pick(ai, L, border)
        except Exception:
            picked = None
        if picked:
            c, conf, why = picked
            return _result(c, "matched", "ai", f"AI {round(conf * 100)}%: {why}")
    if best["score"] >= 55 and best["name_similarity"] >= 0.6:
        return _result(best, "possible", "rules", "needs a human check")
    return None


def apply_match(L: dict, m: Optional[dict]) -> bool:
    """Put a match on a lead; a *matched* record also fills missing phone / email / website. Returns True when
    a contact was added."""
    if not m:
        return False
    L["gov_match"] = m
    L["why"] = [w for w in (L.get("why") or []) if not str(w).startswith(("✓ Government record", "? Possible government"))]
    where = ", ".join(x for x in (m.get("district") or m.get("city"), m.get("state")) if x)
    if m["status"] != "matched":
        L["why"] = (L.get("why") or []) + [f"? Possible government record: {m['name']} ({m['dataset']}"
                                           f"{', ' + where if where else ''}) — check before using"]
        return False
    L["why"] = (L.get("why") or []) + [f"✓ Government record: {m['name']} — {m['dataset']}"
                                       f"{', ' + where if where else ''}"
                                       f"{', reg ' + m['reg_no'] if m.get('reg_no') else ''} "
                                       f"({m['method']}; {', '.join(m['signals'])})"]
    from schema import clean_email, clean_phone
    before = (L.get("phone"), L.get("email"))
    phone = next((clean_phone(p) for p in re.split(r"[,;/]", m.get("phone") or "") if clean_phone(p)), None)
    L["phone"] = L.get("phone") or phone
    L["email"] = L.get("email") or (clean_email(m["email"]) if m.get("email") else None)
    if m.get("website") and not L.get("website"):
        L["website"] = m["website"] if m["website"].startswith("http") else "http://" + m["website"]
    return (L.get("phone"), L.get("email")) != before


async def match_leads(leads: List[dict], ai=None, use_ai: bool = True, max_ai: int = 8) -> Dict[str, int]:
    """Match a batch of leads (in place). AI checks are capped per batch."""
    stats = {"checked": 0, "matched": 0, "possible": 0, "contacts": 0, "ai": 0}
    sem = asyncio.Semaphore(4)
    budget = {"ai": max_ai}

    async def one(L):
        async with sem:
            allow = use_ai and budget["ai"] > 0
            if allow:
                budget["ai"] -= 1
            try:
                return await match_lead(L, ai, allow)
            except Exception:
                return None
    results = await asyncio.gather(*(one(L) for L in leads))
    for L, m in zip(leads, results):
        stats["checked"] += 1
        if not m:
            continue
        stats[m["status"]] += 1
        stats["ai"] += m["method"] == "ai"
        stats["contacts"] += apply_match(L, m)
    return stats


def leads_to_match(lead_ids: List[str], only_unmatched: bool, limit: int, after: str = "") -> List[dict]:
    """Leads in id order after `after` (a cursor, so repeated calls walk through all leads)."""
    from intent_miner import store
    store._ensure()
    _ensure()
    conds, params = ["display_name IS NOT NULL"], []
    if lead_ids:
        conds.append("id::text = ANY(%s)")
        params.append(lead_ids)
    if only_unmatched:
        conds.append("(gov_match IS NULL OR gov_match->>'status' <> 'matched')")
    if after:
        conds.append("id::text > %s")
        params.append(after)
    rows = _q(f"SELECT * FROM im_leads WHERE {' AND '.join(conds)} ORDER BY id::text LIMIT %s", params + [limit], "all")
    return [dict(r) for r in rows]


def save_lead_match(L: dict) -> None:
    _q("""UPDATE im_leads SET gov_match = %s, phone = COALESCE(phone, %s), email = COALESCE(email, %s),
                 website = COALESCE(website, %s), why = %s WHERE lead_key = %s""",
       (json.dumps(L.get("gov_match"), default=str), L.get("phone"), L.get("email"), L.get("website"),
        json.dumps(L.get("why") or []), L["lead_key"]), what="Saving government match")
