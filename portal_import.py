"""
Import candidates from files exported from job-portal employer accounts
(Naukri Resdex / RMS, foundit recruiter, WorkIndia, Indeed, Apna, or any ATS).

Every portal lets a paying employer download applicants or database search results as
Excel / CSV from inside their account. This module reads those files — it never logs in to a
portal: automated logins break the portals' terms and get the recruiter account blocked.

Column names differ by portal and change over time, so columns are matched by keywords
("Mobile No.", "Phone Number", "Contact" → phone, "Key Skills" → skills, …).
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from typing import Dict, List, Optional, Tuple

import dates
from schema import CandidateRecord

PORTALS = {
    "naukri": ("Naukri", "https://www.naukri.com/"),
    "foundit": ("foundit", "https://www.foundit.in/"),
    "workindia": ("WorkIndia", "https://www.workindia.in/"),
    "indeed": ("Indeed", "https://www.indeed.com/"),
    "apna": ("Apna", "https://apna.co/"),
    "naukrigulf": ("Naukrigulf", "https://www.naukrigulf.com/"),
    "other": ("Other portal / ATS", "https://import.invalid/"),
}

# field → keywords that identify its column (first match wins; checked on lower-cased header)
_FIELDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("email", ("e-mail", "email", "mail id", "mail")),
    ("phone", ("mobile", "phone", "contact no", "contact number", "whatsapp", "cell", "tel")),
    ("name", ("candidate name", "full name", "applicant name", "name")),
    ("current_role", ("current designation", "designation", "job title", "current title", "title", "role",
                      "profile headline", "headline", "position")),
    ("skills", ("key skills", "skills", "skill set", "keyskills")),
    ("current_location", ("current location", "location", "city", "address")),
    ("target_countries", ("preferred location", "preferred locations", "desired location", "target")),
    ("experience", ("total experience", "experience", "work exp", "exp")),
    ("company", ("current company", "current employer", "company", "employer", "organization")),
    ("date", ("applied on", "applied date", "application date", "last active", "last modified", "modified on",
              "updated on", "last login", "date")),
    ("profile_url", ("profile url", "profile link", "resume link", "cv link", "url", "link")),
]


# Headers that contain a field's keyword but are something else.
_EXCLUDE = {
    "name": ("company", "employer", "file", "father", "institute", "college", "school", "user name", "job name"),
    "current_location": ("preferred", "desired"),
    "experience": ("expected", "ctc", "salary"),
    "date": ("birth", "dob"),
    "phone": ("alternate email",),
    "current_role": ("job title applied", "applied for"),
}


def _read_rows(filename: str, data: bytes) -> List[List[str]]:
    name = (filename or "").lower()
    if name.endswith(".xlsx") or data[:2] == b"PK":
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        ws = wb.worksheets[0]
        return [["" if v is None else str(v) for v in row] for row in ws.iter_rows(values_only=True)]
    if name.endswith(".xls") or data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        import xlrd
        book = xlrd.open_workbook(file_contents=data)
        sh = book.sheet_by_index(0)
        return [[str(sh.cell_value(r, c)) for c in range(sh.ncols)] for r in range(sh.nrows)]
    if data[:200].lstrip().lower().startswith((b"<html", b"<table", b"<!doctype")):
        # Some portals' "Excel" download is really an HTML table.
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(data, "html.parser")
        return [[c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])] for tr in soup.find_all("tr")]
    text = data.decode("utf-8-sig", errors="replace")
    dialect = csv.Sniffer().sniff(text[:4000], delimiters=",;\t") if text.strip() else csv.excel
    return [row for row in csv.reader(io.StringIO(text), dialect)]


def _map_columns(header: List[str]) -> Dict[str, int]:
    cols: Dict[str, int] = {}
    low = [re.sub(r"\s+", " ", (h or "").strip().lower()) for h in header]
    for field, keys in _FIELDS:
        for key in keys:
            idx = next((i for i, h in enumerate(low) if h and key in h and i not in cols.values()
                        and not any(x in h for x in _EXCLUDE.get(field, ()))), None)
            if idx is not None:
                cols[field] = idx
                break
    return cols


def _header_row(rows: List[List[str]]) -> int:
    """Exports sometimes start with a title / filter block; the header is the first row naming a name
    column and a contact column."""
    for i, row in enumerate(rows[:15]):
        cols = _map_columns(row)
        if "name" in cols and ("phone" in cols or "email" in cols):
            return i
    return 0


def _split(v: str) -> List[str]:
    return [x.strip() for x in re.split(r"[,;|/\n]", v or "") if x.strip()]


def parse_export(filename: str, data: bytes, portal: str, applied: bool) -> Tuple[List[CandidateRecord], dict]:
    """Returns (records, info). `applied`: the file lists people who applied to your job (interested)."""
    label, base = PORTALS.get(portal, PORTALS["other"])
    rows = [r for r in _read_rows(filename, data) if any((c or "").strip() for c in r)]
    if not rows:
        return [], {"rows": 0, "columns": {}, "skipped": 0}
    h = _header_row(rows)
    header, body = rows[h], rows[h + 1:]
    cols = _map_columns(header)
    records, skipped = [], 0
    for row in body:
        get = lambda f: (row[cols[f]].strip() if f in cols and cols[f] < len(row) and row[cols[f]] else "")
        name, email, phone = get("name"), get("email"), get("phone")
        if not (name or email or phone):
            skipped += 1
            continue
        role = get("current_role")
        exp, company = get("experience"), get("company")
        digits = re.sub(r"\D", "", phone)[-10:]
        key = hashlib.sha1(f"{email.lower()}|{digits}|{name.lower()}".encode()).hexdigest()[:16]
        url = get("profile_url")
        evidence = (f"{'Applied via' if applied else 'Found in'} {label}: " +
                    ", ".join(x for x in (role, f"{exp} experience" if exp else "", company, get("current_location"))
                              if x))[:300]
        try:
            records.append(CandidateRecord(
                name=name or None, email=email or None, phone=(phone.split(",")[0] if phone else None),
                current_role=role or None, skills=_split(get("skills"))[:15],
                current_location=get("current_location") or None,
                target_countries=_split(get("target_countries"))[:8],
                evidence_snippet=evidence,
                source_url=f"{base}#import-{portal}-{key}",
                platform=f"{portal}_export",
                profile_url=url if url.startswith("http") else None,
                activity_date=dates.parse(get("date")),
                shows_interest=True if applied else None,
                contact_source=f"portal_export:{portal}" if (email or phone) else None,
            ))
        except Exception:
            skipped += 1
    info = {"rows": len(body), "skipped": skipped,
            "columns": {f: header[i] for f, i in cols.items()}, "header_row": h + 1}
    return records, info
