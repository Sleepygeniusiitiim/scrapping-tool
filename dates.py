"""
When a lead was active: the date of the post or comment it came from.

Sources, best first:
* the comment's / post's own schema.org date (datePublished / dateCreated) on the page,
* the LinkedIn activity id in the URL (the id's top bits are the post's creation time in ms),
* the page's published-date meta tags / <time> element,
* the date the search engine shows next to the result ("3 days ago", "Mar 5, 2024 — …").
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Optional

_LI_ACTIVITY = re.compile(r"(?:activity|ugcPost|share)[-:](\d{18,20})", re.IGNORECASE)
_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_REL = re.compile(r"\b(\d+|an?|one)\s*(min(?:ute)?s?|h(?:ou)?rs?|hours?|d(?:ays?)?|w(?:ee)?ks?|weeks?|mo(?:nths?)?|"
                  r"y(?:ea)?rs?|years?)\s+ago\b", re.IGNORECASE)
_SHORT_REL = re.compile(r"^\s*(\d+)\s*(h|d|w|mo|y|yr)\b", re.IGNORECASE)
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MDY = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b",
                  re.IGNORECASE)
_DMY = re.compile(r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?,?\s+(\d{4})\b",
                  re.IGNORECASE)


def today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def _valid(d: dt.date) -> Optional[str]:
    if dt.date(2000, 1, 1) <= d <= today() + dt.timedelta(days=1):
        return d.isoformat()
    return None


def from_linkedin_url(url: str) -> Optional[str]:
    """linkedin.com/posts/…-activity-7123456789012345678-abcd → 2023-10-…"""
    m = _LI_ACTIVITY.search(url or "")
    if not m:
        return None
    try:
        ms = int(m.group(1)) >> 22
        return _valid(dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date())
    except (ValueError, OverflowError, OSError):
        return None


def parse(text: Optional[str]) -> Optional[str]:
    """ISO date (YYYY-MM-DD) from an ISO timestamp, "Mar 5, 2024", "5 March 2024", "3 days ago", "2w"."""
    if not text:
        return None
    text = str(text).strip()
    m = _ISO.search(text)
    if m:
        try:
            return _valid(dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    for rx, order in ((_MDY, "mdy"), (_DMY, "dmy")):
        m = rx.search(text)
        if m:
            a, b, y = m.groups()
            mon, day = (a, b) if order == "mdy" else (b, a)
            try:
                return _valid(dt.date(int(y), _MONTHS[mon.lower()[:3]], int(day)))
            except (ValueError, KeyError):
                pass
    m = _REL.search(text) or _SHORT_REL.match(text)
    if m:
        n = m.group(1).lower()
        n = 1 if n in ("a", "an", "one") else int(n)
        unit = m.group(2).lower()
        if unit.startswith(("mi", "h")):
            days = 0
        elif unit.startswith("d"):
            days = n
        elif unit.startswith("w"):
            days = 7 * n
        elif unit.startswith("mo"):
            days = 30 * n
        else:
            days = 365 * n
        return _valid(today() - dt.timedelta(days=days))
    return None


def snippet_date(snippet: str) -> Optional[str]:
    """Search engines put the date at the start of a snippet: "Mar 5, 2024 — …" / "3 days ago · …"."""
    head = (snippet or "")[:40]
    if re.search(r"\s[—·-]\s|\.\.\.", head) or _REL.search(head):
        return parse(head)
    return None


def older_than(date_iso: Optional[str], months: float = 0, days: int = 0) -> bool:
    """Older than the window: `days` when given (custom days), else `months`."""
    limit = days or (months * 30.5 if months else 0)
    if not date_iso or not limit:
        return False
    try:
        d = dt.date.fromisoformat(date_iso[:10])
    except ValueError:
        return False
    return (today() - d).days > limit
