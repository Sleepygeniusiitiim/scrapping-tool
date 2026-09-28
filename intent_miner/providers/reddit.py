"""
Reddit through its official API (app-only OAuth: REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET from
reddit.com/prefs/apps, type "script" or "web app"). Without credentials Reddit's public JSON is tried,
which Reddit often refuses from cloud servers; search-engine discovery with the snippet is the fallback.

Reddit's developer terms restrict commercial re-use of Reddit data (data brokerage, ad targeting,
model training). Use this for your own recruiting outreach, not to build or sell a database.
"""

from __future__ import annotations

import datetime as dt
import re
import time
from typing import Dict, List, Optional

import httpx

from ..models import QuerySpec, RawDocument, Unit
from .base import BaseProvider, Capability, ProviderConfig

UA = "web:intent-miner:1.0 (candidate discovery tool)"
_token: Dict[str, object] = {}          # app-only token cache (per warm serverless instance)
_POST_ID = re.compile(r"/comments/([a-z0-9]+)", re.IGNORECASE)


def _day(ts) -> Optional[str]:
    try:
        return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        return None


class RedditProvider(BaseProvider):
    name = "reddit"
    capability = Capability(search=True, fetch=True, comments=True, access="api")
    config = ProviderConfig("reddit", requests_per_second=1.0, max_concurrency=2)

    @property
    def has_api(self) -> bool:
        return bool(self.keys.get("reddit_client_id") and self.keys.get("reddit_client_secret"))

    async def _auth(self, c: httpx.AsyncClient) -> Optional[str]:
        if not self.has_api:
            return None
        if _token.get("value") and float(_token.get("expires", 0)) > time.time() + 60:
            return str(_token["value"])
        r = await c.post("https://www.reddit.com/api/v1/access_token",
                         auth=(self.keys["reddit_client_id"], self.keys["reddit_client_secret"]),
                         data={"grant_type": "client_credentials"}, headers={"User-Agent": UA})
        if r.status_code != 200:
            raise RuntimeError(f"Reddit login failed (HTTP {r.status_code}): {r.text[:120]}")
        data = r.json()
        _token.update(value=data["access_token"], expires=time.time() + float(data.get("expires_in", 3600)))
        return str(_token["value"])

    async def _get(self, path: str, params: dict) -> dict:
        try:
            return await self._get_raw(path, params)
        except RuntimeError:
            raise
        except Exception as exc:                  # timeouts, HTML instead of JSON, connection resets
            self.note("failed")
            raise RuntimeError(f"Reddit {type(exc).__name__}: {str(exc)[:120]}")

    async def _get_raw(self, path: str, params: dict) -> dict:
        async with self.limiter:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
                token = await self._auth(c)
                if token:
                    r = await c.get("https://oauth.reddit.com" + path, params={**params, "raw_json": 1},
                                    headers={"User-Agent": UA, "Authorization": f"bearer {token}"})
                else:
                    r = await c.get("https://www.reddit.com" + path.rstrip("/") + ".json",
                                    params={**params, "raw_json": 1}, headers={"User-Agent": UA})
        if r.status_code in (401, 403, 429):
            self.note("blocked")
            raise RuntimeError(f"Reddit refused ({r.status_code})" +
                               ("" if self.has_api else " — add a Reddit API app (client id + secret)"))
        if r.status_code >= 400:
            self.note("failed")
            raise RuntimeError(f"Reddit HTTP {r.status_code}")
        self.note("ok")
        return r.json()

    @staticmethod
    def _post_doc(d: dict) -> RawDocument:
        url = "https://www.reddit.com" + d.get("permalink", "")
        body = "\n".join(x for x in (d.get("title", ""), d.get("selftext", "")) if x).strip()
        author = d.get("author") if d.get("author") not in (None, "[deleted]", "AutoModerator") else None
        doc = RawDocument(url=url.rstrip("/") + "/", source="reddit", title=d.get("title", ""),
                          date=_day(d.get("created_utc")), via="api",
                          metadata={"subreddit": d.get("subreddit", ""), "score": str(d.get("score", "")),
                                    "comments": str(d.get("num_comments", ""))})
        doc.units.append(Unit("post", author, body, f"https://www.reddit.com/user/{author}" if author else None,
                              doc.date))
        return doc

    async def search(self, query: str, spec: QuerySpec, limit: int) -> List[dict]:
        age = spec.max_age_days or 365
        t = "day" if age <= 1 else "week" if age <= 7 else "month" if age <= 31 else "year" if age <= 366 else "all"
        hits: List[dict] = []
        paths = [("/search", {"q": query, "sort": "new", "t": t, "limit": min(limit, 50), "type": "link"})]
        for sub in spec.subreddits[:3]:
            paths.append((f"/r/{sub}/search", {"q": query, "restrict_sr": 1, "sort": "new", "t": t,
                                                "limit": min(limit, 25)}))
        errors = []
        for path, params in paths:
            try:
                data = await self._get(path, params)
            except RuntimeError as exc:
                errors.append(str(exc))
                continue
            for child in (data.get("data") or {}).get("children", []):
                d = child.get("data") or {}
                if not d.get("permalink"):
                    continue
                doc = self._post_doc(d)
                hits.append({"url": doc.url, "title": doc.title, "snippet": doc.units[0].text[:300],
                             "date": doc.date, "doc": doc})
        if not hits and errors:
            raise RuntimeError(errors[0])
        return hits

    async def fetch(self, url: str, hit: Optional[dict] = None) -> RawDocument:
        m = _POST_ID.search(url)
        if not m:
            return RawDocument(url=url, source="reddit", status="skipped", error="not a Reddit post URL")
        try:
            data = await self._get(f"/comments/{m.group(1)}", {"limit": 200, "depth": 4, "sort": "new"})
        except RuntimeError as exc:
            doc = (hit or {}).get("doc")
            if doc:                        # search already returned the post itself
                doc.error = str(exc)
                return doc
            return RawDocument(url=url, source="reddit", status="blocked", error=str(exc))
        listing = data if isinstance(data, list) else [data]
        post = ((listing[0].get("data") or {}).get("children") or [{}])[0].get("data") or {}
        doc = self._post_doc(post) if post.get("permalink") else RawDocument(url=url, source="reddit", via="api")

        def walk(children):
            for ch in children or []:
                d = ch.get("data") or {}
                if ch.get("kind") == "t1" and d.get("body") and d.get("author") not in ("[deleted]", "AutoModerator"):
                    doc.units.append(Unit("comment", d.get("author"), d["body"],
                                          f"https://www.reddit.com/user/{d.get('author')}", _day(d.get("created_utc"))))
                replies = d.get("replies")
                if isinstance(replies, dict):
                    walk((replies.get("data") or {}).get("children"))

        if len(listing) > 1:
            walk((listing[1].get("data") or {}).get("children"))
        return doc
