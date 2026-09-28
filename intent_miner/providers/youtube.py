"""
YouTube through the official YouTube Data API v3 (key: YOUTUBE_API_KEY — a Google Cloud API key with
"YouTube Data API v3" enabled; the Google Programmable Search key works too if that API is enabled on its
project). Free quota: 10,000 units/day — a video search costs 100, a page of 100 comments costs 1.

Recruitment / "jobs abroad" videos collect comments like "Interested sir, GNM nurse, 98xxxxxxx" — each
comment (and reply) becomes a unit with the author's name, channel link and date.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

import httpx

import dates

from ..models import QuerySpec, RawDocument, Unit
from .base import BaseProvider, Capability, ProviderConfig

API = "https://www.googleapis.com/youtube/v3"
_VIDEO_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")
MAX_COMMENT_PAGES = 2          # 2 × 100 comments per video


def video_id(url: str) -> Optional[str]:
    m = _VIDEO_ID.search(url or "")
    return m.group(1) if m else None


class YouTubeProvider(BaseProvider):
    name = "youtube"
    capability = Capability(search=True, fetch=True, comments=True, access="api")
    config = ProviderConfig("youtube", requests_per_second=4, max_concurrency=3)

    @property
    def key(self) -> str:
        return (self.keys.get("youtube") or os.getenv("YOUTUBE_API_KEY", "") or self.keys.get("google_cse_key") or
                "").strip()

    async def _get(self, path: str, params: dict) -> dict:
        if not self.key:
            raise RuntimeError("No YouTube Data API key (YOUTUBE_API_KEY)")
        async with self.limiter:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.get(f"{API}/{path}", params={**params, "key": self.key})
        if r.status_code == 403 and "commentsDisabled" in r.text:
            self.note("ok")
            return {"items": []}
        if r.status_code >= 400:
            self.note("blocked" if r.status_code in (401, 403, 429) else "failed")
            msg = ((r.json().get("error") or {}).get("message") if "json" in r.headers.get("content-type", "")
                   else r.text[:150])
            raise RuntimeError(f"YouTube API {r.status_code}: {str(msg)[:160]}")
        self.note("ok")
        return r.json()

    async def search(self, query: str, spec: QuerySpec, limit: int) -> List[dict]:
        params = {"part": "snippet", "q": query, "type": "video", "maxResults": min(limit, 25), "order": "relevance"}
        if spec.max_age_days:
            import datetime as dt
            since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=spec.max_age_days)
            params["publishedAfter"] = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        data = await self._get("search", params)
        hits = []
        for it in data.get("items", []):
            vid = (it.get("id") or {}).get("videoId")
            sn = it.get("snippet") or {}
            if vid:
                hits.append({"url": f"https://www.youtube.com/watch?v={vid}", "title": sn.get("title", ""),
                             "snippet": sn.get("description", "")[:300], "date": dates.parse(sn.get("publishedAt"))})
        return hits

    async def fetch(self, url: str, hit: Optional[dict] = None) -> RawDocument:
        vid = video_id(url)
        doc = RawDocument(url=f"https://www.youtube.com/watch?v={vid}" if vid else url, source="youtube",
                          title=(hit or {}).get("title", ""), date=(hit or {}).get("date"), via="api")
        if not vid:
            doc.status, doc.error = "skipped", "not a YouTube video URL"
            return doc
        caption = " ".join(x for x in ((hit or {}).get("title"), (hit or {}).get("snippet")) if x)
        if caption:
            doc.units.append(Unit("post", None, caption, None, doc.date))     # the video = the "post" commented on
        token = None
        try:
            for _ in range(MAX_COMMENT_PAGES):
                params = {"part": "snippet,replies", "videoId": vid, "maxResults": 100, "order": "time",
                          "textFormat": "plainText"}
                if token:
                    params["pageToken"] = token
                data = await self._get("commentThreads", params)
                for th in data.get("items", []):
                    top = ((th.get("snippet") or {}).get("topLevelComment") or {}).get("snippet") or {}
                    comments = [top] + [(r.get("snippet") or {}) for r in ((th.get("replies") or {}).get("comments") or [])]
                    for c in comments:
                        text = (c.get("textOriginal") or c.get("textDisplay") or "").strip()
                        if text:
                            doc.units.append(Unit("comment", (c.get("authorDisplayName") or "").lstrip("@") or None,
                                                  text, c.get("authorChannelUrl"), dates.parse(c.get("publishedAt"))))
                token = data.get("nextPageToken")
                if not token:
                    break
        except RuntimeError as exc:
            doc.error = str(exc)
            if len(doc.units) <= 1:
                doc.status = "blocked"
        return doc
