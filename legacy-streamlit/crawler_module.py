"""
Crawl layer — Crawl4AI (headless Chromium) → clean Markdown.

* One browser is reused for a whole wave (cheap), pages run in small
  concurrent batches (default 5) so memory stays bounded.
* Every page gets its own hard timeout (asyncio.wait_for) on top of
  Crawl4AI's page_timeout, so one hanging page can't stall a batch.
* Login walls / empty pages are detected and reported as "blocked" so the
  pipeline can fall back to the search-result snippet.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

log = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 5
DEFAULT_PAGE_TIMEOUT_S = 30
MIN_USEFUL_CHARS = 250          # below this the page is treated as empty
MAX_MARKDOWN_CHARS = 60_000     # keep memory and LLM cost bounded

# Phrases that indicate an auth wall / bot challenge rather than real content.
_WALL_PATTERNS = re.compile(
    r"(authwall|sign in to view|join linkedin|log in to continue|login to continue|"
    r"please enable javascript|verify you are human|are you a robot|captcha|"
    r"access denied|cf-browser-verification|checking your browser|"
    r"you've been blocked|request blocked|unusual traffic)",
    re.IGNORECASE,
)


@dataclass
class CrawlOutcome:
    url: str
    markdown: str = ""
    ok: bool = False
    blocked: bool = False
    error: Optional[str] = None


def _browser_config() -> BrowserConfig:
    return BrowserConfig(
        headless=True,
        verbose=False,
        text_mode=True,          # skip images → faster, less memory
        light_mode=True,         # disable background features
        user_agent_mode="random",
        extra_args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"],
    )


def _run_config(page_timeout_s: int, respect_robots: bool) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,          # Supabase is our dedup; don't serve stale cache
        page_timeout=page_timeout_s * 1000,   # Crawl4AI expects milliseconds
        wait_until="domcontentloaded",
        word_count_threshold=5,
        excluded_tags=["script", "style", "nav", "footer", "header", "form", "aside", "noscript", "svg"],
        remove_overlay_elements=True,
        remove_consent_popups=True,
        exclude_external_images=True,
        exclude_all_images=True,
        exclude_social_media_links=True,
        check_robots_txt=respect_robots,
        verbose=False,
    )


def _markdown_text(result) -> str:
    """Crawl4AI returns either a str or a MarkdownGenerationResult."""
    md = getattr(result, "markdown", None)
    if md is None:
        return ""
    if isinstance(md, str):
        return md
    for attr in ("fit_markdown", "raw_markdown", "markdown_with_citations"):
        text = getattr(md, attr, None)
        if text:
            return text
    return str(md)


def _clean_markdown(text: str) -> str:
    """Remove link-heavy noise and collapse whitespace."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)            # images
    text = re.sub(r"\[([^\]]+)\]\((?:https?:)?[^)]*\)", r"\1", text)  # keep link text only
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:MAX_MARKDOWN_CHARS]


async def _crawl_one(
    crawler: AsyncWebCrawler, url: str, config: CrawlerRunConfig, hard_timeout_s: float
) -> CrawlOutcome:
    try:
        result = await asyncio.wait_for(crawler.arun(url=url, config=config), timeout=hard_timeout_s)
    except asyncio.TimeoutError:
        return CrawlOutcome(url=url, error=f"timed out after {hard_timeout_s:.0f}s")
    except Exception as exc:  # browser crash, DNS failure, malformed HTML, ...
        return CrawlOutcome(url=url, error=f"{type(exc).__name__}: {str(exc)[:200]}")

    if not getattr(result, "success", False):
        err = (getattr(result, "error_message", "") or "crawl failed").strip()
        low = err.lower()
        blocked = (
            "robots" in low or "anti-bot" in low or "blocked" in low or "captcha" in low
            or getattr(result, "status_code", None) in (401, 403, 429, 999)
        )
        return CrawlOutcome(url=url, blocked=blocked, error=err[:200])

    status = getattr(result, "status_code", None) or 200
    if status in (401, 403, 429, 999):
        return CrawlOutcome(url=url, blocked=True, error=f"HTTP {status}")
    if status >= 400:
        return CrawlOutcome(url=url, error=f"HTTP {status}")

    try:
        text = _clean_markdown(_markdown_text(result))
    except Exception as exc:
        return CrawlOutcome(url=url, error=f"markdown conversion failed: {exc}")

    head = text[:3000]
    if len(text) < MIN_USEFUL_CHARS or (_WALL_PATTERNS.search(head) and len(text) < 4000):
        return CrawlOutcome(url=url, markdown=text, blocked=True, error="login wall / bot check / empty page")

    return CrawlOutcome(url=url, markdown=text, ok=True)


async def crawl_in_batches(
    urls: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    page_timeout_s: int = DEFAULT_PAGE_TIMEOUT_S,
    respect_robots: bool = True,
    pause_between_batches_s: float = 1.0,
) -> AsyncIterator[List[CrawlOutcome]]:
    """
    Async generator: yields a list of CrawlOutcome per batch so the caller
    can extract + persist each batch before the next one starts.
    """
    if not urls:
        return
    config = _run_config(page_timeout_s, respect_robots)
    hard_timeout = page_timeout_s + 15  # browser start-up / teardown slack

    async with AsyncWebCrawler(config=_browser_config()) as crawler:
        for start in range(0, len(urls), batch_size):
            batch = urls[start : start + batch_size]
            outcomes = await asyncio.gather(
                *(_crawl_one(crawler, u, config, hard_timeout) for u in batch),
                return_exceptions=False,
            )
            yield list(outcomes)
            if start + batch_size < len(urls):
                await asyncio.sleep(pause_between_batches_s)


async def crawl_urls(
    urls: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    page_timeout_s: int = DEFAULT_PAGE_TIMEOUT_S,
    respect_robots: bool = True,
) -> Dict[str, str]:
    """Convenience wrapper: crawl everything, return {url: markdown} for successes."""
    out: Dict[str, str] = {}
    async for batch in crawl_in_batches(urls, batch_size, page_timeout_s, respect_robots):
        for o in batch:
            if o.ok:
                out[o.url] = o.markdown
    return out
