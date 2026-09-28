"""
Headless browser fetching (Playwright + Chromium) for background workers.

Used when a plain request is refused or returns an empty JavaScript shell: bot checks (Cloudflare "checking your
browser"), pages that render their content — and their comment threads — with JavaScript, "load more comments"
buttons. Through PROXY_URL (a residential / rotating proxy, http://user:pass@host:port) the request comes from a
home connection instead of a data centre.

What it does NOT do: log in. Content behind a login (LinkedIn profiles, Instagram comments, Facebook groups) stays
out of reach — automating logins breaks those sites' terms and gets accounts banned. Captchas are not solved.

Enabled with BROWSER_FETCH=1 (worker/Dockerfile installs Chromium). Not available on Vercel functions.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from typing import Optional
from urllib.parse import urlparse

_MORE = re.compile(r"^(?:load|show|view|see)\s+(?:more|all|previous|older)(?:\s+(?:comments?|replies|answers?))?|"
                   r"^(?:more|all)\s+(?:comments?|replies)|^\d+\s+more\s+(?:comments?|replies)", re.IGNORECASE)
_state = {"pw": None, "browser": None, "lock": None}
MAX_CLICKS = 6
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/131.0.0.0 Safari/537.36")


def enabled() -> bool:
    if os.getenv("BROWSER_FETCH", "") != "1":
        return False
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False


def _proxy() -> Optional[dict]:
    raw = os.getenv("PROXY_URL", "").strip()
    if not raw:
        return None
    p = urlparse(raw)
    out = {"server": f"{p.scheme}://{p.hostname}:{p.port}"}
    if p.username:
        out.update(username=p.username, password=p.password or "")
    return out


async def _browser():
    if _state["lock"] is None:
        _state["lock"] = asyncio.Lock()
    async with _state["lock"]:
        if _state["browser"] is None or not _state["browser"].is_connected():
            from playwright.async_api import async_playwright
            _state["pw"] = await async_playwright().start()
            exe = os.getenv("CHROMIUM_PATH") or None
            _state["browser"] = await _state["pw"].chromium.launch(
                headless=True, executable_path=exe,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"])
    return _state["browser"]


async def fetch_html(url: str, timeout_s: int = 35) -> tuple[int, str, str]:
    """(status, final_url, html). Scrolls and opens "load more comments" a few times so threads are complete."""
    browser = await _browser()
    ctx = await browser.new_context(user_agent=UA, locale="en-IN", viewport={"width": 1366, "height": 900},
                                    proxy=_proxy(), java_script_enabled=True)
    await ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
    page = await ctx.new_page()
    try:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
        status = resp.status if resp else 0
        try:
            await page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        # bot-check interstitials usually clear themselves in a few seconds
        for _ in range(3):
            title = (await page.title()).lower()
            if not re.search(r"just a moment|checking your browser|attention required|verify you are human", title):
                break
            await page.wait_for_timeout(4000)
        for _ in range(MAX_CLICKS):
            await page.mouse.wheel(0, 2500)
            await page.wait_for_timeout(600 + random.randint(0, 400))
            clicked = False
            for el in await page.query_selector_all("button, a, [role=button]"):
                try:
                    txt = (await el.inner_text()).strip()
                except Exception:
                    continue
                if txt and len(txt) < 40 and _MORE.search(txt) and await el.is_visible():
                    try:
                        await el.click(timeout=1500)
                        clicked = True
                        await page.wait_for_timeout(900)
                        break
                    except Exception:
                        continue
            if not clicked:
                break
        return status, page.url, await page.content()
    finally:
        await ctx.close()


async def close() -> None:
    if _state["browser"] is not None:
        await _state["browser"].close()
        await _state["pw"].stop()
        _state["browser"] = _state["pw"] = None
