"""
Streamlit UI for the autonomous Candidate Sourcing Agent.

Run:  streamlit run app.py

The pipeline runs on a background thread (agent_pipeline.BackgroundRun);
this script polls it, renders progress live, and never blocks on crawling.
Records are saved to Supabase batch-by-batch, so stopping loses nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Shared modules (schema, search, Supabase, Gemini) live in the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import supabase_db as db
from agent_pipeline import BackgroundRun, PipelineSettings
from gemini_client import DEFAULT_MODEL, Gemini
from schema import CandidateRecord

load_dotenv()

st.set_page_config(page_title="Candidate Sourcing Agent", page_icon="🔎", layout="wide")

LIST_COLS = ["skills", "target_countries"]
DISPLAY_COLS = ["name", "current_role", "skills", "current_location", "target_countries",
                "platform", "evidence_snippet", "source_url", "discovered_at"]
POLL_SECONDS = 0.7

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("run", None)            # BackgroundRun | None
ss.setdefault("events", [])           # list[dict]
ss.setdefault("run_records", [])      # list[dict] — candidates found in this run
ss.setdefault("wave_stats", {})       # wave_name -> stats dict
ss.setdefault("plan", None)
ss.setdefault("db_records", None)     # cache of fetch_all_candidates()


# ---------------------------------------------------------------------------
# Sidebar — credentials & connection tests
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔐 Credentials")
    gemini_key = st.text_input("GEMINI_API_KEY", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    supabase_url = st.text_input(
        "DATABASE_URL (Neon DB)",
        value=os.getenv("DATABASE_URL", db.DEFAULT_NEON_DATABASE_URL),
        type="password",
        help="Neon PostgreSQL connection string (postgresql://...).",
    )
    supabase_key = "neon"
    st.caption("Values fall back to your `.env` file or default Neon DB connection.")

    with st.expander("Model settings"):
        model = st.text_input("Gemini model", value=DEFAULT_MODEL)
        gemini_mode = st.selectbox(
            "Key type", ["auto", "gemini", "vertex"], index=0,
            help="auto: detect. gemini = AI Studio key (AIza…). vertex = Vertex AI express-mode key (AQ.…).",
        )

    if st.button("Test connections", width="stretch"):
        with st.spinner("Checking Neon DB…"):
            try:
                db.init_supabase(supabase_url, supabase_key)
                st.success(db.check_connection())
            except Exception as exc:
                st.error(f"Neon DB: {exc}")
        with st.spinner("Checking Gemini…"):
            try:
                st.success(asyncio.run(Gemini(gemini_key, model=model, mode=gemini_mode).ping()))
            except Exception as exc:
                st.error(f"Gemini: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in DISPLAY_COLS:
        if col not in df.columns:
            df[col] = None
    for col in LIST_COLS:
        df[col] = df[col].apply(lambda v: v if isinstance(v, list) else ([] if v is None else [str(v)]))
    extra = [c for c in df.columns if c not in DISPLAY_COLS]
    return df[DISPLAY_COLS + extra]


def apply_filters(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Country / skill / role / platform filters."""
    if df.empty:
        return df
    all_countries = sorted({c for lst in df["target_countries"] for c in lst})
    all_skills = sorted({s for lst in df["skills"] for s in lst}, key=str.lower)
    all_platforms = sorted({p for p in df["platform"].dropna()})

    c1, c2, c3, c4 = st.columns(4)
    countries = c1.multiselect("Target country", all_countries, key=f"{key}_countries")
    skills = c2.multiselect("Skill", all_skills, key=f"{key}_skills")
    role = c3.text_input("Role contains", key=f"{key}_role")
    platforms = c4.multiselect("Platform", all_platforms, key=f"{key}_platforms")

    mask = pd.Series(True, index=df.index)
    if countries:
        wanted = {c.lower() for c in countries}
        mask &= df["target_countries"].apply(lambda lst: any(c.lower() in wanted for c in lst))
    if skills:
        wanted = {s.lower() for s in skills}
        mask &= df["skills"].apply(lambda lst: any(s.lower() in wanted for s in lst))
    if role:
        mask &= df["current_role"].fillna("").str.contains(role, case=False, regex=False)
    if platforms:
        mask &= df["platform"].isin(platforms)
    return df[mask]


def export_buttons(df: pd.DataFrame, key: str) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_df = df.copy()
    for col in LIST_COLS:
        csv_df[col] = csv_df[col].apply(lambda lst: "; ".join(lst) if isinstance(lst, list) else lst)
    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button("⬇️ CSV", csv_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"candidates_{stamp}.csv", mime="text/csv", key=f"{key}_csv")
    c2.download_button("⬇️ JSON", json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str),
                       file_name=f"candidates_{stamp}.json", mime="application/json", key=f"{key}_json")


def editable_table(df: pd.DataFrame, key: str) -> None:
    """Editable grid (lists edited as '; '-separated text) + save back to Supabase."""
    edit_df = df.copy()
    for col in LIST_COLS:
        edit_df[col] = edit_df[col].apply(lambda lst: "; ".join(lst))
    edited = st.data_editor(
        edit_df,
        key=f"{key}_editor",
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        disabled=["source_url", "platform", "discovered_at", "id"],
        column_config={
            "source_url": st.column_config.LinkColumn("source_url", width="medium"),
            "evidence_snippet": st.column_config.TextColumn("evidence_snippet", width="large"),
            "skills": st.column_config.TextColumn("skills", help="Separate with ;"),
            "target_countries": st.column_config.TextColumn("target_countries", help="Separate with ;"),
        },
    )
    if st.button("💾 Save edits to Supabase", key=f"{key}_save"):
        try:
            db.init_supabase(supabase_url, supabase_key)
            recs = []
            for row in edited.to_dict(orient="records"):
                payload = {f: row.get(f) for f in CandidateRecord.model_fields}
                for col in LIST_COLS:
                    payload[col] = [x.strip() for x in str(payload.get(col) or "").split(";") if x.strip()]
                recs.append(CandidateRecord(**payload))
            n = db.save_candidates(recs)
            ss.db_records = None
            st.success(f"Saved {n} rows.")
        except Exception as exc:
            st.error(f"Save failed: {exc}")


def ingest(events: list) -> None:
    """Fold new pipeline events into session state."""
    for ev in events:
        d = ev.as_dict()
        ss.events.append(d)
        if ev.kind == "plan":
            ss.plan = ev.data.get("plan")
        if ev.kind in ("dedup", "wave_end") or (ev.kind == "saved" and "stats" in ev.data):
            stats = ev.data.get("stats", ev.data)
            if ev.wave:
                ss.wave_stats[ev.wave] = {**ss.wave_stats.get(ev.wave, {}), **stats}
        if ev.kind == "wave_start" and ev.wave:
            ss.wave_stats.setdefault(ev.wave, {"wave": ev.wave})
        if ev.kind == "saved":
            ss.run_records.extend(ev.data.get("records", []))


ICONS = {"plan": "🧭", "wave_start": "🌊", "query": "🔍", "dedup": "🧹", "crawl": "🕷️", "saved": "💾",
         "wave_end": "✅", "info": "ℹ️", "warning": "⚠️", "error": "❌", "done": "🏁"}


def render_progress(stats_box, log_box, live_box) -> None:
    if ss.wave_stats:
        cols = ["wave", "found", "duplicates", "fresh", "crawled", "blocked", "failed", "records", "dropped"]
        rows = [{c: s.get(c, 0 if c != "wave" else w) for c in cols} for w, s in ss.wave_stats.items()]
        stats_box.dataframe(
            pd.DataFrame(rows).rename(columns={
                "found": "Found URLs", "duplicates": "Duplicates filtered", "fresh": "Fresh",
                "crawled": "Scraped", "blocked": "Blocked/walled", "failed": "Failed",
                "records": "Extracted records", "dropped": "Rejected (ungrounded)"}),
            hide_index=True, width="stretch",
        )
    lines = []
    for e in ss.events[-250:]:
        t = datetime.fromtimestamp(e["ts"]).strftime("%H:%M:%S")
        wave = f"[{e['wave']}] " if e.get("wave") else ""
        lines.append(f"{t} {ICONS.get(e['kind'], '•')} {wave}{e['message']}")
    log_box.code("\n".join(lines) or "Waiting…", language=None)
    if ss.run_records:
        live_box.dataframe(to_frame(ss.run_records)[DISPLAY_COLS[:-1]], hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Main — search controls
# ---------------------------------------------------------------------------
st.title("🔎 Autonomous Candidate Sourcing Agent")
st.caption("Gemini plans multi-wave search dorks → DuckDuckGo → Supabase dedup → Crawl4AI → "
           "Gemini extraction → Supabase.")

running = ss.run is not None and ss.run.is_alive()

intent = st.text_area(
    "Discovery intent",
    value="Find Indian candidates who are interested for abroad opportunities in CNC",
    height=80, disabled=running,
)

with st.expander("Wave depth & crawler settings", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    num_waves = c1.slider("Number of waves", 1, 6, 3, disabled=running)
    queries_per_wave = c2.slider("Queries per wave", 1, 10, 5, disabled=running)
    max_results = c3.slider("Max results per query", 5, 30, 10, disabled=running)
    max_urls = c4.slider("Max fresh URLs per wave", 5, 100, 30, disabled=running)
    c5, c6, c7, c8 = st.columns(4)
    batch_size = c5.slider("Crawl batch size", 2, 8, 5, disabled=running)
    page_timeout = c6.slider("Page timeout (s)", 10, 60, 30, disabled=running)
    region = c7.selectbox("Search region", ["in-en", "wt-wt", "us-en", "uk-en", "de-de"], disabled=running)
    backend = c8.selectbox("Search backend", ["auto", "duckduckgo"], disabled=running,
                           help="auto rotates engines inside the ddgs library (fewer rate limits); "
                                "duckduckgo forces DuckDuckGo only.")
    c9, c10 = st.columns(2)
    respect_robots = c9.checkbox("Respect robots.txt", value=True, disabled=running)
    snippet_fallback = c10.checkbox("Use search snippet when a page is walled/blocked", value=True,
                                    disabled=running)

b1, b2, _ = st.columns([1, 1, 5])
start_clicked = b1.button("🚀 Start", type="primary", disabled=running, width="stretch")
stop_clicked = b2.button("⏹ Stop", disabled=not running, width="stretch")

if start_clicked:
    missing = [n for n, v in [("GEMINI_API_KEY", gemini_key), ("SUPABASE_URL", supabase_url),
                              ("SUPABASE_KEY", supabase_key)] if not v]
    if missing:
        st.error(f"Missing: {', '.join(missing)}")
    elif not intent.strip():
        st.error("Enter a discovery intent.")
    else:
        settings = PipelineSettings(
            gemini_api_key=gemini_key, supabase_url=supabase_url, supabase_key=supabase_key,
            model=model, gemini_mode=gemini_mode, num_waves=num_waves, queries_per_wave=queries_per_wave,
            max_results_per_query=max_results, max_urls_per_wave=max_urls, batch_size=batch_size,
            page_timeout_s=page_timeout, respect_robots=respect_robots, snippet_fallback=snippet_fallback,
            region=region, search_backend=backend,
        )
        ss.events, ss.run_records, ss.wave_stats, ss.plan, ss.db_records = [], [], {}, None, None
        ss.run = BackgroundRun(intent, settings).start()
        st.rerun()

if stop_clicked and ss.run is not None:
    ss.run.stop()
    st.toast("Stopping after the current batch — everything saved so far is kept.")

# ---------------------------------------------------------------------------
# Progress (live)
# ---------------------------------------------------------------------------
if ss.run is not None or ss.events:
    st.subheader("Progress")
    if ss.plan:
        with st.expander("🧭 Search plan", expanded=False):
            for i, w in enumerate(ss.plan.get("waves", []), 1):
                st.markdown(f"**Wave {i}: {w['wave_name']}** · `{w['platform']}`")
                st.code("\n".join(w["queries"]), language=None)
    stats_box = st.empty()
    log_box = st.empty()
    st.markdown("**Candidates found in this run (live)**")
    live_box = st.empty()

    if ss.run is not None and ss.run.is_alive():
        with st.spinner("Agent running…"):
            while ss.run.is_alive():
                ingest(ss.run.drain())
                render_progress(stats_box, log_box, live_box)
                time.sleep(POLL_SECONDS)
        ingest(ss.run.drain())
        st.rerun()  # redraw once more with editing / export enabled
    else:
        if ss.run is not None:
            ingest(ss.run.drain())
        render_progress(stats_box, log_box, live_box)

# ---------------------------------------------------------------------------
# Results — edit, filter, export
# ---------------------------------------------------------------------------
if not running:
    st.subheader("Results")
    tab_run, tab_db = st.tabs(["This run", "All saved candidates (Supabase)"])

    with tab_run:
        if ss.run_records:
            df = apply_filters(to_frame(ss.run_records), "run")
            st.caption(f"{len(df)} of {len(ss.run_records)} records")
            editable_table(df, "run")
            export_buttons(df, "run")
        else:
            st.info("No records from this session yet.")

    with tab_db:
        if st.button("🔄 Load / refresh from Supabase"):
            try:
                db.init_supabase(supabase_url, supabase_key)
                ss.db_records = db.fetch_all_candidates()
            except Exception as exc:
                st.error(str(exc))
        if ss.db_records is not None:
            try:
                st.caption(f"{len(ss.db_records)} candidates stored · "
                           f"{db.count_scraped_urls()} URLs in the dedup ledger")
            except Exception:
                pass
            if ss.db_records:
                df = apply_filters(to_frame(ss.db_records), "db")
                st.caption(f"Showing {len(df)} after filters")
                editable_table(df, "db")
                export_buttons(df, "db")
