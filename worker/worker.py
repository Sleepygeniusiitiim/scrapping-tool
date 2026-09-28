"""
Persistent background worker: claims jobs queued by the web app (jobs.py) and runs them end to end with no time
limit — full searches, the headless browser for protected pages, the local model, SMTP mailbox checks and the
search-index embeddings. Run any number of them against the same Neon database:

    python worker/worker.py               (or the Docker image in worker/Dockerfile)

Environment (same names as on Vercel): DATABASE_URL, AI / search / unblocker / lead-database keys, and optionally
    BROWSER_FETCH=1        headless Chromium for pages that refuse plain requests (worker/Dockerfile installs it)
    PROXY_URL=...          residential / rotating proxy for the browser, http://user:pass@host:port
    LOCAL_LLM_URL=...      local model server for bulk parsing (docker-compose starts Ollama)
    WORKER_SCALE=3         how much more per batch than a web request (profile visits, lookups, guesses …)
    SMTP_VERIFY_FROM=...   sender for the zero-send mailbox checks (port 25 must be open on this host)
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("WORKER_SCALE", "3")
os.environ.setdefault("RUNTIME", "worker")

import ai_router  # noqa: E402
import integrations  # noqa: E402
import jobs  # noqa: E402

POLL_S = float(os.getenv("WORKER_POLL_SECONDS", "5"))
EMBED_EVERY_S = 300
_stop = {"flag": False}


def _info() -> dict:
    return {"browser": os.getenv("BROWSER_FETCH") == "1", "proxy": bool(os.getenv("PROXY_URL")),
            "local_llm": ai_router.local_available(), "scale": os.getenv("WORKER_SCALE"),
            "smtp_from": bool(os.getenv("SMTP_VERIFY_FROM"))}


async def run_job(job: dict, wid: str) -> None:
    from intent_miner import runner
    job_id = str(job["id"])
    opts = job.get("options") or {}
    keys = integrations.resolve_keys(opts.get("keys") or {})
    llm_keys = opts.get("llm_keys") or {}
    keys.update({k: v for k, v in (("mistral", llm_keys.get("mistral") or os.getenv("MISTRAL_API_KEY")),
                                   ("gemini", llm_keys.get("gemini") or os.getenv("GEMINI_API_KEY"))) if v})
    ai = ai_router.hosted_chain(llm_keys, opts.get("llm_provider", "")) or ai_router.bulk(None)
    buf: list = []
    state = {"cancel": False, "last": 0.0}

    def emit(line: str) -> None:
        buf.append(time.strftime("%H:%M:%S ") + line)
        print(f"[{job_id[:8]}] {line}", flush=True)

    async def pump():
        while True:
            await asyncio.sleep(3)
            lines, buf[:] = buf[:], []
            await asyncio.to_thread(jobs.log, job_id, lines)
            state["cancel"] = await asyncio.to_thread(jobs.heartbeat, wid, _info(), job_id)

    pumper = asyncio.create_task(pump())
    try:
        if ai is None:
            raise RuntimeError("No AI key on the worker (set GEMINI_API_KEY / GROQ_API_KEY / … or LOCAL_LLM_URL)")
        if job["kind"] == "run":
            totals = await runner.run(ai, job.get("command") or "", opts, keys, emit,
                                      lambda: state["cancel"] or _stop["flag"],
                                      lambda t: jobs.update(job_id, stats=t))
            jobs.update(job_id, stats=totals, run_id=totals.get("run_id"))
        elif job["kind"] == "index":
            import vectors
            emit("Embedding the search index…")
            while True:
                r = await vectors.embed_pending(keys, None, 256)
                emit(f"embedded {r.get('embedded', 0)}, {r.get('remaining', 0)} left {r.get('note', '')}")
                if not r.get("embedded") or not r.get("remaining") or state["cancel"]:
                    break
        pumper.cancel()
        jobs.log(job_id, buf)
        jobs.update(job_id, status="cancelled" if state["cancel"] else "done")
    except Exception as exc:
        pumper.cancel()
        emit(f"FAILED: {type(exc).__name__}: {exc}")
        jobs.log(job_id, buf)
        jobs.update(job_id, status="failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        traceback.print_exc()


async def main() -> None:
    wid = jobs.worker_id()
    print(f"worker {wid} started {_info()}", flush=True)
    last_embed = 0.0
    while not _stop["flag"]:
        try:
            jobs.heartbeat(wid, _info())
            jobs.requeue_stale()
            job = jobs.claim(wid)
        except Exception as exc:
            print(f"queue error: {exc}", flush=True)
            await asyncio.sleep(POLL_S * 3)
            continue
        if job:
            print(f"claimed {job['id']} ({job['kind']}): {str(job.get('command'))[:80]}", flush=True)
            await run_job(job, wid)
            continue
        if time.time() - last_embed > EMBED_EVERY_S:       # idle: keep the semantic index embedded
            last_embed = time.time()
            try:
                import vectors
                r = await vectors.embed_pending(integrations.resolve_keys({}), None, 256)
                if r.get("embedded"):
                    print(f"search index: embedded {r['embedded']}, {r.get('remaining')} left", flush=True)
            except Exception as exc:
                print(f"index error: {exc}", flush=True)
        await asyncio.sleep(POLL_S)


def _signal(*_):
    _stop["flag"] = True
    print("stopping after the current step…", flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT, _signal)
    asyncio.run(main())
