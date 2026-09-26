# Autonomous Candidate Sourcing Agent

Turns an open intent ("Find Indian candidates who are interested for abroad opportunities
in CNC") into a multi-wave search plan, fetches only never-before-seen pages, and stores
structured candidate records in **Neon DB (PostgreSQL)**.

This repo deploys to **Vercel** (a static page + a Python serverless API). The original
Streamlit version with headless-Chromium crawling is in [`legacy-streamlit/`](legacy-streamlit/)
for running on your own machine.

```
browser (public/index.html) drives the run step by step:
  POST /api/plan     Gemini 2.5 Flash: multi-wave dork plan
  POST /api/search   one DuckDuckGo query → canonicalised URLs     (×N, 1–2.5 s jitter)
  POST /api/dedup    drop URLs already in Neon DB `scraped_urls`
  POST /api/process  ≈5 URLs: record → HTTP fetch → Gemini extraction
                     → grounding check → Neon DB `candidates` upsert
  GET  /api/candidates, POST /api/candidates   view / edit stored records
```

Each request finishes well inside Vercel's 60 s function limit. Waves still run strictly one
after another, and Stop takes effect after the current batch.

## 1. Neon DB (PostgreSQL)

The app is pre-configured to connect to Neon DB using `psycopg2-binary` and automatically initializes the `scraped_urls` and `candidates` tables (`schema.sql`) on startup.

## 2. Deploy on Vercel

1. vercel.com → **Add New… → Project** → import this GitHub repo (`Sleepygeniusiitiim/scrapping-tool`). Framework preset: **Other**
   (no build command; `public/` is served as the site, `api/index.py` as the function).
2. **Settings → Environment Variables**, add:

   | Name | Value |
   |---|---|
   | `DATABASE_URL` | `postgresql://neondb_owner:npg_jBms9Rc4oHgD@ep-fancy-dust-b5rdd2ee-pooler.c-7.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require` |
   | `APP_PASSWORD` | `CSA-Neon-Vercel-2026!` (also configured as default fallback) |
   | `GEMINI_API_KEY` | AI Studio (`AIza…`) or Vertex express (`AQ.…`) key (can also be entered directly in the UI) |

3. Redeploy so the variables take effect. Open the site, enter the password (`CSA-Neon-Vercel-2026!`), and click **Test connections**.

## Contact details, comments and Scrape.do

- **email / phone columns.** Filled only when the candidate wrote the detail themselves — in their
  own post, profile or comment. Every value must appear verbatim on the page, and on pages with a
  comment thread it must be in that person's own lines (a recruiter's number is never attached to a
  commenter). Invented values are discarded.
- **Comment sections.** LinkedIn posts, Quora answers and many forums embed the post and its comments
  (with authors) as schema.org data; the app reads that first. Logged-out LinkedIn shows about the
  first 10 comments of a post.
- **LinkedIn / Facebook / Reddit / Quora.** Their robots.txt disallows crawlers. While "Respect
  robots.txt" is ticked, the search plan starts with openly crawlable sources (job portals, forums,
  regional pages) and only the search snippets of these sites are read. Untick it to read their
  public posts — check that this fits your use and their terms. These hosts are fetched one page at a
  time with a pause, because parallel requests get a login wall.
- **Scrape.do (optional).** Add `SCRAPEDO_TOKEN` in Vercel → Environment Variables. Pages that come
  back blocked are fetched again through Scrape.do (residential proxies for LinkedIn/Facebook), and
  in "auto" mode every query also runs on Google via Scrape.do's SERP API (merged with DuckDuckGo).
  Without the token only DuckDuckGo is used, and the run log says so.
  Choose "google (Scrape.do)" as the search backend to use Google only. Both use Scrape.do credits.

## Page reading without AI tokens

Fetching pages never uses AI — it is plain HTTP (`fetcher.py`). Reading them is set by
"Page reading" under "Wave depth & crawler settings":

- **rules — no AI tokens (default).** `rule_extractor.py` finds emails (including "name at gmail dot com"),
  phone / WhatsApp numbers (Indian mobiles and Gulf/Europe numbers with country code), names (comment
  authors, profile titles, "my name is …", "posted by …"), role and skills (from the search plan's own
  keywords plus common machines/software), current location (Indian cities/states) and target countries.
  A contact is kept only when the text around it reads like the person talking about themselves — a
  recruiter's "send your CV to …" is skipped.
- **hybrid.** Rules first; the AI reads a page only when rules find nobody but the page looks like it
  has candidates.
- **AI reads every page.** Best at unusual pages; uses the most tokens.

In rules mode the only AI call is the search plan (once per round).

## AI providers and automatic fallback

Supported: **OpenRouter**, **Gemini**, **Groq**, **Cerebras**, **Mistral**, **DeepSeek** and **Kimi (Moonshot)**.
Pick a provider on the home page and paste its key; repeat for as many providers as you like (each key
is remembered in the browser). Or set the keys in Vercel: `OPENROUTER_API_KEY`, `GEMINI_API_KEY`,
`GROQ_API_KEY`, `CEREBRAS_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `MOONSHOT_API_KEY`
(optional model overrides: `<PROVIDER>_MODEL`, e.g. `KIMI_MODEL`).

The chosen provider is used first. When it runs out of credits, hits a daily/rate limit or rejects its
key, the run switches to the next provider that has a key and logs it — free tiers first:
Gemini → Groq → Cerebras → Mistral → OpenRouter → DeepSeek → Kimi. Only when every provider is exhausted
does the run stop; unread pages are retried next run. "Test connections" checks every provider in the chain.

OpenRouter is pay-as-you-go (one balance for many models); its own fallback only switches models
inside your OpenRouter balance. Gemini, Groq, Cerebras and Mistral have free tiers with daily limits;
DeepSeek and Kimi are paid but cheap.

## Assisted outreach (Outreach tab)

1. Fill in your name, agency, the role and location, then **Load candidates** (default filter: no contact yet).
2. Tick candidates → **Draft messages** (AI writes a short, honest message citing the post where they
   showed interest, asking them to share phone/email if interested, with an opt-out line). Edit freely.
3. **Copy & open** copies the message and opens their post/profile — you paste it as a reply or DM.
   The app never sends messages itself: LinkedIn and Facebook forbid automated messaging.
4. When they answer, paste the reply and **Save reply**. The AI reads it; any phone/email that is
   actually in the reply is saved with `contact_source = shared_in_reply` and the date — a record
   that the candidate shared it with you. "Not interested" replies are marked so.

## Local development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt uvicorn python-dotenv
copy .env.example .env         # fill in GEMINI_API_KEY if desired
uvicorn api.index:app --reload --env-file .env
```

Open http://127.0.0.1:8000 — the page and the API are served together locally.

## Files

| File | Purpose |
|---|---|
| `schema.sql` | PostgreSQL tables (`scraped_urls`, `candidates`) + indexes |
| `schema.py` | `CandidateRecord`, `SearchWave`, `ComprehensiveSearchPlan` |
| `supabase_db.py` | Neon PostgreSQL dedup ledger + candidate upserts |
| `search_module.py` | DDGS search with jitter/back-off, URL canonicalisation, platform tagging |
| `gemini_client.py` | Gemini structured output, retries, AI Studio vs Vertex-express key handling |
| `fetcher.py` | HTTP fetch, robots.txt, HTML → text, login-wall detection |
| `pipeline.py` | plan / search / dedup / process-batch steps |
| `api/index.py` | FastAPI app (Vercel function), password & Gemini header check |
| `public/index.html` | UI: run controls, live progress, filters, editing, CSV / JSON export |
| `vercel.json` | function duration + `/api/*` routing |
