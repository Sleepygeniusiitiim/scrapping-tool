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

## Lead databases, Google search and unblockers

In the **“🔑 Lead databases & search API keys”** section of the home page (or as Vercel env vars):

| Purpose | Services (env var) |
|---|---|
| Google search | Serper.dev (`SERPER_API_KEY`), SerpApi (`SERPAPI_KEY`), Google Programmable Search (`GOOGLE_CSE_KEY` + `GOOGLE_CSE_CX`), Brave (`BRAVE_API_KEY`), Scrape.do (`SCRAPEDO_TOKEN`) |
| Sites that refuse the crawler (403 / bot checks) | Scrape.do, ScraperAPI (`SCRAPERAPI_KEY`), ZenRows (`ZENROWS_API_KEY`), ScrapingBee (`SCRAPINGBEE_API_KEY`), Jina Reader (free; `JINA_API_KEY` optional, `DISABLE_JINA=1` to turn off) |
| Phone / email lookup | ContactOut (`CONTACTOUT_API_KEY`), Lusha (`LUSHA_API_KEY`), RocketReach (`ROCKETREACH_API_KEY`), Apollo (`APOLLO_API_KEY`) |

- Without any Google key the app can only use DuckDuckGo, which returns few results from Vercel.
- A page that refuses the direct fetch is retried through up to two unblockers, in the order above.
- **📇 Find missing contacts** (All saved candidates tab) looks up every candidate without a phone or
  email in the lead databases, using their LinkedIn profile link (saved from post comments and profile
  pages). Found contacts are saved with `contact_source = enriched:<service>`. Each lookup uses that
  service's credits; coverage is best for people with a LinkedIn profile. Apollo returns emails only
  (its phone reveal needs a webhook).
- Only API keys are supported, not account passwords: automated logins break these services' terms.

## 🧠 Intent Miner (added alongside the original pipeline)

**Search modes** (Search panel, next to the sources): one command box and one 🚀 Start button run
- **Combined (default)** — Intent Miner understanding, sources and intent scoring, *and* the classic
  page reader on the same pages. Both write to the same place: the run log / per-source table,
  "This run" and "All saved candidates", and the intent-lead cards. A person found by both is saved once
  (matched on email, phone, or name on the same page); the classic reader only fills in missing details.
- **Intent Miner only** — the same run without the classic page reader.
- **Classic waves only** — the original wave pipeline, unchanged.

**People or organizations.** The understanding step decides whether the command asks for *people* who
show intent (job seekers, buyers, students…) or *organizations* (businesses, institutes, training
centres, agencies and their owners — B2B leads). For organizations the searches target business
directories and "contact us" pages; schema.org Organization / LocalBusiness data is read; each listing
entry or business page becomes a lead named after the organization, scored on business type (35%),
location (20%), a public business phone / email (20%), semantic match (15%) and AI confidence (10%).
"Only interested" and the time window do not apply to organizations, and the classic page reader (which
looks for individual candidates) is skipped for them — use Combined or Intent Miner only for B2B searches.

Combined and Intent-Miner runs use the existing settings: sources, "Only leads active in the last…",
"Only leads who say they're interested", Page reading (rules = no AI classification, hybrid / AI = AI on
the shortlist), lead-database lookups, robots.txt, search backend, region, results per query, fresh URLs
per source, batch size, Next round (new searches that avoid earlier ones), and the shared
already-read-pages ledger.

Type what you want ("Find people in India looking for CNC operator jobs in Germany within 12 months").

```
command → understanding (LLM: professions + synonyms, origin, destination, intent type, high-intent and
negative terms, timeline, time window, subreddits, 12–20 searches per source)
→ discovery: search engines (DuckDuckGo + Google APIs), Reddit API, Quora (via search), forums &
  websites, search-indexed LinkedIn / Facebook pages, RSS / Atom feeds
→ ingestion: posts, comments and answers with author and date (Reddit API; schema.org JSON-LD threads;
  page blocks; search snippet when a page is not readable)
→ normalization: clean text, language, canonical URL, content hash, near-duplicate removal
→ intent engine: Stage 1 keyword score → Stage 2 semantic similarity (Mistral / Gemini embeddings,
  lexical fallback) → Stage 3 LLM classification of the shortlist only, contacts redacted
→ lead engine: score, tier, evidence quotes, "why this lead", entity resolution
→ PostgreSQL (im_* tables) → cards, CSV / Excel / JSON, Salesforce
```

- **Scores.** intent = 25% keyword + 20% semantic + 20% explicit need + 15% timeline + 10% location
  + 10% LLM confidence; lead score = 80% intent + 10% freshness (7d 100 · 30d 85 · 90d 65 · 180d 40 ·
  older 20) + 10% source quality (forum 85, Reddit 80, LinkedIn 75, Quora 70, snippet 25). HIGH ≥ 80,
  MEDIUM ≥ 60.
- **Evidence.** Every lead keeps verbatim quotes (checked against the text) and a "why" checklist.
- **Identity.** Same platform + author → one lead; same phone / email → merged; same name on another
  platform → listed as a *possible* match needing verification, never merged automatically.
- **Incremental.** Pages whose content hash was already classified are skipped (no AI cost).
  Failed / blocked pages are kept (with attempts) and can be retried from "♻️ Failed pages".
- **Lifecycle.** QUALIFIED → ENRICHED → EXPORTED → CONTACTED → RESPONDED → CONVERTED (per-lead dropdown).
- **Qualified leads are also copied into “All saved candidates”**, so lead-database lookups and
  outreach work on them unchanged.
- **Reddit API:** create an app at reddit.com/prefs/apps (type "script") and add its client id and secret
  (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`). Without it Reddit threads are found through search
  engines and read from their snippets. Reddit's developer terms forbid reselling / brokering Reddit data.
- **Salesforce:** `SALESFORCE_INSTANCE_URL` + `SALESFORCE_ACCESS_TOKEN`; "Push 80+" creates Leads
  (LeadSource "Intent Miner", evidence in Description) for leads not pushed before.
- **Not built, by design:** logins, CAPTCHA bypass, private groups or messages, fake accounts, email
  guessing, sending every page to an LLM.

## Importing from job-portal employer accounts

Naukri (Resdex / RMS), foundit, WorkIndia, Indeed, Apna and Naukrigulf let employers download applicants
or database search results as Excel / CSV. Upload that file under **All saved candidates → 📥 Import**
(`portal_import.py`; .xlsx, .xls, .csv or HTML-table "Excel" files up to 4 MB). Columns are matched by
name (Mobile No. / Phone → phone, Key Skills → skills, Last Active / Applied On → date, …). Tick "These
people applied to my job" to mark them interested. The tool does not log in to portals with your
password — automated logins break their terms and get recruiter accounts blocked.

## Company contacts from their own website (organization leads)

How tools like Hunter / Apollo get *business* contacts, done on the public web only
(`company_contacts.py`, runs automatically for organization leads, up to 5 per batch):
find the organization's own site (the lead's page, or a search result whose domain / title matches the
name — directories are skipped), read home + contact / about / team / admissions / careers pages, and
collect published emails (text + mailto), phones (text + tel), WhatsApp (wa.me), LinkedIn / Facebook /
Instagram pages and named people with roles (schema.org founder / employee, "Principal: …",
"…, Managing Director"). Each email's domain is checked for MX records ("domain accepts mail" — not proof
that the mailbox exists). Not done: guessing emails from name patterns, SMTP probing, logged-in scraping.

What cannot be copied from those tools: their personal mobile numbers mostly come from contributor
networks (users' address books uploaded via their extensions / apps) and purchased data.

## Re-checking old pages for new comments

**🔁 Re-check old pages for new comments** (🧠 section) re-reads pages from earlier runs of either
engine — oldest first, read more than 1 day / 3 days / 1 week / 1 month ago, optionally only pages that
gave leads or have several posts / comments — with the current intent and settings. Pages whose content
hash has not changed are skipped without any AI cost; changed pages are scored again, so new comments
become new leads while existing people are merged, not duplicated.

## Getting the most contacts out of every page

Ideas taken from open-source tools (theHarvester, Photon, Reacher / check-if-email-exists, email-verifier):

* **Hidden contacts are revealed where they are:** Cloudflare-protected emails (`data-cfemail`, `/cdn-cgi/l/email-
  protection`) are decoded; `mailto:` / `tel:` / WhatsApp (`wa.me`, `api.whatsapp.com`) links behind "Email us" /
  "Call now" buttons get their address written next to the link text; header / footer contacts (dropped from the
  page text) are kept in a separate section marked as the website owner's — used for business leads only, never
  given to commenters.
* **Business websites:** contact / about pages are tried directly when the menu is built by JavaScript; a site
  that shows nothing to a plain request is read rendered (Jina reader); addresses at the company's domain that
  search engines indexed anywhere ("@domain", the theHarvester approach) add evidence for the email format; a
  business with a website but no published email gets info@ / contact@ / enquiry@ / admissions@ … tested with the
  zero-send SMTP check (kept only when the mail server confirms it and the domain is not catch-all).
* **Link-in-bio pages** (Linktree, bio.link, beacons, taplink …) and WhatsApp links on commenters' profiles are
  followed for their email / WhatsApp / LinkedIn.
* **Results:** a WhatsApp chat link for every mobile number, a contact filter (has phone / email / both / none),
  searches run 3 at a time, and a **run summary** after every run: contacts found (phone / email / both /
  verified) and concrete fixes for the next run (missing keys, robots.txt skips, blocked pages, time window,
  dropped partial lookups, rules-only reading).

## AI source planner, Google Maps listings and business directories

With **🤖 Let the AI choose the best sources** ticked (default), the understanding step also ranks every source
for the command (`intent_miner/planner.py`, shown under "Best sources for this command") and runs the top ones:

* businesses / institutes / owners ("truck driving schools in North India") → **Google Maps listings** (name,
  phone, website, address of every place), **business directories** (JustDial, IndiaMART, Sulekha, OLX), the
  businesses' own websites, LinkedIn for owners / directors;
* individuals showing intent (job seekers) → comment sections (Facebook, LinkedIn, YouTube, blogs), forums,
  Reddit, Quora.

Regions are expanded into cities ("North India" → ~40 cities in Delhi, Punjab, Haryana, UP, Uttarakhand, HP,
J&K, Rajasthan) and Maps / directories are searched city by city.

**Google Maps** needs one of: a Serper key, a SerpApi key (both already used for Google search) or
`GOOGLE_PLACES_API_KEY`. Without one, the same searches go to web search instead. **Directories** are read
through your unblocker keys; JustDial / IndiaMART usually hide phone numbers behind a login, so the business
names found there are looked up on Maps for their phone.

For every business lead: Maps lookup (when it came without a phone) → its own website (phones, emails,
WhatsApp, people named with their role) → **decision makers** from search-indexed LinkedIn profiles
("Name - Owner - Business") → government-list match.

## Analysing specific posts (paste links)

🧠 panel → **🔗 Analyse specific posts / pages**: paste Instagram / Facebook / YouTube / Reddit / blog / forum links and
they are read directly (no searching), with the time window ignored. Every commenter showing interest becomes a lead;
the post's own caption (usually the recruiter's) does not.

Instagram posts are read through Instagram's public **embed page** (`/p/<code>/embed/captioned/`), which is served
without login and carries the caption and, for many posts, the first comments. It is not the full list — Instagram
only gives all comments through the Meta API, and only for posts on your own account (auto-reply below). Instagram's
robots.txt does not allow crawlers, so untick "Respect robots.txt" to use it. If the embed page is refused, the normal
reader with your unblocker keys is tried.

## Government lists (India) — matching leads to official records

🧠 panel → **🏛️ Government lists**: import public lists published by government bodies (CSV / Excel, or straight
from **data.gov.in** with the resource id and a free API key: `DATA_GOV_IN_KEY`). Useful lists: MCA company / LLP
master data, state transport lists of motor-driving schools, NCVT / Skill India ITI and training-partner lists, Indian
Nursing Council institutions, Udyam lists, professional registers. Columns are recognised by their names.

Every lead of a new run (and saved leads, with **Match saved leads now**) is matched to these records
(`gov_registry.py`):

1. **blocking** — only records sharing a distinctive name word, or the same phone / email / website;
2. **features** — fuzzy name similarity (spelling variants, initials, Pvt / Ltd / Institute noise; "Sharma Driving
   School" ≠ "Verma Driving School"), place (city / district / pincode / state), type (driving school, nursing …),
   identifiers (registration no. / CIN in the lead's text, same phone / email / domain);
3. **AI check** — borderline candidates go to the AI with both records side by side; it picks one or none;
4. **decision** — organisations match at 80+; a person needs a near-exact name, a second signal and the AI's
   confirmation (or an identifier). Weaker candidates are shown as "possible" and are never applied.

A match adds the record's registered phone / email / website / address to the lead (`contact_source:
government_list:<list>`) and a "✓ Government record" line to "Why this lead?".

Electoral rolls, Aadhaar, e-Shram / UAN and similar ID lists are refused at import: they are not public, the law
restricts their use, and leaked copies are illegal to use for outreach.

## YouTube comments and blog comments

- **YouTube** (source "YouTube comments"): the official YouTube Data API v3 (`YOUTUBE_API_KEY`, or the Google
  Programmable Search key if that API is enabled on its project; free 10,000 units/day — a video search is
  100 units, 100 comments 1 unit). Recruitment / jobs-abroad videos are found, and every comment and reply
  becomes a unit with the viewer's name, channel link and date — "Interested sir, HMV 5 years, 98…".
- **Blogs** (source "Blogs"): comment sections of blog posts on any website — recruitment-agency blogs,
  job-news sites, Blogspot / WordPress / Medium. Comments rendered as HTML (WordPress, wpDiscuz, Blogger
  and similar themes) are read as one comment per author with date and profile link (`fetcher.html_comments`);
  the site's own replies ("send your CV to …") are recruiter messages and never leads.

## Waterfall enrichment and email verification

When a page gives an interested lead no complete contact, `pipeline._enrich_records` cascades through these steps,
each only for the leads the previous ones left without a contact:

1. **own profile page** — real name, LinkedIn link, published phone / email;
2. **search result of that exact profile** — bio contacts, employer from the headline;
3. **employer's website** — employer from the bio, the headline ("Staff Nurse at Fortis Hospital") or the post
   ("working at …"); its site gives the **mail domain**, the **email format** and, if the company lists the
   person, their real address;
4. **lead databases** — by LinkedIn profile, or by **name + employer / domain** (ContactOut, Lusha, RocketReach,
   Apollo, **Hunter** email-finder);
5. **work-email guess + SMTP check** (last resort: employed, nothing found anywhere) — the likely addresses
   (the company's own format first, then first.last, first, firstlast, flast, …) are all tested in one SMTP session;
   a confirmed mailbox is reported as verified, on a catch-all domain only the company's own format is offered
   and marked inconclusive, and if every address bounces no guess is given;
6. **verification of every email** → the `email_status` column.

`email_verify.py` checks without sending anything: syntax → disposable-inbox list → DNS (MX, implicit MX via
the A record, null MX) → SMTP handshake (`EHLO`, `MAIL FROM`, `RCPT TO:<address>`, `QUIT` — no `DATA`) →
catch-all probe (a random address in the same session: if it is accepted too, "250" proves nothing).
Statuses: valid / invalid / catch-all (inconclusive) / disposable / domain takes no mail / unverified.

**Port 25 on Vercel:** outbound SMTP (port 25) is blocked on Vercel's functions, like most cloud platforms, so
there the mailbox step falls back to, in order:
* `SMTP_VERIFY_URL` + `SMTP_VERIFY_TOKEN` — the same file running on any small server with port 25 open:
  `SMTP_VERIFY_TOKEN=… SMTP_VERIFY_FROM=verify@yourdomain SMTP_HELO_DOMAIN=host.yourdomain python email_verify.py serve 8025`
* Hunter / ZeroBounce / NeverBounce verification (`HUNTER_API_KEY`, `ZEROBOUNCE_API_KEY`, `NEVERBOUNCE_API_KEY`)
* DNS only (domain checked, mailbox "unverified").

Use a sender on a domain you own (`SMTP_VERIFY_FROM`) and probe sparingly; many RCPT checks from one IP get it
rate-limited or block-listed. 🔑 keys → **Verify emails / find a work email** tests both by hand.

## Profile bios and work-email guesses

For every interested lead still missing a contact (after their comment and their own profile page):
- **Profile bio via search** (`social_lookup.py`): search engines index public profiles even when the
  site shows servers a login wall. Only the result for the person's *exact* profile URL
  (instagram.com/<handle>/, x.com/<handle>, facebook.com/<handle>, linkedin.com/in/<slug>) is used —
  pages that only mention the handle are ignored — for their name, the phone / email in their bio
  (`contact_source = profile_bio`) and, for LinkedIn, their current employer and job title.
- **Work-email guess** (`email_patterns.py`) — the LAST option, only for a lead who is working (employer
  known) and for whom no phone number and no email was found anywhere (comment, own profile, bio, lead
  databases): the employer's website
  is crawled, its email format is learned from the addresses it publishes (a named person next to their
  address, e.g. Harpreet Kaur ↔ harpreet.kaur@…, or the shape of published addresses), and applied to the
  lead's name. Stored separately in `email_guess` as "x@company.com (guessed, <confidence>, format …)",
  never in `email`. No format evidence → no guess. The same guesses are shown for people named on
  organization websites. MX is checked, but a guess is still unverified — send sparingly.

## Auto-reply on your own Instagram / Facebook posts (Outreach tab)

Someone comments "interested" on a post from your official Instagram Business account or Facebook
Page → the tool replies under the comment, sends them one private message asking for name, WhatsApp,
email and experience, and when they answer, saves their details as a candidate
(`contact_source = shared_in_reply`) and thanks them. "STOP" marks them not interested. Only your own
posts are possible: Meta's API cannot reply on other people's posts, and bots commenting elsewhere are spam.

Setup (once):
1. Make the Instagram account Professional (Business / Creator) and link it to your Facebook Page.
2. developers.facebook.com → Create App (type Business) → add **Webhooks**, **Messenger** and
   **Instagram** (API with Facebook Login). As app admin, your own accounts work without App Review.
3. Get a long-lived **Page access token** with `pages_show_list, pages_read_engagement,
   pages_manage_engagement, pages_messaging, instagram_basic, instagram_manage_comments,
   instagram_manage_messages, business_management` (Graph API Explorer → exchange for a long-lived token).
4. Vercel env vars: `META_VERIFY_TOKEN` (any secret you choose), `META_APP_SECRET`, `META_PAGE_TOKEN`,
   `META_PAGE_ID`, `META_IG_USER_ID` (optional `META_GRAPH_VERSION`, default v23.0). Redeploy.
5. Webhooks → callback URL `https://<your-app>.vercel.app/api/meta/webhook`, verify token =
   `META_VERIFY_TOKEN`. Subscribe **Page**: `feed`, `messages`; **Instagram**: `comments`, `messages`.
   Then subscribe the Page to the app: `POST /{page-id}/subscribed_apps?subscribed_fields=feed,messages`.
6. Instagram app → Settings → Messages → Connected tools → allow access to messages.
7. Outreach tab → 🤖 Auto-reply → edit the texts, tick **Auto-reply is ON**, Save.

Meta's rules the tool follows: one private reply per comment (within 7 days of the comment), further
messages only inside the 24-hour window after the person writes back, no promotional follow-ups.

## How contacts are found for interested commenters

1. **Their own words** — a phone / email the person wrote in their comment or post.
2. **Their own profile** — for every interested lead still missing a contact, the tool opens the
   commenter's public profile page (Instagram / X / TikTok handle, forum or LinkedIn author link from the
   page) and reads their real name, a LinkedIn link in the bio, and any phone / email published there
   (`profile_visit.py`; robots.txt setting respected, no logins — walled profiles are skipped).
3. **Lead databases** — only leads with a LinkedIn profile are looked up in ContactOut / Lusha /
   RocketReach / Apollo, so the lookup is about the right person. No Google name-guessing in runs
   (it often matched the wrong person). Leads without a LinkedIn profile are skipped, costing no credits.

The run log says per batch how many profiles were opened, what they gave, how many leads were looked up
and filled, and why the rest were not. The **Test the lead databases** box runs one lookup per service
and shows its own reply (a 403 usually means the plan has no API access).

## Following the intent exactly (any role, any country)

The plan call also returns `role_keywords` (titles / synonyms in English and the local language, e.g.
"Lehrer", "Ausbilder" for Germany) and `locations` (only when the intent restricts where people are or
want to work). Pages are then matched on those keywords, and when locations are set, a lead is kept
only if their own words, profile or the post they replied to names one of those places.

Crawler settings:
- **Only leads who say they're interested / keen** (default on) — the person's own words must show
  interest ("interested", "looking for a job", "open to work", "ready to join", "my CV", German
  "interessiert", …). Every lead is stored with `shows_interest`.
- **Look up missing contacts in lead databases** (default off) — during the run, interested leads
  (within "Only leads active in the last…") missing a phone or email are looked up in ContactOut /
  Lusha / RocketReach / Apollo, up to 10 per batch.
- **Lead databases: keep only if both phone & email** (default on) — a lookup result is saved only when
  the lead then has both. The lookup itself may still use the service's credit.
- The **📇 Find missing contacts for interested leads** button applies the same three rules to saved leads.

## Lead dates (how recent a lead is)

Every lead gets an `activity_date`: the date of the candidate's own comment when the page carries it,
else the post / page publish date, else the date encoded in a LinkedIn post URL (`…-activity-<id>-…`),
else the date the search engine showed. "Only leads active in the last N months" (crawler settings,
default 6) restricts searches to recent pages and skips dated leads older than that; leads with no date
found are kept. Older saved LinkedIn leads get their date from the post URL when listed. Without a
Google key, a free Google attempt is made per query; it is often refused from cloud IPs.

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
