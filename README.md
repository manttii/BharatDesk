# BharatDesk (pilot build)

Voice-first Telegram bot for Indian B2B jewellery-wholesale field sales. LLMs extract;
deterministic Python decides. Money is integer paise, weight is integer milligrams.

**Status: Phase 4 of 4 (pilot-ready, pending real recordings).**
- Phase 0: scaffold, data model, onboarding + consent, health, CI.
- Phase 1: money/weight types, deterministic numeral parser, policy engine, eval harness.
- Phase 2: the voice pipeline. A voice note (or text) becomes a verified record: Groq speech-to-text
  with a script check, Gemini extraction, deterministic date/number/bounds checks, a Hindi
  read-back with Haan / Nahi / Sudharo buttons, one-question self-healing, policy checks,
  boss approvals (Approve / Reject / Counter, expiry, reminder), and a browser sandbox.
- Phase 3: handwritten-slip photos (vision model, totals recomputed in code), the client dossier
  query, SET_BENCHMARK (`/rate` or by voice, boss only, read back), Google Sheets sync (durable
  outbox, retry with backoff, never in the reply path), and a read-only boss dashboard behind a
  signed expiring link, with corrections.

- Phase 4: demo seed, media retention purge, worker data export and erasure, the pilot report
  (`scripts/pilot_report.py`, real data only), `ARCHITECTURE.md` and `SECURITY.md`.

## 5-minute demo

You need a Gemini key (typed notes) and a Groq key (voice). No Telegram needed.

1. **Install** (1 min)
   ```bash
   python -m venv .venv && .venv\Scriptsctivate      # Linux/macOS: source .venv/bin/activate
   pip install -r requirements-dev.txt
   cp .env.example .env                                 # Windows: copy .env.example .env
   ```
2. **Add keys** to `.env`: `GEMINI_API_KEY` + `GEMINI_MODEL`, and for voice `GROQ_API_KEY` + `WHISPER_MODEL`
   (you pick the model names). Prove them: `python -m scripts.smoke_providers`
3. **Start**: `uvicorn app.main:app` and open <http://127.0.0.1:8000/sandbox>
4. **Log a visit** (as *Rep*): record a voice note or type
   `Rajendra Jewellers gaya tha, dedh sau gram necklace, chaar lakh baarah hazaar ka order, paintalis hazaar advance, pachattar hazaar rate, kal delivery`.
   The stages show the transcript, the parsed numbers, the model's JSON and the validated record. You get
   `Rajendra Jewellers, 150 gram, ₹4,12,000, rate ₹75,000 (10 gram), advance ₹45,000, baaki ₹3,67,000, delivery 3 Nov. Sahi hai?`
   (the date is relative to today); tap **Haan** to save it.
5. **Trigger an approval**: say it again with `sattar hazaar rate` and tap Haan. ₹70,000 is below the demo
   floor (the ₹75,200 benchmark less 1% = ₹74,448), so the **Boss** panel shows Approve / Reject / Counter.
   Switch to *Boss*, tap **Approve**, and the rep is told. (The sandbox's demo rules ask for a rate on every
   order, so include one.)

With Telegram: put the bot token in `.env`, run
`python -m scripts.seed --tenant "Demo Jewellers" --boss "Asha" --workers "Ramesh,Suresh" --history`,
and open the printed links (boss first). Seeded tenants are marked demo and excluded from pilot reports.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                                 # Windows: copy .env.example .env
uvicorn app.main:app --reload
```

The app migrates its own database on startup (`AUTO_MIGRATE=true`) and runs **without** a
Telegram token (`/health` reports `"telegram": "disabled"`).

### Connect a Telegram bot (long polling, dev)
1. Create a bot with @BotFather, put the token in `.env` as `TELEGRAM_BOT_TOKEN`.
2. Create your tenant and boss link: `python -m scripts.bootstrap --tenant "Sharma Jewellers" --boss "Rakesh"`
3. Open the printed link in Telegram, tap Start, accept the consent notice.
4. As boss: `/addworker Ramesh` -> tap Haan -> forward the link to Ramesh. `/team` lists status.

### Voice notes (Groq + Gemini)
Put `GROQ_API_KEY`, `WHISPER_MODEL`, `GEMINI_API_KEY`, `GEMINI_MODEL` in `.env` (you choose the
model names; the app fails at startup with a clear error if the Gemini model is unavailable), then
prove them with one live call each:

```bash
python -m scripts.smoke_providers
```

### Sandbox (no Telegram needed)
Open <http://127.0.0.1:8000/sandbox>: record or upload a voice note, or type one, and watch every
stage (transcript, numerals, raw LLM JSON, validated payload, policy/commit result, replies).
It uses the real Groq/Gemini providers when configured and says so when not; it never fakes
output. Demo tenant only; the benchmark there (Rs 75,200 / 10 g) is demo data. Loopback-only and
disabled when `ENV=prod`.

### Google Sheets
1. Create a Google Cloud service account, download its JSON key, and set
   `GOOGLE_SERVICE_ACCOUNT_FILE` (or `GOOGLE_SERVICE_ACCOUNT_JSON`).
2. Share your Google Sheet with the account's `client_email` as **Editor**.
3. As boss, send the bot `/sheet <link or id>` and tap Haan. It tells you at once if the sheet
   cannot be reached. `python -m scripts.smoke_providers` (with `SMOKE_SHEET_ID`) also checks.
Only confirmed or approved records are sent; they appear within a few seconds as rows keyed by
record id, so a retry updates a row rather than duplicating it.

### Boss dashboard
Set `DASHBOARD_SECRET` (32+ random characters) and `PUBLIC_BASE_URL`, then send the bot
`/dashboard`: it replies with a link valid for an hour. No passwords. The page is read-only except
"Correct" on a confirmed record, which keeps the original, logs it and updates the sheet row.

### Operations
```bash
python -m scripts.bootstrap --tenant "Customer Co" --boss "Owner"     # a REAL tenant (no demo data)
python -m scripts.pilot_report [--since 2026-11-01] [--out report.md] # evidence, real data only
python -m scripts.admin retention --tenant "Customer Co" --audio-days 30 --image-days 30
python -m scripts.admin purge                                         # also runs daily in the app
python -m scripts.admin export-worker --tenant "Customer Co" --worker "Ramesh"
python -m scripts.admin erase-worker  --tenant "Customer Co" --worker "Ramesh" --yes
```
The pilot report prints `no data` until reps have actually used the bot. See `SECURITY.md` for retention
and erasure semantics and `ARCHITECTURE.md` for how the pieces fit.

### Custom policy rules
Beyond the four built-in rule types, the owner can define a rule as an expression that describes the
**violation**. It is parsed with `ast` and interpreted by a whitelist walker: no `eval`, no calls, attributes,
strings or floats; integers only; any error is treated as a violation (fails closed).

```python
PolicyRule(
    rule_type="custom_expression",
    params={
        "expression": "order_value_paise > 5000000 and payment_received_paise * 10 < order_value_paise",
        "reason_hi": "Advance order ka 10% se kam hai.",
        "reason_en": "Advance is under 10% of a large order.",
    },
)
```
Names: `order_value_paise payment_received_paise new_pending_paise list_value_paise quoted_rate_paise
benchmark_paise benchmark_age_seconds outstanding_paise weight_mg`. If a rule reads the order value or the
quoted rate, the rep is asked for it. Rules are validated when loaded; a malformed one routes orders to the boss.

### Daily report
The boss sends `/dailyreport 20` (local hour) to get a short Hindi summary once a day (visits, confirmed order
value, approvals waiting, per-rep counts, failed Sheet syncs), or `/dailyreport off`. Off by default; at most
once a day.

### Webhook mode (deployment)
Set `TELEGRAM_MODE=webhook`, `TELEGRAM_WEBHOOK_URL=https://<host>/telegram/webhook` and
`TELEGRAM_WEBHOOK_SECRET` (1-256 chars of `A-Za-z0-9_-`). The app registers the webhook with
that secret and rejects any request whose `X-Telegram-Bot-Api-Secret-Token` header differs.

## Checks

```bash
ruff check . && mypy && pytest -q
```

## How a voice note is handled
1. Gate: duration >= 1.5 s (configurable), <= 120 s, size, MIME, then magic bytes on the download.
2. Speech-to-text, `language="hi"`, with a vocabulary prompt of the tenant's client names. Urdu
   script triggers one retry with a Devanagari hint, then the note is flagged, never trusted.
3. LLM extraction (temperature 0, strict JSON schema). The transcript is fenced as untrusted data.
4. Code takes over: dates resolved ("kal", "agle somvar"), every number cross-checked against the
   numeral parser, sanity bounds applied, confidence flagged.
5. Read-back with buttons. Commit only on Haan, unless the tenant enabled auto-confirm **and**
   every check passed (it is off by default).
6. Commit runs the policy engine. A violation (or a missing/stale benchmark) creates an approval
   request for the boss; the record stays unconfirmed (never synced) until Approve.

## Bills, PDFs and stock (shops and other small businesses)

Send a PDF to the bot in Telegram (or a photo with a caption starting "bill"), or use the demo page at
<http://127.0.0.1:8000/sandbox/bills>. Anyone on the team can upload; only the boss can confirm,
accept or reject (`/bills` lists what is still pending):

1. **Each page is routed by what it contains.** A page with a real text layer is read as text; a
   scanned page is rendered to an image and read by a vision model (OCR). A PDF can mix both.
2. **The model reads, code checks.** Line = quantity x rate, subtotal, tax and total are all
   recomputed in integer arithmetic. A bill that disagrees with itself, has no date, party or
   bill number, repeats an earlier bill, or looks unreadable goes to **needs review**; nothing
   is defaulted or guessed. Quantities are stored in thousandths and money in paise.
3. **You confirm once.** Clean bills wait for one "Confirm clean bills" tap; flagged ones are
   accepted or rejected individually, and the flags you saw are kept in the audit log.
4. **Many rows per call.** Confirmed bills are written to your sheet in a handful of API calls:
   a `Bills` tab, a `Bill items` tab and a live `Stock` tab (purchases minus sales per item).

Ask the bot about the shop: `/stock [item]` (everyone) shows bought minus sold from confirmed bills,
and `/today` (boss only) shows today's sales, purchases and top-selling items. Short plain questions
such as "aaj kya bika?" or "rice ka stock kitna hai?" give the same answers; the matching is
deterministic and deliberately narrow, so a visit note that merely mentions stock is never mistaken
for a question.

Made a mistake? The boss runs `/reverse` (or `/reverse <bill no>`) and taps "Ulta karo", then confirms
with a second tap. The bill leaves stock and every report at once, and its sheet rows are marked
`reversed` (kept as the record, never deleted); who did it, when and why is in the audit log.
Re-uploading a corrected bill is not treated as a duplicate of the reversed one.

Limits you should know: one bill per page (a bill spanning pages becomes separate records);
stock groups by exact normalised item name and unit, so "Rice 25kg" and "Basmati rice" are
different items until you standardise names; uploaded files are not kept, only the extracted
data.
Tune with `MAX_PDF_PAGES`, `MAX_PDF_BYTES`, `BILL_CONCURRENCY`.

## Deploying the dashboard

The boss dashboard can be hosted on Vercel against a hosted Postgres database while the bot keeps
running on a normal machine: see [DEPLOY_VERCEL.md](DEPLOY_VERCEL.md).

## Known limitations
- **Not yet tested on real recordings.** All accuracy-sensitive code (numerals, script check, the read-back
  gate) is tested on synthetic data. Collect 50-100 real clips (`evals/README.md`) and run
  `python -m evals.run_pipeline` before trusting any number; expect bug fixes from that.
- Auto-confirm is off by default and should stay off until the eval harness shows a false auto-confirm rate of 0.
- Transcript retention is opt-in per tenant (`--transcript-days`); by default transcripts are kept (see `SECURITY.md`).
- Worker export/erasure are operator CLI commands, not bot commands.
- Slip dates are read as past or present dates; a slip dated in the future would resolve to last
  year. Slip lines are read per piece (quantity x rate), so weight-priced lines may show a
  harmless "check" flag.
- One slip per photo. Photos and images sent as files are both handled.
- The dashboard loads Tailwind and htmx from CDNs (htmx is version-pinned, without SRI hashes).
  Serve it over HTTPS; the session cookie is `Secure` when `PUBLIC_BASE_URL` is https.
- A dashboard correction is authoritative: the policy engine is not re-run on it.
- The dashboard shows rows of the whole tenant (no per-rep filter) and the last 50 records.
- Photos/slips, client dossier, SET_BENCHMARK, Sheets sync and the dashboard arrive in Phase 3.
  Until SET_BENCHMARK exists, benchmarks and policy rules are inserted directly in the database.
- The live Groq and Gemini calls are tested with mocks only; run `scripts/smoke_providers.py`.
  Whether Groq accepts Telegram's raw OGG/Opus is unverified (see `STT_FORCE_CONVERT`).
- Voice notes are processed by one worker, one at a time. A note stranded by a crash is given up on after
  `STRANDED_AFTER_MINUTES` (default 10), marked failed, and the rep is told to resend.
- Client matching works within one script: a Devanagari name will not match a Roman one.
- Outstanding balance = sum of unpaid portions of confirmed orders (no payment application yet).
- A counter can change a rate or a discount only; other violations are Approve/Reject.
- Custom rules use a deliberately small language (integers, comparisons, and/or/not); see "Custom rules".
- Only the PostgreSQL DDL and migration SQL are checked (no live Postgres run).
- 1 tola = 11.664 g is an assumption (`app/units.py`); confirm with the customer.
- The numeral parser reads a stray "do" (Hindi verb) as 2; it only cross-checks LLM output.
- Hindi strings are Roman-script ("Hinglish"); they live in `app/pack/field_sales_jewellery.yaml`.
- One worker process: rate limiting and the queue are in-memory.
- Audit-log immutability is enforced by SQLite triggers; on PostgreSQL use a role without
  UPDATE/DELETE on `audit_log` (the ORM guard applies on every backend).
- The consent notice wording must be reviewed by counsel before a real pilot.
