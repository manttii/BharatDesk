# Deploying the dashboard on Vercel

Only the **boss dashboard** runs on Vercel. The Telegram bot, the model calls and the background
jobs (approval sweep, Google Sheets sync, retention purge) need a long-lived process, so they keep
running on your computer or a small server. Both talk to the **same hosted Postgres database**.

```
 Telegram ── bot (your PC / a server) ──┐
                                        ├── hosted Postgres ──┐
 Boss's browser ── Vercel dashboard ────┘                     │ same data
```

## What you need
1. A hosted Postgres database (any provider that gives you a `postgres://...` connection URL).
2. A Vercel account with this GitHub repository imported.
3. The same `DASHBOARD_SECRET` on the bot and on Vercel (32+ random characters).

## Steps

**1. Create the database** and copy its connection URL. Provider URLs such as
`postgres://user:pass@host/db?sslmode=require` work as they are; the app converts them.

**2. Point the bot at it.** In the bot's local `.env` (never in Git):
```
DATABASE_URL=postgres://user:pass@host/db?sslmode=require
DASHBOARD_SECRET=<the same long random value you will give Vercel>
PUBLIC_BASE_URL=https://<your-project>.vercel.app
```
Start the bot. It creates the tables on startup (`AUTO_MIGRATE`), or run `alembic upgrade head`.
Your existing SQLite data is not copied over; the Postgres database starts empty.

**3. Import the repo in Vercel** (Add New → Project → pick this repository). Keep the defaults;
`vercel.json` selects the FastAPI preset and the root `index.py` is the entry point (the bot's own
`app/main.py` is never served).

**4. Add environment variables in Vercel** (Project → Settings → Environment Variables), never in
the repository:

| Name | Value |
|---|---|
| `DATABASE_URL` | the same Postgres URL |
| `DASHBOARD_SECRET` | the same secret as the bot |
| `ENV` | `prod` |

`PUBLIC_BASE_URL` is optional on Vercel (the production hostname is used automatically), but it
**must be set on the bot** so the `/dashboard` link it sends points at Vercel.

**5. Redeploy** (environment variables only apply to new deployments), then in Telegram send `/dashboard` to the bot (as the boss) and open the link.
`https://<your-project>.vercel.app/health` should say `"status": "ok"`.

## How the build works
Vercel installs from the `dependencies` list in `pyproject.toml` (not `requirements.txt`). That list
is deliberately small: only what the dashboard needs, so the function stays light and cold starts
are quick. `requirements.txt` stays the full pinned set for the bot, the tests and CI, and a test
keeps the two in sync. The bot's heavy libraries (aiogram, google-genai, PDF/OCR) are not installed
on Vercel.

## What works and what does not
- Works: the live summary, visit feed, approvals view, sheet/sync status, and corrections, all
  through the same signed, expiring sign-in link as before.
- Not available on Vercel: stored audio and photos (there is no persistent disk), SQLite, and the
  `/sandbox` pages. Bills and stock are in the database and the Google Sheet, not in this dashboard.
- The first request after idle time is slower (a cold start).

## Safety
- `.vercelignore` and `.gitignore` keep `.env`, databases, uploads and key files out of every
  upload. Check before pushing with `python -m scripts.check_secrets`.
- Rotate any key that was ever pasted into a chat or a ticket.

## Not yet verified
The Postgres path is covered by offline SQL and unit tests only, not against a live database, and
this deployment has not been run on Vercel yet. Treat the first deploy as a test: check `/health`,
open a dashboard link, and confirm a visit logged in Telegram shows up.
