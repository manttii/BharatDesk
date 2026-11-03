# Security

This is a pilot build holding commercial data and the voices of employees. It is **not**
security-reviewed by a third party. Legal and privacy points below need confirmation from counsel
before a real deployment.

## Threat model

Assets: the customer's order and pricing data, workers' personal data (voice, Telegram id), the
bot token and provider keys, and the integrity of approvals (money decisions).

| Threat | Who | Mitigation | Enforced / tested in |
|---|---|---|---|
| Forged webhook calls | internet | secret-token header, constant-time compare, 403 otherwise; webhook mode refuses to start without a valid secret | `test_webhook_and_health.py` |
| Replay / duplicate delivery | Telegram, network | `update_id` and callback-id claims; approvals by compare-and-set | `test_idempotency.py`, `test_approvals_flow.py` |
| Stranger messages the bot | internet | unregistered chats get one polite reply; nothing is stored | `test_onboarding.py` |
| Stolen or reused invite link | anyone | single use, 24 h expiry, stored hashed, atomic redemption | `test_onboarding.py` |
| Processing before consent | system | consent gate before any message is processed; `/stop` withdraws | `test_consent.py` |
| A rep approves their own exception | insider | approvals require the boss role, re-checked on every tap; wrong tenant refused | `test_approvals_flow.py` |
| Cross-tenant data access | tenant / bug | single scoped repository; static scan for unscoped queries; media and corrections are tenant-scoped | `test_tenant_isolation.py`, `test_dashboard.py` |
| **Prompt injection** via speech, slip text or typed text | rep, customer | transcript/image treated as data and fenced; model output validated strictly; **no model output can change a rate, role or rule** because code decides from stored rules | `test_voice_pipeline.py`, `test_slips.py` |
| Model misreads a number | model | every number cross-checked against a deterministic parser; bounds; confidence; read-back | `test_numerals.py`, `test_extract_assess_clients.py` |
| Floating-point money errors | developer | integer paise/mg, strict schemas, AST scan | `test_no_float_money.py` |
| Spreadsheet formula injection | rep | RAW cell input plus a leading apostrophe on formula-like text | `test_sheets_sync.py` |
| Dashboard link theft | anyone with the link | HMAC-signed, 1 h expiry, swapped for an HttpOnly SameSite=Strict cookie and redirected away from the URL; `Referrer-Policy: no-referrer`; boss re-checked on every request | `test_dashboard.py` |
| CSRF on the one write action | web attacker | per-record CSRF token bound to the session; SameSite=Strict | `test_dashboard.py` |
| XSS from stored text | rep | Jinja2 autoescape; `textContent` in the sandbox | `test_dashboard.py`, `test_sandbox.py` |
| PII in logs | operator | chat ids masked; transcripts, names and tokens never logged | `test_hygiene.py` |
| Secrets in the repo | developer | env-only config; scan for key-shaped strings; `.env.example` ships empty | `test_hygiene.py` |
| Code execution | developer | no `eval`/`exec`/`compile` (AST scan) | `test_hygiene.py` |
| Tampering with the audit trail | insider | append-only in the ORM and via SQLite triggers | `test_audit_log.py` |
| Abuse / flooding | rep, stranger | per-chat rate limit; size, duration and MIME limits; magic-byte checks | `test_security.py`, `test_voice_pipeline.py` |

## Security checklist

Done and tested:

- [x] Secrets come from the environment only; webhook secret validated; dashboard secret >= 32 chars
- [x] Identity by Telegram chat id bound through single-use invite links; consent stored with version
- [x] Boss-only commands checked against the verified chat id and read back before applying
- [x] Tenant-scoped repository and isolation tests; cross-tenant dashboard/media access refused
- [x] Rate limit per chat id; media size/duration/MIME/magic-byte limits
- [x] Strict validation of all model output; numbers cross-checked; policy engine fails closed
- [x] Append-only audit log; every sensitive action audited
- [x] Per-tenant retention of audio and photos with an automatic purge job (below); opt-in transcript retention
- [x] Notes stranded by a crash are recovered automatically (marked failed, rep told to resend)
- [x] Custom policy rules run in a whitelisted, integer-only evaluator (no `eval`), failing closed
- [x] Worker data export and erasure on request (below)
- [x] Structured JSON logs with masked chat ids and no transcripts
- [x] Dependencies pinned
- [x] Dashboard: signed expiring link, cookie session, CSRF, escaping, security headers

Open (not done):

- [ ] Legal review of the consent notice, retention periods and cross-border processing (audio goes
      to Groq and text/images go to Google). Needs counsel.
- [ ] Third-party penetration test and dependency vulnerability scanning in CI
- [ ] Subresource-integrity hashes for the dashboard's CDN scripts (htmx is version-pinned; Tailwind
      uses its play CDN). Self-host both for a hardened deployment.
- [ ] Encryption at rest for the database and media directory (use disk/volume encryption)
- [ ] Backups and a tested restore procedure
- [ ] On PostgreSQL: a database role without UPDATE/DELETE on `audit_log` (the ORM guard applies everywhere,
      the SQLite triggers only on SQLite)
- [ ] A live PostgreSQL run (only the DDL and migration SQL are verified)

## Uploaded documents (bills, PDFs, photos)

- File type is decided from the bytes, never the filename; size and page count are capped;
  encrypted or damaged PDFs are refused with a reason. Text and images from a page are data,
  never instructions, and page text is fenced with angle brackets neutralised.
- The model's numbers are never trusted on their own: amounts are recomputed in code, and an
  inconsistent or incomplete bill is held for a person. Cell text written to Sheets is neutralised
  against formula injection and sent RAW.
- Uploaded files are not stored; only the extracted fields are (tenant-scoped, audited).
- **Student or other personal data about minors** (school use) is children's data under the DPDP
  Act 2023, with consent and processing duties that must be confirmed with counsel before any
  real school data is processed. Do not run real student records through this build.

## Retention

Each tenant has `audio_retention_days` and `image_retention_days` (default **90**, `None` keeps
forever). The purge job runs every `RETENTION_SWEEP_HOURS` (default 24) and on demand:

```bash
python -m scripts.admin retention --tenant "Sharma Jewellers" --audio-days 30 --image-days 30
python -m scripts.admin purge
```

Only media files are deleted by default; the message, transcript and business record remain, and each purge
is audited. A tenant may also opt in to transcript retention (`--transcript-days N`), which blanks transcripts
and the model's raw output (the validated business payload stays). Files are content-addressed, so a file is removed only when every message that references it
is past retention. The dashboard shows "Recording deleted (retention policy)".

## Erasure and export

A worker may ask to see or delete their data. These are operator commands (run on the host holding the
database), deliberately not bot commands:

```bash
python -m scripts.admin export-worker --tenant "Sharma Jewellers" --worker "Ramesh" --out ramesh.json
python -m scripts.admin erase-worker  --tenant "Sharma Jewellers" --worker "Ramesh" --yes
```

**Export** returns the profile, consent, every message with its transcript and hashes, records,
corrections and audit events as JSON.

**Erase** deletes the worker's stored media, Telegram file ids, hashes, transcripts and raw model output,
open conversation state, name, chat id and consent, and re-syncs their Google Sheet rows so the rep column
reads "Deleted worker". It **keeps** the company's business records (what was sold to whom) and the
append-only audit log, which holds ids rather than names. The tenant owner cannot be erased this way.
Erasure is idempotent. Data already copied into the customer's own spreadsheet exports or Telegram
history is outside this system's control.

## Reporting a problem

This is a pilot: report suspected vulnerabilities to the project owner directly, not in a public issue.
