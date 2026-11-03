# Real-recording collection plan

**Purpose:** get enough real, labelled voice notes to decide, with evidence, whether BharatDesk is
safe to run in front of a customer, and whether auto-confirm may ever be turned on.

Everything below marked *proposed* is a starting point for you and the customer to agree, not a fact.
The commands refer to tools that already exist (`scripts/label_clip.py`, `evals/run_pipeline.py`,
`evals/run_evals.py`).

## 1. The decisions this data must support

| Decision | Evidence needed |
|---|---|
| Can reps' notes be read back correctly most of the time? | numeric exact-match and confirmation-trigger rate on the **test** set |
| May auto-confirm ever be enabled for a tenant? | **false auto-confirm rate = 0** with a large enough sample (section 3) |
| Is Groq happy with Telegram's OGG/Opus? | the first live run (`scripts.smoke_providers`, then a real voice note) |
| Which numbers does the model mishear? | failure analysis by number type (section 7) |
| Are the thresholds and bounds sensible? | the distribution of confidences and of out-of-bounds flags |

## 2. How many clips, and why

You cannot prove "zero false auto-confirms" with a small sample. If you observe 0 failures in `n`
independent trials, the 95% upper bound on the true failure rate is roughly `3 / n` (the "rule of three").

| Auto-confirm-eligible clips with 0 failures | You can claim, at 95% confidence, a rate below |
|---|---|
| 50 | 6% |
| 100 | 3% |
| 300 | 1% |
| 1000 | 0.3% |

So: **100 clips is enough to find bugs; about 300 is the minimum to say anything about auto-confirm.**
Only clips the system *would* auto-confirm count toward that figure, so collect more than 300 (plan for
~500, since many clips will be asked about and therefore do not count).

Split the clips **by speaker**, not at random, so the test set measures generalisation to people the
system was tuned on **not** hearing:

- **Dev set (about half):** use freely to find and fix bugs and to tune bounds and thresholds.
- **Test set (about half, different speakers):** freeze it. Run it **only** when you want a verdict.
  If you tune against it, it stops being a test.

With only 3-5 reps, hold out at least one or two entire speakers for the test set.

## 3. Who, where, what (coverage matrix)

Aim for every cell to have at least a handful of clips; do not collect only easy ones.

| Dimension | Cover |
|---|---|
| Speakers | every rep who will use it, plus the owner; different ages and accents |
| Place | the real counter, a noisy market, a moving two-wheeler, a quiet office, a phone call in the background |
| Speed and style | slow and careful, fast and clipped, trailing off, mid-sentence correction ("nahi nahi, paintalis") |
| Length | 2-5 s, 10-30 s, a 60 s ramble |
| Language | Hindi, Hinglish with English product words, a little pure English |
| Content | simple visit; large amount (lakh); fractional words (dedh, dhai, sava, saade, paune); weights in gram, kg and tola; advance and pending; relative dates (kal, parso, agle somvar, a named date); rate quotes; owner setting today's rate; "status batao" queries |
| Slips | clear, smudged, dark, tilted, glare, two columns, totals that do not add up |

### Hard cases to elicit on purpose

Ask reps to read or speak scripts containing the pairs most likely to be confused:

- chalis / paintalis, sattar / saintis, paintis / paintalis, unchas / unsath, tees / teen
- amounts that sound alike: 45 hazaar vs 54 hazaar, 1.5 lakh vs 15 lakh, 25 vs 52
- a bare number with no unit ("pachas") vs "pachas gram" vs "pachas hazaar"
- weights in tola: the tola value (assumed 11.664 g) must be confirmed with the customer before this step

Real, unscripted notes matter most. Scripted clips are for coverage; aim for **at least 60% unscripted**.

## 4. Consent and data handling (do this before recording anything)

- Get each rep's agreement, in writing or on tape, that recordings will be used to test the system, who sees
  them, how long they are kept, and that they can ask for deletion (`scripts.admin erase-worker`).
- Voice is personal data. Keep audio **out of git** (`evals/datasets/` is git-ignored), on an encrypted disk, with
  access limited to the people labelling.
- Client names and amounts are commercially sensitive: if clips leave the customer's machines, strip or mask
  them first, and confirm the customer agrees.
- Real clips will be sent to the speech and extraction providers during evaluation. Confirm the customer and
  counsel accept that, exactly as for production.
- Record only work-related speech. Do not record personal calls.

## 5. Recording protocol

1. **Use the real channel.** Reps send notes to a *test* Telegram chat with the bot; export the OGG files. That
   keeps the codec identical to production.
2. **One visit per note.** Say it the way you normally would; do not "speak for the machine".
3. **Note the context** in the label `notes` field: speaker code (e.g. `R1`, not a name), place, noise, scripted
   or real, and later which split it belongs to.
4. **Collect over several days**, not one afternoon: voices and conditions drift.
5. Collect the dev set first, run it, fix what you find, then collect the test set. Do not look at test results
   while collecting.

## 6. Labelling

Label **from the audio**, never from the model's output (otherwise the labels inherit its mistakes and the
measured accuracy is inflated).

1. For each clip: `python -m scripts.label_clip --dataset evals/datasets/dev --audio clip.ogg`
   Type exactly what was **said** as the transcript, then the fields. Money is integer paise and weight is
   integer milligrams; the script echoes `₹` and `gram` back so a slipped zero is visible.
2. Use the field names `evals/run_pipeline.py` writes: `intent`, `client`, `order.weight_mg`,
   `order.value_paise`, `order.rate_paise_per_10g`, `payment.received_paise`, `payment.pending_paise`,
   `delivery_date`, `next_followup_date`, `benchmark_paise`.
3. **Two people label the same 10-20% of clips independently.** Disagreements show where the *task* is ambiguous
   (e.g. is "dedh sau" a weight or a rate?), which is a spec problem, not a model problem. Resolve them and write
   the rule down.
4. Label what was said, including the rep's own mistakes ("sattar hazaar" when they meant 75): the system cannot
   know, and the read-back is what catches it. Mark such clips in `notes`.

Tooling gap to be aware of: the manifest has no `split` field yet. For now keep **two directories**
(`evals/datasets/dev`, `evals/datasets/test`) and never copy clips between them.

## 7. Running and reading the results

```bash
python -m scripts.smoke_providers                                   # keys and model names work
python -m evals.run_pipeline --dataset evals/datasets/dev  --out dev_preds.jsonl
python -m evals.run_evals    --dataset evals/datasets/dev  --predictions dev_preds.jsonl --out dev_report.md
```

Read the report in this order:

1. **Failure cases first.** Every wrong numeric field is listed. Classify each one: STT misheard / LLM
   misextracted / numeral parser gap / bounds or threshold wrong / label error. Fix by category, cheapest first.
2. **False auto-confirm count.** Anything above 0 is a bug in the gate, not noise. Find out which check should have
   caught it.
3. **Numeric exact-match** (the headline) and the **confirmation-trigger rate** (how often reps are asked). The
   trade-off is deliberate: asking more is safer and more annoying.
4. **Numerals parser vs transcripts.** The "not found in transcript" list separates parser gaps from numbers the rep
   never said aloud.
5. Latency p50/p95 and cost per message, which decide whether the experience is usable.

Fix the dev set until it is good, then run the **test** set once and record the result. Do not iterate against it.

## 8. Proposed go / no-go gates (agree these with the customer first)

These are *proposed* starting points. They are not validated and should be argued about.

| Gate | Proposed threshold | Measured on |
|---|---|---|
| No wrong number reaches a saved record without the rep seeing it | **false auto-confirm = 0 at n >= 300 eligible clips**; otherwise keep auto-confirm off | test set |
| Read-back catches errors | the share of wrong numbers that **do** trigger a read-back warning is high (track it; decide the bar with the customer) | test set |
| The bot is usable | confirmation-trigger rate low enough that reps do not abandon it (decide with the reps; track weekly in the pilot) | test set and pilot |
| Stability | p95 latency acceptable to reps; no stranded notes | test set and pilot |

If the numbers fall short: **keep auto-confirm off.** The read-back already makes the product safe; auto-confirm is a
convenience, not a requirement.

## 9. Suggested timeline

| Step | Output |
|---|---|
| Day 1 | keys, `smoke_providers`, one real voice note through the sandbox; settle the open questions (benchmark unit and purity, tola value, "list price") |
| Days 2-3 | consent; record the **dev** set (about 150 clips across speakers, places, hard cases) |
| Days 3-4 | label (two-person overlap); run; failure analysis; fix |
| Days 5-6 | record the **test** set from held-out speakers (about 150+) and label it |
| Day 7 | run the test set once; write down the verdict and the decision on auto-confirm |
| Then | start the live pilot with auto-confirm off; use `python -m scripts.pilot_report` weekly |

## 10. Risks to watch

- **Leakage:** tuning on the test set, or labelling from model output. Both inflate accuracy.
- **Selection bias:** reps re-recording until it "works". Collect what really happened first time.
- **Unrepresentative speakers:** one careful speaker is not the team.
- **Scripted-only data:** reading numbers aloud is easier than a real rushed note.
- **Drifting rules:** changing bounds or prompts mid-evaluation. Freeze them for the verdict run.
- **Small-n confidence:** do not quote a percentage from 30 clips as if it were a finding.
