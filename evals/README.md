# Evals

The headline metric is **numeric exact-match**: of every amount, weight and rate in the gold
labels, how many did the pipeline get exactly right. The safety metric is the **false
auto-confirm rate** (target 0): messages auto-confirmed without asking the rep although a
number was wrong.

`python -m evals.run_evals` prints `no data` until you add a labelled dataset. Nothing here
is ever fabricated.

## Collecting realistic recordings

Aim for 50-100 clips before trusting any number, and 200+ before comparing models.

- **Speakers:** every rep who will use it (3-10 people), not just one. Voices, accents and
  speaking speed differ; Whisper does not fail uniformly across them.
- **Places:** the real counter, a noisy market, a moving two-wheeler, a quiet office.
- **Content mix:** simple visits; large amounts (lakh); fractional words (dedh, dhai, sava,
  saade, paune); weights in gram, kg and tola; Hinglish mixed with English product words;
  corrections mid-sentence ("nahi nahi, paintalis"); similar-sounding numbers (chalis /
  paintalis, sattar / saintis).
- **Hard cases on purpose:** phone call in the background, fast speech, very short notes
  (1-2 s), long notes (60 s+).
- **Consent:** record only with the rep's agreement, and keep audio out of git (the
  `datasets/` folder is git-ignored). Strip client names if the data leaves the company.
- **Source:** export real Telegram voice notes from a test chat, so the codec (OGG/Opus)
  matches production.

## Labelling

```bash
python -m scripts.label_clip --dataset evals/datasets/pilot1 --audio clip.ogg
```

Type exactly what was *said* as the transcript. Numeric fields are integers (paise,
milligrams); the script echoes them as rupees/grams so a slipped zero is visible.
₹45,000 is `4500000` paise; 150 g is `150000` mg.

## Running

```bash
# Parser vs gold transcripts only (no STT/LLM; works today)
python -m evals.run_evals --dataset evals/datasets/pilot1

# Full pipeline metrics, from a predictions file (the runner arrives in Phase 2)
python -m evals.run_evals --dataset evals/datasets/pilot1 --predictions preds.jsonl --out report.md
```
Formats of `manifest.jsonl` and `predictions.jsonl` are documented at the top of
`evals/metrics.py`.
