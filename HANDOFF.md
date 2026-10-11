# Where this project stands

Last updated 2026-10-11. Written to be readable on a phone, so you can pick this up
anywhere. If you are starting a fresh Claude session, point it here first.

---

## The one-paragraph version

We asked whether a song can have a star sign. We scraped 1,325 real Spotify zodiac
playlists (87,907 tracks), measured it honestly, and the answer is **barely** — about
1.3–1.4× better than random guessing, against a 1-in-12 baseline. A control where we
gave unrelated playlists random zodiac labels scored exactly at chance, which proves
the small effect is real rather than an artefact of our pipeline. Real but far too
small to build a product on. So the product became an honestly-labelled taste
fingerprint instead of a prediction, and the next step is testing the premise directly
with a survey.

---

## The numbers that matter

Balanced accuracy, cluster-grouped cross-validation. Chance = 0.083.

| | score | vs chance |
|---|---|---|
| control — real playlists, randomly relabelled | 0.0785 | **0.94×** |
| popularity + year + genre | 0.1000 | 1.20× |
| audio features | 0.1067 | 1.28× |
| everything combined | 0.1156 | 1.39× |
| *pre-registered "too weak to ship" line* | *0.1200* | *1.44×* |

Every feature set lands **below** the line. That line was written into the code before
anything was measured.

**The strongest single chart for a presentation:** identical features, identical
method — predicting *genre* reaches ~7.3× chance while predicting *zodiac* reaches
~1.4×. That isolates the label as the problem and kills the "maybe your features or
models were bad" objection.

---

## What exists

```
ml/notebooks/   00 data · 01 zodiac · 02 similarity · 03 recommender
                04 end-to-end · 05 bias · 06 real-data validation · 07 archetypes
ml/src/         16 modules behind them
ml/tests/       205 tests, including a leakage canary
```

Open the notebooks on GitHub and read them — charts and results are saved in, no setup
needed.

Key documents: `ml/README.md` (findings and method), `ml/PREREGISTRATION.md` (the
survey's locked-in analysis plan), `ml/SURVEY.md` (form design and power analysis),
`ml/BIAS.md` (the bias register).

---

## Open pull requests

- **#1** — refuses to boot without `SECRET_KEY`, hardens the session cookie.
  ⚠️ **Set `SECRET_KEY` in Railway before redeploying or the app will not start.** That
  is deliberate: crashing beats running with a key that is published in git history.
- **#2** — the entire `ml/` folder, 11 commits.

Both need your approval; `main` is protected.

---

## Blocked right now

**Spotify API is rate-limited until roughly 05:00 UTC on 2026-10-12.** About 8,000
requests in 90 minutes earned a `Retry-After` of 74,655 seconds — 20.7 hours. Nothing
that touches the API can run before then.

When it lifts, the one job waiting is finishing the control scrape. It got 71 of 480
playlists:

```
cd ml && .venv/bin/python scripts/scrape_zodiac.py --control --resume
```

---

## What is in flight

**The redesign.** Rather than learning what an "Aries song" is from strangers'
playlists — which we proved does not work — we are *designing* it, grounded in the
actual astrological tradition (4 elements × 3 modalities = exactly 12 signs) and in the
documented psychology of why astrology engages people. Research agent running.

Important framing: that makes it a **designed rubric, not a trained model.** It must
never carry an accuracy number, because there is no held-out truth to measure against.
Claiming otherwise would recreate the exact circularity that already cost us one
headline finding.

**The survey.** The only part of this project with real ground truth. A Google Form
asking a person's actual birth sign and their actual top artists, then testing whether
one predicts the other. Analysis pipeline and pre-registration are built and committed.
Form is half-built; see below.

---

## Your to-do list

1. **Approve PRs #1 and #2.** `main` requires one approval; as repo owner you may need
   the "bypass rules" checkbox since GitHub resists self-approval.
2. **Finish the Google Form** — two things left:
   - Settings → Responses → set *Collect email addresses* to **Do not collect**. It is
     currently on, which hurts response rate and collects data we do not want.
   - Add the remaining questions (star sign, top artists, top songs, belief 1–5,
     library size, where-did-you-find-this). Exact wording is in `ml/SURVEY.md`.
3. **Post it to r/SampleSize**, not only to astrology subreddits — a believers-only
   sample cannot separate "astrology is real" from "believers curate toward their sign".
4. **Set `SECRET_KEY` in Railway** before the next deploy.

---

## Things we got wrong, and fixed

Kept here deliberately. Each one was caught by measurement rather than review, and the
pattern is the most useful thing the project learned.

| claim | what happened |
|---|---|
| "Genre overlap beats every audio metric" | **Withdrawn.** Circular — the test data was built from genres, so genre could not lose. |
| "The leakage trap inflates this result" | **Overturned on real data.** Huge in simulation (81% vs 9%), nearly absent in reality (13.1% vs 10.7%). |
| "1.45× chance, borderline" | **Stale number** from before a data repair. The real figure is 1.39×, which is below the line. |
| "`recall@20` measures the recommender" | **Overturned.** Exactly zero at real catalog scale. Replaced with percentile rank. |
| "`is_distinctive` fires for 100% of users" | **Half wrong.** 52% on the zodiac model; the 100% was a different quantity. |

Four separate assumptions baked into the synthetic data generator turned out false —
feature distributions, how mainstream listeners are, what defines taste, and how much
playlists overlap. Every one was wrong in the direction its author expected. That is
the lesson worth putting on a slide.
