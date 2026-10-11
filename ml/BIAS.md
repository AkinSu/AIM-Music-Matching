# Bias register — Lovify ML

Every entry is either **measured** (there is a function in `lovify_ml.bias` that
produces the number, and it runs in `notebooks/05_bias_audit.ipynb`) or **stated**
(a known limitation we cannot currently quantify, recorded so it is not mistaken for
an absence).

All current values come from **synthetic data**. They demonstrate that the measurement
works and give a rough sense of magnitude; they are not claims about the live product.
Re-run notebook 05 against real data and replace this column before quoting any of it.

Two entries — **#1 (coverage)** and **#6 (library size)** — are engineering problems
rather than disclosures. #6 in particular is a live bug.

---

## The register

| # | Bias | Severity | Status | Measurement | Value (synthetic) | Mitigation |
|---|---|---|---|---|---|---|
| 1 | **Feature-store coverage is non-random** — audio features come from a static dump joined on `track_id`; the join fails on obscure, non-Western and recent music | Medium | Measured | `simulate_store_coverage`, `coverage_by_user`, `coverage_skew`, `coverage_error_report`, `PartialCoverageScorer` | Coverage ρ=0.97 with listener mainstream-ness; bottom decile 0.44 vs top decile 0.90. Score error 2.1 pts (blend) vs 5.4 pts (audio-only) | Blend already weights genre/artist over audio. `PartialCoverageScorer` renormalises weights over surviving components and drops audio below `min_resolved=10` tracks. **Keep the blend weights — reverting to audio-heavy weights triples the error.** |
| 2 | **Catalog bias** — can only recommend what the catalog holds; the Kaggle dump is Western, English-heavy, and stops ~2021 | Medium | Measured (distortion) / Stated (absence) | `distribution_shift`, `catalog_bias_report` | Genre total variation 0.16, but per-genre ratios span 0.33× to 2.0× | Report per-genre ratio, not just aggregate TV. The *absence* half needs a better catalog, not a better model — document catalog provenance in the final report. |
| 3 | **Curator demographics** — zodiac labels are individual opinions, skewed young/Western/pop; a few prolific accounts may have written most of the data | **High** | Measurable, awaiting real scrape | `provenance_report`, `synthetic_provenance` | Demonstrated on synthetic provenance only | Run `provenance_report` on the scrape **before training**. Treat `owner_gini > 0.6` and `top5_owner_share > 0.3` as go/no-go, not footnotes. Compounds with playlist copying, handled separately in the scraper. |
| 4 | **Popularity / exposure concentration** — a recommender can score well on recall while serving the same few hundred songs to everyone | Medium | Measured | `recommendation_exposure`, `popularity_bias_report`, `gini` | No popularity skew in the mean (24.7 recommended vs 24.3 catalog — a genuine null). But `coverage_vs_random` 0.54 and a small track set takes half of all slots | Report `coverage_vs_random`, not raw catalog coverage (raw coverage is mostly arithmetic when slots ≪ catalog). `popularity_penalty` exists as a knob but costs recall — see notebook 03. |
| 5 | **Audio features are culturally trained** — `valence`, `danceability`, `mode` encode Western-pop assumptions; minor key ≠ sad outside Western tonality | Medium | **Stated, unmeasured** | — | Not measurable from inside the pipeline | Would need labelled cross-cultural ground truth we do not have; the synthetic catalog cannot stand in, since it was generated from the same assumptions. Partly mitigated by the blend leaning on genre/artist over audio. Disclose as a limitation. |
| 6 | **Library-size instability** — the percentile is displayed to one decimal regardless of how little listening history it rests on | **High** | **Measured and fixed** | `percentile_stability`, `stability_with_matched_null`, `fit_size_matched_null`, `recommend_min_library` | Population null: bias **−47 pts** at 10 tracks. Size-matched null: **+2 pts**. Min library at a 10-pt budget: 200 → 100 tracks | **Two guards.** (a) Build the null at the pair's library size. (b) Below the measured minimum, show a range or "not enough listening history yet" — never a confident decimal. |
| 7 | **Social framing** — "12% compatible" reads as a verdict on a relationship | Medium | Product decision | — | — | Percentile, never raw similarity. Frame as about music, not people. No hard rejection — surface what the pair *does* share via `explain()`. |

---

## Notes on three of these

### #1 — mitigated by accident, keep it deliberate

The blend survives the loss of most audio features only because notebook 02 found that
genre overlap and artist Jaccard carry the signal and audio features contribute least.
That was a modelling decision made for accuracy, and it happens to also be the thing
that makes the system robust to Spotify's 2024 API deprecation.

This is worth stating plainly because it is fragile: anyone re-weighting the blend
towards audio features reintroduces the bias at roughly triple the error. The
blend-vs-audio-only ablation in notebook 05 should be treated as a **regression test**,
not a one-time finding.

### #6 — why the bug existed and why spread hid it

The null distribution was fitted on full-size libraries. A user with twenty tracks
scores lower on nearly every blend component for reasons unrelated to their taste —
artist overlap collapses towards zero, genre profiles turn sparse — so their raw score
lands near the bottom of a full-library null regardless of who they are paired with.
Every small-library user was being told they were incompatible with everyone.

The bootstrap *spread* hides this completely. At tiny library sizes the score degenerates
and pins against the end of the null, so variance **falls** while error rises: the answer
becomes reliably wrong. That is why `recommend_min_library` selects on RMSE against the
full-library percentile rather than on interval width — interval width is not monotonic
in library size, and selecting on it returns an absurdly small threshold.

A percentile is only meaningful against a reference population that resembles the thing
being scored.

### #5 — what "stated, unmeasured" is for

Listing an unmeasurable bias is not box-ticking. An empty row here would imply we
checked and found nothing, when the truth is that we cannot check with the data we have.
The distinction matters to anyone deciding how much to trust the rest of the register.

---

## Running the audit

```bash
cd ml
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  notebooks/05_bias_audit.ipynb
```

Every measurement is a plain function returning a DataFrame or dict, so the same checks
can run in CI and be diffed between releases. The three worth gating on:

| Check | Gate |
|---|---|
| `coverage_skew(...)["worst_decile_coverage"]` | alert if it falls below ~0.3 |
| `provenance_report(...)["owner_gini"]` | block training above ~0.6 |
| `recommend_min_library(...)["min_library_size"]` | must match the threshold the service enforces |

The last one is the important one: if the model changes and the measured minimum library
size moves, the service's guard has to move with it, or the product silently starts
showing numbers it cannot support again.
