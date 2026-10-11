# Pre-registration — does your birth sign predict your music taste?

**Written 2026-10-11, before a single response was collected.**

Everything below — the hypotheses, the tests, the thresholds, the minimum
sample sizes, and what counts as a null — was fixed in advance and committed
to git with a timestamp. With six tests and a free hand, anyone can find a
pattern in anything. The only thing that makes the result of this survey
worth reading is that we could not move the goalposts after seeing it.

If any of this changes later, the change goes in the "Amendments" section at
the bottom with a date and a reason. It does not get quietly edited.

---

## Why this survey exists

Everything else in this project either learns from labels that strangers
disagree about, or uses a mapping we designed ourselves and therefore cannot
be wrong in any testable sense.

This is the one question with an actual answer: **does a person's real sun
sign predict their real music taste?**

We expect it does not. A clean, adequately powered null is the goal.

---

## Hypotheses

| | statement |
|---|---|
| **H1 (primary)** | Sun sign is predictable from a listener's genre profile at above the 1-in-12 chance rate of 8.33%. |
| **H0 (primary)** | It is not. |
| **H2 (secondary)** | Two people who share a sun sign have more similar music taste than two who do not. |
| **H3 (moderator)** | Any effect under H1 or H2 is stronger among people who say they believe in astrology. |

**H3 is the one worth watching.** A believers-only effect would be real and
would have nothing to do with astrology — it would mean people shape their
listening around a sign they already know about themselves. That is a finding
about identity and self-concept, and it is the most interesting outcome
available here. It must not be reported as evidence for astrology.

---

## What we collect

Birth **month** and **day** (no year); self-reported sign; top artists as
pasted free text; top songs (optional); belief in astrology on 1–5; library
size bucket (optional); where they found the form.

The sign used in every analysis is **computed from the birth date**, not the
self-reported one. The agreement between them is reported separately.

No names, no emails, no Spotify login.

---

## Analysis plan

Implemented in `scripts/analyse_survey.py`. Run as-is; do not re-specify.

**Primary (H1).** Twelve-class classification of computed sign from the genre
profile. Random forest, 300 trees, stratified k-fold with k = min(5, smallest
class size). Report accuracy with a Wilson 95% interval, balanced accuracy,
and a one-sided binomial test against 8.33%. **Reject H0 at p < 0.05.**

**Control (mandatory).** Shuffle the sign labels, re-run the primary five
times, average. Real labels must beat shuffled ones by **more than 0.02
balanced accuracy**. If they do not, H1 is not supported *whatever the
p-value says* — we measured the pipeline, not the people. This is not
optional and it is not a robustness check; it is the result.

**Secondary (H2).** Mean pairwise similarity of same-sign pairs minus
different-sign pairs, using `survey.SurveyScorer`. Significance by
**permutation over respondent labels** (2,000 permutations), not a t-test:
pairwise scores are not independent and a t-test would overstate
significance.

**Moderator (H3).** Re-run the primary separately for believers (belief ≥ 4)
and sceptics (belief ≤ 2). Reported only if **both groups have ≥ 40
respondents**.

**Agreement.** Rate at which the self-reported sign matches the computed one.
Descriptive, no test.

**Power.** Reported alongside the result, always, so a null can be stated as
"we could have detected X and did not" rather than "we found nothing".

---

## Thresholds, fixed in advance

| | value |
|---|---|
| alpha | 0.05, one-sided for the primary |
| minimum n for the primary | **60** |
| minimum n per group for the belief split | **40** |
| minimum artists per respondent | **3** |
| control margin | **0.02** balanced accuracy |
| permutations | 2,000 |

Respondents with an unreadable birth date, or fewer than three parsable
artists, are excluded. Exclusions are counted and reported.

---

## Power

Simulated, at alpha = 0.05 one-sided against a 1-in-12 baseline:

| respondents | true acc 10% | 12% | 15% | 20% |
|---|---|---|---|---|
| 50 | 6% | 13% | 33% | 69% |
| 100 | 12% | 31% | 65% | 95% |
| 200 | 20% | 52% | 90% | 100% |
| 300 | 25% | 64% | 97% | 100% |
| 500 | 34% | 85% | 100% | 100% |
| 1,000 | 54% | 99% | 100% | 100% |

Responses needed for 80% power:

| if the true effect is | you need |
|---|---|
| 20% accuracy (large) | ~61 |
| 15% (moderate) | ~148 |
| 12% (small) | ~430 |
| 10% (tiny) | ~1,986 |

**Target: 150+.** Below about 60 the primary test is not reported at all.

---

## What counts as what

**Null (expected).** Primary p ≥ 0.05, or the control gap ≤ 0.02. Reported
as: *"With n = N we had 80% power to detect an effect of X% accuracy or
greater. We observed Y%. We can rule out an effect larger than X%; a smaller
one would require a larger sample."*

**Support for H1.** Primary p < 0.05 **and** control gap > 0.02. Both. A
significant p-value with a failing control is a null.

**Support for H3 only.** Believers show the effect, sceptics do not. Reported
as evidence about **self-concept and curation**, explicitly not about
astrology.

**Inconclusive.** n below 60, or power under 80% even for a 25% effect. Said
plainly as inconclusive — not dressed up as a null.

---

## Known limitations, stated now rather than discovered later

**Recruitment is self-selected.** Posting to Reddit draws people interested
enough in astrology or music to click. The sample is not representative of
anything, and sign distribution will not be uniform. This limits
generalisation; it does not invalidate a within-sample null.

**Artists, not libraries.** We ask for a pasted top-artists list, which is
shorter and more top-of-mind than a real library. Taste vectors are thinner
than those built from listening history, which **biases toward the null**.
Any positive result survives this; a null is weakened by it.

**Three of five similarity components are unavailable.** `SurveyScorer` uses
genre overlap, artist Jaccard and obscurity match. Audio-feature distance and
era overlap need per-track data we do not collect. The secondary test is a
weaker instrument than the one used on scraped playlists.

**Genre resolution is incomplete.** Artist genres come from a 2021 snapshot
plus a live API that now returns an empty genre list for 61% of artists.
Unresolved artists contribute nothing to a taste vector. The resolution rate
is reported with every run.

**Our cusp convention is one of several.** People born within a day of a cusp
may disagree with our computed sign. We use one published set of dates for
everybody and report the disagreement rate.

**Self-report is unverified.** Nobody checks that a respondent typed their
real birth date, or that the artists are really theirs.

---

## Amendments

*None. Any change to the above goes here with a date and a reason.*
