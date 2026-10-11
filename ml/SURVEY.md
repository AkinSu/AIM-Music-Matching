# The survey: does a person's actual birth sign predict their music taste?

Every other part of this project asks a narrower question — *can we learn what playlist
curators think an Aries song sounds like?* That is a question about curators. This one
asks the question the product actually implies, and it is the only experiment here that
can answer it.

It is also the only one that needs no Spotify API access at all, which matters: the app
is capped at 25 allowlisted users in development mode, so libraries cannot be collected
at survey scale. A Google Form has no such cap.

---

## The hypothesis, stated so it can fail

> **H1.** A person's sun sign is predictable from their music taste at a rate above the
> 1-in-12 chance baseline of 8.33%.
>
> **H0.** It is not.

Pre-register H0 as the expected outcome. The literature on astrology and personality is
consistent and unkind, and music taste is a weaker proxy for personality than the
instruments that have already failed to find an effect. Going in expecting a null and
reporting one honestly is a legitimate result — and a better talk than an accuracy
figure nobody can reproduce.

---

## What you can actually detect

Power to reject H0 at α = 0.05, one-sided, simulated:

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

**Read this before collecting anything.** A realistic UTD AIS survey gets 100–300
responses. At that size you can detect a *moderate-to-large* effect and nothing
smaller. So decide now what the null result will mean, and say it in the write-up:

> *"With n responses we had 80% power to detect an effect of 15% accuracy or greater.
> We observed X%. We can therefore rule out an effect larger than moderate; a very
> small effect would require roughly 2,000 respondents to detect and is outside the
> scope of this study."*

That sentence is the deliverable. It is honest, it is quantitative, and it is more than
most undergraduate projects manage.

---

## The form

Keep it under 90 seconds or the response rate collapses.

1. **What is your star sign?** — dropdown, 12 options, plus "I don't know". Ask for the
   sign rather than the birth date: asking for a date collects more personal data than
   the study needs, and people know their sign.
2. **Paste your top artists.** Spotify has a built-in share for this
   (Profile → Top artists this month → Share), and people can paste the list. Free
   text; parse later.
3. **Paste your top songs** — same mechanism, optional.
4. **Roughly how many songs are in your library?** — buckets. Lets you check whether
   the light-listener bias from `service.py` applies here too.
5. **Optional:** age bracket and whether they believe in astrology. The second one is
   the interesting covariate — people who believe may *curate* toward their sign, which
   would be a real effect with a non-astrological cause, and separating the two is the
   kind of nuance that makes the analysis worth reading.

No names, no emails, no Spotify login.

---

## Analysis plan — write this down before you look at the data

Pre-registering stops you from discovering a pattern after the fact, which is the main
way studies this size go wrong.

1. **Resolve the pasted artists** to Spotify artist IDs via `/v1/search`, then pull
   genres via `/v1/artists`. Both endpoints still work — no user auth required, so the
   25-user cap never comes into play.
2. **Build a taste vector per respondent**, the same genre histogram
   `features.genre_profile` already produces.
3. **Primary test.** 12-class classification of sun sign from the taste vector, with
   stratified cross-validation. Compare accuracy against 8.33% with a one-sided
   binomial test.
4. **Secondary test.** Do same-sign pairs score higher on `CompatibilityScorer` than
   different-sign pairs? Use a permutation test over shuffled sign labels, because
   pairwise scores are not independent and a naive t-test would overstate significance.
5. **The control that makes it credible.** Shuffle the sign labels and re-run the whole
   pipeline. If shuffled labels score as well as real ones, whatever you measured was
   pipeline artefact, not signal.
6. **Report the confidence interval, not just the point estimate.** At n = 200 the
   interval on an accuracy estimate is wide, and hiding that would misrepresent the
   result in exactly the direction the project has already been burned by.

---

## Why this is worth doing even though it will probably find nothing

Three reasons.

A measured null **kills or confirms the product premise** before more is built on it.
If birth sign genuinely does not predict taste, the zodiac feature must be framed as a
playful taste summary — which is what `taste_profile`'s lift-over-base-rate output
already does — and never as a reading.

It is **the only part of the project with a real ground truth.** Everything else is
validated against curator opinion or synthetic data. Here the label is a fact about a
person.

And it is **cheap**: a form, a few hundred responses, and analysis code that already
exists in this repo.

---

## Privacy

Collect no identifiers. Store responses in a sheet the team controls, not in the repo.
Say on the form what the data is for and that it is anonymous, and mean it. If you
later publish the analysis, publish aggregates — never the raw artist lists, which are
quite identifying in combination.
