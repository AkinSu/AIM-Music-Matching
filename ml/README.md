# Lovify ML

The machine-learning side of Lovify, rebuilt as runnable notebooks.

The 2023 project proposed three models — zodiac taste classification, user similarity,
and song recommendation — and presented them in a deck. The figures on those slides
described the system that was *intended*: an attempt was made, it did not work, and the
code and data are gone. So there is no prior result here to reproduce or correct.

This folder builds the three models properly, measures them against baselines, and
leaves behind something the Flask app can import. Much of its value is in the traps it
finds **before** data collection rather than after.

---

## Quick start

Everything runs with **no Spotify credentials and no downloads**. The notebooks
generate synthetic data whose statistical structure matches the real thing.

```bash
cd ml
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # first time only
.venv/bin/jupyter lab notebooks/
```

The notebooks are committed **with their outputs already executed**, so you can also
just open them on GitHub or in VS Code and read straight through without running
anything.

Read them in order:

| notebook | what it establishes |
|---|---|
| `00_data_foundations.ipynb` | What the data is, and the three properties that decide whether any later result can be believed |
| `01_zodiac_classifier.ipynb` | The leakage trap this design walks into, found before we collect data, and how to make the feature shippable anyway |
| `02_user_similarity.ipynb` | A bakeoff of seven similarity metrics against ground truth, plus percentile calibration |
| `03_recommender.ipynb` | The scaling bug, taste modes, diversity, and the relevance/discovery tradeoff |
| `04_end_to_end.ipynb` | The three models assembled into one service the Flask app can call |
| `05_bias_audit.ipynb` | Where bias enters, which parts are measured, and the guards that follow |

---

## What the notebooks found

**Zodiac sign is barely predictable from a song, and that is the whole story.** We
scraped it: 1,325 real playlists, 169,010 rows, 87,907 unique tracks. Under a
cluster-grouped split with balanced accuracy:

| features | balanced accuracy | vs chance |
|---|---|---|
| popularity + year + genre (100% coverage) | 0.100 | 1.20× |
| audio features (41% coverage) | 0.107 | 1.28× |
| everything combined | **0.116** | **1.39×** |
| chance (1 of 12) | 0.083 | 1.00× |

**A caution against our own earlier conclusion.** Notebook 01 demonstrates a leakage
trap — duplicate tracks straddling a random split — and on synthetic data it is
enormous: 81.0% on tracks seen in training against 7.0% on unseen ones. On the real
scrape it almost vanishes: **13.1% seen against 10.7% unseen**, a gap of 2.4 points.
(Both are raw accuracy on one split, which is what the probe measures; across six
seeds, 13.2% ± 0.1 seen and 10.9% ± 0.5 unseen.)

Real zodiac playlists turn out to overlap far less than the generator assumed. Only
25.2% of tracks appear more than once (the generator predicted 76.4%), and when a track
does recur it is usually under a *different* sign, which is label noise rather than
leakage fuel. Effective sample size after collapsing copied playlists came out at
**~100% for every sign** — the generator predicted 63.6%.

So the deduplication machinery is good hygiene that changes little here, and the weak
result is not an artefact of anything. It is the ceiling.

**For user similarity, the blend is the only metric that survives changing our
assumptions.** An earlier version of this README claimed genre overlap beat every
audio-feature metric (ρ 0.942 against 0.371). That comparison was circular and has been
withdrawn: the synthetic users were generated from a genre mixture and "true similarity"
was defined as the distance between those same mixtures, so the genre histogram was the
maximum-likelihood estimator of the answer and could not lose.

Re-running the identical bakeoff against a counter-generator — latent taste living in
audio space, genre playing no part in sampling — inverts the ranking:

| metric | ρ, genre-defined truth | ρ, audio-defined truth | worst case |
|---|---|---|---|
| genre overlap | **0.924** | 0.412 | 0.412 |
| mean euclidean (the deck's KNN) | 0.357 | **0.611** | 0.357 |
| blend of both | 0.879 | 0.523 | **0.523** |

Whichever view of taste the generator privileges, the metric aligned with that view
wins. So neither single metric is a finding. What *is* a finding is the last column:
the blend holds up best when you are wrong about which view matters, and since nobody
knows which view real listeners inhabit, that is the property worth buying.

The held-out-half protocol does not rescue the original claim. It reproduces whichever
generator it is run on, which is why it agreed. **Any claim comparing views of taste is
unresolved until real scraped data lands.**

**Distance measurements were being run on unscaled columns.** `duration_ms` spans about
half a million and `valence` spans one, so raw-column cosine accounts for >99.99% song
length. The original recommender's top-10 had an intra-list diversity of ~3e-11: ten
effectively identical songs.

**Spotify deprecated the endpoint the whole pipeline depended on.** `/v1/audio-features`,
`/v1/audio-analysis`, `/v1/recommendations`, `/v1/artists/{id}/related-artists` and 30s
previews were removed on **2024-11-27** for apps without pre-existing extended access.
Confirmed against this project's own app by `scripts/check_spotify_access.py`: 403 on
audio-features, 404 on recommendations and related-artists, while search, playlist
tracks, artists and tracks all return 200.

An earlier version of this README said the deprecation cost us little because the signal
was never in the removed features. That rested on the withdrawn comparison above and is
**not currently supported** — under an audio-defined notion of taste the deprecated
features are the signal. The honest position: we do not yet know how much the
deprecation costs, and the first real scrape is what will tell us. Build the local
feature store regardless.

---

## Spotify access

Check what the app can still do before planning anything:

```bash
# put credentials in ml/.env (gitignored)
#   SPOTIFY_CLIENT_ID=...
#   SPOTIFY_CLIENT_SECRET=...
# from https://developer.spotify.com/dashboard -> Lovify -> Settings

cd ml && .venv/bin/python scripts/check_spotify_access.py
```

It probes every endpoint the project needs and prints a verdict with the consequences
either way. It uses the Client Credentials flow — an app-level token, no user login —
which is also what the scraper uses.

Two operational constraints worth knowing up front:

- **Development mode caps the app at 25 users**, each allowlisted by Spotify email in
  the dashboard. Collecting user libraries at survey scale is not possible. Scraping
  public playlists has no such cap.
- **Extended quota mode** needs a review, is rarely granted to student projects, and
  would not restore the deprecated endpoints anyway. Don't plan around it.

---

## Data collection: the rigorous re-scrape

The finding is that **the labels are the problem, not the model**. On identical
features with identical method and group discipline, predicting *genre* reaches ~7.3×
chance while predicting *zodiac* reaches ~1.4×. That isolates the label as the cause
and rules out "our features or models were bad". So the collection design is built
around label quality rather than around modelling.

A label is worth training on if it is **independent** (two labels are two opinions, not
one copied twice), **consistent** (curators agree), and **representative** (of music,
not of Spotify's search ranking). The obvious approach — search the sign name, take the
top results — fails all three, which is most likely why the 2023 attempt did not work.

### What the corpus actually looks like

Measured against the live API on 2026-10-11, not assumed:

- **Spotify's own per-sign "Cosmic Playlists" are invisible to the API** — zero
  Spotify-owned results in the top 50 for "Aries" or "Taurus", despite being the first
  hit in the app. They cannot be used as a label source at all.
- **~130 user playlists per sign carry the sign in the title; 1,557 in total, roughly
  263,000 track rows**, found with six query templates per sign. Enough to work with.
- `/v1/search` returns a **useless `total`** — a bare "aries" query reports 32 hits
  while multi-template pagination finds 135. Never plan against that field.
- **"Cancer" and "Leo" are heavily contaminated**, because they are also ordinary
  words. Most "cancer" hits are about the disease ("Cancer Survivor Fight Songs");
  most "leo" hits are artists named Leo or León, skewing hard to Latin music.
  Unfiltered, a model learns *"Cancers like cancer-ward ballads"* and *"Leos like
  regional Mexican"*.
- **Three more collisions turned up on a larger sample**: **"Grupo Libra"** (a Mexican
  band, 5 of 11 at-risk Libra titles), **"Romain Virgo"** (a Jamaican reggae singer)
  together with **"Virgo's Groove"** (a Beyoncé track), and **"Gemini Home
  Entertainment"** (a horror series). An earlier claim here that Virgo was 9 of 9 clean
  came from nine titles and is **retracted** — it did not survive 107.
- Two more junk classes: healing/frequency playlists that are sine tones rather than
  music, and astrological *placements* ("Aries Moon", "Taurus Venus") which are a
  different claim from sun signs.

The zodiac emoji (♈♉♊♋♌♍♎♏♐♑♒♓) turns out to be a strong disambiguator — genuine
zodiac playlists use it constantly, contaminated ones essentially never do.

### What the new scraper does differently:

1. **Hundreds of playlists per sign**, via deep search pagination, instead of the top
   page of results.
2. **Near-duplicate playlist clustering.** Playlists copy each other, so ten agreeing
   playlists may be one opinion counted ten times. Playlists are clustered by Jaccard
   similarity over their track sets, and **each cluster counts as one vote**. The
   resulting *effective sample size* is the number that says how much evidence there
   actually is.
3. **Per-track label confidence** — the share of independent clusters that agree on a
   sign — so training can be restricted to tracks curators actually agreed about.
4. **Full provenance** on every row: playlist id, owner, follower count, description,
   snapshot, scrape timestamp. Needed to filter spam and to audit curator bias.
5. **Grouped cross-validation by playlist cluster**, not just by track. Track-level
   grouping is necessary but not sufficient — correlated playlists leak too.
6. **A title filter** built from the contamination above: require the zodiac emoji or a
   co-occurring astrology term, exclude disease and person-name contexts, exclude
   frequency/healing playlists, cap playlist size at 15–500 tracks, and tag placements
   separately rather than mixing them in.
7. **A negative control.** Scrape a matched set of playlists with nothing to do with
   astrology — study, gym, road trip, shower songs — assign them *fake zodiac labels at
   random*, and run the identical pipeline. If the real data scores 15% and the fake
   one also scores 15%, there is no signal and we know immediately. This is the single
   experiment that decides whether the supervised model is worth shipping, and nothing
   else substitutes for it.

### One trap the filter creates

The title filter removes far more Cancer and Leo playlists than any other sign, because
those are the two sign names that are also ordinary English words. In the dry run that
leaves Cancer at 16 playlists and Leo at 13 against roughly 32 elsewhere — **an
imbalance we introduced, not a property of astrology.**

It matters because it moves the goalposts. With uneven classes the floor is no longer
1/12 but the majority-class rate, and on a filter-shaped imbalance that is 9.5% rather
than 8.3% — turning an apparent 1.53x chance into an honest 1.34x. `zodiac.benchmark()`
therefore reports **balanced accuracy** (macro-average recall, whose chance level stays
at 1/12 however uneven the classes) alongside raw accuracy, plus `vs_majority`.
**Quote balanced accuracy when the real scrape lands.**

### What the scrape actually found

Run on 2026-10-11: **1,325 playlists, 87,907 unique tracks**, all twelve signs.
Balanced accuracy under a cluster-grouped split, chance = 0.083:

| | balanced accuracy | vs chance |
|---|---|---|
| **control — real playlists, randomly relabelled** | **0.0785** | **0.94×** |
| popularity + year + genre (100% coverage) | 0.1000 | 1.20× |
| audio features (41% coverage) | 0.1067 | 1.28× |
| everything combined | 0.1156 | 1.39× |
| *pre-registered "dead" threshold* | *0.1200* | *1.44×* |

Two things follow, and they point in opposite directions.

The control is the result that makes the rest mean anything: non-astrology playlists
given random zodiac labels score **at chance**, so the pipeline is not manufacturing
the signal. The labels really do carry something, and the gap (+0.0296) clears the
pre-registered +0.0200 bar.

But **every feature set lands below the "dead" threshold.** The signal is real and too
small to build on. That is not a borderline call; it is the verdict the gate was
written to deliver, with the threshold fixed before any of this was measured.

A quirk worth recording: per-sign playlist counts sum to 1,345 against 1,325 distinct,
so 20 playlists are returned by more than one sign's search.

*(The control is preliminary — 71 playlists before we hit the rate limit below.)*

### An operational limit worth knowing

The full scrape plus probes ran roughly **8,000 requests in 90 minutes**, and Spotify
responded with a `Retry-After` of **74,655 seconds — 20.7 hours**. Plan collection in
one sitting per day, and resume rather than restart.

The scraper caps `Retry-After` at 180s and raises `RateLimitedTooLong` past it, naming
`--resume`. That cap exists because without it the process obeyed the header literally
and sat silent for 15 minutes before anyone noticed; with a 20.7-hour pause it would
have looked hung until the next morning.

### The decision gate

Don't pick the product now; let the measurement pick it.

| honest accuracy, cluster-grouped CV | what it means |
|---|---|
| < 12%, or the control scores the same | supervised zodiac is dead → ship data-defined archetypes |
| 12–20% | weak but real → ship as a "taste mix" with lift framing, never as a prediction |
| > 20% | genuinely interesting, worth writing up properly |

The same scrape feeds either path, so nothing is foreclosed by collecting first.

### The experiment that needs no API

Development mode caps the app at 25 users, so libraries cannot be collected at survey
scale. A **Google Form** has no such limit: ask for someone's sign and have them paste
their Spotify top artists. A hundred-plus responses tests the hypothesis the product
actually implies — *does a person's real birth sign predict their music taste?* The
answer is probably no, and a clean null result with data behind it is a stronger
finding than an accuracy figure nobody can reproduce.

Run it:

```bash
cd ml
.venv/bin/python scripts/scrape_zodiac.py --dry-run              # no credentials needed
.venv/bin/python scripts/scrape_zodiac.py --playlists-per-sign 200
```

Expect a trustworthy number, and expect it to be low. That is the point.

---

## Using real data

Drop files into `ml/data/raw/` and every notebook switches over on the next run with no
code change:

| file | what it is |
|---|---|
| `tracks.csv` | the Kaggle 600k song dump — column names are auto-mapped to our schema |
| `zodiac.csv` | output of the scraper |
| `listens.csv` | per-user listening histories |

`data.using_real_data()` reports which are live. Each notebook prints whether it is
running on real or synthetic data in its first cell.

---

## Layout

```
ml/
├── notebooks/        the analysis, committed with outputs
├── src/lovify_ml/
│   ├── schema.py     the data contract every module agrees on
│   ├── synthetic.py  structurally realistic fake data, so nothing blocks on credentials
│   ├── data.py       loaders: real data if present, synthetic otherwise
│   ├── features.py   scaling and the four user-profile representations
│   ├── zodiac.py     the classifier, the splits, and the leakage demonstration
│   ├── similarity.py seven metrics, the blend, and percentile calibration
│   ├── recommend.py  retrieval with each fix switchable, plus holdout evaluation
│   ├── scrape.py     the playlist scraper and independence analysis
│   ├── bias.py       the bias measurements
│   ├── spotify.py    API client + what the deprecation means in practice
│   ├── service.py    the object the Flask app imports
│   └── viz.py        shared plotting
├── scripts/          CLI entry points
├── tests/
├── data/raw/         your real data goes here (gitignored)
└── artifacts/        trained models (gitignored)
```

## Why there is synthetic data

Not as a substitute for having data — as a way to test the *methods*. Synthetic users
are generated from a known latent taste vector, which means notebook 02 can score
similarity metrics against ground truth. You never get that with real users. Having it
is what turned "results seem to be quite well" into a number you can improve.

The protocols then get re-validated without it: notebook 02's held-out-half test splits
each user's library in two and checks whether a metric ranks your own other half above
every stranger. That needs no labels, runs on live data, and reproduces the same
ranking — so it is the one that ships.
