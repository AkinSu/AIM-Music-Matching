#!/usr/bin/env python
"""Generate a Linear-importable backlog from the real state of this project.

Linear takes a CSV at Settings -> Import/Export -> Import -> CSV. Running this
writes `linear_backlog.csv` next to the repo root; upload that and the whole
team starts with real tickets instead of a blank board.

Every ticket here traces to something measured in the notebooks or to a
constraint we confirmed against the live Spotify API, so the backlog is not
invented work.
"""

from __future__ import annotations

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "linear_backlog.csv"

# Already shipped. Kept in the backlog rather than deleted so the import gives
# the team a record of what was done, not just what is left.
DONE = {
    "Rotate the Spotify client secret",
    "Enable branch protection on main",
    "Add CI: pytest plus a notebook smoke test",
    "Add the held-out-half self-match rate as a CI regression test",
}

# (title, description, priority, estimate, labels)
T = [
    # ── security: do these first ─────────────────────────────────────────────
    ("Rotate the Spotify client secret",
     "The client secret is committed in git history (commit d90fc16, 'LOVIFY ME APP/.env') "
     "and this repository is PUBLIC. The exposed secret is the one the app is using today. "
     "Assume it is harvested; scraping history clean does not undo publication.\n\n"
     "Fix: Spotify dashboard -> Lovify -> Settings -> rotate client secret. Update ml/.env "
     "and the Railway environment variables. Do not commit the new value anywhere.",
     "Urgent", 1, "security"),

    ("Remove the hardcoded Flask SECRET_KEY fallback",
     "lovify.py line ~19 falls back to a literal secret key when SECRET_KEY is unset, and "
     "that literal is in the public repo. Flask signs session cookies with it, and our "
     "sessions carry Spotify access and refresh tokens, so anyone can forge a session.\n\n"
     "Fix: raise on startup if SECRET_KEY is missing rather than falling back. A missing "
     "key should be a crash, never a silent downgrade to a published one.",
     "Urgent", 1, "security"),

    ("Audit Railway env vars and redeploy with rotated credentials",
     "After rotating, confirm SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SECRET_KEY, SCOPE "
     "and BASE_URL are all set in Railway, and that the deployed app still completes the "
     "OAuth round trip for two users.",
     "Urgent", 2, "security,app"),

    # ── infra ────────────────────────────────────────────────────────────────
    ("Enable branch protection on main",
     "Require a pull request and one approving review before merge; block direct pushes. "
     "GitHub -> Settings -> Branches -> Add rule for main.",
     "High", 1, "infra"),

    ("Add CI: pytest plus a notebook smoke test",
     "GitHub Action on every PR: install ml/requirements.txt, run pytest, and execute each "
     "notebook with a small synthetic dataset to prove it still runs end to end. Notebooks "
     "that silently stop running are how results rot.",
     "Medium", 3, "infra"),

    ("Add the held-out-half self-match rate as a CI regression test",
     "Notebook 02's protocol needs no labels, so it can run in CI on synthetic users. Fail "
     "the build if the blended metric's self-match rate drops below a threshold. This is the "
     "test that would have caught the original leakage bug.",
     "High", 3, "infra,eval"),

    # ── data collection ──────────────────────────────────────────────────────
    ("Download the Kaggle 600k song dump to ml/data/raw/tracks.csv",
     "Spotify's /v1/audio-features returns 403 for this app (confirmed by probe), so audio "
     "features must come from a local store keyed on track_id. Any Kaggle Spotify dump that "
     "carries the Spotify track id and the 13 audio features will work; data.py auto-maps "
     "common column namings.",
     "High", 2, "data"),

    ("Run the full zodiac scrape with provenance",
     "scripts/scrape_zodiac.py --playlists-per-sign 200. Capture playlist id, owner, follower "
     "count, description and snapshot for every row; we need these to filter spam and to "
     "audit curator bias later.",
     "High", 3, "data"),

    ("Validate near-duplicate playlist clustering on the real scrape",
     "Playlists copy each other, so ten agreeing playlists may be one opinion counted ten "
     "times. Cluster by Jaccard over track sets and hand-check a sample of clusters to "
     "confirm the threshold is sensible before trusting it.",
     "High", 3, "data"),

    ("Report the effective sample size after collapsing copies",
     "The headline number from the scrape: how much apparent evidence survives once each "
     "cluster of copied playlists counts as a single vote. Expect a large drop. This figure "
     "determines whether the supervised zodiac model is worth training at all.",
     "High", 2, "data,eval"),

    ("Cache artist genres during the scrape",
     "Batch /v1/artists (50 ids per call) and dedupe artist ids across the whole run.\n\n"
     "Note the justification changed: notebook 02's 'genre overlap is the strongest signal' "
     "finding was WITHDRAWN as circular (the generator defined truth in genre space, so "
     "genre could not lose). Do this anyway, for two reasons that survive - the endpoint "
     "still works where /audio-features does not, and the calls are nearly free once artist "
     "ids are deduped across the run. Genre is a feature we can get, not a feature we have "
     "shown to be best.",
     "High", 2, "data"),

    ("Measure feature-store coverage of scraped tracks",
     "What fraction of scraped track_ids resolve against the Kaggle dump? Break it down by "
     "track popularity and release year. Coverage failures will concentrate on obscure and "
     "recent music, which is a bias problem, not just a data gap.",
     "High", 2, "data,bias"),

    ("Choose a minimum label-confidence threshold from the data",
     "Plot kept-data versus threshold and pick the knee. Train only on tracks curators "
     "actually agreed about, and state the chosen threshold in the report.",
     "Medium", 2, "data,models"),

    # ── models ───────────────────────────────────────────────────────────────
    ("DECISION: supervised zodiac vs unsupervised archetypes",
     "Supervised needs curator labels and is capped by their disagreement; notebook 01 "
     "measured ~15% against 8.3% chance on synthetic data with the same structure. "
     "Unsupervised archetypes (cluster the catalog, name the clusters after signs) have no "
     "label problem at all and produce the same user-facing output.\n\n"
     "Decide after the effective-sample-size number lands. Write the decision and its "
     "reasoning into ml/README.md either way.",
     "High", 2, "models,decision"),

    ("Retrain the zodiac model with GroupKFold by playlist cluster",
     "Track-level grouping is necessary but not sufficient: correlated playlists leak too. "
     "Group by playlist cluster and report accuracy against the 8.3% chance rate with a "
     "confidence interval.",
     "High", 3, "models"),

    ("Ship lift-over-base-rate and the is_distinctive guard",
     "A raw '11.9% Capricorn' is mostly a statement about the class prior. Report how far "
     "each sign deviates from what the model says about an average listener, and return "
     "is_distinctive=false when the top lift is under 1.25x rather than inventing a "
     "personality. Already implemented in service.taste_profile; needs wiring to the UI.",
     "Medium", 2, "models,app"),

    ("Re-run the similarity bakeoff on real user libraries",
     "Notebook 02's ranking turned out to be an artefact of the generator, and the "
     "held-out-half protocol does NOT rescue it - it inverts under a counter-generator too "
     "(genre overlap 0.980 -> 0.107; the deck's method 0.297 -> 0.843), because it "
     "reproduces whichever generator it runs on.\n\n"
     "So this is the ticket that finally answers the question: run the bakeoff on real "
     "libraries, where no generator is choosing the winner. Until it lands, the only "
     "defensible claim is that the blend has the best worst case.",
     "High", 3, "models,eval"),

    ("Tune blend weights on real users",
     "similarity.tune_weights searches the simplex against a target. On synthetic data this "
     "lifted Spearman from 0.889 to 0.945. Re-tune against the held-out-half objective and "
     "freeze the weights into service.LovifyModels.",
     "Medium", 2, "models"),

    ("Validate taste-mode retrieval on the real catalog",
     "Confirm that per-mode recommendation still beats the single mean vector once the "
     "catalog is 600k real songs rather than synthetic. Report recall@20 against the random "
     "baseline and intra-list diversity.",
     "Medium", 3, "models"),

    ("Pick popularity_penalty and mmr_lambda from the measured frontier",
     "These trade recall against discovery, and the tradeoff is real: on synthetic data the "
     "popularity penalty halved mean recommendation popularity but cut recall below random. "
     "Choose from the curve in notebook 03, and state the reasoning.",
     "Low", 2, "models"),

    # ── bias ─────────────────────────────────────────────────────────────────
    ("Wire the coverage-bias measurement into CI",
     "Users whose music the feature store cannot resolve get silently degraded scores, and "
     "those users are disproportionately listeners of obscure and non-Western music. Track "
     "per-user coverage and alert when the distribution shifts.",
     "High", 2, "bias"),

    ("Set a minimum library size from the bootstrap curve",
     "Small libraries produce noisy profiles, so their compatibility percentile is noise "
     "presented as precision. Take the threshold from the measured confidence-interval "
     "curve in notebook 05, not from intuition, and have the service decline rather than "
     "show a confident wrong number.",
     "Medium", 2, "bias,app"),

    ("Report long-tail coverage and Gini for recommendations",
     "What share of the catalog is ever recommended to anyone? A recommender that only "
     "surfaces the head is not doing its job regardless of its recall score.",
     "Low", 2, "bias,eval"),

    ("Finalise BIAS.md for the report",
     "One row per bias: severity, whether it is measured or merely stated, the measurement "
     "function, the current value, and the mitigation. Mark the unmeasurable ones honestly "
     "as unmeasured rather than quietly omitting them.",
     "Medium", 2, "bias,docs"),

    # ── app ──────────────────────────────────────────────────────────────────
    ("Allowlist team and testers in the Spotify dashboard",
     "Development mode caps the app at 25 users, each added manually by their Spotify "
     "account email. Anyone not on the list gets a confusing auth failure, so collect the "
     "emails before demo day.",
     "High", 1, "app"),

    ("Load the model artifact at Flask startup",
     "LovifyModels.load() once at boot, not per request. Add a health check that fails "
     "loudly if the artifact is missing or was trained against a different schema version.",
     "Medium", 2, "app"),

    ("Build the /api/compatibility route",
     "Read both users' tokens from the session, pull their libraries, join features from the "
     "local store, score the pair, and return the explain() payload. Notebook 04 contains "
     "the proposed implementation.",
     "High", 3, "app"),

    ("Design UI states for low coverage and small libraries",
     "Two cases the current design has no answer for: we cannot resolve enough of your music, "
     "and you have not listened to enough for us to be confident. Both need real screens; "
     "showing a confident number anyway is the wrong call.",
     "Medium", 3, "app,bias"),

    ("Replace the compatibility page placeholder with the real report",
     "compatibilityV2.html currently renders nothing but the login prompt. Wire it to the "
     "percentile, the component breakdown, shared artists, and the blended playlist.",
     "Medium", 3, "app"),

    # ── docs ─────────────────────────────────────────────────────────────────
    ("Rewrite the deck around what was actually measured",
     "The 2023 deck's ML figures were illustrative, written to describe the system that was "
     "intended rather than one that ran. So the new deck is not a correction of a past "
     "result - there is no past result - it is the first honest set of numbers.\n\n"
     "Lead with the leakage demonstration as PREVENTION: a design like this reports 60-80% "
     "and means nothing, here is the mechanism (81% on tracks seen in training vs ~10% on "
     "unseen, against 8.3% chance), and here is the pipeline built to avoid it. Showing the "
     "trap caught before data collection is a better talk than any accuracy figure.",
     "Medium", 3, "docs"),

    # ── added after measuring the live API and the QA pass ───────────────────
    ("Build the zodiac playlist title filter",
     "Measured against the live API: 'cancer' searches return mostly disease playlists "
     "('Cancer Survivor Fight Songs') and 'leo' returns mostly artists named Leo/Leon, "
     "skewing hard to Latin music. Virgo came back 9/9 genuine, so contamination is "
     "concentrated in signs that are also ordinary words.\n\n"
     "Filter: require the zodiac emoji or a co-occurring astrology term; reject disease "
     "and person-name contexts; reject frequency/healing playlists (sine tones, not "
     "music); cap size 15-500 tracks; tag placements ('Aries Moon') separately.",
     "Urgent", 3, "data"),

    ("Scrape the negative control and run the decision gate",
     "Scrape ~1,500 playlists with nothing to do with astrology (study, gym, road trip, "
     "shower songs), assign FAKE zodiac labels at random, and run the identical "
     "pipeline. If the real data and the fake-labelled control score the same, there is "
     "no signal.\n\n"
     "This single experiment decides whether the supervised zodiac model ships. Gate: "
     "<12% or control-equal -> archetypes instead; 12-20% -> taste mix with lift "
     "framing only; >20% -> write it up properly.",
     "Urgent", 3, "data,eval,decision"),

    ("Hand-label 300 scraped playlist titles",
     "Sample scraped playlists and label each genuine-zodiac or junk, so the title "
     "filter's precision becomes a measured number instead of a guess. "
     "scripts/label_playlists.py produces the sheet and scores it once filled in. "
     "Parallelisable and needs no ML background.",
     "High", 2, "data"),

    ("Handle the class imbalance the title filter creates",
     "The filter strips far more Cancer and Leo playlists than any other sign, because "
     "those sign names are also ordinary words. Dry run: Cancer 16 and Leo 13 against ~32 "
     "elsewhere. We created that imbalance, so we have to account for it.\n\n"
     "With uneven classes the baseline is the majority-class rate, not 1/12 - measured at "
     "9.5% vs 8.3%, which turns an apparent 1.53x chance into 1.34x. Quote balanced "
     "accuracy (zodiac.benchmark already reports it), consider class_weight='balanced', "
     "and state the class sizes in the writeup.",
     "High", 2, "models,eval"),

    ("Finish the fake-label control scrape",
     "The control is the experiment that makes every other zodiac number mean "
     "something, and it only got 71 of 480 playlists before Spotify rate-limited the "
     "app for 20.7 hours.\n\n"
     "Preliminary result is encouraging: random labels score 0.0785 balanced accuracy "
     "(0.94x chance) against 0.1042 for size-matched real zodiac data, a +0.026 gap "
     "versus the pre-registered +0.020 bar. So the signal is not a pipeline artefact. "
     "Finish it with: scripts/scrape_zodiac.py --control --resume. Budget a full day; "
     "~8,000 requests in 90 minutes is what triggered the ban.",
     "High", 2, "data,eval"),

    ("Run the zodiac survey",
     "The only experiment with real ground truth, and it needs no Spotify API, so the "
     "25-user cap does not apply. See ml/SURVEY.md for the form, the pre-registered "
     "analysis plan, and the power analysis: ~148 responses gives 80% power to detect "
     "a moderate effect, ~430 for a small one. Expect and report a null.",
     "High", 3, "eval,docs"),

    ("Rebuild notebook 02 around robustness, not ranking",
     "The 'genre overlap beats audio features' finding was withdrawn as circular - the "
     "generator defined truth in genre space, so genre could not lose. A counter-"
     "generator with taste in audio space inverts the ranking in every seed. What "
     "survives is that the blend has the best worst-case score, which is the right "
     "argument anyway: nobody knows which view of taste real listeners inhabit.",
     "High", 3, "models,docs"),

    ("Re-validate every synthetic-data conclusion against the real scrape",
     "Every finding in this project is currently measured against one generator written "
     "by the person who expected those conclusions, and that has already produced one "
     "withdrawn headline. The held-out-half protocol does not rescue it - it reproduces "
     "whichever generator it runs on. Re-run the similarity bakeoff, the coverage-bias "
     "null, and the recommender ablation on real data and report which survive.",
     "High", 5, "eval,models,bias"),
]


def main() -> None:
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Title", "Description", "Priority", "Estimate", "Labels", "Status"])
        for title, desc, prio, est, labels in T:
            w.writerow([title, desc, prio, est, labels,
                        "Done" if title in DONE else "Backlog"])

    print(f"wrote {OUT}  ({len(T)} issues)")
    by_label: dict[str, int] = {}
    for *_, labels in T:
        for l in labels.split(","):
            by_label[l] = by_label.get(l, 0) + 1
    print("\nby label:")
    for l, n in sorted(by_label.items(), key=lambda kv: -kv[1]):
        print(f"  {l:<10} {n}")
    print("\nby priority:")
    for p in ["Urgent", "High", "Medium", "Low"]:
        print(f"  {p:<8} {sum(1 for *_x, pr, _e, _l in [(t, d, pr, e, l) for t, d, pr, e, l in T] if pr == p)}")
    print(f"\ntotal estimate: {sum(e for *_, e, _ in [(t, d, p, e, l) for t, d, p, e, l in T])} points")
    print("\nImport at: Linear -> Settings -> Import / Export -> Import -> CSV")


if __name__ == "__main__":
    main()
