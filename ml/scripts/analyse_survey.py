#!/usr/bin/env python
"""The pre-registered survey analysis. Run it the moment a CSV exists.

    cd ml && .venv/bin/python scripts/analyse_survey.py data/raw/survey.csv

Every test below, and every threshold, was fixed in PREREGISTRATION.md before
any response was collected. That is the whole reason a null result from this
script is worth anything: with six tests and a free hand, somebody will always
find a pattern. Nothing here is chosen after seeing the data.

Six outputs, in the order that matters:

  0. Who answered           -- sample, sources, resolution rate
  1. PRIMARY                -- can a model predict real sign from taste?
  2. CONTROL                -- shuffle the labels; does it score the same?
  3. SECONDARY              -- do same-sign pairs resemble each other?
  4. BELIEF SPLIT           -- do believers curate toward their sign?
  5. SELF-REPORT AGREEMENT  -- do people know their own sign?
  6. POWER                  -- what could we have detected?

Step 2 is not optional. On the playlist scrape the equivalent control is what
separated "a real but tiny effect" from "we measured our own pipeline", and
nothing else in the analysis can substitute for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lovify_ml import schema as S  # noqa: E402
from lovify_ml import survey as SV  # noqa: E402

OK, WARN, DEAD, DIM, BOLD, END = (
    "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[1m", "\033[0m")

# ── pre-registered thresholds (see PREREGISTRATION.md) ───────────────────────
ALPHA = 0.05
MIN_N_PRIMARY = 60          # below this the primary test is not reported
MIN_N_BELIEF = 40           # per group, for the believer/non-believer split
MIN_ARTISTS = 3             # a respondent with fewer is not a taste profile
CONTROL_MARGIN = 0.02       # real must beat shuffled by this in balanced accuracy
N_PERMUTATIONS = 2000
N_CV_SPLITS = 5


def hr(title: str) -> None:
    print(f"\n{BOLD}{'=' * 72}\n{title}\n{'=' * 72}{END}")


# ── the model used everywhere below ──────────────────────────────────────────

def _classify(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Stratified CV, 12-class. Returns accuracy and balanced accuracy."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    counts = pd.Series(y).value_counts()
    n_splits = int(min(N_CV_SPLITS, counts.min())) if len(counts) else 0
    if n_splits < 2:
        return {"accuracy": float("nan"), "balanced": float("nan"), "n_splits": 0}

    accs, baccs = [], []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y):
        m = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                   random_state=seed).fit(X[tr], y[tr])
        p = m.predict(X[te])
        accs.append(float((p == y[te]).mean()))
        baccs.append(float(balanced_accuracy_score(y[te], p)))
    return {"accuracy": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "balanced": float(np.mean(baccs)), "balanced_std": float(np.std(baccs)),
            "n_splits": n_splits}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- behaves near 0, which normal approximation does not."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _power_for(n: int, true_acc: float, sims: int = 3000, seed: int = 0) -> float:
    """Probability of rejecting chance at ALPHA if the true accuracy is true_acc."""
    from scipy.stats import binomtest

    rng = np.random.default_rng(seed)
    k_crit = next((k for k in range(n + 1)
                   if binomtest(k, n, S.CHANCE_ACCURACY, "greater").pvalue < ALPHA), n + 1)
    return float((rng.binomial(n, true_acc, sims) >= k_crit).mean())


def main() -> int:
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "survey.csv"
    if not csv.exists():
        print(f"{WARN}No responses at {csv}{END}")
        print("Export the Google Form as CSV and drop it there, then re-run.")
        return 1

    # ── 0 · who answered ─────────────────────────────────────────────────────
    hr("0 · WHO ANSWERED")
    resp = SV.load_responses(csv)
    resp = resp[resp["n_artists"] >= MIN_ARTISTS].reset_index(drop=True)
    print(f"  usable respondents with >= {MIN_ARTISTS} artists: {len(resp)}")

    if "source" in resp.columns:
        print(f"\n  {DIM}where they came from:{END}")
        for src, n in resp["source"].value_counts().head(6).items():
            print(f"    {str(src)[:46]:<46} {n}")

    dist = resp[S.STAR_SIGN].value_counts().reindex(S.ZODIAC_SIGNS, fill_value=0)
    print(f"\n  {DIM}sign distribution (expect roughly uniform; it will not be){END}")
    print("   ", "  ".join(f"{s[:3]} {n}" for s, n in dist.items()))

    resolver = SV.ArtistResolver()
    vectors, long = SV.build_taste_vectors(resp, resolver)
    # Count lookups, not distinct artists: the cache absorbs repeats, so
    # `snapshot` alone undercounts resolutions badly once an artist recurs.
    resolved = resolver.stats["snapshot"] + resolver.stats["cache"]
    unresolved = resolver.stats["unresolved"]
    print(f"\n  artist lookups resolved to a genre: {resolved:,}"
          f"   unresolved: {unresolved:,}"
          f"   ({resolved / max(resolved + unresolved, 1):.1%})")
    print(f"  taste vector dimensions (distinct genres): {vectors.shape[1]}")
    if unresolved > resolved:
        print(f"  {WARN}more artists unresolved than resolved -- the taste vectors are thin, "
              f"which biases every test below toward the null{END}")

    users = [u for u in vectors.index if u in set(resp["respondent_id"])]
    y = resp.set_index("respondent_id").loc[users, S.STAR_SIGN].to_numpy()
    X = vectors.loc[users].to_numpy()

    # ── 1 · primary ──────────────────────────────────────────────────────────
    hr("1 · PRIMARY — can a model predict real sign from music taste?")
    if len(users) < MIN_N_PRIMARY:
        print(f"  {WARN}n = {len(users)}, below the pre-registered minimum of "
              f"{MIN_N_PRIMARY}. Not reporting a result.{END}")
        print(f"  {DIM}Collect more responses. This is the pre-registered rule, "
              f"not a judgement about the data.{END}")
        return 0

    from scipy.stats import binomtest

    real = _classify(X, y)
    k = int(round(real["accuracy"] * len(users)))
    lo, hi = _wilson(k, len(users))
    bt = binomtest(k, len(users), S.CHANCE_ACCURACY, "greater")

    print(f"  n = {len(users)},  {real['n_splits']}-fold stratified CV")
    print(f"  accuracy          {real['accuracy']:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  balanced accuracy {real['balanced']:.4f} +/- {real['balanced_std']:.4f}")
    print(f"  chance            {S.CHANCE_ACCURACY:.4f}")
    print(f"  p (one-sided)     {bt.pvalue:.4g}")
    verdict_primary = bt.pvalue < ALPHA
    print(f"\n  {OK if verdict_primary else DIM}"
          f"{'above chance' if verdict_primary else 'indistinguishable from chance'}{END}")

    # ── 2 · control ──────────────────────────────────────────────────────────
    hr("2 · CONTROL — shuffle the signs and run it again")
    rng = np.random.default_rng(0)
    shuffled = [_classify(X, rng.permutation(y), seed=i)["balanced"] for i in range(5)]
    ctrl = float(np.nanmean(shuffled))
    gap = real["balanced"] - ctrl
    print(f"  real labels     {real['balanced']:.4f}")
    print(f"  shuffled labels {ctrl:.4f} +/- {np.nanstd(shuffled):.4f}   (5 shuffles)")
    print(f"  gap             {gap:+.4f}   (pre-registered bar: > {CONTROL_MARGIN:+.4f})")
    control_passed = gap > CONTROL_MARGIN
    print(f"\n  {OK if control_passed else DEAD}"
          f"{'the labels carry something the pipeline alone does not'
             if control_passed else
             'INDISTINGUISHABLE FROM RANDOM LABELS -- no signal'}{END}")

    # ── 3 · secondary ────────────────────────────────────────────────────────
    hr("3 · SECONDARY — do same-sign pairs have more similar taste?")
    scorer = SV.SurveyScorer(long)
    idx = {u: i for i, u in enumerate(users)}
    pairs = [(a, b) for i, a in enumerate(users) for b in users[i + 1:]]
    scores = np.array([scorer.raw_score(a, b) for a, b in pairs])
    same = np.array([y[idx[a]] == y[idx[b]] for a, b in pairs])

    if same.sum() < 10:
        print(f"  {WARN}only {same.sum()} same-sign pairs; skipping{END}")
    else:
        observed = scores[same].mean() - scores[~same].mean()
        # Permutation over respondent labels, not over pairs: pair scores are
        # not independent, and shuffling pairs would understate the null spread.
        null = np.empty(N_PERMUTATIONS)
        for i in range(N_PERMUTATIONS):
            perm = rng.permutation(y)
            s = np.array([perm[idx[a]] == perm[idx[b]] for a, b in pairs])
            null[i] = scores[s].mean() - scores[~s].mean() if 0 < s.sum() < len(s) else 0.0
        p = float((null >= observed).mean())
        print(f"  same-sign pairs      {same.sum():,}   mean score {scores[same].mean():.4f}")
        print(f"  different-sign pairs {(~same).sum():,}   mean score {scores[~same].mean():.4f}")
        print(f"  difference           {observed:+.4f}")
        print(f"  p (permutation, {N_PERMUTATIONS})  {p:.4g}")
        print(f"\n  {OK if p < ALPHA else DIM}"
              f"{'same-sign pairs are more alike' if p < ALPHA else 'no effect'}{END}")

    # ── 4 · belief split ─────────────────────────────────────────────────────
    hr("4 · BELIEF SPLIT — do believers curate toward their sign?")
    if "belief" not in resp.columns or resp["belief"].isna().all():
        print(f"  {DIM}no belief question answered; skipping{END}")
    else:
        bel = resp.set_index("respondent_id").loc[users, "belief"]
        believers = [u for u in users if bel.get(u, np.nan) >= 4]
        sceptics = [u for u in users if bel.get(u, np.nan) <= 2]
        print(f"  believers (4-5): {len(believers)}    sceptics (1-2): {len(sceptics)}")
        if min(len(believers), len(sceptics)) < MIN_N_BELIEF:
            print(f"  {WARN}below the pre-registered {MIN_N_BELIEF} per group; "
                  f"not reporting{END}")
        else:
            for label, group in (("believers", believers), ("sceptics", sceptics)):
                gi = [idx[u] for u in group]
                r = _classify(X[gi], y[gi])
                print(f"    {label:<12} balanced accuracy {r['balanced']:.4f}  (n={len(group)})")
            print(f"\n  {DIM}A believer-only effect would be real but NOT astrological -- "
                  f"it would mean people{END}")
            print(f"  {DIM}shape their listening around a sign they already know. "
                  f"That is the most{END}")
            print(f"  {DIM}interesting outcome available here, and it is a finding about "
                  f"identity, not stars.{END}")

    # ── 5 · self-report agreement ────────────────────────────────────────────
    hr("5 · SELF-REPORT — do people know their own sign?")
    if "self_sign" in resp.columns:
        both = resp[resp["self_sign"].isin(S.ZODIAC_SIGNS)]
        if len(both):
            agree = (both["self_sign"] == both[S.STAR_SIGN]).mean()
            print(f"  n = {len(both)}   agreement with the computed sign: {agree:.1%}")
            dis = both[both["self_sign"] != both[S.STAR_SIGN]]
            if len(dis):
                print(f"  {DIM}{len(dis)} disagreements, mostly cusp births -- "
                      f"our date convention is one of several{END}")
        else:
            print(f"  {DIM}no parsable self-reported signs{END}")
    else:
        print(f"  {DIM}question not in the export; skipping{END}")

    # ── 6 · power ────────────────────────────────────────────────────────────
    hr("6 · POWER — what could this sample have detected?")
    n = len(users)
    print(f"  with n = {n}, probability of detecting a true accuracy of:")
    detectable = None
    for acc in (0.10, 0.12, 0.15, 0.20, 0.25):
        pw = _power_for(n, acc)
        mark = ""
        if pw >= 0.80 and detectable is None:
            detectable, mark = acc, "   <- 80% power"
        print(f"    {acc:.0%}  ->  {pw:.0%}{mark}")

    print()
    if not verdict_primary:
        if detectable:
            print(f"  {BOLD}The honest statement of this null:{END}")
            print(f"  \"With n = {n} we had 80% power to detect an effect of "
                  f"{detectable:.0%} accuracy or")
            print(f"   greater. We observed {real['accuracy']:.1%} against a "
                  f"{S.CHANCE_ACCURACY:.1%} chance rate. We can therefore rule out an")
            print(f"   effect larger than that; a smaller one would need a bigger sample.\"")
        else:
            print(f"  {WARN}This sample had under 80% power even for a 25% effect. "
                  f"A null here means{END}")
            print(f"  {WARN}very little -- report it as inconclusive, not as evidence "
                  f"of no effect.{END}")

    print(f"\n{DIM}Thresholds and tests were fixed in PREREGISTRATION.md "
          f"before collection.{END}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
