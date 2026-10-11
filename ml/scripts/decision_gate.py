#!/usr/bin/env python
"""The decision gate: does the supervised zodiac model survive real data?

This is the script the whole project has been building toward. Everything
before it was measured against data we generated; this measures the real
scrape and returns a verdict that picks the product.

    cd ml && .venv/bin/python scripts/decision_gate.py

Four questions, in order, each of which can end the matter:

1. **How much independent evidence is there really?** Playlists copy each
   other, so apparent evidence overstates real evidence. If a sign has fewer
   than MIN_CLUSTERS independent playlist clusters, there is nothing to train
   on no matter what the accuracy says.

2. **Does the model beat chance on an honest split?** Grouped by playlist
   cluster, not just by track -- correlated playlists leak too. Reported as
   balanced accuracy, whose chance level stays at 1/12 however uneven the
   classes, because our own title filter makes them uneven.

3. **Does a control with FAKE labels score the same?** The experiment that
   makes the rest credible. If randomly-labelled playlists score as well as
   zodiac-labelled ones, we measured the pipeline, not astrology. Run
   `scrape_zodiac.py --control` first; without it this step is skipped and
   the verdict is explicitly marked as unconfirmed.

4. **Is the edge statistically real?** A binomial test against the honest
   baseline, and cross-validated so it is not one lucky split.

The thresholds below were chosen before seeing the data. They are written
here rather than decided afterwards for the obvious reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lovify_ml import data, schema as S, zodiac as Z  # noqa: E402

OK, WARN, DEAD, DIM, BOLD, END = (
    "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[1m", "\033[0m")

# ── pre-registered thresholds ────────────────────────────────────────────────
MIN_CLUSTERS_PER_SIGN = 30      # below this, a class has no independent evidence
GATE_DEAD = 0.12                # at or below -> ship unsupervised archetypes
GATE_WEAK = 0.20                # between -> taste mix with lift framing only
CONTROL_MARGIN = 0.02           # real data must beat the control by this much


def hr(title: str) -> None:
    print(f"\n{BOLD}{'=' * 70}\n{title}\n{'=' * 70}{END}")


def honest_accuracy(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Balanced accuracy under a split grouped by playlist cluster."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = [f for f in S.AUDIO_FEATURES if f in df.columns]
    if not feats:
        raise SystemExit("no audio features on the scrape -- run prepare_catalog.py "
                         "and join the feature store first")

    X = df[feats].to_numpy(float)
    y = df[S.STAR_SIGN].to_numpy()
    groups = df[S.PLAYLIST_CLUSTER].to_numpy() if S.PLAYLIST_CLUSTER in df.columns \
        else df[S.TRACK_ID].to_numpy()

    accs, baccs = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        accs.append(float((p == y[te]).mean()))
        baccs.append(float(balanced_accuracy_score(y[te], p)))
    return {
        "accuracy": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "balanced": float(np.mean(baccs)), "balanced_std": float(np.std(baccs)),
        "n": len(df), "baselines": Z.baselines(y),
    }


def main() -> None:
    flags = data.using_real_data()
    if not flags["zodiac"]:
        raise SystemExit(f"{DEAD}No real scrape at data/raw/zodiac.csv.{END}\n"
                         "Run: .venv/bin/python scripts/scrape_zodiac.py")

    z = data.load_zodiac()

    hr("0 · CAN WE EVEN GET FEATURES FOR THESE TRACKS?")
    # The scrape carries genre, popularity and year from the live API, but
    # /v1/audio-features is dead, so the 13 audio columns can only come from
    # the local store. The join rate is not housekeeping -- it is the single
    # number that decides how much of this design is still possible, and the
    # misses are not random: they concentrate on obscure and post-2021 music,
    # which is exactly what makes a listener distinctive.
    missing_audio = [f for f in S.AUDIO_FEATURES if f not in z.columns]
    store_path = ROOT / "data" / "raw" / "feature_store.csv"
    if missing_audio and (store_path.exists() or flags["catalog"]):
        # Prefer the feature store: it is the union of every dump we have and
        # resolves 42.9% of scraped tracks against the curated catalog's 25.8%.
        if store_path.exists():
            cat = pd.read_csv(store_path, low_memory=False)
            print(f"  {DIM}using feature_store.csv ({len(cat):,} tracks){END}")
        else:
            cat = data.load_catalog(verbose=False)
            print(f"  {DIM}using tracks.csv -- run prepare_catalog.py for the larger store{END}")
        have = [f for f in S.AUDIO_FEATURES if f in cat.columns]
        before = z[S.TRACK_ID].nunique()
        z = z.merge(cat[[S.TRACK_ID] + have].drop_duplicates(S.TRACK_ID),
                    on=S.TRACK_ID, how="left")
        cov_rows = z[have[0]].notna().mean()
        resolved = z.loc[z[have[0]].notna(), S.TRACK_ID].nunique()
        print(f"  scraped tracks      : {before:,}")
        print(f"  resolved in store   : {resolved:,}  ({resolved/before:.1%})")
        print(f"  rows with features  : {cov_rows:.1%}")

        miss = z[z[have[0]].isna()]
        if len(miss):
            print(f"\n  {DIM}who falls in the gap:{END}")
            if S.YEAR in z.columns:
                print(f"    median year   : resolved {z.loc[z[have[0]].notna(), S.YEAR].median():.0f}"
                      f"   missing {miss[S.YEAR].median():.0f}")
            if S.POPULARITY in z.columns:
                print(f"    median popular: resolved {z.loc[z[have[0]].notna(), S.POPULARITY].median():.0f}"
                      f"   missing {miss[S.POPULARITY].median():.0f}")
        z = z[z[have[0]].notna()].copy()
        print(f"\n  {OK if resolved/before > 0.5 else WARN}"
              f"proceeding on {len(z):,} rows with full features{END}")
    elif missing_audio:
        raise SystemExit(f"{DEAD}No audio features and no catalog to join from.{END}\n"
                         "Run: .venv/bin/python scripts/prepare_catalog.py")

    hr("1 · HOW MUCH INDEPENDENT EVIDENCE IS THERE?")

    if S.PLAYLIST_CLUSTER in z.columns:
        per = z.groupby(S.STAR_SIGN).agg(
            playlists=("playlist_id", "nunique"),
            clusters=(S.PLAYLIST_CLUSTER, "nunique"),
            tracks=(S.TRACK_ID, "nunique"))
        per["ESS"] = (per.clusters / per.playlists * 100).round(1)
        print(per.to_string())
        thin = per[per.clusters < MIN_CLUSTERS_PER_SIGN]
        print(f"\n  pre-registered floor: {MIN_CLUSTERS_PER_SIGN} independent clusters per sign")
        if len(thin):
            print(f"  {WARN}{len(thin)} sign(s) below it: "
                  f"{', '.join(thin.index)}{END}")
        else:
            print(f"  {OK}every sign clears it{END}")
    else:
        print(f"{WARN}no cluster column -- grouping by track only, which is weaker{END}")

    hr("2 · DOES IT BEAT CHANCE ON AN HONEST SPLIT?")
    clean = Z.deduplicate(z, "majority")
    print(f"  {len(z):,} rows -> {len(clean):,} deduplicated tracks")
    counts = clean[S.STAR_SIGN].value_counts()
    print(f"  class sizes: max {counts.max():,}  min {counts.min():,} "
          f"(ratio {counts.max()/counts.min():.2f})")

    r = honest_accuracy(clean)
    print(f"\n  accuracy          {r['accuracy']:.4f} +/- {r['accuracy_std']:.4f}")
    print(f"  balanced accuracy {r['balanced']:.4f} +/- {r['balanced_std']:.4f}   {DIM}<- the one to quote{END}")
    print(f"  uniform chance    {r['baselines']['uniform']:.4f}")
    print(f"  majority baseline {r['baselines']['majority']:.4f}")
    print(f"\n  balanced accuracy vs chance: {r['balanced']/S.CHANCE_ACCURACY:.2f}x")

    hr("3 · DOES A CONTROL WITH FAKE LABELS SCORE THE SAME?")
    control_path = ROOT / "data" / "raw" / "zodiac_control.csv"
    control_score = None
    if control_path.exists():
        c = pd.read_csv(control_path)
        rng = np.random.default_rng(0)
        c[S.STAR_SIGN] = rng.choice(S.ZODIAC_SIGNS, size=len(c))
        c_clean = Z.deduplicate(c, "majority")
        cr = honest_accuracy(c_clean)
        control_score = cr["balanced"]
        print(f"  control, randomly relabelled: {control_score:.4f} +/- {cr['balanced_std']:.4f}")
        print(f"  real zodiac data            : {r['balanced']:.4f}")
        gap = r["balanced"] - control_score
        print(f"  gap                         : {gap:+.4f}  "
              f"(need > {CONTROL_MARGIN:+.4f})")
        print(f"\n  {OK if gap > CONTROL_MARGIN else DEAD}"
              f"{'the labels carry something the pipeline alone does not'
                 if gap > CONTROL_MARGIN else
                 'INDISTINGUISHABLE FROM RANDOM LABELS -- there is no signal'}{END}")
    else:
        print(f"  {WARN}no control set.{END} Run:")
        print(f"    .venv/bin/python scripts/scrape_zodiac.py --control")
        print(f"  {DIM}without it, any result below is unconfirmed: a pipeline that{END}")
        print(f"  {DIM}scores above chance on random labels would look identical.{END}")

    hr("4 · IS THE EDGE REAL?")
    from scipy.stats import binomtest

    n = len(clean)
    k = int(round(r["accuracy"] * n))
    bt = binomtest(k, n, r["baselines"]["majority"], alternative="greater")
    print(f"  {k:,} of {n:,} correct against the {r['baselines']['majority']:.4f} "
          f"majority baseline")
    print(f"  p = {bt.pvalue:.3g}  -> "
          f"{'a real edge' if bt.pvalue < 0.05 else 'not distinguishable from the baseline'}")

    # ── verdict ──────────────────────────────────────────────────────────────
    hr("VERDICT")
    b = r["balanced"]
    if control_score is not None and (b - control_score) <= CONTROL_MARGIN:
        print(f"{DEAD}SUPERVISED ZODIAC IS DEAD.{END}")
        print("""
Random labels score as well as real ones, so the model is reading the
pipeline rather than the music. Ship the unsupervised archetypes instead:
cluster the catalog into 12 taste groups and name them after the signs. Same
pie chart, no label problem, nothing to leak.""")
    elif b <= GATE_DEAD:
        print(f"{DEAD}SUPERVISED ZODIAC IS DEAD{END}  (balanced accuracy {b:.3f} <= {GATE_DEAD})")
        print("\nShip unsupervised archetypes. The labels do not support a classifier.")
    elif b <= GATE_WEAK:
        print(f"{WARN}WEAK BUT REAL{END}  (balanced accuracy {b:.3f}, chance {S.CHANCE_ACCURACY:.3f})")
        print("""
Ship it as a taste mix with lift-over-base-rate framing, never as a
prediction, and keep the is_distinctive guard so it stays quiet when it has
nothing to say. service.taste_profile already does this. Do not put an
accuracy number on a slide without the chance rate beside it.""")
    else:
        print(f"{OK}GENUINELY INTERESTING{END}  (balanced accuracy {b:.3f})")
        print("""
Well above what the label noise should permit. Before believing it: confirm
the control gap, re-run with a different cluster threshold, and check no
single sign is carrying the result.""")
    print()


if __name__ == "__main__":
    main()
