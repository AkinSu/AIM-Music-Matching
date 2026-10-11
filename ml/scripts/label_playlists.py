#!/usr/bin/env python
"""Turn the title filter's guesses into measured precision and recall.

filters.py is a table of empirical rules written from a sample of live search
results. It is a hypothesis. This script samples real scraped playlists, hands
them out for hand-review, and then scores the filter against the verdicts --
so "the filter is about 90% precise" becomes a number with a denominator
instead of an impression.

USAGE
    cd ml

    # 1. draw a review sheet (works on the dry-run file too)
    .venv/bin/python scripts/label_playlists.py --n 200

    # 2. fill in the human_class column: keep | reject | placement
    #    (a person or an agent; one column, 200 rows, ~20 minutes)

    # 3. score the filter against it
    .venv/bin/python scripts/label_playlists.py --score data/interim/title_review.csv

WHY THE SAMPLE IS STRATIFIED. A simple random sample of kept playlists can
measure precision but not recall: to find the GENUINE zodiac playlists the
filter wrongly rejected, the rejected ones have to be in the sheet. So the
sheet draws from every (auto_class, auto_reason) group, which makes the raw
rates in the sample unrepresentative -- rare reasons are over-sampled on
purpose. Each row therefore carries `stratum_weight`, the number of real
playlists each sampled row stands for, and score_filter() weights by it. The
unweighted rates are printed alongside, and the gap between the two is the
sampling design showing itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lovify_ml import filters as F            # noqa: E402
from lovify_ml import schema as S             # noqa: E402
from lovify_ml.data import INTERIM, RAW       # noqa: E402

OK, DEAD, WARN = "\033[92m", "\033[91m", "\033[93m"
DIM, END = "\033[2m", "\033[0m"

REVIEW_COLUMNS = [
    S.PLAYLIST_ID, S.PLAYLIST_NAME, S.PLAYLIST_OWNER, "n_tracks",
    S.STAR_SIGN, "auto_class", "auto_reason", "stratum_weight", "human_class",
]
CLASSES = (F.KEEP, F.REJECT, F.PLACEMENT)


# ── sampling ──────────────────────────────────────────────────────────────────

def sample_for_review(df: pd.DataFrame, n: int = 200, seed: int = 0,
                      strict: bool | None = None, control: bool = False) -> pd.DataFrame:
    """One row per playlist, stratified over (auto_class, auto_reason).

    Equal allocation per stratum rather than proportional: the point of the
    sheet is to audit every RULE, and a rule that fired four times is exactly
    the one most likely to be wrong.
    """
    verdicts = F.classify_frame(df, strict=strict, control=control)
    if verdicts.empty:
        return pd.DataFrame(columns=REVIEW_COLUMNS)

    owners = (df.groupby(S.PLAYLIST_ID)[S.PLAYLIST_OWNER].first()
              if S.PLAYLIST_OWNER in df.columns else None)
    verdicts[S.PLAYLIST_OWNER] = (verdicts[S.PLAYLIST_ID].map(owners)
                                  if owners is not None else "")

    groups = list(verdicts.groupby([S.TITLE_CLASS, S.TITLE_REASON], sort=True))
    per_group = max(1, n // max(1, len(groups)))
    picks = []
    for (cls, reason), block in groups:
        take = block.sample(n=min(per_group, len(block)), random_state=seed)
        take = take.copy()
        take["auto_class"] = cls
        take["auto_reason"] = reason
        # Each sampled row stands for this many real playlists.
        take["stratum_weight"] = len(block) / len(take)
        picks.append(take)

    out = pd.concat(picks, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
        # Re-derive weights after trimming, or they no longer sum to the
        # population.
        sizes = verdicts.groupby([S.TITLE_CLASS, S.TITLE_REASON])[S.PLAYLIST_ID].nunique()
        taken = out.groupby(["auto_class", "auto_reason"])[S.PLAYLIST_ID].transform("nunique")
        out["stratum_weight"] = [
            sizes[(c, r)] / t for c, r, t in
            zip(out["auto_class"], out["auto_reason"], taken)
        ]
    out["human_class"] = ""
    out = out.rename(columns={S.PLAYLIST_NAME: S.PLAYLIST_NAME})
    return out.reindex(columns=REVIEW_COLUMNS).sort_values(
        ["auto_class", "auto_reason", S.PLAYLIST_NAME]).reset_index(drop=True)


# ── scoring ───────────────────────────────────────────────────────────────────

def score_filter(reviewed: pd.DataFrame) -> dict:
    """Score the filter against hand labels.

    The filter's positive decision is "this playlist is about this sign",
    which covers both keep and placement -- a placement playlist is genuine
    astrology, just a different question, so counting it as a rejection would
    understate recall.

    Returns weighted and unweighted precision/recall/F1, exact 3-class
    agreement, and the per-reason error table, which is the actionable part:
    it names the rule to go and edit.
    """
    df = reviewed.copy()
    df["human_class"] = df["human_class"].astype(str).str.strip().str.lower()
    df = df[df["human_class"].isin(CLASSES)]
    if df.empty:
        return {"reviewed": 0}

    # A sheet hand-made without the weights column scores unweighted, rather
    # than failing: the column is this script's bookkeeping, not the reviewer's
    # responsibility.
    w = (pd.to_numeric(df["stratum_weight"], errors="coerce").fillna(1.0)
         if "stratum_weight" in df.columns else pd.Series(1.0, index=df.index))
    auto_pos = df["auto_class"].isin((F.KEEP, F.PLACEMENT))
    human_pos = df["human_class"].isin((F.KEEP, F.PLACEMENT))

    def rates(weights):
        tp = float(weights[auto_pos & human_pos].sum())
        fp = float(weights[auto_pos & ~human_pos].sum())
        fn = float(weights[~auto_pos & human_pos].sum())
        tn = float(weights[~auto_pos & ~human_pos].sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": prec, "recall": rec, "f1": f1}

    errors = (df.assign(wrong=(auto_pos != human_pos))
              .groupby(["auto_class", "auto_reason"])
              .agg(reviewed=("wrong", "size"), disagreements=("wrong", "sum"),
                   example=(S.PLAYLIST_NAME, "first"))
              .reset_index())
    errors["error_rate"] = errors["disagreements"] / errors["reviewed"]

    return {
        "reviewed": int(len(df)),
        "exact_agreement": float((df["auto_class"] == df["human_class"]).mean()),
        "weighted": rates(w),
        "unweighted": rates(pd.Series(1.0, index=df.index)),
        "by_reason": errors.sort_values("disagreements", ascending=False)
                           .reset_index(drop=True),
        "confusion": pd.crosstab(df["auto_class"], df["human_class"]),
    }


def print_score(score: dict) -> None:
    if not score.get("reviewed"):
        print(f"{DEAD}no rows with a human_class of keep/reject/placement.{END}  "
              f"Fill that column in and re-run.")
        return
    print("\n" + "=" * 68)
    print(f"FILTER SCORED AGAINST {score['reviewed']} HAND LABELS")
    print("=" * 68)
    print(f"  exact 3-class agreement   {score['exact_agreement']:.1%}")
    for label in ("weighted", "unweighted"):
        r = score[label]
        note = (f"   {DIM}<- the population estimate{END}" if label == "weighted"
                else f"   {DIM}(raw sample; strata are deliberately uneven){END}")
        print(f"\n  {label}: precision {r['precision']:.1%}  recall {r['recall']:.1%}  "
              f"F1 {r['f1']:.1%}{note}")
        print(f"    {DIM}kept-and-right {r['tp']:.0f}  kept-but-wrong {r['fp']:.0f}  "
              f"rejected-but-genuine {r['fn']:.0f}  rejected-and-right {r['tn']:.0f}{END}")
    print("\n  PER RULE  (which table row to go and edit)")
    print(f"  {'class':<10}{'reason':<22}{'seen':>5}{'wrong':>7}{'rate':>8}  example")
    for _, r in score["by_reason"].iterrows():
        print(f"  {r['auto_class']:<10}{r['auto_reason']:<22}{int(r['reviewed']):>5}"
              f"{int(r['disagreements']):>7}{r['error_rate']:>8.0%}  "
              f"{DIM}{str(r['example'])[:34]}{END}")
    print("\n  CONFUSION  (rows: filter, columns: human)")
    print("   " + score["confusion"].to_string().replace("\n", "\n   "))
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Sample scraped playlists for hand-review, and score the "
                    "title filter once the human_class column is filled in.")
    p.add_argument("--input", type=Path, default=None,
                   help="scraped CSV (default: data/raw/zodiac.csv, else the "
                        "dry-run file in data/interim/)")
    p.add_argument("--n", type=int, default=200, help="playlists to sample (default 200)")
    p.add_argument("--out", type=Path, default=INTERIM / "title_review.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strict", action="store_true",
                   help="sample the verdicts of the all-signs-strict filter")
    p.add_argument("--control", action="store_true",
                   help="the input is the negative control (no sign expected)")
    p.add_argument("--score", type=Path, default=None,
                   help="score a filled-in review sheet instead of drawing a new one")
    args = p.parse_args(argv)

    if args.score is not None:
        if not args.score.exists():
            print(f"{DEAD}{args.score} not found.{END}")
            sys.exit(1)
        print_score(score_filter(pd.read_csv(args.score)))
        return

    src = args.input
    if src is None:
        for candidate in (RAW / "zodiac.csv", INTERIM / "zodiac_dryrun.csv",
                          INTERIM / "zodiac_control_dryrun.csv"):
            if candidate.exists():
                src = candidate
                break
    if src is None or not Path(src).exists():
        print(f"{DEAD}no scraped CSV found.{END}  Run scripts/scrape_zodiac.py "
              f"(or --dry-run) first, or pass --input.")
        sys.exit(1)

    df = pd.read_csv(src)
    sheet = sample_for_review(df, n=args.n, seed=args.seed,
                              strict=True if args.strict else None,
                              control=args.control)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.out, index=False)

    print(f"\nsampled {len(sheet)} playlists from {src.name} "
          f"({df[S.PLAYLIST_ID].nunique():,} total)")
    print(f"{OK}wrote{END} {args.out}")
    print(f"\n  strata sampled ({sheet.groupby(['auto_class', 'auto_reason']).ngroups}):")
    for (cls, reason), block in sheet.groupby(["auto_class", "auto_reason"]):
        print(f"    {cls:<10}{reason:<22}{len(block):>4} rows"
              f"   {DIM}x{block['stratum_weight'].iloc[0]:.1f} weight{END}")
    print(f"\nNow fill in {WARN}human_class{END} with keep / reject / placement "
          f"-- judging ONLY from the title, the way the filter has to.")
    print(f"Then: .venv/bin/python scripts/label_playlists.py --score {args.out}\n")


if __name__ == "__main__":
    main()
