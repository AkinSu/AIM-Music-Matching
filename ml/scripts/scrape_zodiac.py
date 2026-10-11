#!/usr/bin/env python
"""Scrape zodiac playlists, then report how much evidence the scrape contains.

The point of this script is the second half. Pulling 2,400 playlists is
plumbing; the finding is what happens to the apparent evidence once copied
playlists are collapsed into clusters and one cluster counts as one vote.
The 2023 run reported 76% accuracy on 12 classes off a dataset where the same
song appeared under several labels in several near-identical playlists; this
prints the number that makes that result readable.

USAGE
    cd ml

    # No credentials needed -- fake scrape, exercises the whole pipeline:
    .venv/bin/python scripts/scrape_zodiac.py --dry-run

    # The real thing (needs ml/.env, see scripts/check_spotify_access.py):
    .venv/bin/python scripts/scrape_zodiac.py --playlists-per-sign 200
    .venv/bin/python scripts/scrape_zodiac.py --resume     # after a stall

    # The negative control -- non-astrological playlists, random sign labels:
    .venv/bin/python scripts/scrape_zodiac.py --control
    .venv/bin/python scripts/scrape_zodiac.py --control --dry-run

THE CONTROL IS THE IMPORTANT RUN. Train the same model on
data/raw/zodiac_control.csv and whatever accuracy comes out is what this
method yields from a dataset with no signal in it by construction. The real
zodiac number only means something as a distance from that one. The control
is written to its own file and data.load_zodiac() reads only zodiac.csv, so
it can never be picked up by accident.

CREDENTIALS come from ml/.env, falling back to the Flask app's own .env --
same loading as scripts/check_spotify_access.py:

        SPOTIFY_CLIENT_ID=xxxxxxxxxxxxxxxxxxxx
        SPOTIFY_CLIENT_SECRET=xxxxxxxxxxxxxxxx

OUTPUT is a CSV that data.load_zodiac() reads directly: star_sign and
track_id are present, plus the provenance columns (owner, followers, snapshot,
the query that found it) and the two analysis columns playlist_cluster and
label_confidence. Downstream code can then train on the confident subset
rather than on every row -- see dedup.confident_labels.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lovify_ml import dedup as D              # noqa: E402
from lovify_ml import filters as F            # noqa: E402
from lovify_ml import schema as S             # noqa: E402
from lovify_ml import scrape as sc            # noqa: E402
from lovify_ml.data import INTERIM, RAW       # noqa: E402

OK, DEAD, WARN = "\033[92m", "\033[91m", "\033[93m"
DIM, END = "\033[2m", "\033[0m"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deep zodiac playlist scrape with playlist-independence analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--playlists-per-sign", type=int, default=200,
                   help="target distinct playlists per sign (default 200). The old "
                        "scrape used 10, which is why it saw only mega-playlists.")
    p.add_argument("--signs", nargs="+", default=list(S.ZODIAC_SIGNS),
                   metavar="SIGN", help="signs to scrape (default: all 12)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"output CSV (default {RAW / 'zodiac.csv'}; a --dry-run "
                        "writes to data/interim/ instead unless you pass this)")
    p.add_argument("--resume", action="store_true",
                   help="reuse the checkpoint in data/interim/ and skip finished signs")
    p.add_argument("--dry-run", action="store_true",
                   help="no network, no credentials: generate a fake scrape with "
                        "planted copied playlists and run the full analysis on it")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Jaccard threshold for calling two playlists copies (default 0.5)")
    p.add_argument("--linkage", choices=["single", "complete"], default="single",
                   help="single = connected components, conservative about evidence "
                        "(default); complete = every pair within threshold")
    p.add_argument("--min-tracks", type=int, default=F.MIN_TRACKS,
                   help=f"skip playlists with fewer tracks than this "
                        f"(default {F.MIN_TRACKS}: below that there is nothing to "
                        f"learn and Jaccard misbehaves)")
    p.add_argument("--max-tracks", type=int, default=F.MAX_TRACKS,
                   help=f"skip playlists with more tracks than this (default "
                        f"{F.MAX_TRACKS}: one 4,078-track playlist would dominate "
                        f"its class on its own)")
    p.add_argument("--min-followers", type=int, default=0,
                   help="skip playlists below this follower count (default 0: record "
                        "followers, filter later -- scrape-time filtering destroys "
                        "the evidence that something was spam)")
    p.add_argument("--max-tracks-per-playlist", type=int, default=500)
    p.add_argument("--control", action="store_true",
                   help="scrape the NEGATIVE CONTROL instead of zodiac playlists: "
                        "study/gym/road-trip playlists with uniformly random fake "
                        "sign labels. Train on it and you measure what this method "
                        "produces from zero signal.")
    p.add_argument("--control-match-sizes", type=Path, default=None,
                   help="a zodiac CSV whose playlist-size distribution the control "
                        "should match (default: the scrape's own --out if it exists)")
    p.add_argument("--no-title-filter", action="store_true",
                   help="keep every playlist whatever its title. Only for measuring "
                        "what the filter removes -- 8 of 9 'cancer' results are "
                        "cancer-ward playlists and 8 of 9 'leo' are artists.")
    p.add_argument("--strict-titles", action="store_true",
                   help="demand an emoji or an astrology word for ALL signs, not "
                        "just the ambiguous ones (Cancer, Leo). Costs real data on "
                        "the ten signs that measured clean -- the report prints how "
                        "much.")
    p.add_argument("--drop-placements", action="store_true",
                   help="drop 'Aries Moon' style placement playlists instead of "
                        "tagging them in title_class")
    p.add_argument("--seed", type=int, default=11, help="--dry-run only")
    return p.parse_args(argv)


# ── credentials ───────────────────────────────────────────────────────────────

def load_credentials() -> tuple[str, str]:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / "LOVIFY ME APP" / ".env")  # fall back to the app's own .env

    cid = os.getenv("SPOTIFY_CLIENT_ID")
    secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not secret:
        print(f"{DEAD}No credentials found.{END}\n")
        print(f"Create {ROOT / '.env'} containing:\n")
        print("    SPOTIFY_CLIENT_ID=your_id_here")
        print("    SPOTIFY_CLIENT_SECRET=your_secret_here\n")
        print("From https://developer.spotify.com/dashboard -> Lovify -> Settings.")
        print(f"{WARN}Or run with --dry-run{END} to exercise the pipeline on "
              "synthetic playlists with no credentials at all.")
        sys.exit(1)
    print(f"\nclient id {cid[:6]}...{cid[-4:]}   (secret loaded, {len(secret)} chars)")
    return cid, secret


# ── collection ────────────────────────────────────────────────────────────────

def size_reference(args: argparse.Namespace) -> pd.DataFrame | None:
    """The zodiac frame whose playlist-size distribution the control matches."""
    path = args.control_match_sizes
    if path is None:
        for candidate in (RAW / "zodiac.csv", INTERIM / "zodiac_dryrun.csv"):
            if candidate.exists():
                path = candidate
                break
    if path is None or not Path(path).exists():
        print(f"{WARN}no zodiac CSV to match sizes against{END}  the control will "
              f"use its default playlist length; run the zodiac scrape first for a "
              f"size-matched control.")
        return None
    print(f"{DIM}matching control playlist sizes to {path}{END}")
    return pd.read_csv(path)


def collect(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.dry_run:
        if args.resume:
            print(f"{WARN}--resume ignored{END}  a dry run has no checkpoint to "
                  f"resume: it regenerates the same playlists from --seed.")
        n = max(4, args.playlists_per_sign)
        if args.playlists_per_sign > 40:
            # 200 fake playlists/sign is 2,400 playlists and an O(P^2) Jaccard
            # matrix; the structure under test shows up fine at 20.
            n = 20
            print(f"{DIM}  (capping fake playlists at {n}/sign -- the Jaccard "
                  f"matrix is O(P^2) and the structure is visible at 20){END}")
        if args.control:
            print(f"\n{WARN}DRY RUN + CONTROL{END}  synthetic non-astrological "
                  f"playlists with uniformly random fake sign labels "
                  f"{DIM}(--signs is ignored: the labels are random over all 12){END}")
            return sc.synthetic_control(n_playlists=n * len(args.signs),
                                        match_size_to=size_reference(args),
                                        seed=args.seed), _no_rejects()
        print(f"\n{WARN}DRY RUN{END}  no network, no credentials. Playlists are "
              f"synthetic, with copies deliberately planted so the clustering "
              f"path is exercised, and the REAL contaminated titles the live API "
              f"returned for cancer/leo planted so the filter is exercised too.")
        return (sc.synthetic_scrape(signs=args.signs, n_playlists_per_sign=n,
                                    seed=args.seed), _no_rejects())

    cid, secret = load_credentials()
    tag = "zodiac_control" if args.control else "zodiac"
    ckpt = sc.ScrapeCheckpoint(tag=tag)
    if args.resume:
        done = [s for s in args.signs if ckpt.is_done(s)]
        have = len(ckpt.all_done_playlists())
        print(f"{OK}resuming{END}  {len(done)} signs complete, "
              f"{have} playlists already in {ckpt.dir}")
    scraper = sc.SpotifyScraper(cid, secret)
    print(f"{OK}app token acquired{END}  (Client Credentials flow)\n")
    if args.control:
        df = scraper.scrape_control(
            n_playlists=args.playlists_per_sign * len(args.signs),
            checkpoint=ckpt, seed=args.seed,
            match_size_to=size_reference(args),
            min_tracks=args.min_tracks, max_tracks=args.max_tracks,
            max_tracks_per_playlist=args.max_tracks_per_playlist,
        )
    else:
        df = scraper.scrape_all(
            signs=args.signs, n_playlists=args.playlists_per_sign, checkpoint=ckpt,
            min_tracks=args.min_tracks, max_tracks=args.max_tracks,
            min_followers=args.min_followers,
            max_tracks_per_playlist=args.max_tracks_per_playlist,
            filter_titles=not args.no_title_filter,
            strict_titles=True if args.strict_titles else None,
        )
    print(f"\n{DIM}{scraper.stats['requests']} requests, "
          f"{scraper.stats['rate_limited']} rate limits, "
          f"{scraper.stats['server_errors']} server errors, "
          f"{scraper.stats['not_found']} gone, "
          f"{scraper.stats['titles_rejected']} playlists rejected by title{END}")
    rej = scraper.rejection_frame()
    if len(rej):
        print("\n  REJECTED BEFORE FETCHING (per sign)")
        _print_rejections(rej, by_sign=True)
    # The rejected playlists never reach `df` -- their tracks were never
    # fetched -- so carry them separately or the audit file would only ever
    # contain the survivors, and recall could not be measured at all.
    prefetch = pd.DataFrame(scraper.rejected) if scraper.rejected else _no_rejects()
    if len(prefetch):
        prefetch[S.TITLE_CLASS] = F.REJECT
    return df, prefetch


def _no_rejects() -> pd.DataFrame:
    return pd.DataFrame(columns=[S.STAR_SIGN, S.PLAYLIST_ID, S.PLAYLIST_NAME,
                                 S.PLAYLIST_OWNER, "n_tracks", S.TITLE_REASON,
                                 S.TITLE_CLASS])


# ── title filter ──────────────────────────────────────────────────────────────

def _print_rejections(rej: pd.DataFrame, by_sign: bool = False) -> None:
    """The audit table. The example title is the point: a reason sitting next
    to a title that looks fine is a rule to go and fix."""
    cols = ([S.STAR_SIGN] if by_sign else []) + [S.TITLE_REASON, "playlists"]
    print(f"\n  {'sign':<12}" if by_sign else "\n  ", end="")
    print(f"{'reason':<22}{'lists':>6}  example")
    for _, r in rej.sort_values("playlists", ascending=False).iterrows():
        head = f"{str(r[S.STAR_SIGN]):<12}" if by_sign else ""
        example = str(r.get("example", ""))[:38]
        print(f"  {head}{r[S.TITLE_REASON]:<22}{int(r['playlists']):>6}  "
              f"{DIM}{example}{END}")


def filter_stage(df: pd.DataFrame, args: argparse.Namespace,
                 prefetch_rejected: pd.DataFrame | None = None,
                 audit_path: Path | None = None):
    """Apply filters.py and print the audit. Returns the kept rows.

    Run on BOTH paths: in a real scrape the filter already ran before any
    track page was fetched, so re-applying it here finds nothing new and the
    breakdown simply reproduces what was skipped. In a dry run (and with
    --no-title-filter) this is where the decision actually happens.
    """
    strict = True if args.strict_titles else None
    kept, verdicts = F.apply_filter(df, strict=strict, control=args.control,
                                    drop_placements=args.drop_placements,
                                    min_tracks=args.min_tracks,
                                    max_tracks=args.max_tracks)
    if prefetch_rejected is not None and len(prefetch_rejected):
        # Union the two sources of verdicts: titles rejected before the fetch
        # and titles judged from the rows we hold.
        verdicts = pd.concat([verdicts, prefetch_rejected.reindex(
            columns=verdicts.columns)], ignore_index=True)
    rej = F.rejection_breakdown(verdicts)
    dropped = rej[rej[S.TITLE_CLASS] == F.REJECT]

    print("\n" + "=" * 68)
    print("TITLE FILTER  (what these playlists are actually about)")
    print("=" * 68)
    n_before = df[S.PLAYLIST_ID].nunique()
    n_after = kept[S.PLAYLIST_ID].nunique() if len(kept) else 0
    print(f"  playlists in        {n_before:,}")
    print(f"  rejected            {int(dropped['playlists'].sum()):,}"
          f"   {DIM}({int(dropped['tracks'].sum()):,} track rows){END}")
    placements = rej[rej[S.TITLE_CLASS] == F.PLACEMENT]["playlists"].sum()
    print(f"  tagged 'placement'  {int(placements):,}"
          f"   {DIM}(Aries Moon etc -- kept, but separable at training time){END}")
    print(f"  kept                {n_after:,}   {DIM}({len(kept):,} rows){END}")

    if len(dropped):
        print(f"\n  WHY REJECTED")
        _print_rejections(dropped)
    if len(rej[rej[S.TITLE_CLASS] == F.KEEP]):
        print(f"\n  WHY KEPT")
        _print_rejections(rej[rej[S.TITLE_CLASS] != F.REJECT])

    if not args.control:
        cost = F.strictness_cost(df)
        print(f"\n  STRICTNESS  {DIM}emoji or astrology word required for...{END}")
        print(f"    just Cancer and Leo (default)  {cost['kept_lenient']:,} playlists kept")
        print(f"    all twelve (--strict-titles)   {cost['kept_strict']:,} playlists kept"
              f"   {DIM}-{cost['would_drop']:,} ({cost['would_drop_tracks']:,} rows){END}")
        worst = sorted(cost["by_sign"].items(), key=lambda kv: -kv[1])[:4]
        if worst:
            print(f"    {DIM}dropped from: "
                  f"{', '.join(f'{k} {v}' for k, v in worst)}{END}")

    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        verdicts.to_csv(audit_path, index=False)
        print(f"\n  {DIM}every verdict, rejects included -> {audit_path}")
        print(f"  sample it for hand-review with scripts/label_playlists.py "
              f"--input {audit_path.name}{END}")

    if args.no_title_filter:
        print(f"\n{WARN}--no-title-filter: keeping everything above anyway.{END} "
              f"Expect 'Cancer' to mean cancer-ward ballads and 'Leo' to mean "
              f"regional Mexican.")
        return df.assign(**{S.TITLE_CLASS: F.KEEP, S.TITLE_REASON: "unfiltered"})
    return kept


# ── report ────────────────────────────────────────────────────────────────────

def report(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Print the scrape report and return the frame with analysis columns."""
    print("\n" + "=" * 68)
    print("WHAT WAS COLLECTED")
    print("=" * 68)
    print(f"  signs                 {df[S.STAR_SIGN].nunique()}")
    print(f"  playlists             {df[S.PLAYLIST_ID].nunique():,}")
    print(f"  rows (track-in-list)  {len(df):,}")
    print(f"  unique tracks         {df[S.TRACK_ID].nunique():,}")
    if S.PLAYLIST_OWNER_ID in df.columns:
        print(f"  distinct owners       {df[S.PLAYLIST_OWNER_ID].nunique():,}")
    per_sign = df.groupby(S.STAR_SIGN)[S.PLAYLIST_ID].nunique()
    print(f"  playlists per sign    min {per_sign.min()}, median "
          f"{int(per_sign.median())}, max {per_sign.max()}")

    print("\n" + "=" * 68)
    print(f"PLAYLIST INDEPENDENCE  (Jaccard >= {args.threshold}, {args.linkage} linkage)")
    print("=" * 68)
    clusters = D.cluster_playlists(df, threshold=args.threshold, linkage=args.linkage)
    sizes = D.cluster_sizes(clusters)
    ess = D.effective_sample_size(df, clusters)
    print(f"  playlists                 {ess['playlists']:,}")
    print(f"  independent clusters      {ess['clusters']:,}   "
          f"{DIM}({ess['playlist_collapse']:.0%} of the playlist count){END}")
    print(f"  largest cluster           {int(sizes.iloc[0]) if len(sizes) else 0} playlists"
          f"   {DIM}(copies of one opinion){END}")
    print(f"  clusters with >1 playlist {int((sizes > 1).sum()):,}")

    print(f"\n  apparent evidence   {ess['apparent_votes']:,} (track, playlist) pairs"
          f"   {DIM}<- what the old pipeline trained on{END}")
    print(f"  effective evidence  {ess['effective_votes']:,} (track, cluster) pairs")
    colour = OK if ess["ess_ratio"] > 0.8 else (WARN if ess["ess_ratio"] > 0.5 else DEAD)
    print(f"  {colour}effective sample size   {ess['ess_ratio']:.1%} of apparent"
          f"   (evidence overstated {ess['inflation']:.2f}x){END}")

    print(f"\n  (track, sign) pairs            {ess['track_sign_pairs']:,}")
    print(f"  backed by >=2 clusters         {ess['independent_label_rate']:.1%}"
          f"   {DIM}<- the only rows where 'curators agree' means anything{END}")
    print(f"  mean label confidence          {ess['mean_label_confidence']:.3f}")
    print(f"  labels the copy count moved    {ess['share_changed']:.1%}"
          f"   {DIM}(mean shift {ess['confidence_shift']:.3f}){END}")
    print(f"  moved DOWN (was overstated)    {ess['share_overstated']:.1%}"
          f"   {DIM}<- counting playlists flattered these labels{END}")

    conf = D.label_confidence(df, clusters)
    print("\n" + "=" * 68)
    print("LABEL CONFIDENCE  = clusters of this sign / clusters of any sign")
    print("=" * 68)
    dist = D.confidence_distribution(conf)
    for _, r in dist.iterrows():
        bar = "#" * int(round(r["share"] * 40))
        print(f"  {r['bucket']:>9}  {int(r['n']):>7,}  {r['share']:>6.1%}  {bar}")
    trainable = D.confident_labels(conf, min_confidence=1.0, min_clusters=2)
    print(f"\n  unanimous AND backed by >=2 clusters: {len(trainable):,} "
          f"({len(trainable) / max(1, len(conf)):.1%}) "
          f"covering {trainable[S.TRACK_ID].nunique():,} tracks")
    print(f"  {DIM}that is the honest training set; everything else is a "
          f"contested or single-source label{END}")

    print("\n" + "=" * 68)
    print("PER SIGN")
    print("=" * 68)
    ps = D.per_sign_ess(df, clusters)
    print(f"  {'sign':<12}{'lists':>7}{'clusters':>10}{'tracks':>9}{'ESS':>8}")
    for _, r in ps.iterrows():
        print(f"  {r[S.STAR_SIGN]:<12}{r['playlists']:>7,}{r['clusters']:>10,}"
              f"{r['unique_tracks']:>9,}{r['ess_ratio']:>8.1%}")

    print("\n" + "=" * 68)
    print("THRESHOLD SENSITIVITY  (is the shrink real, or an artefact?)")
    print("=" * 68)
    sweep = D.threshold_sweep(df, linkage=args.linkage)
    print(f"  {'J>=':>5}{'clusters':>10}{'largest':>9}{'ESS':>8}")
    for _, r in sweep.iterrows():
        print(f"  {r['threshold']:>5.1f}{int(r['clusters']):>10,}"
              f"{int(r['largest_cluster']):>9,}{r['ess_ratio']:>8.1%}")

    if "copy_of" in df.columns:
        _validate_against_planted_copies(df, clusters)

    return D.attach_confidence(D.attach_clusters(df, clusters), conf)


def _validate_against_planted_copies(df: pd.DataFrame, clusters: pd.Series) -> None:
    """Dry-run only: score the clustering against the copies we planted.

    The synthetic scrape knows which playlists are clones of which, so this is
    a genuine check that cluster_playlists works -- not just that it runs.
    """
    truth = (df[[S.PLAYLIST_ID, "copy_of"]].astype(str)
             .drop_duplicates().set_index(S.PLAYLIST_ID)["copy_of"])
    assigned = clusters.reindex(truth.index)
    groups = truth.groupby(truth).groups
    planted = {g: m for g, m in groups.items() if len(m) > 1}
    recovered = sum(1 for m in planted.values() if assigned.loc[list(m)].nunique() == 1)
    # A cluster that merges two unrelated originals is the opposite failure.
    merged = (pd.DataFrame({"cluster": assigned, "truth": truth})
              .groupby("cluster")["truth"].nunique())
    print("\n" + "=" * 68)
    print("CLUSTERING CHECKED AGAINST GROUND TRUTH  (dry run only)")
    print("=" * 68)
    print(f"  planted copy groups          {len(planted):,}")
    print(f"  recovered in one cluster     {recovered:,}"
          f"  ({recovered / max(1, len(planted)):.0%})")
    print(f"  clusters mixing unrelated    {int((merged > 1).sum()):,}"
          f"   {DIM}(over-merging: single linkage is transitive, so some of "
          f"this is expected and is the safe direction){END}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = parse_args(argv)
    bad = [s for s in args.signs if s not in S.ZODIAC_SIGNS]
    if bad:
        print(f"{DEAD}unknown sign(s): {bad}{END}  valid: {', '.join(S.ZODIAC_SIGNS)}")
        sys.exit(1)

    out = args.out
    if out is None:
        # Two separations enforced here. A dry run must never land in
        # data/raw/zodiac.csv -- data.load_zodiac() would announce "REAL
        # zodiac set" and every notebook would silently train on invented
        # playlists. And the control must never land there either, because a
        # dataset of deliberately random labels loaded as the real thing is
        # the worst failure mode in this whole project.
        stem = "zodiac_control" if args.control else "zodiac"
        out = (INTERIM / f"{stem}_dryrun.csv") if args.dry_run else (RAW / f"{stem}.csv")
    if args.control and out.name == "zodiac.csv":
        print(f"{DEAD}refusing to write the control set to {out}{END}  "
              f"data.load_zodiac() reads that file and would train on random "
              f"labels. Pick another --out.")
        sys.exit(1)

    df, prefetch_rejected = collect(args)
    if df.empty:
        print(f"{DEAD}nothing collected.{END}")
        sys.exit(1)

    df = filter_stage(df, args, prefetch_rejected,
                      audit_path=INTERIM / f"{out.stem}_title_verdicts.csv")
    if df.empty:
        print(f"{DEAD}the title filter rejected everything.{END}  Check the "
              f"breakdown above -- that is a filter bug, not a data finding.")
        sys.exit(1)

    df = report(df, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{OK}wrote{END} {out}  ({len(df):,} rows, {len(df.columns)} columns)")
    if args.control:
        print(f"{WARN}NEGATIVE CONTROL{END} -- the star_sign column is uniformly "
              f"RANDOM. Train the same model on this file and compare: that "
              f"number, not 8.3% chance, is the floor the real zodiac result has "
              f"to clear. data.load_zodiac() reads only zodiac.csv, so this file "
              f"can never be picked up by accident.")
    if args.dry_run:
        print(f"{WARN}synthetic data{END} -- deliberately NOT written to "
              f"{RAW / 'zodiac.csv'}, so the notebooks keep using their own "
              f"generator rather than mistaking this for a real scrape.")
    print()


if __name__ == "__main__":
    main()
