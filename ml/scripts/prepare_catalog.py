#!/usr/bin/env python
"""Turn the Kaggle dump into the catalog the rest of the project expects.

Source: "Spotify Dataset 1921-2020, 600k+ Tracks" (Yamac Eren Ay), the same
dataset the 2023 deck used. Download the 202 MB zip from

    https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks

and unzip it into ml/data/raw/kaggle_600k/. Then:

    cd ml && .venv/bin/python scripts/prepare_catalog.py

Two jobs.

**Rename to our schema.** `data.load_catalog()` can auto-map most of it, but
doing it once here means every notebook reads a clean file instead of
re-deriving the same columns.

**Backfill genre, which is the interesting part.** Spotify's live API now
returns an empty genre list for 61% of artists -- measured on 285 real artists
pulled from our own scrape, and the gaps include Taylor Swift, The Weeknd,
Rihanna and Billie Eilish. This dump was captured in 2021, before that, so its
artists.csv is a snapshot of genre data that is no longer obtainable. Joining
it back is the only way to get genre for a large fraction of the catalog.

Note what that implies: genre coverage is now a function of *when* a dataset
was captured, not of what the artist is. That is worth saying out loud in the
report, because it is the kind of silent drift that makes a model trained in
2021 and one trained today incomparable.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "kaggle_600k"
OUT = ROOT / "data" / "raw" / "tracks.csv"
ARTIST_OUT = ROOT / "data" / "raw" / "artist_genres.csv"
STORE_OUT = ROOT / "data" / "raw" / "feature_store.csv"

OK, WARN, DIM, END = "\033[92m", "\033[93m", "\033[2m", "\033[0m"


def first_id(raw: str) -> str | None:
    """`id_artists` arrives as the string "['abc', 'def']"."""
    if not isinstance(raw, str) or not raw.startswith("["):
        return raw if isinstance(raw, str) and raw else None
    try:
        ids = ast.literal_eval(raw)
        return ids[0] if ids else None
    except (ValueError, SyntaxError):
        return None


def first_genre(raw: str) -> str:
    if not isinstance(raw, str) or not raw.startswith("["):
        return "unknown"
    try:
        g = ast.literal_eval(raw)
        return g[0] if g else "unknown"
    except (ValueError, SyntaxError):
        return "unknown"


def clean_artist(raw: str) -> str:
    """`artists` is also a stringified list; keep it readable."""
    if not isinstance(raw, str) or not raw.startswith("["):
        return raw if isinstance(raw, str) else "unknown"
    try:
        a = ast.literal_eval(raw)
        return ", ".join(a) if a else "unknown"
    except (ValueError, SyntaxError):
        return raw


def merge_extra_sources(base: pd.DataFrame) -> pd.DataFrame:
    """Union additional feature dumps in, keyed on Spotify track id.

    Coverage is the binding constraint on this whole design, and it is worse
    than it looks: only 25.8% of our scraped tracks resolve against the 600k
    dump, because that dump stops at 2021 while zodiac playlists are full of
    recent music (missing tracks have a median release year of 2022 against
    2012 for resolved ones).

    No single dump fixes that, and none ever will -- /v1/audio-features died
    on 2024-11-27, so no dataset can contain features for anything released
    after it. Unioning several dumps captured at different times is the only
    lever we have. Drop any CSV with an `id` (or `track_id`) column plus audio
    features into data/raw/extra/ and it gets folded in here; rows already
    present win, so the base dump stays authoritative.
    """
    extra_dir = ROOT / "data" / "raw" / "extra"
    if not extra_dir.exists():
        return base

    files = sorted(p for p in extra_dir.glob("*.csv"))
    if not files:
        return base

    feats = ["danceability", "energy", "speechiness", "acousticness",
             "instrumentalness", "liveness", "valence", "tempo", "loudness",
             "duration_ms", "key", "mode", "time_signature"]

    for path in files:
        print(f"{DIM}folding in {path.name} …{END}")
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(f"  {WARN}skipped: {exc}{END}")
            continue

        df = df.rename(columns={"id": "track_id", "name": "track_name",
                                "artists": "artist", "album_name": "album"})
        missing = [f for f in feats if f not in df.columns]
        if "track_id" not in df.columns or len(missing) > 3:
            print(f"  {WARN}skipped: not a usable feature source "
                  f"(missing {missing[:4]}){END}")
            continue

        keep = ["track_id"] + [f for f in feats if f in df.columns]
        for c in ("track_name", "artist", "album", "genre", "popularity", "year"):
            if c in df.columns:
                keep.append(c)
        df = df[keep].drop_duplicates("track_id")

        new = df[~df["track_id"].isin(set(base["track_id"]))]
        print(f"  {len(df):,} rows, {len(new):,} track ids we did not already have")
        if len(new):
            base = pd.concat([base, new], ignore_index=True)

    return base


def main() -> None:
    if not (SRC / "tracks.csv").exists():
        print(f"{WARN}No Kaggle dump at {SRC}{END}")
        print("Download and unzip it there first -- see this file's docstring.")
        sys.exit(1)

    print(f"{DIM}reading tracks.csv …{END}")
    tracks = pd.read_csv(SRC / "tracks.csv")
    print(f"  {len(tracks):,} tracks, {len(tracks.columns)} columns")

    print(f"{DIM}reading artists.csv …{END}")
    artists = pd.read_csv(SRC / "artists.csv")
    print(f"  {len(artists):,} artists")

    artists["genre"] = artists["genres"].map(first_genre)
    have = (artists["genre"] != "unknown")
    print(f"  artists with a genre in the 2021 snapshot: {have.mean():.1%} "
          f"({have.sum():,})")

    # Keep the full artist->genre map; the scraper can use it to patch the
    # live API's gaps for artists that are not in the 600k track subset.
    artists.loc[have, ["id", "name", "genre", "genres", "followers", "popularity"]] \
        .to_csv(ARTIST_OUT, index=False)
    print(f"  wrote {ARTIST_OUT.name} ({have.sum():,} artists with genres)")

    print(f"{DIM}joining genre onto tracks …{END}")
    tracks["primary_artist_id"] = tracks["id_artists"].map(first_id)
    gmap = dict(zip(artists["id"], artists["genre"]))
    tracks["genre"] = tracks["primary_artist_id"].map(gmap).fillna("unknown")

    out = pd.DataFrame({
        "track_id": tracks["id"],
        "track_name": tracks["name"],
        "artist": tracks["artists"].map(clean_artist),
        "artist_id": tracks["primary_artist_id"],
        "album": "",
        "genre": tracks["genre"],
        "popularity": tracks["popularity"],
        "year": pd.to_numeric(tracks["release_date"].astype(str).str.slice(0, 4),
                              errors="coerce"),
    })
    for f in ["danceability", "energy", "speechiness", "acousticness",
              "instrumentalness", "liveness", "valence", "tempo", "loudness",
              "duration_ms", "key", "mode", "time_signature"]:
        out[f] = tracks[f]

    # Two outputs, deliberately. The CATALOG is the curated 600k: every row
    # has genre, year and popularity, and it is what the recommender draws
    # candidates from and what the notebooks analyse. The FEATURE STORE is the
    # union of every dump we have, audio features only, and exists for one
    # job -- looking up "what does this track sound like" for a track someone
    # actually listened to.
    #
    # Folding the extras into the catalog instead was a mistake I made and
    # reverted: it lifted scraped-track coverage from 25.8% to 42.9%, but
    # dropped catalog genre coverage from 89.6% to 37%, because the extra
    # dumps carry features without genre, year or popularity. Those are
    # different jobs and they want different tables.
    store = merge_extra_sources(out.copy())
    store_cols = ["track_id"] + [c for c in [
        "danceability", "energy", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "loudness",
        "duration_ms", "key", "mode", "time_signature"] if c in store.columns]
    store[store_cols].drop_duplicates("track_id").to_csv(STORE_OUT, index=False)
    print(f"  wrote {STORE_OUT.name}: {len(store):,} tracks with audio features")

    before = len(out)
    # Only track_id is mandatory. Extra sources often carry audio features
    # without a release year, and features are the reason we are here -- a
    # row with no year still answers "what does this song sound like", which
    # is what the deprecated endpoint used to tell us.
    out = out.dropna(subset=["track_id"]).drop_duplicates("track_id")
    print(f"  dropped {before - len(out):,} rows (missing id, or duplicate track)")
    no_year = out["year"].isna().sum()
    if no_year:
        print(f"  {no_year:,} rows have features but no release year")

    out.to_csv(OUT, index=False)

    out["genre"] = out["genre"].fillna("unknown")
    cov = (out["genre"] != "unknown").mean()
    print(f"\n{OK}wrote{END} {OUT.relative_to(ROOT)}  ({len(out):,} tracks)")
    print(f"  genre coverage      : {cov:.1%}")
    print(f"  year range          : {int(out.year.min())}-{int(out.year.max())}")
    print(f"  popularity (mean)   : {out.popularity.mean():.1f}")
    print(f"  distinct genres     : {out.genre.nunique():,}")
    print(f"\n  top genres: {', '.join(out[out.genre != 'unknown'].genre.value_counts().head(8).index)}")
    print(f"\n{DIM}Every notebook will now load this instead of synthetic data.{END}")


if __name__ == "__main__":
    main()
