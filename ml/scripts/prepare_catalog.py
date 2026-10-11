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

    before = len(out)
    out = out.dropna(subset=["track_id", "year"])
    out = out.drop_duplicates("track_id")
    print(f"  dropped {before - len(out):,} rows (missing id/year, or duplicate track)")

    out.to_csv(OUT, index=False)

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
