"""Loaders. Prefer real data on disk; fall back to synthetic so nothing blocks.

Drop real files into ml/data/raw/ and every notebook picks them up on the next
run with no code change:

    ml/data/raw/tracks.csv    -- the Kaggle 600k song dump (any column naming
                                 close to Spotify's is auto-mapped)
    ml/data/raw/zodiac.csv    -- your scraped zodiac playlists
    ml/data/raw/listens.csv   -- per-user track histories
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schema as S
from . import synthetic as syn

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
INTERIM = Path(__file__).resolve().parents[2] / "data" / "interim"
ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"

# Kaggle dumps name things inconsistently; normalise to our schema.
_ALIASES = {
    "id": S.TRACK_ID, "spotify_id": S.TRACK_ID, "track id": S.TRACK_ID,
    "name": S.TRACK_NAME, "song": S.TRACK_NAME, "title": S.TRACK_NAME, "track": S.TRACK_NAME,
    "artists": S.ARTIST, "artist_name": S.ARTIST, "artistname": S.ARTIST,
    "album_name": S.ALBUM, "albumname": S.ALBUM,
    "track_genre": S.GENRE, "genres": S.GENRE,
    "release_year": S.YEAR, "release_date": S.YEAR,
    "duration": "duration_ms", "duration_min": "duration_ms",
}


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df = df.rename(columns={k: v for k, v in _ALIASES.items() if k in df.columns})
    if S.YEAR in df.columns:
        df[S.YEAR] = pd.to_numeric(
            df[S.YEAR].astype(str).str.slice(0, 4), errors="coerce"
        )
    return df


def load_catalog(n_synthetic: int = 6000, seed: int = 7, verbose: bool = True) -> pd.DataFrame:
    """The song catalog -- real Kaggle dump if present, else synthetic."""
    path = RAW / "tracks.csv"
    if path.exists():
        df = _normalise(pd.read_csv(path))
        missing = [c for c in S.AUDIO_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path} is missing required audio features: {missing}. "
                "Either add them or remove the file to fall back to synthetic data."
            )
        for col, default in [(S.GENRE, "unknown"), (S.POPULARITY, 50), (S.YEAR, 2015),
                             (S.ALBUM, ""), (S.ARTIST, "unknown"), (S.TRACK_NAME, "")]:
            if col not in df.columns:
                df[col] = default
        if S.TRACK_ID not in df.columns:
            df[S.TRACK_ID] = [f"R{i:07d}" for i in range(len(df))]
        if verbose:
            print(f"[data] REAL catalog: {len(df):,} tracks from {path.name}")
        return df
    if verbose:
        print(f"[data] SYNTHETIC catalog ({n_synthetic:,} tracks). "
              f"Drop a Kaggle dump at {path} to use real songs.")
    return syn.make_catalog(n_synthetic, seed=seed)


def attach_audio_features(df: pd.DataFrame, verbose: bool = True,
                          drop_unresolved: bool = True) -> pd.DataFrame:
    """Join audio features onto a frame that only has track ids.

    The scraper cannot collect audio features -- /v1/audio-features has
    returned 403 since 2024-11-27 -- so a real scrape arrives with genre,
    popularity and year but none of the 13 audio columns. Synthetic data has
    always had all of them. Without this join, every notebook written against
    synthetic data breaks the moment real data appears, which is exactly what
    happened.

    Prefers feature_store.csv (the union of every dump, ~1.4M tracks) over
    tracks.csv (the curated 586k), because the store resolves materially more.

    `drop_unresolved` keeps the contract that downstream code can rely on
    the audio columns being present and non-null, the way synthetic data
    always could. It is lossy and the loss is not random -- unresolved tracks
    skew recent and obscure -- so the coverage figure is printed every time
    rather than buried.
    """
    missing = [f for f in S.AUDIO_FEATURES if f not in df.columns]
    if not missing:
        return df

    for source in (RAW / "feature_store.csv", RAW / "tracks.csv"):
        if not source.exists():
            continue
        store = pd.read_csv(source, low_memory=False)
        # Only bring in what is actually absent, or pandas appends _x/_y
        # suffixes and the audio columns silently vanish under new names.
        bring = [f for f in S.AUDIO_FEATURES if f in store.columns and f not in df.columns]
        if not bring:
            continue

        before = df[S.TRACK_ID].nunique()
        out = df.merge(store[[S.TRACK_ID] + bring].drop_duplicates(S.TRACK_ID),
                       on=S.TRACK_ID, how="left")
        resolved = out.loc[out[bring[0]].notna(), S.TRACK_ID].nunique()
        if verbose:
            print(f"[data] audio features joined from {source.name}: "
                  f"{resolved:,}/{before:,} tracks ({resolved/max(before,1):.1%})")
        if drop_unresolved:
            out = out.dropna(subset=bring)
            if verbose:
                print(f"[data] kept {len(out):,} rows with complete features "
                      f"-- unresolved tracks skew recent and obscure")
        return out

    raise FileNotFoundError(
        "This frame has no audio features and there is no feature store to join "
        "from. Run: .venv/bin/python scripts/prepare_catalog.py")


def load_zodiac(catalog: pd.DataFrame | None = None, verbose: bool = True,
                with_features: bool = True, **kw) -> pd.DataFrame:
    """Zodiac-labelled playlist tracks, with audio features attached."""
    path = RAW / "zodiac.csv"
    if path.exists():
        df = _normalise(pd.read_csv(path))
        if S.STAR_SIGN not in df.columns:
            raise ValueError(f"{path} needs a '{S.STAR_SIGN}' column.")
        if verbose:
            print(f"[data] REAL zodiac set: {len(df):,} rows, "
                  f"{df[S.TRACK_ID].nunique():,} unique tracks")
        return attach_audio_features(df, verbose=verbose) if with_features else df
    catalog = load_catalog(verbose=False) if catalog is None else catalog
    if verbose:
        print(f"[data] SYNTHETIC zodiac playlists. Drop your scrape at {path} to use real ones.")
    return syn.make_zodiac_playlists(catalog, **kw)


def load_listens(catalog: pd.DataFrame | None = None, verbose: bool = True, **kw):
    """Per-user listening histories. Returns (listens, taste_or_None).

    `taste` is the ground-truth latent taste matrix -- available only for
    synthetic users. With real users it is None, and the similarity notebook
    falls back to its held-out-half evaluation instead.
    """
    path = RAW / "listens.csv"
    if path.exists():
        df = _normalise(pd.read_csv(path))
        if verbose:
            print(f"[data] REAL listens: {df[S.USER_ID].nunique():,} users, {len(df):,} rows "
                  "(no ground-truth taste matrix -- using held-out-half evaluation)")
        return df, None
    catalog = load_catalog(verbose=False) if catalog is None else catalog
    if verbose:
        print(f"[data] SYNTHETIC users. Drop real histories at {path} to use yours.")
    return syn.make_users(catalog, **kw)


def using_real_data() -> dict[str, bool]:
    return {
        "catalog": (RAW / "tracks.csv").exists(),
        "zodiac": (RAW / "zodiac.csv").exists(),
        "listens": (RAW / "listens.csv").exists(),
    }
