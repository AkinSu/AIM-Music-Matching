"""Pulling real data -- and what Spotify took away on 2024-11-27.

READ THIS BEFORE PLANNING THE NEXT VERSION.

On 27 November 2024 Spotify removed these Web API endpoints for applications
that did not already hold extended access:

    /v1/audio-features        <- danceability, energy, valence, tempo, ...
    /v1/audio-analysis
    /v1/recommendations
    /v1/artists/{id}/related-artists
    /v1/browse/featured-playlists, /v1/browse/categories/{id}/playlists
    30-second preview_url on track objects

Every feature in the 2023 Lovify pipeline came from the first of those. A new
app key calling it today gets 403 Forbidden. The models are not wrong; their
input tap is closed.

WHAT STILL WORKS (and is enough to rebuild on):
    /v1/me/top/tracks, /v1/me/top/artists   (short/medium/long_term)
    /v1/me/tracks                            (saved/liked songs)
    /v1/me/player/recently-played
    /v1/artists?ids=...                      -> genres, popularity, followers
    /v1/tracks?ids=...                       -> popularity, release date, duration
    /v1/search

THE WAY FORWARD -- a local feature store. Audio features for the catalog are
still obtainable in bulk, just not from the live API:

    a) the Kaggle 600k dump you already use (it predates the shutdown and
       carries all 13 features, keyed by Spotify track_id) -- join on track_id
       and you recover features for any catalog song a user listens to;
    b) AcousticBrainz / MusicBrainz for open, permanently downloadable
       descriptors;
    c) compute your own with Essentia or librosa if you ever hold audio.

Anything the store misses falls back to the genre/popularity/era features.

A caution on how much that fallback costs. An earlier version of this comment
claimed the deprecation barely mattered, because genre overlap outscored every
audio-feature metric in the similarity notebook. That comparison turned out to
be circular -- the synthetic users were built from genre mixtures and truth was
defined as the distance between those mixtures, so the genre histogram was
estimating the answer directly. Under a generator where taste lives in audio
space the ranking inverts and the deprecated features are the signal.

So the real cost of losing /v1/audio-features is UNKNOWN until we measure it on
real scraped data. Build the feature store on that basis: not because the audio
features are expendable, but because they are no longer obtainable live and we
cannot yet rule out needing them.

Nothing in this module is required to run the notebooks; they work entirely
off data.py.
"""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd

from . import schema as S

API = "https://api.spotify.com/v1"


class SpotifyClient:
    """Thin read-only client over an existing OAuth access token.

    Get the token from the Flask app's session (lovify.py already stores it as
    session['first_user_token']).
    """

    def __init__(self, access_token: str, session=None):
        import requests

        self.s = session or requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {access_token}"})

    def _get(self, path: str, **params):
        for attempt in range(5):
            r = self.s.get(f"{API}{path}", params=params, timeout=30)
            if r.status_code == 429:  # rate limited -- obey Retry-After
                time.sleep(int(r.headers.get("Retry-After", 2)) + 1)
                continue
            if r.status_code == 403 and "audio-features" in path:
                raise PermissionError(
                    "/v1/audio-features is 403. This endpoint was deprecated on "
                    "2024-11-27 for apps without pre-existing extended access. "
                    "Use a local feature store instead -- see this module's docstring."
                )
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"giving up on {path} after repeated rate limits")

    # ── still available ──────────────────────────────────────────────────────
    def top_tracks(self, time_range: str = "medium_term", limit: int = 50) -> list[dict]:
        return self._get("/me/top/tracks", time_range=time_range, limit=limit)["items"]

    def saved_tracks(self, limit: int = 50, max_items: int = 200) -> list[dict]:
        out, offset = [], 0
        while len(out) < max_items:
            page = self._get("/me/tracks", limit=limit, offset=offset)
            items = [i["track"] for i in page["items"]]
            if not items:
                break
            out += items
            offset += len(items)
        return out[:max_items]

    def artists(self, ids: Iterable[str]) -> list[dict]:
        ids = list(dict.fromkeys(ids))
        out = []
        for i in range(0, len(ids), 50):
            out += self._get("/artists", ids=",".join(ids[i:i + 50]))["artists"]
        return out

    # ── deprecated; kept so the failure is explicit, not mysterious ──────────
    def audio_features(self, ids: Iterable[str]) -> list[dict]:
        """Will raise PermissionError on any app created after 2024-11-27."""
        ids = list(dict.fromkeys(ids))
        out = []
        for i in range(0, len(ids), 100):
            out += self._get("/audio-features", ids=",".join(ids[i:i + 100]))["audio_features"]
        return out


def library_to_frame(tracks: list[dict], user_id: str) -> pd.DataFrame:
    """Spotify track objects -> our schema, using only fields still served."""
    rows = []
    for t in tracks:
        if not t:
            continue
        rows.append({
            S.USER_ID: user_id,
            S.TRACK_ID: t["id"],
            S.TRACK_NAME: t.get("name", ""),
            S.ARTIST: ", ".join(a["name"] for a in t.get("artists", [])),
            S.ALBUM: (t.get("album") or {}).get("name", ""),
            S.POPULARITY: t.get("popularity", 0),
            S.YEAR: int(str((t.get("album") or {}).get("release_date", "2015"))[:4] or 2015),
            "duration_ms": t.get("duration_ms", 0),
            "artist_ids": [a["id"] for a in t.get("artists", [])],
        })
    return pd.DataFrame(rows)


def attach_genres(df: pd.DataFrame, client: SpotifyClient) -> pd.DataFrame:
    """Genre per track, via its primary artist.

    Worth collecting, though not for the reason originally given here: the
    claim that genre was our strongest similarity signal has been withdrawn
    as circular (notebook 02). The reasons that survive are duller and more
    durable -- /artists still returns 200 where /audio-features does not, and
    the calls cost almost nothing once artist ids are deduped across a run.
    Genre is a feature we can still get, not one shown to be best.
    """
    ids = {a for lst in df["artist_ids"] for a in lst}
    lookup = {a["id"]: (a.get("genres") or ["unknown"])[0] for a in client.artists(ids)}
    df = df.copy()
    df[S.GENRE] = [lookup.get(lst[0], "unknown") if lst else "unknown" for lst in df["artist_ids"]]
    return df


def attach_features_from_store(df: pd.DataFrame, store: pd.DataFrame,
                               on: str = S.TRACK_ID) -> tuple[pd.DataFrame, float]:
    """Join audio features from a local store (the Kaggle dump).

    This is the replacement for the dead /v1/audio-features call. Returns the
    joined frame and the COVERAGE -- the fraction of the user's tracks the
    store could resolve. Watch that number: coverage is the real constraint on
    this design, and it will be worst for exactly the obscure music that makes
    a user distinctive.
    """
    cols = [c for c in S.AUDIO_FEATURES if c in store.columns]
    merged = df.merge(store[[on] + cols].drop_duplicates(on), on=on, how="left")
    coverage = float(merged[cols[0]].notna().mean()) if cols else 0.0
    return merged, coverage
