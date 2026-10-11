"""Scraping zodiac playlists so the labels mean something.

The 2023 scrape took the top ~10 playlists per sign and stopped. Three things
went wrong, and this module is organised around fixing them.

1. SHALLOW SAMPLING. Ten playlists per sign is whatever Spotify's relevance
   ranking happens to put on page one, which is dominated by a handful of
   mega-playlists. We page deep (hundreds per sign) and across several query
   phrasings, because search rank is not a random sample of opinion.

2. NON-INDEPENDENT LABELS. Playlists copy each other. Ten "Aries" playlists
   agreeing is not ten votes if nine were cloned from the first. This module
   records enough provenance for dedup.py to collapse copies into clusters and
   count one cluster as one vote. That correction is the whole point of the
   rebuild; everything else here is plumbing that feeds it.

3. NO PROVENANCE. The old CSV was (track, sign). You could not tell a
   50-follower bot playlist from an editorial one, nor audit anything. Every
   row we write now carries the playlist id, owner, follower count,
   description, snapshot id, the query that surfaced it and its rank in those
   results -- see schema.PROVENANCE_COLUMNS.

4. CONTAMINATED CLASSES. Measured on the live API: searching "cancer"
   returns cancer-ward playlists and healing-frequency sine tones, "leo"
   returns artists called Leo and Leon. Unfiltered, the model learns those
   instead of astrology. filters.py decides what a title is actually about,
   and this module applies it BEFORE spending requests on a playlist's tracks.

5. NO CONTROL. There was nothing to compare 76% against except chance.
   `scrape_control` collects structurally similar, non-astrological playlists
   (study, gym, road trip...) and labels them with uniformly random signs. Run
   the identical pipeline on it: whatever accuracy comes out is the accuracy
   this method produces from NO signal at all. The zodiac result only means
   something as a distance from that number.

AUTH: Client Credentials, i.e. an app token with no user attached. That is the
right flow here -- we read public playlists, not anyone's library -- and it is
the flow scripts/check_spotify_access.py probes, so a green result there means
a scrape is viable.

WHAT WE CAN AND CANNOT COLLECT. Probed live against this app's credentials:

    200  /search type=track, /search type=playlist
    200  /playlists/{id}/tracks
    200  /artists   -> genres, popularity, followers
    200  /tracks    -> popularity, release date
    403  /audio-features, /audio-analysis
    404  /recommendations, /artists/{id}/related-artists,
         /browse/featured-playlists

So the audio features the 2023 pipeline was built on are gone (deprecated
2024-11-27 for apps without pre-existing extended access -- see spotify.py),
and this scraper is now the only place features for a scraped track can come
from. It therefore also collects, inline and cached:

    genre, artist_genres   from /artists (batched 50, deduped across the whole
                           scrape -- tracks share artists heavily, so this
                           costs a small number of calls)
    popularity, year       from the playlist-tracks payload, backfilled from
                           /tracks (batched 50) where it is missing

Collect them for the duller, more durable reasons: /artists and /tracks still
return 200 where /audio-features returns 403, and both are nearly free once
ids are deduped across the run. So the output CSV is directly modellable
without a Kaggle join -- not because these fields were shown to be the better
signal.

An earlier version of this comment justified them with notebook 02's finding
that genre overlap beat every audio-feature metric. That comparison has been
withdrawn as circular: the synthetic users were drawn from genre mixtures and
truth was defined as the distance between those same mixtures, so the genre
histogram was estimating the target directly. Under a generator with taste in
audio space the ranking inverts. What survives is that a BLEND has the best
worst-case score across both generators -- nobody yet knows which view of
taste real listeners inhabit, which is an argument for collecting genre
alongside a feature store, not instead of one.

Note the deprecated routes answer with a MIX of 403 and 404, so detection
keys on the path, not on the status code.

Two more measured quirks of /v1/search, both of which would otherwise cost a
day: the `total` field is useless (a bare "aries" query reports 32 hits while
paging six phrasings finds 135), so it is never used for planning or for
pagination bounds -- we page until Spotify stops returning items; and
Spotify's own-branded "Cosmic Playlists" never appear in search results at
all, so the owner-id guard against them is kept as cheap insurance rather
than relied upon.

SCALE: 12 signs x 200 playlists is ~2,400 playlist-metadata calls, ~5,000
track-page calls and a few hundred search calls. That run WILL be rate
limited part-way through, so it checkpoints to data/interim/ after every
playlist and can resume (see ScrapeCheckpoint).
"""

from __future__ import annotations

import base64
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import filters as F
from . import schema as S
from . import synthetic as syn
from .data import INTERIM

ACCOUNTS = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"

# Spotify rejects /v1/search when offset + limit > 1000, so no single query
# can ever yield more than 1000 playlists however patiently you page. That is
# a hard ceiling on one phrasing, which is the second reason for querying
# several phrasings per sign (the first being ranking bias).
SEARCH_OFFSET_CAP = 1000
SEARCH_PAGE = 50     # max limit accepted by /v1/search
TRACKS_PAGE = 100    # max limit accepted by /v1/playlists/{id}/tracks
ID_BATCH = 50        # max ids accepted by /v1/artists and /v1/tracks

# Paths removed on 2024-11-27 for apps without pre-existing extended access.
# Facts and wording follow spotify.py -- do not add to this list without a
# source. They answer with 403 (audio-features, audio-analysis) or 404
# (recommendations, related-artists, browse) depending on the route, which is
# why _get decides by path rather than by status code, and why a hit here is
# raised immediately instead of retried: expected-dead is not a transient
# failure.
DEPRECATED_PATHS = (
    "/audio-features",
    "/audio-analysis",
    "/recommendations",
    "related-artists",
    "/browse/featured-playlists",
    "/browse/categories",
)

# Several phrasings per sign, because one phrasing buys you one corner of the
# playlist space (and at most 1000 results). "songs for a leo" and "leo core"
# surface genuinely different curators; both are how people actually name
# these playlists.
QUERY_TEMPLATES = (
    "{sign}",
    "{sign} playlist",
    "{sign} vibes",
    "{sign} season",
    "{sign} energy",
    "{sign} aesthetic",
    "{sign} zodiac",
    "{sign} astrology",
    "{sign} mood",
    "{sign} core",
    "songs for a {sign}",
    "{sign} sun",
)

# Playlist themes for the negative control: ordinary, popular, structurally
# just like a zodiac playlist (one curator, one mood, 20-200 tracks) and with
# nothing whatever to do with astrology. Spread deliberately across activity,
# mood and setting, because a control drawn from one corner of the catalogue
# would be its own kind of confound.
CONTROL_THEMES = (
    "study", "gym", "road trip", "shower songs", "cleaning", "cooking",
    "rainy day", "party", "focus", "sad", "hype", "late night",
)

# Fixed column order: the checkpoint appends to CSV parts, and appending
# frames whose columns drift would silently misalign the file.
SCRAPE_COLUMNS = [
    S.STAR_SIGN, S.IS_CONTROL,
    S.PLAYLIST_ID, S.PLAYLIST_NAME, S.PLAYLIST_OWNER_ID, S.PLAYLIST_OWNER,
    S.PLAYLIST_FOLLOWERS, S.PLAYLIST_DESCRIPTION, S.PLAYLIST_SNAPSHOT,
    S.PLAYLIST_TRACK_TOTAL, S.SEARCH_QUERY, S.SEARCH_RANK, S.SCRAPED_AT,
    S.TRACK_ID, S.TRACK_NAME, S.ARTIST, S.ARTIST_ID, S.ALBUM,
    S.GENRE, S.ARTIST_GENRES,
    S.POPULARITY, S.YEAR, "duration_ms", S.TRACK_POSITION, S.ADDED_AT,
    S.TITLE_CLASS, S.TITLE_REASON,
]


class ScrapeError(RuntimeError):
    """Gave up on a request after exhausting retries."""


# ── token ─────────────────────────────────────────────────────────────────────

@dataclass
class _Token:
    value: str
    expires_at: float  # monotonic seconds

    def alive(self, now: float, margin: float = 60.0) -> bool:
        # Refresh a minute early: a token that expires mid-page turns into a
        # 401 in the middle of a 2,000-request scrape, which is noise we can
        # just avoid.
        return bool(self.value) and now < self.expires_at - margin


# ── the scraper ───────────────────────────────────────────────────────────────

class RateLimitedTooLong(RuntimeError):
    """Raised when Spotify's requested pause exceeds MAX_RETRY_AFTER."""


class SpotifyScraper:
    """Read-only Spotify client for a deep, resumable playlist scrape.

    `session`, `sleep` and `clock` are injectable so the whole class can be
    tested against a fake HTTP session with no network and no real waiting --
    see tests/test_scrape.py. Everything else defaults to requests + time.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        session=None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        max_attempts: int = 8,
        timeout: float = 30.0,
        verbose: bool = True,
    ):
        if session is None:
            import requests  # imported lazily so tests need no network stack
            session = requests.Session()
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = session
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.verbose = verbose
        self._token: _Token | None = None
        # Enrichment caches. Tracks share artists heavily -- a few thousand
        # playlists resolve to a few thousand distinct artists -- so caching
        # turns what looks like one lookup per track into a few hundred calls
        # for the whole scrape.
        self._artist_genres: dict[str, list[str]] = {}
        self._track_meta: dict[str, dict] = {}
        self.stats = {"requests": 0, "rate_limited": 0, "server_errors": 0,
                      "token_fetches": 0, "not_found": 0,
                      "artists_resolved": 0, "tracks_backfilled": 0,
                      "titles_rejected": 0}
        # Rejections are audit data, not noise: a filter nobody can inspect is
        # a filter nobody can correct. `rejections` counts (sign, reason) and
        # `rejected` keeps the titles themselves, capped so a 2,400-playlist
        # run cannot grow it without bound.
        self.rejections: Counter = Counter()
        self.rejected: list[dict] = []
        self._authenticate()

    # ── auth ────────────────────────────────────────────────────────────────
    def _authenticate(self) -> None:
        basic = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()).decode()
        r = self.session.post(
            ACCOUNTS,
            data={"grant_type": "client_credentials"},
            headers={"Authorization": f"Basic {basic}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout,
        )
        self.stats["token_fetches"] += 1
        if not getattr(r, "ok", False):
            body = str(getattr(r, "text", ""))[:300]
            raise ScrapeError(
                f"could not get an app token: {getattr(r, 'status_code', '?')} {body}\n"
                "Usual causes: wrong client id/secret, or the app was deleted."
            )
        payload = r.json()
        # Tokens last 3600s. Trust the response rather than the constant.
        ttl = float(payload.get("expires_in", 3600))
        self._token = _Token(payload["access_token"], self._clock() + ttl)

    def _auth_header(self) -> dict[str, str]:
        if self._token is None or not self._token.alive(self._clock()):
            self._authenticate()
        assert self._token is not None
        return {"Authorization": f"Bearer {self._token.value}"}

    # ── HTTP ────────────────────────────────────────────────────────────────
    # Longest single pause we will take on a 429 before giving up and
    # letting the caller resume later.
    #
    # Obeying Retry-After is not optional -- ignoring it is how an app key
    # earns a longer ban. But obeying it *without a cap* is how a scrape
    # turns into a silent hang: after ~7,500 requests in one session Spotify
    # handed us a Retry-After long enough that the process sat there for 15
    # minutes with no output, two sockets in CLOSE_WAIT and 2.8 seconds of
    # CPU. It looked exactly like a deadlock.
    #
    # Past this cap, stopping is better than sleeping. The checkpoint already
    # makes the work resumable, so --resume after the window expires costs
    # nothing, and the operator finds out immediately instead of an hour later.
    MAX_RETRY_AFTER = 180.0

    @staticmethod
    def _retry_after(response, default: float = 2.0) -> float:
        """Seconds to wait after a 429, from the header Spotify sends."""
        raw = (getattr(response, "headers", None) or {}).get("Retry-After")
        try:
            return float(raw) + 1.0  # +1s of slack for clock skew
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _explain_dead(path: str, code: int) -> str:
        return (
            f"{path} is {code}. This endpoint was deprecated on 2024-11-27 for "
            "apps without pre-existing extended access (a live probe of this "
            "app returned 403 for audio-features/audio-analysis and 404 for "
            "recommendations/related-artists/browse -- same cause, different "
            "status). Use a local feature store instead, and the genre, "
            "popularity and year this scraper collects from /artists and "
            "/tracks -- see spotify.py's module docstring."
        )

    def _explain_403(self, path: str) -> str:
        # Nothing the scraper needs was on the deprecated list, so a 403 here
        # is about these credentials, not the deprecation. Saying so saves an
        # hour of chasing the wrong cause.
        return (
            f"{path} is 403, and this endpoint was NOT among those removed on "
            "2024-11-27 (that list is /audio-features, /audio-analysis, "
            "/recommendations, related-artists and the /browse playlist "
            "endpoints -- see spotify.py). So this is about the app token, not "
            "the deprecation: check SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET "
            "in ml/.env and re-run scripts/check_spotify_access.py."
        )

    def _get(self, path: str, none_on_404: bool = False, **params):
        """One GET, with the retry policy the whole scrape depends on."""
        url = f"{API}{path}"
        backoff = 1.0
        refreshed = False
        for _ in range(self.max_attempts):
            r = self.session.get(url, params=params, headers=self._auth_header(),
                                 timeout=self.timeout)
            code = int(getattr(r, "status_code", 0))
            self.stats["requests"] += 1

            if code == 429:
                self.stats["rate_limited"] += 1
                wait = self._retry_after(r)
                if wait > self.MAX_RETRY_AFTER:
                    raise RateLimitedTooLong(
                        f"Spotify asked for a {wait:.0f}s pause, past the "
                        f"{self.MAX_RETRY_AFTER:.0f}s cap. Stopping so the wait is "
                        "visible rather than silent -- progress is checkpointed, so "
                        "re-run with --resume once the window has passed.")
                print(f"    rate limited, waiting {wait:.0f}s", flush=True)
                self._sleep(wait)
                continue
            if code == 401 and not refreshed:
                # Token died early (revoked, or clock skew). One free refresh.
                refreshed = True
                self._token = None
                continue
            if code in (403, 404) and any(dead in path for dead in DEPRECATED_PATHS):
                # Expected-dead, and dead in two different ways depending on
                # the route. Never retried.
                raise PermissionError(self._explain_dead(path, code))
            if code == 403:
                raise PermissionError(self._explain_403(path))
            if code == 404:
                self.stats["not_found"] += 1
                if none_on_404:
                    return None
                raise ScrapeError(f"404 on {path}")
            if 500 <= code < 600:
                self.stats["server_errors"] += 1
                self._sleep(backoff)
                backoff *= 2  # exponential: 1, 2, 4, 8 ...
                continue
            r.raise_for_status()
            return r.json()
        raise ScrapeError(
            f"giving up on {path} after {self.max_attempts} attempts "
            f"({self.stats['rate_limited']} rate limits, "
            f"{self.stats['server_errors']} server errors so far)"
        )

    # ── search ──────────────────────────────────────────────────────────────
    def search_playlists(self, query: str, max_results: int = 200) -> list[dict]:
        """Deep-page /v1/search for playlists.

        Two Spotify quirks are handled here rather than by every caller:

        * offset + limit > 1000 is rejected outright, so we clamp the final
          page's limit and stop at the cap. One query can therefore never
          return more than 1000 playlists no matter what `max_results` says.
        * `playlists.items` contains None entries -- Spotify's own response
          includes holes where a playlist has become unavailable. They are
          filtered, not counted.

        Returns dicts carrying the playlist's rank in these results, since
        rank is itself provenance (position 1 of "aries" is a very different
        artefact from position 700).
        """
        out: list[dict] = []
        seen: set[str] = set()
        offset = 0
        while len(out) < max_results and offset < SEARCH_OFFSET_CAP:
            limit = min(SEARCH_PAGE, max_results - len(out),
                        SEARCH_OFFSET_CAP - offset)
            if limit <= 0:
                break
            page = self._get("/search", q=query, type="playlist",
                             limit=limit, offset=offset)
            block = (page or {}).get("playlists") or {}
            raw_items = block.get("items") or []
            # Enumerate over the RAW page so a rank still means "position in
            # these search results" even where Spotify left None holes.
            for slot, it in enumerate(raw_items):
                if not it:
                    continue
                pid = it.get("id")
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                out.append({
                    "id": pid,
                    "name": it.get("name") or "",
                    "owner_id": ((it.get("owner") or {}).get("id") or ""),
                    "owner_name": ((it.get("owner") or {}).get("display_name") or ""),
                    "track_total": ((it.get("tracks") or {}).get("total") or 0),
                    "query": query,
                    "rank": offset + slot,
                })
            # Trust the page size we asked for, not len(items): None entries
            # still occupy their offset slot, so advancing by len(items) would
            # re-request the same window forever.
            offset += limit
            if not block.get("next") or not raw_items:
                break
        return out[:max_results]

    # ── playlists ───────────────────────────────────────────────────────────
    def playlist(self, playlist_id: str) -> dict | None:
        """Metadata for one playlist, or None if it has vanished (404).

        `fields` keeps the response small: the default playlist object embeds
        the first 100 tracks, which we fetch separately anyway. snapshot_id is
        requested because it is the only honest way to say WHICH version of a
        playlist a row came from -- playlists mutate, and a scrape without it
        is unreproducible.
        """
        fields = ("id,name,description,snapshot_id,public,collaborative,"
                  "followers(total),owner(id,display_name),tracks(total)")
        raw = self._get(f"/playlists/{playlist_id}", none_on_404=True, fields=fields)
        if not raw:
            return None
        return {
            "id": raw.get("id") or playlist_id,
            "name": raw.get("name") or "",
            "description": raw.get("description") or "",
            "snapshot_id": raw.get("snapshot_id") or "",
            "owner_id": ((raw.get("owner") or {}).get("id") or ""),
            "owner_name": ((raw.get("owner") or {}).get("display_name") or ""),
            "followers": int(((raw.get("followers") or {}).get("total") or 0)),
            "track_total": int(((raw.get("tracks") or {}).get("total") or 0)),
            "public": raw.get("public"),
            "collaborative": bool(raw.get("collaborative")),
        }

    def playlist_tracks(self, playlist_id: str, max_tracks: int = 1000) -> list[dict]:
        """Paginated track objects for a playlist.

        Skipped, with reasons, because each would otherwise poison the data:
          * item is None, or item["track"] is None -- deleted/unavailable
            tracks leave holes in the response;
          * is_local -- a file on the curator's machine. It has no Spotify id,
            so it can never join to a feature store and it is not a track
            anyone else can hear;
          * type != "track" -- podcast episodes live in playlists too;
          * no id -- nothing downstream can key on it.
        """
        out: list[dict] = []
        offset = 0
        fields = ("items(added_at,is_local,track(id,name,type,duration_ms,popularity,"
                  "artists(id,name),album(name,release_date))),next,total")
        while len(out) < max_tracks:
            limit = min(TRACKS_PAGE, max_tracks - len(out))
            page = self._get(f"/playlists/{playlist_id}/tracks", none_on_404=True,
                             limit=limit, offset=offset, fields=fields,
                             additional_types="track")
            if not page:
                break
            items = page.get("items") or []
            for pos, item in enumerate(items):
                if not item or item.get("is_local"):
                    continue
                t = item.get("track")
                if not t or not t.get("id") or (t.get("type") or "track") != "track":
                    continue
                out.append({**t, "added_at": item.get("added_at"),
                            "position": offset + pos})
            offset += limit
            if not page.get("next") or not items:
                break
        return out[:max_tracks]

    # ── enrichment: the features that still exist ───────────────────────────
    def artist_genres(self, artist_ids: Iterable[str]) -> dict[str, list[str]]:
        """artist_id -> genre list, via /v1/artists in batches of 50.

        Worth collecting, though not for the reason first given here: the claim
        that genre was our strongest similarity signal has been withdrawn as
        circular (notebook 02). The surviving reasons are plainer -- /artists
        still returns 200 where /audio-features does not, and the calls cost
        almost nothing once artist ids are deduped across the run. Genre is a
        feature we can still get, not one shown to be best.

        Results are cached for the lifetime of the scraper (and persisted by
        ScrapeCheckpoint), so a track by an already-seen artist costs zero
        requests.
        """
        wanted = [a for a in dict.fromkeys(artist_ids) if a and a not in self._artist_genres]
        for i in range(0, len(wanted), ID_BATCH):
            batch = wanted[i:i + ID_BATCH]
            payload = self._get("/artists", ids=",".join(batch)) or {}
            for art in payload.get("artists") or []:
                if art and art.get("id"):
                    self._artist_genres[art["id"]] = list(art.get("genres") or [])
            # Spotify silently omits ids it cannot resolve. Cache the miss as
            # an empty list, or every later playlist retries the same dead id.
            for aid in batch:
                self._artist_genres.setdefault(aid, [])
            self.stats["artists_resolved"] += len(batch)
        return {a: self._artist_genres.get(a, []) for a in artist_ids if a}

    def track_meta(self, track_ids: Iterable[str]) -> dict[str, dict]:
        """track_id -> {popularity, year}, via /v1/tracks in batches of 50.

        Only used as a BACKFILL: the playlist-tracks payload already carries
        popularity and album.release_date for most tracks, so asking /tracks
        about all of them would double the request count for nothing.
        """
        wanted = [t for t in dict.fromkeys(track_ids) if t and t not in self._track_meta]
        for i in range(0, len(wanted), ID_BATCH):
            batch = wanted[i:i + ID_BATCH]
            payload = self._get("/tracks", ids=",".join(batch)) or {}
            for t in payload.get("tracks") or []:
                if not t or not t.get("id"):
                    continue
                year = str((t.get("album") or {}).get("release_date") or "")[:4]
                self._track_meta[t["id"]] = {
                    S.POPULARITY: t.get("popularity"),
                    S.YEAR: int(year) if year.isdigit() else None,
                }
            for tid in batch:
                self._track_meta.setdefault(tid, {})
            self.stats["tracks_backfilled"] += len(batch)
        return {t: self._track_meta.get(t, {}) for t in track_ids if t}

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fill genre / artist_genres, and backfill popularity + year.

        Genre comes from the PRIMARY artist, matching spotify.attach_genres so
        the two paths agree; artist_genres keeps that artist's whole list,
        because collapsing five genres to one is most of the signal thrown
        away. An artist with no genres at all is common on Spotify (small or
        very new artists) and becomes "unknown" rather than NaN, so grouping
        by genre never silently drops those rows.
        """
        if frame.empty:
            return frame
        out = frame.copy()
        lookup = self.artist_genres(out[S.ARTIST_ID].dropna().astype(str).unique())
        genres = out[S.ARTIST_ID].astype(str).map(lambda a: lookup.get(a) or [])
        out[S.GENRE] = [g[0] if g else "unknown" for g in genres]
        out[S.ARTIST_GENRES] = ["|".join(g) for g in genres]

        missing = out[out[S.POPULARITY].isna() | out[S.YEAR].isna()][S.TRACK_ID]
        if len(missing):
            meta = self.track_meta(missing.dropna().astype(str).unique())
            for col in (S.POPULARITY, S.YEAR):
                filled = out[S.TRACK_ID].astype(str).map(
                    lambda t: (meta.get(t) or {}).get(col))
                out[col] = out[col].where(out[col].notna(), filled)
        return out

    # ── one sign ────────────────────────────────────────────────────────────
    def scrape_sign(
        self,
        sign: str,
        n_playlists: int = 200,
        checkpoint: "ScrapeCheckpoint | None" = None,
        queries: Sequence[str] | None = None,
        min_tracks: int = F.MIN_TRACKS,
        max_tracks: int = F.MAX_TRACKS,
        min_followers: int = 0,
        max_tracks_per_playlist: int = 500,
        filter_titles: bool = True,
        strict_titles: bool | None = None,
    ) -> pd.DataFrame:
        """All rows for one sign: (sign, playlist provenance, track).

        Candidates come from every query template, interleaved round-robin
        rather than concatenated, so `n_playlists` is spread across phrasings
        instead of being eaten by the first query's first 200 hits.

        `min_tracks` drops near-empty playlists (nothing to learn, and they
        distort Jaccard: a 2-track playlist is a near-duplicate of anything
        containing those 2 tracks). `min_followers` defaults to 0 -- we RECORD
        follower counts and let the analysis decide, because filtering spam at
        scrape time throws away the evidence that it was spam.

        Each playlist's rows are enriched (genre from /artists, popularity and
        year backfilled from /tracks) BEFORE being checkpointed, so the parts
        on disk are already modellable and a resumed run never has to revisit
        them.

        The title filter runs on the playlist METADATA, before any track page
        is requested. That ordering is worth most of the scrape's cost: ~8 of
        9 "cancer" results and ~8 of 9 "leo" results are rejected, and each
        rejection saves one to five track-page calls. Rejections are recorded
        on the scraper (`rejections`, `rejected`) so the CLI can print an
        audit instead of silently shrinking the dataset.
        """
        queries = list(queries or [t.format(sign=sign.lower()) for t in QUERY_TEMPLATES])
        per_query = max(SEARCH_PAGE, math.ceil(2.0 * n_playlists / max(1, len(queries))))

        pools = []
        for q in queries:
            try:
                pools.append(self.search_playlists(q, max_results=per_query))
            except ScrapeError as exc:   # one dead phrasing must not kill the sign
                self._log(f"    search {q!r} failed: {exc}")
                pools.append([])

        candidates: list[dict] = []
        seen: set[str] = set()
        for row in _round_robin(pools):
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            candidates.append(row)

        if checkpoint:
            # Re-use artist genres resolved by earlier runs: on a resume this
            # is the difference between a few calls and a few hundred.
            self._artist_genres.update(checkpoint.load_artist_genres())

        already = checkpoint.done_playlists(sign) if checkpoint else set()
        todo = [c for c in candidates if c["id"] not in already][:max(0, n_playlists - len(already))]
        self._log(f"  {sign}: {len(candidates)} distinct playlists found, "
                  f"{len(already)} already on disk, fetching {len(todo)}")

        frames: list[pd.DataFrame] = []
        for i, cand in enumerate(todo, 1):
            meta = self.playlist(cand["id"])
            if meta is None:
                continue
            if meta["followers"] < min_followers:
                continue
            # Spotify's own branded playlists are editorial, not opinions from
            # 12 independent curators -- and measurement says they never
            # surface in search anyway, so this is cheap insurance.
            if meta.get("owner_id") == "spotify":
                self._reject(sign, meta, "spotify_owned")
                continue
            title_class, reason = F.classify_playlist(
                meta["name"], sign if filter_titles else None,
                meta["track_total"], strict=strict_titles,
                min_tracks=min_tracks, max_tracks=max_tracks)
            if title_class == F.REJECT:
                self._reject(sign, meta, reason)
                continue
            tracks = self.playlist_tracks(cand["id"], max_tracks=max_tracks_per_playlist)
            # Re-check the size against what we actually hold: `tracks.total`
            # counts entries we drop (local files, episodes, dead tracks).
            if len(tracks) < min_tracks:
                self._reject(sign, meta, "too_few_tracks")
                continue
            frame = self.enrich(tracks_to_frame(
                sign, meta, tracks, query=cand["query"], rank=cand["rank"],
                title_class=title_class, title_reason=reason))
            frames.append(frame)
            if checkpoint:
                # Flush per playlist. A scrape that dies at playlist 1,900 of
                # 2,400 must not lose the first 1,899.
                checkpoint.append(sign, frame, playlist_id=cand["id"])
            if self.verbose and i % 25 == 0:
                self._log(f"    {i}/{len(todo)} playlists, {self.stats['requests']} requests, "
                          f"{self.stats['rate_limited']} rate limits")

        if checkpoint:
            checkpoint.save_artist_genres(self._artist_genres)
            checkpoint.mark_done(sign)
            return checkpoint.load_sign(sign)
        if not frames:
            return pd.DataFrame(columns=SCRAPE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    # ── the negative control ────────────────────────────────────────────────
    def scrape_control(
        self,
        n_playlists: int = 200,
        themes: Sequence[str] | None = None,
        checkpoint: "ScrapeCheckpoint | None" = None,
        seed: int = 0,
        match_size_to: pd.DataFrame | None = None,
        min_tracks: int = F.MIN_TRACKS,
        max_tracks: int = F.MAX_TRACKS,
        max_tracks_per_playlist: int = 500,
    ) -> pd.DataFrame:
        """Playlists with nothing to do with astrology, labelled at random.

        THIS IS THE EXPERIMENT THAT DECIDES THE PROJECT. Run the identical
        pipeline on this frame. Whatever accuracy comes out is what this
        method produces from a dataset containing, by construction, zero
        signal -- so it is the floor the real zodiac result has to clear.
        Chance is 8.3%, but the honest floor is usually above chance, because
        popularity structure, duplicate tracks and class imbalance all leak a
        little information that has nothing to do with the label.

        Three design points that matter for the comparison to be fair:

        * Labels are assigned PER PLAYLIST, not per row. A curator is the unit
          of observation in the real set, so the control must share that
          structure or the group-aware split is not testing the same thing.
        * `match_size_to` takes the real zodiac frame and draws each control
          playlist's length from ITS distribution of playlist sizes. Without
          that the control could be systematically shorter or longer, and
          playlist length alone changes both Jaccard clustering and how often
          a track recurs.
        * Only the universal title rules apply (sign=None in filters), so a
          "study frequencies 528hz" playlist is still excluded -- it is sine
          tones in either dataset.
        """
        rng = np.random.default_rng(seed)
        themes = list(themes or CONTROL_THEMES)
        per_theme = max(SEARCH_PAGE, math.ceil(2.0 * n_playlists / max(1, len(themes))))

        pools = []
        for theme in themes:
            try:
                pools.append(self.search_playlists(theme, max_results=per_theme))
            except ScrapeError as exc:
                self._log(f"    control search {theme!r} failed: {exc}")
                pools.append([])

        candidates, seen = [], set()
        for row in _round_robin(pools):
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            candidates.append(row)

        if checkpoint:
            self._artist_genres.update(checkpoint.load_artist_genres())
            already = checkpoint.all_done_playlists()
            candidates = [c for c in candidates if c["id"] not in already]
            n_playlists = max(0, n_playlists - len(already))

        sizes = None
        if match_size_to is not None and len(match_size_to):
            sizes = (match_size_to.groupby(S.PLAYLIST_ID)[S.TRACK_ID]
                     .nunique().to_numpy())

        self._log(f"  control: {len(candidates)} distinct playlists found, "
                  f"fetching up to {n_playlists}")
        frames: list[pd.DataFrame] = []
        kept = 0
        for cand in candidates:
            if kept >= n_playlists:
                break
            meta = self.playlist(cand["id"])
            if meta is None:
                continue
            title_class, reason = F.classify_playlist(
                meta["name"], None, meta["track_total"],
                min_tracks=min_tracks, max_tracks=max_tracks)
            if title_class == F.REJECT:
                self._reject(None, meta, reason)
                continue
            want = max_tracks_per_playlist
            if sizes is not None:
                want = int(min(max_tracks_per_playlist,
                               rng.choice(sizes)))      # match the real size mix
            tracks = self.playlist_tracks(cand["id"], max_tracks=want)
            if len(tracks) < min_tracks:
                self._reject(None, meta, "too_few_tracks")
                continue
            fake_sign = str(rng.choice(S.ZODIAC_SIGNS))  # the whole point
            frame = self.enrich(tracks_to_frame(
                fake_sign, meta, tracks, query=cand["query"], rank=cand["rank"],
                is_control=True, title_class=title_class, title_reason=reason))
            frames.append(frame)
            kept += 1
            if checkpoint:
                checkpoint.append(fake_sign, frame, playlist_id=cand["id"])
            if self.verbose and kept % 25 == 0:
                self._log(f"    {kept} control playlists, "
                          f"{self.stats['requests']} requests")

        if checkpoint:
            checkpoint.save_artist_genres(self._artist_genres)
            for sign in S.ZODIAC_SIGNS:
                checkpoint.mark_done(sign)
            return checkpoint.load_all()
        if not frames:
            return pd.DataFrame(columns=SCRAPE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def scrape_all(
        self,
        signs: Sequence[str] | None = None,
        n_playlists: int = 200,
        checkpoint: "ScrapeCheckpoint | None" = None,
        **kw,
    ) -> pd.DataFrame:
        """Every sign, checkpointing as it goes. Safe to re-run after a crash."""
        signs = list(signs or S.ZODIAC_SIGNS)
        frames = []
        for sign in signs:
            if checkpoint and checkpoint.is_done(sign):
                self._log(f"  {sign}: already complete, skipping")
                frames.append(checkpoint.load_sign(sign))
                continue
            frames.append(self.scrape_sign(sign, n_playlists=n_playlists,
                                           checkpoint=checkpoint, **kw))
        frames = [f for f in frames if len(f)]
        if not frames:
            return pd.DataFrame(columns=SCRAPE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _reject(self, sign: str | None, meta: dict, reason: str) -> None:
        self.stats["titles_rejected"] += 1
        self.rejections[(sign or "control", reason)] += 1
        if len(self.rejected) < 500:   # enough to audit, bounded for memory
            self.rejected.append({
                S.STAR_SIGN: sign or "control",
                S.PLAYLIST_ID: meta.get("id", ""),
                S.PLAYLIST_NAME: meta.get("name", ""),
                S.PLAYLIST_OWNER: meta.get("owner_name", ""),
                "n_tracks": meta.get("track_total", 0),
                S.TITLE_REASON: reason,
            })

    def rejection_frame(self) -> pd.DataFrame:
        """Rejections per (sign, reason), with an example title each."""
        rows = [{S.STAR_SIGN: sign, S.TITLE_REASON: reason, "playlists": n}
                for (sign, reason), n in sorted(self.rejections.items())]
        out = pd.DataFrame(rows, columns=[S.STAR_SIGN, S.TITLE_REASON, "playlists"])
        if not len(out) or not self.rejected:
            return out
        examples = (pd.DataFrame(self.rejected)
                    .groupby([S.STAR_SIGN, S.TITLE_REASON])[S.PLAYLIST_NAME]
                    .first().rename("example").reset_index())
        return out.merge(examples, on=[S.STAR_SIGN, S.TITLE_REASON], how="left")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


def _round_robin(pools: Iterable[Sequence[dict]]) -> list[dict]:
    """Interleave ranked lists so every query contributes its best hits first."""
    pools = [list(p) for p in pools]
    out = []
    for i in range(max((len(p) for p in pools), default=0)):
        for p in pools:
            if i < len(p):
                out.append(p[i])
    return out


def tracks_to_frame(sign: str, meta: dict, tracks: list[dict],
                    query: str = "", rank: int = -1, is_control: bool = False,
                    title_class: str = F.KEEP, title_reason: str = "") -> pd.DataFrame:
    """Spotify track objects + playlist metadata -> rows in our schema.

    `title_class` travels with the rows rather than being recomputed later:
    the verdict belongs to the title at scrape time, and titles change.
    """
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for t in tracks:
        if not t or not t.get("id"):
            continue
        artists = t.get("artists") or []
        release = str((t.get("album") or {}).get("release_date") or "")[:4]
        rows.append({
            S.STAR_SIGN: sign,
            S.IS_CONTROL: is_control,
            S.PLAYLIST_ID: meta["id"],
            S.PLAYLIST_NAME: meta.get("name", ""),
            S.PLAYLIST_OWNER_ID: meta.get("owner_id", ""),
            S.PLAYLIST_OWNER: meta.get("owner_name", ""),
            S.PLAYLIST_FOLLOWERS: meta.get("followers", 0),
            # Newlines in a description would break the CSV row; curators use
            # them freely.
            S.PLAYLIST_DESCRIPTION: " ".join((meta.get("description") or "").split()),
            S.PLAYLIST_SNAPSHOT: meta.get("snapshot_id", ""),
            S.PLAYLIST_TRACK_TOTAL: meta.get("track_total", 0),
            S.SEARCH_QUERY: query,
            S.SEARCH_RANK: rank,
            S.SCRAPED_AT: stamp,
            S.TRACK_ID: t["id"],
            S.TRACK_NAME: t.get("name") or "",
            S.ARTIST: ", ".join(a.get("name", "") for a in artists),
            S.ARTIST_ID: (artists[0].get("id") if artists else "") or "",
            S.ALBUM: ((t.get("album") or {}).get("name") or ""),
            S.POPULARITY: t.get("popularity"),
            S.YEAR: int(release) if release.isdigit() else None,
            "duration_ms": t.get("duration_ms"),
            S.TRACK_POSITION: t.get("position", -1),
            S.ADDED_AT: t.get("added_at") or "",
            S.TITLE_CLASS: title_class,
            S.TITLE_REASON: title_reason,
        })
    return pd.DataFrame(rows, columns=SCRAPE_COLUMNS)


# ── checkpointing ─────────────────────────────────────────────────────────────

class ScrapeCheckpoint:
    """Crash-and-resume state for a long scrape, under data/interim/.

    One CSV part per sign plus a small JSON index. Appending to a per-sign CSV
    after every playlist costs nothing next to the HTTP round trips it
    protects, and the parts stay inspectable mid-run -- which matters when the
    run takes an hour and you want to know whether it is working.

    The index records which playlist ids are already on disk, so a resumed run
    re-searches (cheap, and search results drift anyway) but never re-fetches
    a playlist's tracks.
    """

    def __init__(self, root: Path | None = None, tag: str = "zodiac"):
        self.dir = Path(root or INTERIM) / f"{tag}_scrape"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self.state: dict = {"signs": {}, "updated": None}
        if self.index_path.exists():
            try:
                self.state = json.loads(self.index_path.read_text())
            except json.JSONDecodeError:
                # A half-written index must not be fatal; the CSV parts are
                # the real data and can be re-indexed from disk.
                self.state = {"signs": {}, "updated": None}
            self._reindex_from_parts()

    # ── paths / state ───────────────────────────────────────────────────────
    def part(self, sign: str) -> Path:
        return self.dir / f"{sign.lower()}.csv"

    def _sign_state(self, sign: str) -> dict:
        return self.state["signs"].setdefault(sign, {"done": False, "playlists": []})

    def _reindex_from_parts(self) -> None:
        for sign in list(S.ZODIAC_SIGNS):
            path = self.part(sign)
            if not path.exists():
                continue
            st = self._sign_state(sign)
            try:
                ids = pd.read_csv(path, usecols=[S.PLAYLIST_ID])[S.PLAYLIST_ID]
            except (ValueError, pd.errors.EmptyDataError):
                continue
            st["playlists"] = sorted(set(st["playlists"]) | set(ids.astype(str)))

    def done_playlists(self, sign: str) -> set[str]:
        return set(self._sign_state(sign)["playlists"])

    def is_done(self, sign: str) -> bool:
        return bool(self._sign_state(sign)["done"])

    def mark_done(self, sign: str) -> None:
        self._sign_state(sign)["done"] = True
        self._save()

    def _save(self) -> None:
        self.state["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.index_path.write_text(json.dumps(self.state, indent=1))

    # ── data ────────────────────────────────────────────────────────────────
    def append(self, sign: str, frame: pd.DataFrame, playlist_id: str | None = None) -> None:
        if len(frame):
            path = self.part(sign)
            frame[SCRAPE_COLUMNS].to_csv(path, mode="a", index=False,
                                         header=not path.exists())
        st = self._sign_state(sign)
        for pid in ({playlist_id} if playlist_id else set(frame[S.PLAYLIST_ID].unique())):
            if pid and pid not in st["playlists"]:
                st["playlists"].append(str(pid))
        self._save()

    # ── the artist-genre cache ──────────────────────────────────────────────
    # Kept with the checkpoint rather than in the output CSV because it is
    # keyed by artist, not by row: a resumed run should not pay again for
    # genres it has already resolved.
    @property
    def artist_cache_path(self) -> Path:
        return self.dir / "artist_genres.json"

    def load_artist_genres(self) -> dict[str, list[str]]:
        if not self.artist_cache_path.exists():
            return {}
        try:
            return json.loads(self.artist_cache_path.read_text())
        except json.JSONDecodeError:
            return {}

    def save_artist_genres(self, cache: dict[str, list[str]]) -> None:
        merged = {**self.load_artist_genres(), **cache}
        self.artist_cache_path.write_text(json.dumps(merged))

    def all_done_playlists(self) -> set[str]:
        """Every playlist id on disk, whatever sign it landed under.

        The control set needs this: its labels are random, so a playlist
        already fetched under a fake "Leo" must not be fetched again when the
        coin comes up Virgo.
        """
        out: set[str] = set()
        for sign in S.ZODIAC_SIGNS:
            out |= self.done_playlists(sign)
        return out

    def load_sign(self, sign: str) -> pd.DataFrame:
        path = self.part(sign)
        if not path.exists():
            return pd.DataFrame(columns=SCRAPE_COLUMNS)
        df = pd.read_csv(path)
        # Forward compatibility: a part written before a column existed must
        # still load, rather than failing the whole resumed run.
        for col in SCRAPE_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[SCRAPE_COLUMNS + [c for c in df.columns if c not in SCRAPE_COLUMNS]]

    def load_all(self) -> pd.DataFrame:
        frames = [self.load_sign(s) for s in S.ZODIAC_SIGNS]
        frames = [f for f in frames if len(f)]
        if not frames:
            return pd.DataFrame(columns=SCRAPE_COLUMNS)
        return pd.concat(frames, ignore_index=True)


# ── offline stand-in ──────────────────────────────────────────────────────────

# Real titles, verbatim from the live API, for the dry run to filter. Using
# invented contamination would test the filter against our own imagination;
# these are the ones that actually came back.
CONTAMINATED_TITLES: dict[str, tuple[str, ...]] = {
    "Cancer": (
        "pov: my mom has cancer",
        "Cancer fighting songs",
        "Cancer Survivor Fight Songs",
        "Healing Cancer Frequencies",
        "POWERFUL KILLING CANCER CELLS WITH SOUND 10000HZ+741",
    ),
    "Leo": (
        "LEO DAN MIX😍EXITOS",
        "CARIN LEON MIX",
        "LEONARDO FAVIO ÉXITOS",
        "BEST OF LEON BRIDGES",
        "LOS INVASORES DE NUEVO LEON",
    ),
    # From the second, larger measurement (85-131 titles per sign). These are
    # the collisions a sign's own name does NOT give away -- the reason
    # filters.SIGN_REJECT needs targeted patterns rather than strict mode.
    "Libra": (
        "Grupo Libra – Puras Romanticas",
        "GRUPO LIBRA RANCHERAS",
        "El Grupo Libra-Éxitos Inmortales",
        "Sergio Mendivil y Grupo Libra-Puras Romanticas-",
        "Ladrón,grupo libra,Mandingo,los rehenes,bronco",
    ),
    "Virgo": (
        "Best Of Romain Virgo",
        "VIRGO’S GROOVE",
    ),
    "Gemini": (
        "Gemini Home Entertainment",
    ),
}

# Genuine titles from the same measurement, planted so the dry run also proves
# the filter does NOT eat them. Strict mode would reject every one of these,
# which is exactly why strict mode is not the default.
GENUINE_TITLES: dict[str, tuple[str, ...]] = {
    "Taurus": ("Taurus Playlist (Made by a Taurus)", "✨Taurus Shit✨",
               "2026 TAURUS"),
    "Scorpio": ("she's a scorpio",
                "scorpio woman tries to enter her dark feminine main character"),
    "Pisces": ("Can you tell I'm a Pisces? 🤍🥲",
               "the real pisces playlist", "Pisces core💞"),
    "Libra": ("songs for a libra",),
    "Gemini": ("gemini radio",),
}

# The real one has 4,078 tracks. Any size over filters.MAX_TRACKS exercises
# the same rule, and the dry run's catalogue is smaller than 4,078.
OVERSIZED_TITLE = "cancer energy♋️💥"
OVERSIZED_TRACKS = 600


def synthetic_scrape(
    catalog: pd.DataFrame | None = None,
    signs: Sequence[str] | None = None,
    n_playlists_per_sign: int = 20,
    tracks_per_playlist: int = 40,
    copy_fraction: float = 0.35,
    copies_per_source: tuple[int, int] = (1, 3),
    copy_fidelity: tuple[float, float] = (0.85, 1.0),
    n_catalog: int = 4000,
    contaminate: bool = True,
    seed: int = 11,
) -> pd.DataFrame:
    """A fake scrape with the same columns as the real one, for --dry-run.

    synthetic.make_zodiac_playlists already reproduces the within-sign overlap
    that comes from curators drawing on the same small pool of hits. What it
    does not reproduce is outright COPYING -- a curator duplicating someone
    else's playlist and renaming it -- which is the specific pathology
    dedup.py exists to catch. So we plant it here: `copy_fraction` of each
    sign's playlists become sources, each spawning 1-3 near-identical clones
    that keep 85-100% of the parent's tracks under a new id and owner.

    Each row carries `copy_of` (the source playlist id, or its own id for an
    original), which is GROUND TRUTH for the clustering: the dry run can check
    whether cluster_playlists actually recovers the planted groups. A real
    scrape has no such column.

    Audio features ride along from the synthetic catalog, which a real scrape
    cannot supply (/v1/audio-features died on 2024-11-27 -- see spotify.py);
    treat the dry-run CSV as "scrape already joined to the feature store".
    genre and artist_genres are filled from the catalog's own genre, standing
    in for the /artists lookup the real scraper does.

    `contaminate` plants playlists carrying REAL titles the live API returned:
    the contaminated ones for cancer, leo, libra, virgo and gemini, the
    oversized "cancer energy" playlist, and -- just as important -- genuine
    titles like "she's a scorpio" that a careless filter would also eat. So
    the dry run exercises filters.py on the titles that motivated it, in both
    directions, rather than on clean invented ones.
    """
    rng = np.random.default_rng(seed)
    catalog = syn.make_catalog(n_catalog, seed=seed) if catalog is None else catalog
    signs = list(signs or S.ZODIAC_SIGNS)

    base = syn.make_zodiac_playlists(
        catalog, n_playlists_per_sign=n_playlists_per_sign,
        tracks_per_playlist=tracks_per_playlist, seed=seed,
    )
    base = base[base[S.STAR_SIGN].isin(signs)]

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    owners = [f"{syn._word(rng)}{rng.integers(10, 9999)}" for _ in range(400)]
    frames: list[pd.DataFrame] = []
    n_pl = 0

    for sign in signs:
        sign_rows = base[base[S.STAR_SIGN] == sign]
        originals = list(sign_rows.groupby(S.PLAYLIST_NAME, sort=True))
        pool = sign_rows[S.TRACK_ID].unique()

        for _, pl in originals:
            pid = f"PL{n_pl:06d}"
            n_pl += 1
            frames.append(_fake_playlist_frame(pl, sign, pid, pid, owners, rng, stamp, len(frames)))

            if rng.random() >= copy_fraction:
                continue
            for _ in range(int(rng.integers(copies_per_source[0], copies_per_source[1] + 1))):
                keep = float(rng.uniform(*copy_fidelity))
                k = max(1, int(round(len(pl) * keep)))
                clone = pl.sample(n=k, random_state=int(rng.integers(0, 2**31))).copy()
                # A copier often adds a track or two of their own -- Jaccard
                # must survive that, which is why the threshold is not 1.0.
                extra = rng.choice(pool, size=int(rng.integers(0, 3)), replace=False)
                if len(extra):
                    add = base[base[S.TRACK_ID].isin(extra)].drop_duplicates(S.TRACK_ID)
                    clone = pd.concat([clone, add.reindex(columns=clone.columns)],
                                      ignore_index=True)
                cid = f"PL{n_pl:06d}"
                n_pl += 1
                frames.append(_fake_playlist_frame(clone, sign, cid, pid, owners,
                                                   rng, stamp, len(frames)))

    if contaminate:
        for sign in signs:
            titles = list(CONTAMINATED_TITLES.get(sign, ()))
            titles += list(GENUINE_TITLES.get(sign, ()))
            if sign == "Cancer":
                titles.append(OVERSIZED_TITLE)
            for title in titles:
                k = OVERSIZED_TRACKS if title == OVERSIZED_TITLE else tracks_per_playlist
                k = min(k, len(catalog))
                # Drawn from the whole catalogue, not the sign's pool: a
                # cancer-ward playlist has no relationship to an Aries one.
                take = rng.choice(len(catalog), size=k, replace=False)
                junk = catalog.iloc[take].copy()
                junk[S.STAR_SIGN] = sign
                cid = f"PL{n_pl:06d}"
                n_pl += 1
                frames.append(_fake_playlist_frame(junk, sign, cid, cid, owners,
                                                   rng, stamp, len(frames),
                                                   name=title))

    out = pd.concat(frames, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def synthetic_control(
    catalog: pd.DataFrame | None = None,
    n_playlists: int = 200,
    themes: Sequence[str] | None = None,
    tracks_per_playlist: int = 40,
    copy_fraction: float = 0.2,
    match_size_to: pd.DataFrame | None = None,
    n_catalog: int = 4000,
    seed: int = 29,
) -> pd.DataFrame:
    """The offline negative control: themed playlists, random sign labels.

    Built from the catalogue with NO sign direction at all -- only the
    popularity pull that makes every real playlist reach for the same hits.
    So it reproduces the structure of the zodiac set (shared hits, some
    copying, lognormal followers, matched sizes) while containing, by
    construction, nothing for a sign classifier to find.

    If a model scores the same here as on the real zodiac scrape, the real
    score was never about astrology. That is the comparison this function
    exists to make possible without credentials.
    """
    rng = np.random.default_rng(seed)
    catalog = syn.make_catalog(n_catalog, seed=seed) if catalog is None else catalog
    themes = list(themes or CONTROL_THEMES)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    owners = [f"{syn._word(rng)}{rng.integers(10, 9999)}" for _ in range(400)]

    pop = catalog[S.POPULARITY].to_numpy(float) / 100.0
    sizes = None
    if match_size_to is not None and len(match_size_to):
        sizes = (match_size_to.groupby(S.PLAYLIST_ID)[S.TRACK_ID]
                 .nunique().to_numpy())

    frames: list[pd.DataFrame] = []
    n_pl = 0
    for i in range(n_playlists):
        theme = themes[i % len(themes)]
        # One pool per theme, so playlists on the same theme overlap the way
        # real ones do -- the control needs the same duplication structure or
        # dedup.py has nothing to collapse.
        w = pop + 0.25
        w = w / w.sum()
        k = int(min(len(catalog), rng.choice(sizes) if sizes is not None
                    else tracks_per_playlist))
        take = rng.choice(len(catalog), size=max(1, k), replace=False, p=w)
        rows = catalog.iloc[take].copy()
        fake_sign = str(rng.choice(S.ZODIAC_SIGNS))   # label at random, per playlist
        rows[S.STAR_SIGN] = fake_sign
        pid = f"CT{n_pl:06d}"
        n_pl += 1
        frames.append(_fake_playlist_frame(
            rows, fake_sign, pid, pid, owners, rng, stamp, i,
            name=f"{theme} {_fake_control_title(rng)}", query=theme,
            is_control=True))
        if rng.random() < copy_fraction:
            clone = rows.sample(frac=float(rng.uniform(0.85, 1.0)),
                                random_state=int(rng.integers(0, 2**31))).copy()
            # A copied playlist keeps the PARENT's random label: copying is
            # what makes two playlists one opinion, label included.
            clone[S.STAR_SIGN] = fake_sign
            cid = f"CT{n_pl:06d}"
            n_pl += 1
            frames.append(_fake_playlist_frame(
                clone, fake_sign, cid, pid, owners, rng, stamp, i,
                name=f"{theme} {_fake_control_title(rng)}", query=theme,
                is_control=True))

    out = pd.concat(frames, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


_CONTROL_TITLES = ("playlist", "mix", "songs", "hits", "mood", "tunes",
                   "bangers", "essentials", "favourites", "rotation")


def _fake_control_title(rng) -> str:
    return str(rng.choice(_CONTROL_TITLES))


def _fake_playlist_frame(rows: pd.DataFrame, sign: str, pid: str, source_id: str,
                         owners: list[str], rng, stamp: str, rank: int,
                         name: str | None = None, query: str | None = None,
                         is_control: bool = False) -> pd.DataFrame:
    owner = owners[int(rng.integers(0, len(owners)))]
    # Follower counts are lognormal in reality: a few huge playlists, a long
    # tail of near-zero ones (including the bots).
    followers = int(np.clip(rng.lognormal(3.2, 2.3), 0, 4_000_000))
    frame = rows.copy()
    frame[S.PLAYLIST_ID] = pid
    frame[S.PLAYLIST_NAME] = name or f"{sign.lower()} {_fake_title(rng)}"
    frame[S.PLAYLIST_OWNER_ID] = owner
    frame[S.PLAYLIST_OWNER] = owner.title()
    frame[S.PLAYLIST_FOLLOWERS] = followers
    frame[S.PLAYLIST_DESCRIPTION] = f"for my fellow {sign.lower()}s"
    frame[S.PLAYLIST_SNAPSHOT] = f"snap{pid}"
    frame[S.PLAYLIST_TRACK_TOTAL] = len(frame)
    frame[S.SEARCH_QUERY] = query or f"{sign.lower()} playlist"
    frame[S.SEARCH_RANK] = rank
    frame[S.SCRAPED_AT] = stamp
    frame[S.ARTIST_ID] = "A" + frame[S.ARTIST].astype(str).str.replace(" ", "", regex=False)
    # The real scraper fills these from /artists; the synthetic catalog already
    # knows each track's genre, so mirror the column shape.
    frame[S.ARTIST_GENRES] = frame[S.GENRE]
    frame[S.TRACK_POSITION] = range(len(frame))
    frame[S.ADDED_AT] = stamp
    frame["copy_of"] = source_id
    frame[S.STAR_SIGN] = sign
    frame[S.IS_CONTROL] = is_control
    # The real scraper decides this from the title before fetching anything;
    # here the pipeline re-decides it from the planted name, which is exactly
    # what we want the dry run to test.
    frame[S.TITLE_CLASS] = F.KEEP
    frame[S.TITLE_REASON] = ""
    return frame


_TITLES = ("vibes", "season", "energy", "core", "mood", "aesthetic", "playlist",
           "sun placement", "rising", "hours", "only", "anthems")


def _fake_title(rng) -> str:
    return str(rng.choice(_TITLES))
