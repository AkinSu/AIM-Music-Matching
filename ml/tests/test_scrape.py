#!/usr/bin/env python
"""Offline tests for scrape.py and dedup.py.

NO NETWORK. Every HTTP call goes through FakeSession, every sleep through a
recorder, and the clock is a list we increment by hand -- so the retry,
pagination and token-refresh paths are tested in milliseconds instead of being
hoped about.

The clustering tests are built on hand-computed examples: the Jaccard values
and cluster memberships in the assertions were worked out on paper, not read
off a run. That matters because label_confidence is the number the whole
rebuild rests on -- a test that merely records current behaviour would not
catch it being wrong.

RUN
    cd ml && .venv/bin/python -m unittest discover -s tests -t tests -v
    cd ml && .venv/bin/python tests/test_scrape.py
"""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lovify_ml import data as D_data          # noqa: E402
from lovify_ml import dedup as D              # noqa: E402
from lovify_ml import filters as F            # noqa: E402
from lovify_ml import schema as S             # noqa: E402
from lovify_ml import scrape as sc            # noqa: E402


# ══ fake HTTP ════════════════════════════════════════════════════════════════

class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = text

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise AssertionError(f"unexpected raise_for_status on {self.status_code}")


class FakeSession:
    """Stands in for requests.Session: routes by path substring."""

    def __init__(self):
        self.headers = {}
        self.calls: list[tuple[str, dict, dict]] = []
        self.routes: list[tuple[str, object]] = []
        self.token_calls = 0
        self.token_status = 200
        self.last_url = ""

    # token endpoint
    def post(self, url, data=None, headers=None, timeout=None):
        self.token_calls += 1
        if self.token_status != 200:
            return FakeResponse(self.token_status, text="bad client")
        return FakeResponse(200, {"access_token": f"tok-{self.token_calls}",
                                  "expires_in": 3600, "token_type": "Bearer"})

    def route(self, needle: str, handler):
        """handler(params, call_index) -> FakeResponse | dict. First match wins."""
        self.routes.append((needle, handler))
        return self

    def get(self, url, params=None, headers=None, timeout=None):
        params = dict(params or {})
        self.last_url = url      # handlers dispatch on it; params carry no path
        self.calls.append((url, params, dict(headers or {})))
        for needle, handler in self.routes:
            if needle in url:
                n = sum(1 for c in self.calls[:-1] if needle in c[0])
                out = handler(params, n) if callable(handler) else handler
                return out if isinstance(out, FakeResponse) else FakeResponse(200, out)
        raise AssertionError(f"no fake route for {url}")

    def calls_to(self, needle: str) -> list[dict]:
        return [p for u, p, _ in self.calls if needle in u]

    def auth_tokens(self) -> list[str]:
        return [h.get("Authorization", "") for _, _, h in self.calls]


class Clock:
    def __init__(self, t=1000.0):
        self.t = t
        self.sleeps: list[float] = []

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(float(seconds))
        self.t += float(seconds)


def make_scraper(session, clock=None, **kw):
    clock = clock or Clock()
    s = sc.SpotifyScraper("cid", "secret", session=session, sleep=clock.sleep,
                          clock=clock, verbose=False, **kw)
    s.clock = clock
    return s


# ══ payload builders ═════════════════════════════════════════════════════════

def search_page(query, offset, limit, total=5000, holes=(), dup_of=None):
    items = []
    for i in range(max(0, min(limit, total - offset))):
        idx = offset + i
        if idx in holes:
            items.append(None)          # Spotify really does return these
            continue
        pid = f"p{dup_of}" if dup_of is not None and i == 0 else f"p{idx}"
        items.append({
            "id": pid,
            "name": f"{query} mix {idx}",
            "owner": {"id": f"owner{idx % 5}", "display_name": f"Owner {idx % 5}"},
            "tracks": {"total": 40},
        })
    return {"playlists": {"items": items, "total": total,
                          "next": "nextpage" if offset + limit < total else None}}


def track_obj(i, popularity=55, release="2019-05-01", artist=None):
    return {
        "id": f"t{i}", "name": f"Track {i}", "type": "track",
        "duration_ms": 200_000 + i,
        "popularity": popularity,
        "artists": [{"id": artist or f"a{i % 3}", "name": f"Artist {i % 3}"}],
        "album": {"name": f"Album {i}", "release_date": release},
    }


def tracks_page(indices, next_page=None, **kw):
    return {"items": [{"added_at": "2024-01-01T00:00:00Z", "is_local": False,
                       "track": track_obj(i, **kw)} for i in indices],
            "next": next_page, "total": len(indices)}


def meta_obj(pid="p0", followers=1234, total=3, name=None):
    """What SpotifyScraper.playlist() returns -- flattened, which is what
    tracks_to_frame consumes."""
    return {"id": pid, "name": name or f"name {pid}",
            "description": "line one\nline two",
            "snapshot_id": f"snap-{pid}", "owner_id": "ow1", "owner_name": "Ow",
            "followers": followers, "track_total": total}


def playlist_obj(pid="p0", followers=1234, total=3, name=None, owner="ow1"):
    return {"id": pid, "name": name or f"name {pid}",
            "description": "line one\nline two",
            "snapshot_id": f"snap-{pid}", "public": True, "collaborative": False,
            "followers": {"total": followers},
            "owner": {"id": owner, "display_name": "Ow"},
            "tracks": {"total": total}}


# ══ auth / HTTP ══════════════════════════════════════════════════════════════

class TestAuth(unittest.TestCase):
    def test_token_is_fetched_once_and_reused(self):
        sess = FakeSession().route("/search", lambda p, n: search_page("q", 0, 1, total=1))
        sc_ = make_scraper(sess)
        sc_._get("/search", q="x", type="playlist", limit=1, offset=0)
        sc_._get("/search", q="y", type="playlist", limit=1, offset=0)
        self.assertEqual(sess.token_calls, 1)
        self.assertEqual(sess.auth_tokens(), ["Bearer tok-1", "Bearer tok-1"])

    def test_token_auto_refreshes_when_expired(self):
        clock = Clock()
        sess = FakeSession().route("/search", lambda p, n: search_page("q", 0, 1, total=1))
        sc_ = make_scraper(sess, clock)
        sc_._get("/search", q="x", type="playlist", limit=1, offset=0)
        clock.t += 3600          # past the 3600s lifetime
        sc_._get("/search", q="x", type="playlist", limit=1, offset=0)
        self.assertEqual(sess.token_calls, 2)
        self.assertEqual(sess.auth_tokens()[-1], "Bearer tok-2")

    def test_token_refreshes_early_inside_the_margin(self):
        """A token with 30s left is refreshed now, not mid-page."""
        clock = Clock()
        sess = FakeSession().route("/search", lambda p, n: search_page("q", 0, 1, total=1))
        sc_ = make_scraper(sess, clock)
        clock.t += 3600 - 30
        sc_._get("/search", q="x", type="playlist", limit=1, offset=0)
        self.assertEqual(sess.token_calls, 2)

    def test_401_triggers_one_refresh_then_retries(self):
        def handler(params, n):
            return FakeResponse(401) if n == 0 else FakeResponse(200, search_page("q", 0, 1, total=1))
        sess = FakeSession().route("/search", handler)
        sc_ = make_scraper(sess)
        sc_._get("/search", q="x", type="playlist", limit=1, offset=0)
        self.assertEqual(sess.token_calls, 2)

    def test_bad_credentials_raise_a_clear_error(self):
        sess = FakeSession()
        sess.token_status = 400
        with self.assertRaises(sc.ScrapeError) as cm:
            make_scraper(sess)
        self.assertIn("client id/secret", str(cm.exception))


class TestRetries(unittest.TestCase):
    def test_429_sleeps_for_retry_after_then_succeeds(self):
        def handler(params, n):
            if n == 0:
                return FakeResponse(429, headers={"Retry-After": "7"})
            return FakeResponse(200, search_page("q", 0, 1, total=1))
        sess = FakeSession().route("/search", handler)
        clock = Clock()
        sc_ = make_scraper(sess, clock)
        out = sc_._get("/search", q="x", type="playlist", limit=1, offset=0)
        self.assertIn("playlists", out)
        self.assertEqual(clock.sleeps, [8.0])          # Retry-After + 1s slack
        self.assertEqual(sc_.stats["rate_limited"], 1)

    def test_429_without_a_header_still_waits(self):
        def handler(params, n):
            return FakeResponse(429) if n == 0 else FakeResponse(200, search_page("q", 0, 1, total=1))
        clock = Clock()
        sc_ = make_scraper(FakeSession().route("/search", handler), clock)
        sc_._get("/search", q="x", type="playlist", limit=1, offset=0)
        self.assertEqual(clock.sleeps, [2.0])

    def test_5xx_backs_off_exponentially(self):
        def handler(params, n):
            return FakeResponse(503) if n < 3 else FakeResponse(200, search_page("q", 0, 1, total=1))
        clock = Clock()
        sc_ = make_scraper(FakeSession().route("/search", handler), clock)
        sc_._get("/search", q="x", type="playlist", limit=1, offset=0)
        self.assertEqual(clock.sleeps, [1.0, 2.0, 4.0])
        self.assertEqual(sc_.stats["server_errors"], 3)

    def test_gives_up_after_max_attempts(self):
        sess = FakeSession().route("/search", lambda p, n: FakeResponse(503))
        sc_ = make_scraper(sess, max_attempts=3)
        with self.assertRaises(sc.ScrapeError):
            sc_._get("/search", q="x", type="playlist", limit=1, offset=0)
        self.assertEqual(len(sess.calls_to("/search")), 3)

    def test_403_on_deprecated_endpoint_explains_the_deprecation(self):
        sess = FakeSession().route("/audio-features", lambda p, n: FakeResponse(403))
        sc_ = make_scraper(sess)
        with self.assertRaises(PermissionError) as cm:
            sc_._get("/audio-features", ids="t1")
        msg = str(cm.exception)
        self.assertIn("2024-11-27", msg)
        self.assertIn("extended access", msg)
        self.assertEqual(len(sess.calls_to("/audio-features")), 1)  # never retried

    def test_404_on_deprecated_endpoint_is_also_explained(self):
        """The dead routes answer with a mix of 403 and 404, so detection
        cannot key on the status code alone."""
        sess = FakeSession().route("/recommendations", lambda p, n: FakeResponse(404))
        sc_ = make_scraper(sess)
        with self.assertRaises(PermissionError) as cm:
            sc_._get("/recommendations", seed_tracks="t1")
        self.assertIn("2024-11-27", str(cm.exception))

    def test_403_elsewhere_blames_credentials_not_the_deprecation(self):
        sess = FakeSession().route("/search", lambda p, n: FakeResponse(403))
        sc_ = make_scraper(sess)
        with self.assertRaises(PermissionError) as cm:
            sc_._get("/search", q="x", type="playlist")
        msg = str(cm.exception)
        self.assertIn("NOT among those removed", msg)
        self.assertIn(".env", msg)


# ══ search ═══════════════════════════════════════════════════════════════════

class TestSearchPagination(unittest.TestCase):
    def test_pages_until_max_results(self):
        sess = FakeSession().route(
            "/search", lambda p, n: search_page(p["q"], int(p["offset"]), int(p["limit"])))
        sc_ = make_scraper(sess)
        out = sc_.search_playlists("aries", max_results=120)
        self.assertEqual(len(out), 120)
        self.assertEqual([(c["offset"], c["limit"]) for c in sess.calls_to("/search")],
                         [(0, 50), (50, 50), (100, 20)])
        self.assertEqual(out[0]["id"], "p0")
        self.assertEqual(out[-1]["id"], "p119")

    def test_stops_when_next_is_null(self):
        sess = FakeSession().route(
            "/search", lambda p, n: search_page(p["q"], int(p["offset"]), 30, total=30))
        out = make_scraper(sess).search_playlists("aries", max_results=500)
        self.assertEqual(len(out), 30)
        self.assertEqual(len(sess.calls_to("/search")), 1)

    def test_never_exceeds_the_1000_offset_cap(self):
        """Spotify rejects offset + limit > 1000, so one query can yield at
        most 1000 playlists however high max_results is set."""
        sess = FakeSession().route(
            "/search", lambda p, n: search_page(p["q"], int(p["offset"]), int(p["limit"]),
                                                total=100_000))
        out = make_scraper(sess).search_playlists("aries", max_results=5000)
        self.assertEqual(len(out), sc.SEARCH_OFFSET_CAP)
        windows = [(c["offset"], c["limit"]) for c in sess.calls_to("/search")]
        self.assertEqual(len(windows), 20)
        for offset, limit in windows:
            self.assertLessEqual(offset + limit, sc.SEARCH_OFFSET_CAP,
                                 f"offset {offset} + limit {limit} would 400")

    def test_cap_clamps_the_last_page_instead_of_overshooting(self):
        """With the cap at 120 the third page must ask for 20, not 50: the
        clamp is on offset + limit, not on offset alone."""
        sess = FakeSession().route(
            "/search", lambda p, n: search_page(p["q"], int(p["offset"]), int(p["limit"]),
                                                total=100_000))
        with mock.patch.object(sc, "SEARCH_OFFSET_CAP", 120):
            out = make_scraper(sess).search_playlists("aries", max_results=5000)
        self.assertEqual([(c["offset"], c["limit"]) for c in sess.calls_to("/search")],
                         [(0, 50), (50, 50), (100, 20)])
        self.assertEqual(len(out), 120)

    def test_filters_none_items_without_losing_the_window(self):
        holes = {0, 3, 7}
        sess = FakeSession().route(
            "/search", lambda p, n: search_page(p["q"], int(p["offset"]), int(p["limit"]),
                                                total=10, holes=holes))
        out = make_scraper(sess).search_playlists("aries", max_results=50)
        self.assertEqual([r["id"] for r in out],
                         [f"p{i}" for i in range(10) if i not in holes])
        # Rank is the position in the RAW results, holes included.
        self.assertEqual(out[0]["rank"], 1)
        self.assertEqual(out[-1]["rank"], 9)

    def test_deduplicates_playlists_repeated_across_pages(self):
        def handler(params, n):
            off = int(params["offset"])
            return search_page(params["q"], off, int(params["limit"]),
                               total=100, dup_of=0 if off else None)
        sess = FakeSession().route("/search", handler)
        out = make_scraper(sess).search_playlists("aries", max_results=100)
        ids = [r["id"] for r in out]
        self.assertEqual(len(ids), len(set(ids)))


# ══ playlists and tracks ═════════════════════════════════════════════════════

class TestPlaylistFetching(unittest.TestCase):
    def test_playlist_metadata_is_flattened(self):
        sess = FakeSession().route("/playlists/", lambda p, n: playlist_obj("p9", followers=77))
        meta = make_scraper(sess).playlist("p9")
        self.assertEqual(meta["followers"], 77)
        self.assertEqual(meta["owner_id"], "ow1")
        self.assertEqual(meta["snapshot_id"], "snap-p9")
        # snapshot_id must be requested, or the scrape is unreproducible.
        self.assertIn("snapshot_id", sess.calls_to("/playlists/")[0]["fields"])

    def test_vanished_playlist_returns_none_rather_than_raising(self):
        sess = FakeSession().route("/playlists/", lambda p, n: FakeResponse(404))
        self.assertIsNone(make_scraper(sess).playlist("gone"))

    def test_playlist_tracks_paginate(self):
        def handler(params, n):
            off = int(params["offset"])
            return tracks_page(range(off, off + 100),
                               next_page="more" if off < 100 else None)
        sess = FakeSession().route("/tracks", handler)   # matches .../tracks
        out = make_scraper(sess).playlist_tracks("p1", max_tracks=500)
        self.assertEqual(len(out), 200)
        self.assertEqual([c["limit"] for c in sess.calls_to("/tracks")], [100, 100])
        self.assertEqual(out[0]["position"], 0)
        self.assertEqual(out[-1]["position"], 199)

    def test_playlist_tracks_skips_nulls_locals_and_episodes(self):
        items = [
            None,                                                   # hole
            {"track": None, "is_local": False},                     # deleted track
            {"track": track_obj(1), "is_local": True},              # local file
            {"track": {**track_obj(2), "type": "episode"}, "is_local": False},
            {"track": {**track_obj(3), "id": None}, "is_local": False},
            {"track": track_obj(4), "is_local": False, "added_at": "2020-01-01T00:00:00Z"},
        ]
        sess = FakeSession().route("/tracks", lambda p, n: {"items": items, "next": None})
        out = make_scraper(sess).playlist_tracks("p1")
        self.assertEqual([t["id"] for t in out], ["t4"])
        self.assertEqual(out[0]["added_at"], "2020-01-01T00:00:00Z")

    def test_max_tracks_caps_a_runaway_playlist(self):
        sess = FakeSession().route(
            "/tracks", lambda p, n: tracks_page(range(int(p["offset"]), int(p["offset"]) + 100),
                                                next_page="more"))
        out = make_scraper(sess).playlist_tracks("p1", max_tracks=150)
        self.assertEqual(len(out), 150)


# ══ enrichment ═══════════════════════════════════════════════════════════════

class TestEnrichment(unittest.TestCase):
    def _artist_route(self, sess, genres=("hip hop", "trap")):
        def handler(params, n):
            ids = params["ids"].split(",")
            return {"artists": [{"id": i, "genres": list(genres)} for i in ids]}
        return sess.route("/artists", handler)

    def test_artist_genres_batch_by_50_and_cache(self):
        sess = self._artist_route(FakeSession())
        sc_ = make_scraper(sess)
        ids = [f"a{i}" for i in range(120)]
        got = sc_.artist_genres(ids)
        self.assertEqual(len(got), 120)
        self.assertEqual([len(c["ids"].split(",")) for c in sess.calls_to("/artists")],
                         [50, 50, 20])
        sc_.artist_genres(ids)                      # second pass: all cached
        self.assertEqual(len(sess.calls_to("/artists")), 3)

    def test_unresolvable_artist_is_cached_as_empty_not_retried(self):
        sess = FakeSession().route("/artists", lambda p, n: {"artists": [None]})
        sc_ = make_scraper(sess)
        sc_.artist_genres(["a1"])
        sc_.artist_genres(["a1"])
        self.assertEqual(len(sess.calls_to("/artists")), 1)

    def test_enrich_fills_genre_and_artist_genres(self):
        sess = self._artist_route(FakeSession(), genres=("indie pop", "bedroom pop"))
        sc_ = make_scraper(sess)
        frame = sc.tracks_to_frame("Aries", meta_obj("p1"), [track_obj(1), track_obj(2)])
        out = sc_.enrich(frame)
        self.assertEqual(list(out[S.GENRE].unique()), ["indie pop"])
        self.assertEqual(out[S.ARTIST_GENRES].iloc[0], "indie pop|bedroom pop")

    def test_artist_with_no_genres_becomes_unknown(self):
        sess = FakeSession().route("/artists", lambda p, n: {
            "artists": [{"id": i, "genres": []} for i in p["ids"].split(",")]})
        frame = sc.tracks_to_frame("Aries", meta_obj("p1"), [track_obj(1)])
        out = make_scraper(sess).enrich(frame)
        self.assertEqual(out[S.GENRE].iloc[0], "unknown")

    def test_popularity_and_year_backfill_from_tracks_endpoint(self):
        sess = self._artist_route(FakeSession())
        sess.route("/tracks", lambda p, n: {"tracks": [
            {"id": i, "popularity": 42, "album": {"release_date": "1999-03-04"}}
            for i in p["ids"].split(",")]})
        # The playlist payload sometimes omits both -- that is what triggers it.
        bare = {"id": "t1", "name": "x", "type": "track", "duration_ms": 1,
                "artists": [{"id": "a1", "name": "A"}], "album": {"name": "al"}}
        frame = sc.tracks_to_frame("Aries", meta_obj("p1"), [bare])
        self.assertTrue(frame[S.POPULARITY].isna().all())
        out = make_scraper(sess).enrich(frame)
        self.assertEqual(int(out[S.POPULARITY].iloc[0]), 42)
        self.assertEqual(int(out[S.YEAR].iloc[0]), 1999)

    def test_no_backfill_call_when_the_payload_already_had_them(self):
        sess = self._artist_route(FakeSession())
        sess.route("/tracks", lambda p, n: {"tracks": []})
        frame = sc.tracks_to_frame("Aries", meta_obj("p1"), [track_obj(1)])
        make_scraper(sess).enrich(frame)
        self.assertEqual(sess.calls_to("/tracks"), [])


# ══ row building ═════════════════════════════════════════════════════════════

class TestRowBuilding(unittest.TestCase):
    def test_columns_match_the_declared_order(self):
        frame = sc.tracks_to_frame("Leo", meta_obj("p3"), [track_obj(1)],
                                   query="leo vibes", rank=4)
        self.assertEqual(list(frame.columns), sc.SCRAPE_COLUMNS)

    def test_provenance_is_recorded(self):
        frame = sc.tracks_to_frame("Leo", meta_obj("p3", followers=9), [track_obj(1)],
                                   query="leo vibes", rank=4)
        row = frame.iloc[0]
        self.assertEqual(row[S.PLAYLIST_ID], "p3")
        self.assertEqual(row[S.PLAYLIST_FOLLOWERS], 9)
        self.assertEqual(row[S.SEARCH_QUERY], "leo vibes")
        self.assertEqual(row[S.SEARCH_RANK], 4)
        self.assertEqual(row[S.PLAYLIST_SNAPSHOT], "snap-p3")
        # Newlines would break the CSV row.
        self.assertEqual(row[S.PLAYLIST_DESCRIPTION], "line one line two")

    def test_year_is_parsed_and_survives_a_bad_release_date(self):
        frame = sc.tracks_to_frame("Leo", meta_obj(), [
            track_obj(1, release="2019-05-01"),
            track_obj(2, release=""),
        ])
        self.assertEqual(frame[S.YEAR].iloc[0], 2019)
        self.assertTrue(pd.isna(frame[S.YEAR].iloc[1]))


# ══ one sign, end to end ═════════════════════════════════════════════════════

def wire_full_scrape(sess, n_playlists=3, followers=1234, tracks_per=20,
                     sign="aries", title=None, owner="ow1"):
    """A fake Spotify that can answer everything scrape_sign asks.

    Responses are keyed on the playlist id in the URL, not on call order, so
    asking twice gives the same answer -- which is what makes the resume test
    meaningful. Titles are astrologically valid and playlists long enough to
    clear filters.MIN_TRACKS, so the title filter is a pass-through here and
    the filtering tests can set up their own contamination deliberately.
    """
    sess.route("/search", lambda p, n: search_page(p["q"], int(p["offset"]),
                                                   int(p["limit"]), total=n_playlists))

    def playlists_handler(params, n):
        pid = re.search(r"/playlists/([^/?]+)", sess.last_url).group(1)
        k = int(re.sub(r"\D", "", pid) or 0)
        if sess.last_url.endswith("/tracks"):
            # Three shared hits plus its own: realistic overlap, and enough
            # tracks to survive the size filter.
            own = range(100 + k * tracks_per, 100 + k * tracks_per + tracks_per - 3)
            return tracks_page([0, 1, 2, *own])
        return playlist_obj(pid, followers=followers, total=tracks_per, owner=owner,
                            name=title or f"{sign} season {pid}")

    sess.route("/playlists/", playlists_handler)
    sess.route("/artists", lambda p, n: {"artists": [
        {"id": i, "genres": ["pop", "dance pop"]} for i in p["ids"].split(",")]})
    return sess


class TestScrapeSign(unittest.TestCase):
    def test_builds_a_frame_with_provenance_and_genre(self):
        sess = wire_full_scrape(FakeSession())
        sc_ = make_scraper(sess)
        df = sc_.scrape_sign("Aries", n_playlists=3, queries=["aries"], min_tracks=2)
        self.assertEqual(list(df.columns), sc.SCRAPE_COLUMNS)
        self.assertEqual(df[S.STAR_SIGN].unique().tolist(), ["Aries"])
        self.assertEqual(df[S.PLAYLIST_ID].nunique(), 3)
        self.assertEqual(df[S.GENRE].iloc[0], "pop")
        self.assertEqual(df[S.SEARCH_QUERY].iloc[0], "aries")

    def test_min_tracks_and_min_followers_filters(self):
        sess = wire_full_scrape(FakeSession(), followers=5)
        df = make_scraper(sess).scrape_sign("Aries", n_playlists=2, queries=["aries"],
                                            min_followers=1000)
        self.assertTrue(df.empty)

    def test_queries_are_interleaved_not_concatenated(self):
        """Two phrasings must each contribute, or deep paging buys nothing."""
        pools = [[{"id": "a1"}, {"id": "a2"}, {"id": "a3"}],
                 [{"id": "b1"}, {"id": "b2"}]]
        self.assertEqual([r["id"] for r in sc._round_robin(pools)],
                         ["a1", "b1", "a2", "b2", "a3"])

    def test_output_is_loadable_by_data_load_zodiac(self):
        """data.load_zodiac lowercases columns and applies Kaggle aliases; our
        column names must survive that round trip (e.g. nothing we emit may be
        called 'name' or 'id', which would be renamed out from under us)."""
        sess = wire_full_scrape(FakeSession())
        df = make_scraper(sess).scrape_sign("Aries", n_playlists=2, queries=["aries"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zodiac.csv"
            df.to_csv(path, index=False)
            loaded = D_data._normalise(pd.read_csv(path))
        self.assertIn(S.STAR_SIGN, loaded.columns)      # mandatory
        self.assertIn(S.TRACK_ID, loaded.columns)       # needed for dedup
        self.assertEqual(set(loaded.columns), set(sc.SCRAPE_COLUMNS))


# ══ checkpointing ════════════════════════════════════════════════════════════

class TestCheckpoint(unittest.TestCase):
    def test_appends_one_header_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = sc.ScrapeCheckpoint(root=Path(tmp))
            for pid in ("p1", "p2"):
                ck.append("Aries", sc.tracks_to_frame("Aries", meta_obj(pid),
                                                      [track_obj(1)]), playlist_id=pid)
            text = ck.part("Aries").read_text()
            self.assertEqual(text.count(S.STAR_SIGN), 1)
            self.assertEqual(len(ck.load_sign("Aries")), 2)

    def test_resume_skips_playlists_already_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            sess = wire_full_scrape(FakeSession())
            ck = sc.ScrapeCheckpoint(root=Path(tmp))
            first = make_scraper(sess).scrape_sign("Aries", n_playlists=3,
                                                   queries=["aries"], checkpoint=ck)
            self.assertEqual(first[S.PLAYLIST_ID].nunique(), 3)

            # A fresh process: new checkpoint object reading the same directory.
            sess2 = wire_full_scrape(FakeSession())
            ck2 = sc.ScrapeCheckpoint(root=Path(tmp))
            self.assertEqual(ck2.done_playlists("Aries"), {"p0", "p1", "p2"})
            self.assertTrue(ck2.is_done("Aries"))
            again = make_scraper(sess2).scrape_sign("Aries", n_playlists=3,
                                                    queries=["aries"], checkpoint=ck2)
            # Searched again (cheap) but fetched no playlist bodies.
            self.assertEqual(sess2.calls_to("/playlists/"), [])
            self.assertEqual(len(again), len(first))

    def test_scrape_all_skips_completed_signs(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = sc.ScrapeCheckpoint(root=Path(tmp))
            ck.append("Aries", sc.tracks_to_frame("Aries", meta_obj("p0"),
                                                  [track_obj(1)]), playlist_id="p0")
            ck.mark_done("Aries")
            sess = wire_full_scrape(FakeSession())
            df = make_scraper(sess).scrape_all(signs=["Aries"], n_playlists=3,
                                               checkpoint=ck)
            self.assertEqual(sess.calls_to("/search"), [])
            self.assertEqual(len(df), 1)

    def test_artist_genre_cache_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = sc.ScrapeCheckpoint(root=Path(tmp))
            ck.save_artist_genres({"a0": ["pop"], "a1": ["rock"]})
            sess = wire_full_scrape(FakeSession(), sign="taurus")
            sc2 = make_scraper(sess)
            sc2.scrape_sign("Taurus", n_playlists=1, queries=["taurus"], checkpoint=ck)
            # a0/a1 came off disk; only the unseen artist ids were requested.
            asked = {i for c in sess.calls_to("/artists") for i in c["ids"].split(",")}
            self.assertNotIn("a0", asked)

    def test_corrupt_index_falls_back_to_the_csv_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = sc.ScrapeCheckpoint(root=Path(tmp))
            ck.append("Aries", sc.tracks_to_frame("Aries", meta_obj("p7"),
                                                  [track_obj(1)]), playlist_id="p7")
            ck.index_path.write_text("{not json")
            ck2 = sc.ScrapeCheckpoint(root=Path(tmp))
            self.assertEqual(ck2.done_playlists("Aries"), {"p7"})


# ══ Jaccard and clustering: hand-computed ════════════════════════════════════

def frame_from_sets(spec: dict[str, tuple[str, list[str]]]) -> pd.DataFrame:
    """spec: playlist_id -> (sign, [track ids])."""
    rows = [{S.PLAYLIST_ID: pid, S.STAR_SIGN: sign, S.TRACK_ID: t}
            for pid, (sign, tracks) in spec.items() for t in tracks]
    return pd.DataFrame(rows)


class TestJaccard(unittest.TestCase):
    def test_pairwise_values_are_the_hand_computed_ones(self):
        df = frame_from_sets({
            "A": ("Aries", ["1", "2", "3"]),
            "B": ("Aries", ["1", "2", "3", "4"]),   # A n B = 3, A u B = 4 -> 0.75
            "C": ("Leo", ["5", "6"]),               # disjoint from both -> 0.0
        })
        J = D.playlist_jaccard_matrix(df)
        self.assertAlmostEqual(J.loc["A", "B"], 0.75)
        self.assertAlmostEqual(J.loc["B", "A"], 0.75)
        self.assertAlmostEqual(J.loc["A", "C"], 0.0)
        self.assertAlmostEqual(J.loc["A", "A"], 1.0)
        self.assertAlmostEqual(J.loc["C", "C"], 1.0)

    def test_duplicate_rows_do_not_inflate_a_set(self):
        df = frame_from_sets({"A": ("Aries", ["1", "1", "2"]),
                              "B": ("Aries", ["1", "2"])})
        self.assertAlmostEqual(D.playlist_jaccard_matrix(df).loc["A", "B"], 1.0)

    def test_helper_matches_the_matrix(self):
        self.assertAlmostEqual(D.jaccard({"1", "2", "3"}, {"1", "2", "3", "4"}), 0.75)
        self.assertEqual(D.jaccard(set(), set()), 0.0)

    def test_min_tracks_drops_tiny_playlists(self):
        df = frame_from_sets({"A": ("Aries", [str(i) for i in range(10)]),
                              "T": ("Aries", ["0", "1"])})
        J = D.playlist_jaccard_matrix(df, min_tracks=5)
        self.assertEqual(list(J.index), ["A"])


class TestClustering(unittest.TestCase):
    def test_copies_cluster_and_merely_overlapping_playlists_do_not(self):
        # P1 n P2 = 8 of 10 -> J = 8/12 = 0.667 (a copy with two edits).
        # P1 n P3 = 2       -> J = 2/18 = 0.111 (same hits, different curator).
        df = frame_from_sets({
            "P1": ("Aries", [str(i) for i in range(10)]),
            "P2": ("Aries", [str(i) for i in range(8)] + ["90", "91"]),
            "P3": ("Aries", ["0", "1"] + [str(i) for i in range(50, 58)]),
        })
        J = D.playlist_jaccard_matrix(df)
        self.assertAlmostEqual(J.loc["P1", "P2"], 8 / 12)
        self.assertAlmostEqual(J.loc["P1", "P3"], 2 / 18)
        cl = D.cluster_playlists(df, threshold=0.5)
        self.assertEqual(cl["P1"], cl["P2"])
        self.assertNotEqual(cl["P1"], cl["P3"])
        self.assertEqual(cl.nunique(), 2)

    def test_single_linkage_chains_where_complete_linkage_refuses(self):
        """A-B 0.538, B-C 0.538, A-C 0.25: single linkage merges all three
        through B, complete linkage keeps A and C apart. Documented in the
        module docstring as the deliberate conservative choice."""
        df = frame_from_sets({
            "A": ("Aries", [str(i) for i in range(1, 11)]),
            "B": ("Aries", [str(i) for i in range(4, 14)]),
            "C": ("Aries", [str(i) for i in range(7, 17)]),
        })
        J = D.playlist_jaccard_matrix(df)
        self.assertAlmostEqual(J.loc["A", "B"], 7 / 13)
        self.assertAlmostEqual(J.loc["B", "C"], 7 / 13)
        self.assertAlmostEqual(J.loc["A", "C"], 4 / 16)
        self.assertEqual(D.cluster_playlists(df, 0.5, linkage="single").nunique(), 1)
        self.assertEqual(D.cluster_playlists(df, 0.5, linkage="complete").nunique(), 2)

    def test_cluster_ids_are_contiguous_and_deterministic(self):
        df = frame_from_sets({f"P{i}": ("Aries", [str(i)]) for i in range(5)})
        cl = D.cluster_playlists(df, 0.5)
        self.assertEqual(sorted(cl.unique()), list(range(5)))
        self.assertTrue(cl.equals(D.cluster_playlists(df, 0.5)))

    def test_cluster_sizes_counts_copies(self):
        df = frame_from_sets({p: ("Aries", ["1", "2", "3"]) for p in ("A", "B", "C")})
        sizes = D.cluster_sizes(D.cluster_playlists(df, 0.5))
        self.assertEqual(int(sizes.iloc[0]), 3)

    def test_filtered_playlists_get_singleton_clusters_not_a_shared_one(self):
        df = frame_from_sets({"A": ("Aries", [str(i) for i in range(10)]),
                              "T1": ("Aries", ["0", "1"]),
                              "T2": ("Leo", ["0", "1"])})
        cl = D.cluster_playlists(df, 0.5, min_tracks=5)
        tagged = D.attach_clusters(df, cl)
        by_pl = tagged.groupby(S.PLAYLIST_ID)[S.PLAYLIST_CLUSTER].first()
        self.assertEqual(by_pl.nunique(), 3)


# ══ label confidence: the headline maths ═════════════════════════════════════

class TestLabelConfidence(unittest.TestCase):
    def setUp(self):
        # Track X sits in three IDENTICAL Aries playlists and one Leo
        # playlist. Counting playlists says Aries 0.75. Counting independent
        # clusters says 0.50, because the three Aries playlists are one
        # opinion copied twice.
        self.df = frame_from_sets({
            "A1": ("Aries", ["X", "t1", "t2", "t3"]),
            "A2": ("Aries", ["X", "t1", "t2", "t3"]),
            "A3": ("Aries", ["X", "t1", "t2", "t3"]),
            "L1": ("Leo", ["X", "z1", "z2", "z3"]),
        })
        self.clusters = D.cluster_playlists(self.df, threshold=0.5)
        self.conf = D.label_confidence(self.df, self.clusters)

    def test_three_copies_count_as_one_vote(self):
        self.assertEqual(self.clusters.nunique(), 2)
        x = self.conf[self.conf[S.TRACK_ID] == "X"].set_index(S.STAR_SIGN)
        self.assertAlmostEqual(x.loc["Aries", S.LABEL_CONFIDENCE], 0.5)
        self.assertAlmostEqual(x.loc["Leo", S.LABEL_CONFIDENCE], 0.5)
        self.assertAlmostEqual(x.loc["Aries", "naive_confidence"], 0.75)
        self.assertEqual(int(x.loc["Aries", S.N_CLUSTERS_SIGN]), 1)
        self.assertEqual(int(x.loc["Aries", "n_playlists_sign"]), 3)
        self.assertEqual(int(x.loc["Aries", S.N_CLUSTERS_TOTAL]), 2)

    def test_confidences_for_a_track_sum_to_one(self):
        totals = self.conf.groupby(S.TRACK_ID)[S.LABEL_CONFIDENCE].sum()
        for t, v in totals.items():
            self.assertAlmostEqual(v, 1.0, msg=f"track {t}")

    def test_uncontested_track_scores_one(self):
        t1 = self.conf[(self.conf[S.TRACK_ID] == "t1")].iloc[0]
        self.assertAlmostEqual(t1[S.LABEL_CONFIDENCE], 1.0)
        self.assertEqual(int(t1[S.N_CLUSTERS_TOTAL]), 1)

    def test_confident_labels_requires_two_independent_clusters(self):
        """A track seen by one curator is unanimous by construction, not by
        agreement -- min_clusters=2 is what makes the subset meaningful."""
        keep = D.confident_labels(self.conf, min_confidence=1.0, min_clusters=2)
        self.assertTrue(keep.empty)
        loose = D.confident_labels(self.conf, min_confidence=1.0, min_clusters=1)
        self.assertEqual(set(loose[S.TRACK_ID]), {"t1", "t2", "t3", "z1", "z2", "z3"})

    def test_two_independent_clusters_agreeing_is_kept(self):
        df = frame_from_sets({
            "A1": ("Aries", ["X", "t1", "t2", "t3"]),
            "A2": ("Aries", ["X", "u1", "u2", "u3"]),   # J with A1 = 1/7, independent
        })
        cl = D.cluster_playlists(df, 0.5)
        keep = D.confident_labels(D.label_confidence(df, cl), 1.0, min_clusters=2)
        self.assertEqual(set(keep[S.TRACK_ID]), {"X"})

    def test_attach_confidence_merges_onto_rows(self):
        out = D.attach_confidence(D.attach_clusters(self.df, self.clusters), self.conf)
        self.assertEqual(len(out), len(self.df))
        self.assertFalse(out[S.LABEL_CONFIDENCE].isna().any())
        x_aries = out[(out[S.TRACK_ID] == "X") & (out[S.STAR_SIGN] == "Aries")]
        self.assertTrue((x_aries[S.LABEL_CONFIDENCE] == 0.5).all())

    def test_confidence_distribution_shares_sum_to_one(self):
        dist = D.confidence_distribution(self.conf)
        self.assertAlmostEqual(dist["share"].sum(), 1.0)
        self.assertEqual(int(dist["n"].sum()), len(self.conf))


class TestEffectiveSampleSize(unittest.TestCase):
    def test_copies_halve_the_evidence(self):
        df = frame_from_sets({
            "A1": ("Aries", ["X", "t1", "t2", "t3"]),
            "A2": ("Aries", ["X", "t1", "t2", "t3"]),
            "A3": ("Aries", ["X", "t1", "t2", "t3"]),
            "L1": ("Leo", ["X", "z1", "z2", "z3"]),
        })
        ess = D.effective_sample_size(df, D.cluster_playlists(df, 0.5))
        self.assertEqual(ess["playlists"], 4)
        self.assertEqual(ess["clusters"], 2)
        self.assertEqual(ess["apparent_votes"], 16)   # 4 playlists x 4 tracks
        self.assertEqual(ess["effective_votes"], 8)   # 2 clusters x 4 tracks
        self.assertAlmostEqual(ess["ess_ratio"], 0.5)
        self.assertAlmostEqual(ess["inflation"], 2.0)
        self.assertEqual(ess["unique_tracks"], 7)
        self.assertAlmostEqual(ess["independent_label_rate"], 0.25)  # only X

    def test_naive_and_cluster_confidence_have_the_same_mean_by_construction(self):
        """Why the report prints the per-row shift and not a mean comparison:
        both measures normalise per track, so each track's confidences sum to
        1 either way and the two means are forced equal. Comparing them would
        look like a finding and be an identity."""
        df = frame_from_sets({
            "A1": ("Aries", ["X", "t1", "t2"]),
            "A2": ("Aries", ["X", "t1", "t2"]),
            "A3": ("Aries", ["X", "t1", "t2"]),
            "L1": ("Leo", ["X", "z1", "z2"]),
        })
        clusters = D.cluster_playlists(df, 0.5)
        conf = D.label_confidence(df, clusters)
        self.assertAlmostEqual(conf[S.LABEL_CONFIDENCE].mean(),
                               conf["naive_confidence"].mean())
        ess = D.effective_sample_size(df, clusters)
        # The shift is where the difference actually lives.
        self.assertGreater(ess["share_changed"], 0.0)
        self.assertGreater(ess["confidence_shift"], 0.0)
        # Aries X was 0.75 naive, 0.5 by cluster -> overstated.
        self.assertAlmostEqual(ess["share_overstated"], 1 / len(conf))

    def test_independent_playlists_lose_nothing(self):
        df = frame_from_sets({
            "A1": ("Aries", ["1", "2", "3"]),
            "A2": ("Aries", ["4", "5", "6"]),
        })
        ess = D.effective_sample_size(df, D.cluster_playlists(df, 0.5))
        self.assertAlmostEqual(ess["ess_ratio"], 1.0)
        self.assertAlmostEqual(ess["inflation"], 1.0)

    def test_per_sign_breakdown_covers_every_sign(self):
        df = frame_from_sets({
            "A1": ("Aries", ["1", "2", "3"]), "A2": ("Aries", ["1", "2", "3"]),
            "L1": ("Leo", ["7", "8", "9"]),
        })
        ps = D.per_sign_ess(df, D.cluster_playlists(df, 0.5))
        self.assertEqual(set(ps[S.STAR_SIGN]), {"Aries", "Leo"})
        self.assertAlmostEqual(
            float(ps.loc[ps[S.STAR_SIGN] == "Aries", "ess_ratio"].iloc[0]), 0.5)
        self.assertAlmostEqual(
            float(ps.loc[ps[S.STAR_SIGN] == "Leo", "ess_ratio"].iloc[0]), 1.0)

    def test_threshold_sweep_is_monotone_in_clusters(self):
        """A stricter threshold can only split clusters, never merge them."""
        df = sc.synthetic_scrape(signs=["Aries", "Leo"], n_playlists_per_sign=6,
                                 n_catalog=400, seed=3)
        sweep = D.threshold_sweep(df, thresholds=(0.3, 0.5, 0.7, 0.9))
        self.assertTrue((sweep["clusters"].diff().dropna() >= 0).all(),
                        f"clusters fell as the threshold rose:\n{sweep}")


# ══ the synthetic stand-in ═══════════════════════════════════════════════════

class TestSyntheticScrape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = sc.synthetic_scrape(signs=["Aries", "Leo", "Virgo"],
                                     n_playlists_per_sign=8, n_catalog=600, seed=5)

    def test_has_every_column_the_real_scrape_writes(self):
        for col in sc.SCRAPE_COLUMNS:
            self.assertIn(col, self.df.columns)

    def test_plants_copies_that_clustering_recovers(self):
        truth = (self.df[[S.PLAYLIST_ID, "copy_of"]].astype(str).drop_duplicates()
                 .set_index(S.PLAYLIST_ID)["copy_of"])
        groups = {g: m for g, m in truth.groupby(truth).groups.items() if len(m) > 1}
        self.assertGreater(len(groups), 0, "no copies were planted")
        clusters = D.cluster_playlists(self.df, threshold=0.5)
        recovered = sum(1 for m in groups.values()
                        if clusters.loc[list(m)].nunique() == 1)
        self.assertEqual(recovered, len(groups),
                         "planted copy groups were not recovered as single clusters")

    def test_evidence_shrinks_once_copies_collapse(self):
        ess = D.effective_sample_size(self.df, D.cluster_playlists(self.df, 0.5))
        self.assertLess(ess["ess_ratio"], 1.0)
        self.assertGreater(ess["inflation"], 1.0)

    def test_is_reproducible_from_the_seed(self):
        a = sc.synthetic_scrape(signs=["Aries"], n_playlists_per_sign=4,
                                n_catalog=300, seed=1)
        b = sc.synthetic_scrape(signs=["Aries"], n_playlists_per_sign=4,
                                n_catalog=300, seed=1)
        self.assertEqual(len(a), len(b))
        self.assertEqual(a[S.TRACK_ID].tolist(), b[S.TRACK_ID].tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ══ the title filter: real titles from the live API ══════════════════════════

def _load_script(name: str):
    """Import a file in scripts/ as a module (they are CLIs, not a package)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Verbatim from the live API. These are ground truth, not fixtures we invented,
# which is the only reason the filter's rules are worth anything.
CANCER_CONTAMINATION = (
    "pov: my mom has cancer",
    "Cancer fighting songs",
    "Cancer Survivor Fight Songs",
    "Healing Cancer Frequencies",
    "POWERFUL KILLING CANCER CELLS WITH SOUND 10000HZ+741",
)
LEO_CONTAMINATION = (
    "LEO DAN MIX\U0001f60dEXITOS",
    "CARIN LEON MIX",
    "LEONARDO FAVIO ÉXITOS",
    "BEST OF LEON BRIDGES",
    "LOS INVASORES DE NUEVO LEON",
)


class TestTitleFilterRealTitles(unittest.TestCase):
    def test_every_measured_cancer_contamination_is_rejected(self):
        for title in CANCER_CONTAMINATION:
            cls, reason = F.classify_playlist_title(title, "Cancer")
            self.assertEqual(cls, F.REJECT, f"{title!r} -> {reason}")

    def test_every_measured_leo_contamination_is_rejected(self):
        for title in LEO_CONTAMINATION:
            cls, reason = F.classify_playlist_title(title, "Leo")
            self.assertEqual(cls, F.REJECT, f"{title!r} -> {reason}")

    def test_the_reasons_name_the_right_rule(self):
        """A reason that misattributes the cause is worse than no reason: the
        audit would send you to edit the wrong table."""
        expect = {
            "pov: my mom has cancer": "disease_context",
            "Cancer Survivor Fight Songs": "disease_context",
            "Healing Cancer Frequencies": "frequency_healing",
            "POWERFUL KILLING CANCER CELLS WITH SOUND 10000HZ+741": "frequency_healing",
        }
        for title, reason in expect.items():
            self.assertEqual(F.classify_playlist_title(title, "Cancer")[1], reason, title)
        for title in LEO_CONTAMINATION:
            self.assertEqual(F.classify_playlist_title(title, "Leo")[1],
                             "name_collision", title)

    def test_genuine_zodiac_titles_survive(self):
        for title, sign in [("cancer season ♋", "Cancer"),
                            ("leo energy", "Leo"),
                            ("aries vibes", "Aries"),
                            ("virgo", "Virgo"),
                            ("♓ pisces zodiac playlist", "Pisces")]:
            cls, reason = F.classify_playlist_title(title, sign)
            self.assertEqual(cls, F.KEEP, f"{title!r} -> {reason}")

    def test_the_emoji_alone_is_enough(self):
        # Measured as the strongest single signal: genuine playlists use it,
        # contaminated ones never do.
        self.assertEqual(F.classify_playlist_title("♋ playlist", "Cancer"),
                         (F.KEEP, "sign_emoji"))
        # The variation-selector form is the one that actually appears.
        self.assertEqual(F.classify_playlist_title("vibes ♋️", "Cancer")[0],
                         F.KEEP)
        # Another sign's emoji is not evidence for this one.
        self.assertEqual(F.classify_playlist_title("♌ stuff", "Cancer"),
                         (F.REJECT, "sign_not_in_title"))

    def test_ambiguous_signs_need_context_and_clean_ones_do_not(self):
        """Cancer and Leo are also a disease and a name; the other ten
        measured clean, so a bare 'virgo' is believed."""
        self.assertEqual(F.classify_playlist_title("cancer", "Cancer")[1],
                         "no_astro_context")
        self.assertEqual(F.classify_playlist_title("leo", "Leo")[1], "no_astro_context")
        self.assertEqual(F.classify_playlist_title("virgo", "Virgo"),
                         (F.KEEP, "sign_in_title"))
        self.assertEqual(F.classify_playlist_title("taurus", "Taurus")[0], F.KEEP)

    def test_strict_demands_context_everywhere(self):
        self.assertEqual(F.classify_playlist_title("virgo", "Virgo", strict=True)[1],
                         "no_astro_context")
        self.assertEqual(F.classify_playlist_title("virgo season", "Virgo",
                                                   strict=True)[0], F.KEEP)
        # ...and strict=False believes even Cancer on its name.
        self.assertEqual(F.classify_playlist_title("cancer", "Cancer", strict=False)[0],
                         F.KEEP)

    def test_word_boundaries_stop_leon_from_matching_leo(self):
        """Substring matching would call LEON BRIDGES an astrology playlist --
        this is why the tables compile to \\b-anchored patterns."""
        self.assertFalse(F.SIGN_WORD_RE["Leo"].search(F.normalise_title("LEON BRIDGES")))
        self.assertTrue(F.SIGN_WORD_RE["Leo"].search(F.normalise_title("leo season")))

    def test_accents_are_normalised_away(self):
        self.assertEqual(F.normalise_title("LEONARDO FAVIO ÉXITOS"),
                         "leonardo favio exitos")
        self.assertEqual(F.normalise_title("CARIN LEÓN MIX"), "carin leon mix")
        self.assertEqual(F.classify_playlist_title("CARIN LEÓN MIX", "Leo")[1],
                         "name_collision")

    def test_healing_frequencies_are_rejected_for_every_sign(self):
        """Sine tones, not music. Their audio features are extreme and
        consistent, so they are exactly what a feature model would latch on to.
        The rule beats even a sign emoji, which is why it runs first."""
        for sign in S.ZODIAC_SIGNS:
            for title in (f"{sign} 528hz healing", f"{sign} solfeggio frequencies",
                          f"432 hz {sign}", f"{sign} binaural beats"):
                self.assertEqual(F.classify_playlist_title(title, sign),
                                 (F.REJECT, "frequency_healing"), title)
        self.assertEqual(
            F.classify_playlist_title("cancer season ♋ 528hz", "Cancer"),
            (F.REJECT, "frequency_healing"))

    def test_hz_matches_when_glued_to_a_number(self):
        # "10000HZ+741" has no word boundary before the h.
        self.assertEqual(F.classify_playlist_title("aries 10000hz mix", "Aries")[1],
                         "frequency_healing")
        # ...but an ordinary number must not trip it.
        self.assertEqual(F.classify_playlist_title("aries 2024 vibes", "Aries")[0],
                         F.KEEP)

    def test_placements_are_tagged_not_rejected(self):
        for title, sign in [("Aries Moon", "Aries"), ("Taurus Venus", "Taurus"),
                            ("virgo rising", "Virgo"), ("leo mars playlist", "Leo")]:
            cls, reason = F.classify_playlist_title(title, sign)
            self.assertEqual(cls, F.PLACEMENT, f"{title!r} -> {reason}")
            self.assertTrue(reason.startswith("placement:"))

    def test_a_sun_placement_is_the_sun_sign(self):
        """'aries sun placement' is a sun sign described as a placement;
        'aries placements' is not."""
        self.assertEqual(F.classify_playlist_title("aries sun placement", "Aries")[0],
                         F.KEEP)
        self.assertEqual(F.classify_playlist_title("aries placements", "Aries")[0],
                         F.PLACEMENT)

    def test_a_twelve_sign_compilation_is_not_evidence_for_one_sign(self):
        self.assertEqual(
            F.classify_playlist_title("zodiac mix ♈♉♊♋", "Aries"),
            (F.REJECT, "multi_sign"))
        self.assertEqual(
            F.classify_playlist_title("aries taurus gemini cancer songs", "Aries")[1],
            "multi_sign")
        # Two signs is still a playlist about those signs.
        self.assertEqual(F.classify_playlist_title("aries and taurus vibes", "Aries")[0],
                         F.KEEP)

    def test_unrelated_titles_are_rejected(self):
        self.assertEqual(F.classify_playlist_title("Top Hits 2024", "Aries"),
                         (F.REJECT, "sign_not_in_title"))

    def test_size_bounds(self):
        """The 4,078-track 'cancer energy♋️💥' playlist is real and would
        supply more rows than a hundred ordinary playlists."""
        monster = "cancer energy♋️\U0001f4a5"
        self.assertEqual(F.classify_playlist(monster, "Cancer", 4078),
                         (F.REJECT, "too_many_tracks"))
        self.assertEqual(F.classify_playlist(monster, "Cancer", 9),
                         (F.REJECT, "too_few_tracks"))
        self.assertEqual(F.classify_playlist(monster, "Cancer", 120)[0], F.KEEP)
        self.assertEqual(F.classify_playlist(monster, "Cancer", 4078,
                                             max_tracks=5000)[0], F.KEEP)

    def test_title_rules_are_reported_before_size_rules(self):
        """So the audit says why the title was not believable, rather than
        blaming its length."""
        self.assertEqual(F.classify_playlist("pov: my mom has cancer", "Cancer", 4078),
                         (F.REJECT, "disease_context"))

    def test_control_titles_only_meet_the_universal_rules(self):
        for title in ("shower songs", "gym motivation", "road trip 2024",
                      "late night study"):
            self.assertEqual(F.classify_playlist_title(title, None),
                             (F.KEEP, "control"), title)
        self.assertEqual(F.classify_playlist_title("study frequencies 528hz", None),
                         (F.REJECT, "frequency_healing"))

    def test_tables_can_be_edited_at_runtime(self):
        """The rules are empirical guesses that will need tuning, so a table
        edit plus rebuild_patterns() must be all it takes."""
        original = F.ASTRO_TERMS
        try:
            F.ASTRO_TERMS = original + ("supremacy",)
            F.rebuild_patterns()
            self.assertEqual(F.classify_playlist_title("cancer supremacy", "Cancer"),
                             (F.KEEP, "astro_context"))
        finally:
            F.ASTRO_TERMS = original
            F.rebuild_patterns()
        self.assertEqual(F.classify_playlist_title("cancer supremacy", "Cancer")[1],
                         "no_astro_context")


class TestFilterOnFrames(unittest.TestCase):
    def frame(self):
        rows = []
        spec = [("g1", "Cancer", "cancer season ♋", 30),
                ("g2", "Virgo", "virgo", 30),
                ("p1", "Aries", "aries moon", 30),
                ("b1", "Cancer", "Cancer Survivor Fight Songs", 30),
                ("b2", "Leo", "CARIN LEON MIX", 30),
                ("b3", "Aries", "aries energy 528hz", 30),
                ("b4", "Cancer", "cancer energy ♋", 600),
                ("b5", "Taurus", "taurus vibes", 4)]
        for pid, sign, name, n in spec:
            for i in range(n):
                rows.append({S.PLAYLIST_ID: pid, S.STAR_SIGN: sign,
                             S.PLAYLIST_NAME: name, S.TRACK_ID: f"{pid}t{i}"})
        return pd.DataFrame(rows)

    def test_classify_frame_one_row_per_playlist(self):
        v = F.classify_frame(self.frame())
        self.assertEqual(len(v), 8)
        by_id = v.set_index(S.PLAYLIST_ID)
        self.assertEqual(by_id.loc["g1", S.TITLE_CLASS], F.KEEP)
        self.assertEqual(by_id.loc["p1", S.TITLE_CLASS], F.PLACEMENT)
        self.assertEqual(by_id.loc["b1", S.TITLE_REASON], "disease_context")
        self.assertEqual(by_id.loc["b4", S.TITLE_REASON], "too_many_tracks")
        self.assertEqual(by_id.loc["b5", S.TITLE_REASON], "too_few_tracks")

    def test_size_comes_from_tracks_held_not_from_the_claimed_total(self):
        df = self.frame()
        df[S.PLAYLIST_TRACK_TOTAL] = 9999      # what Spotify claimed
        v = F.classify_frame(df).set_index(S.PLAYLIST_ID)
        self.assertEqual(v.loc["g1", "n_tracks"], 30)
        self.assertEqual(v.loc["g1", S.TITLE_CLASS], F.KEEP)

    def test_classify_frame_accepts_an_audit_shaped_frame(self):
        audit = pd.DataFrame([
            {S.PLAYLIST_ID: "x1", S.STAR_SIGN: "Cancer",
             S.PLAYLIST_NAME: "Cancer fighting songs", "n_tracks": 40},
            {S.PLAYLIST_ID: "x2", S.STAR_SIGN: "Virgo",
             S.PLAYLIST_NAME: "virgo season", "n_tracks": 40},
        ])
        v = F.classify_frame(audit).set_index(S.PLAYLIST_ID)
        self.assertEqual(v.loc["x1", S.TITLE_CLASS], F.REJECT)
        self.assertEqual(v.loc["x2", S.TITLE_CLASS], F.KEEP)

    def test_classify_frame_rejects_a_frame_it_cannot_size(self):
        with self.assertRaises(ValueError):
            F.classify_frame(pd.DataFrame({S.PLAYLIST_ID: ["a"],
                                           S.STAR_SIGN: ["Aries"],
                                           S.PLAYLIST_NAME: ["aries vibes"]}))

    def test_apply_filter_keeps_placements_by_default(self):
        kept, verdicts = F.apply_filter(self.frame())
        self.assertEqual(set(kept[S.PLAYLIST_ID]), {"g1", "g2", "p1"})
        self.assertIn(S.TITLE_CLASS, kept.columns)
        self.assertEqual(kept.loc[kept[S.PLAYLIST_ID] == "p1", S.TITLE_CLASS].iloc[0],
                         F.PLACEMENT)
        dropped, _ = F.apply_filter(self.frame(), drop_placements=True)
        self.assertEqual(set(dropped[S.PLAYLIST_ID]), {"g1", "g2"})

    def test_rejection_breakdown_names_a_reason_and_an_example(self):
        _, verdicts = F.apply_filter(self.frame())
        rej = F.rejection_breakdown(verdicts)
        reasons = dict(zip(rej[S.TITLE_REASON], rej["playlists"]))
        self.assertEqual(reasons["disease_context"], 1)
        self.assertEqual(reasons["name_collision"], 1)
        self.assertEqual(reasons["frequency_healing"], 1)
        row = rej[rej[S.TITLE_REASON] == "name_collision"].iloc[0]
        self.assertEqual(row["example"], "CARIN LEON MIX")

    def test_strictness_cost_counts_what_the_stricter_rule_would_drop(self):
        cost = F.strictness_cost(self.frame())
        # "virgo" survives the lenient rule and not the strict one.
        self.assertEqual(cost["would_drop"], 1)
        self.assertEqual(cost["by_sign"], {"Virgo": 1})
        self.assertEqual(cost["kept_strict"], cost["kept_lenient"] - 1)


# ══ filtering during the scrape ══════════════════════════════════════════════

class TestScrapeSignFiltering(unittest.TestCase):
    def test_contaminated_playlists_are_rejected_before_their_tracks_are_fetched(self):
        """This ordering is most of the scrape's cost: ~8 of 9 'cancer'
        results are rejected, and each rejection saves a track page."""
        sess = wire_full_scrape(FakeSession(), title="Cancer Survivor Fight Songs")
        sc_ = make_scraper(sess)
        df = sc_.scrape_sign("Cancer", n_playlists=3, queries=["cancer"])
        self.assertTrue(df.empty)
        self.assertEqual([c for c in sess.calls if c[0].endswith("/tracks")], [])
        self.assertEqual(sc_.stats["titles_rejected"], 3)
        self.assertEqual(sc_.rejections[("Cancer", "disease_context")], 3)

    def test_rejections_are_auditable(self):
        sess = wire_full_scrape(FakeSession(), title="LEO DAN MIX")
        sc_ = make_scraper(sess)
        sc_.scrape_sign("Leo", n_playlists=2, queries=["leo"])
        rej = sc_.rejection_frame()
        self.assertEqual(list(rej[S.TITLE_REASON]), ["name_collision"])
        self.assertEqual(int(rej["playlists"].iloc[0]), 2)
        self.assertEqual(rej["example"].iloc[0], "LEO DAN MIX")

    def test_kept_rows_carry_the_verdict(self):
        sess = wire_full_scrape(FakeSession(), title="aries season ♈")
        df = make_scraper(sess).scrape_sign("Aries", n_playlists=2, queries=["aries"])
        self.assertEqual(set(df[S.TITLE_CLASS]), {F.KEEP})
        self.assertEqual(set(df[S.TITLE_REASON]), {"sign_emoji"})
        self.assertFalse(df[S.IS_CONTROL].any())

    def test_placements_are_kept_and_tagged(self):
        sess = wire_full_scrape(FakeSession(), title="aries moon mix")
        df = make_scraper(sess).scrape_sign("Aries", n_playlists=2, queries=["aries"])
        self.assertFalse(df.empty)
        self.assertEqual(set(df[S.TITLE_CLASS]), {F.PLACEMENT})

    def test_oversized_playlist_is_skipped_without_a_fetch(self):
        sess = wire_full_scrape(FakeSession(), title="aries vibes", tracks_per=20)
        sess.routes = [r for r in sess.routes if r[0] != "/playlists/"]
        sess.route("/playlists/", lambda p, n: playlist_obj(
            "p0", total=4078, name="cancer energy ♋"))
        sc_ = make_scraper(sess)
        df = sc_.scrape_sign("Cancer", n_playlists=1, queries=["cancer"])
        self.assertTrue(df.empty)
        self.assertEqual(sc_.rejections[("Cancer", "too_many_tracks")], 1)

    def test_spotify_owned_playlists_are_skipped(self):
        """A no-op in practice -- measurement says Spotify's own Cosmic
        Playlists never surface in search -- but they are editorial, not one
        of twelve independent curators."""
        sess = wire_full_scrape(FakeSession(), owner="spotify")
        sc_ = make_scraper(sess)
        df = sc_.scrape_sign("Aries", n_playlists=2, queries=["aries"])
        self.assertTrue(df.empty)
        self.assertEqual(sc_.rejections[("Aries", "spotify_owned")], 2)

    def test_filter_can_be_turned_off_for_measurement(self):
        sess = wire_full_scrape(FakeSession(), title="Cancer Survivor Fight Songs")
        df = make_scraper(sess).scrape_sign("Cancer", n_playlists=2, queries=["cancer"],
                                            filter_titles=False)
        self.assertFalse(df.empty)

    def test_short_playlist_is_rejected_after_the_fetch_too(self):
        """tracks.total counts entries we drop (local files, episodes), so the
        size has to be re-checked against what we actually hold."""
        sess = wire_full_scrape(FakeSession(), tracks_per=20)
        sess.routes = [r for r in sess.routes if r[0] != "/playlists/"]

        def handler(params, n):
            pid = re.search(r"/playlists/([^/?]+)", sess.last_url).group(1)
            if sess.last_url.endswith("/tracks"):
                return {"items": [{"track": track_obj(1), "is_local": True}] * 20,
                        "next": None}
            return playlist_obj(pid, total=40, name="aries season")
        sess.route("/playlists/", handler)
        sc_ = make_scraper(sess)
        self.assertTrue(sc_.scrape_sign("Aries", n_playlists=1, queries=["aries"]).empty)
        self.assertEqual(sc_.rejections[("Aries", "too_few_tracks")], 1)


# ══ the negative control ════════════════════════════════════════════════════

class TestScrapeControl(unittest.TestCase):
    def test_labels_are_random_per_playlist_and_rows_are_marked(self):
        sess = wire_full_scrape(FakeSession(), n_playlists=12, title="study mood")
        sc_ = make_scraper(sess)
        df = sc_.scrape_control(n_playlists=12, themes=["study"], seed=1)
        self.assertTrue(df[S.IS_CONTROL].all())
        # One label per playlist: a curator is the unit of observation, so a
        # per-row label would not reproduce the real set's structure.
        per_playlist = df.groupby(S.PLAYLIST_ID)[S.STAR_SIGN].nunique()
        self.assertTrue((per_playlist == 1).all())
        self.assertGreater(df[S.STAR_SIGN].nunique(), 1)
        for sign in df[S.STAR_SIGN].unique():
            self.assertIn(sign, S.ZODIAC_SIGNS)

    def test_control_labels_are_reproducible_from_the_seed(self):
        def run(seed):
            sess = wire_full_scrape(FakeSession(), n_playlists=8, title="gym mood")
            df = make_scraper(sess).scrape_control(n_playlists=8, themes=["gym"],
                                                   seed=seed)
            return df.groupby(S.PLAYLIST_ID)[S.STAR_SIGN].first().to_dict()
        self.assertEqual(run(3), run(3))
        self.assertNotEqual(run(3), run(4))

    def test_control_labels_are_uniform_over_all_twelve_signs(self):
        """Not just 'random' -- uniform. A skewed control would make the real
        set's class imbalance look like signal."""
        sess = wire_full_scrape(FakeSession(), n_playlists=400, title="study mood")
        df = make_scraper(sess).scrape_control(n_playlists=400, themes=["study"], seed=7)
        counts = df.groupby(S.PLAYLIST_ID)[S.STAR_SIGN].first().value_counts()
        self.assertEqual(len(counts), 12)
        self.assertLess(counts.max() / counts.min(), 2.5)

    def test_control_applies_the_universal_title_rules(self):
        sess = wire_full_scrape(FakeSession(), n_playlists=4,
                                title="study frequencies 528hz")
        sc_ = make_scraper(sess)
        df = sc_.scrape_control(n_playlists=4, themes=["study"], seed=1)
        self.assertTrue(df.empty)
        self.assertEqual(sc_.rejections[("control", "frequency_healing")], 4)

    def test_control_does_not_apply_the_zodiac_title_rules(self):
        """'study' has no sign in it and must not be rejected for that."""
        sess = wire_full_scrape(FakeSession(), n_playlists=3, title="study mood")
        df = make_scraper(sess).scrape_control(n_playlists=3, themes=["study"], seed=1)
        self.assertEqual(df[S.PLAYLIST_ID].nunique(), 3)
        self.assertEqual(set(df[S.TITLE_REASON]), {"control"})

    def test_control_matches_the_zodiac_size_distribution(self):
        """Playlist length alone changes Jaccard clustering and how often a
        track recurs, so an unmatched control would not be a fair comparison."""
        reference = pd.DataFrame([
            {S.PLAYLIST_ID: f"z{p}", S.TRACK_ID: f"t{p}_{i}"}
            for p in range(5) for i in range(18)])
        sess = wire_full_scrape(FakeSession(), n_playlists=4, title="gym mood",
                                tracks_per=50)
        df = make_scraper(sess).scrape_control(n_playlists=4, themes=["gym"], seed=2,
                                               match_size_to=reference)
        sizes = df.groupby(S.PLAYLIST_ID)[S.TRACK_ID].nunique()
        self.assertTrue((sizes == 18).all(), f"got {sizes.to_dict()}")

    def test_control_resume_dedupes_across_fake_signs(self):
        """A playlist already fetched under a random 'Leo' must not be fetched
        again when the coin comes up Virgo."""
        with tempfile.TemporaryDirectory() as tmp:
            ck = sc.ScrapeCheckpoint(root=Path(tmp), tag="zodiac_control")
            sess = wire_full_scrape(FakeSession(), n_playlists=6, title="study mood")
            first = make_scraper(sess).scrape_control(n_playlists=6, themes=["study"],
                                                      checkpoint=ck, seed=5)
            self.assertEqual(first[S.PLAYLIST_ID].nunique(), 6)

            ck2 = sc.ScrapeCheckpoint(root=Path(tmp), tag="zodiac_control")
            self.assertEqual(len(ck2.all_done_playlists()), 6)
            sess2 = wire_full_scrape(FakeSession(), n_playlists=6, title="study mood")
            make_scraper(sess2).scrape_control(n_playlists=6, themes=["study"],
                                               checkpoint=ck2, seed=5)
            self.assertEqual([c for c in sess2.calls if "/playlists/" in c[0]], [])


class TestSyntheticControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = sc.synthetic_control(n_playlists=40, n_catalog=600, seed=4)

    def test_has_the_scrape_schema_and_is_marked(self):
        for col in sc.SCRAPE_COLUMNS:
            self.assertIn(col, self.df.columns)
        self.assertTrue(self.df[S.IS_CONTROL].all())

    def test_titles_are_not_astrological(self):
        for name in self.df[S.PLAYLIST_NAME].unique():
            self.assertEqual(F.classify_playlist_title(name, None)[0], F.KEEP, name)
            # And none of them accidentally names a sign.
            for sign in S.ZODIAC_SIGNS:
                self.assertEqual(
                    F.classify_playlist_title(name, sign)[1], "sign_not_in_title", name)

    def test_labels_are_per_playlist_and_cover_the_signs(self):
        per = self.df.groupby(S.PLAYLIST_ID)[S.STAR_SIGN].nunique()
        self.assertTrue((per == 1).all())
        self.assertGreaterEqual(self.df[S.STAR_SIGN].nunique(), 8)

    def test_copies_keep_their_parents_label(self):
        """Copying is what makes two playlists one opinion -- label included.
        A clone with a fresh random label would break that."""
        pairs = (self.df[[S.PLAYLIST_ID, "copy_of", S.STAR_SIGN]]
                 .astype(str).drop_duplicates())
        by_id = pairs.set_index(S.PLAYLIST_ID)[S.STAR_SIGN].to_dict()
        for _, r in pairs.iterrows():
            self.assertEqual(r[S.STAR_SIGN], by_id[r["copy_of"]])

    def test_the_whole_pipeline_runs_on_it(self):
        clusters = D.cluster_playlists(self.df, threshold=0.5)
        ess = D.effective_sample_size(self.df, clusters)
        self.assertGreater(ess["clusters"], 0)
        self.assertLessEqual(ess["ess_ratio"], 1.0)

    def test_is_size_matched_when_given_a_reference(self):
        ref = sc.synthetic_scrape(signs=["Aries"], n_playlists_per_sign=4,
                                  n_catalog=300, contaminate=False, seed=2)
        ctl = sc.synthetic_control(n_playlists=10, n_catalog=300,
                                   match_size_to=ref, seed=6)
        ref_sizes = ref.groupby(S.PLAYLIST_ID)[S.TRACK_ID].nunique()
        ctl_sizes = ctl.groupby(S.PLAYLIST_ID)[S.TRACK_ID].nunique()
        self.assertLessEqual(ctl_sizes.max(), ref_sizes.max())
        self.assertGreaterEqual(ctl_sizes.min(), 1)


class TestControlNeverLoadsAsRealData(unittest.TestCase):
    def test_load_zodiac_ignores_the_control_file(self):
        """The worst failure mode in the project would be training on random
        labels believing they were real, so this is enforced by filename and
        checked here."""
        catalog = sc.syn.make_catalog(200, seed=1)
        control = sc.synthetic_control(n_playlists=4, catalog=catalog, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            control.to_csv(raw / "zodiac_control.csv", index=False)
            with mock.patch.object(D_data, "RAW", raw):
                self.assertFalse(D_data.using_real_data()["zodiac"])
                loaded = D_data.load_zodiac(catalog=catalog, verbose=False,
                                            n_playlists_per_sign=2,
                                            tracks_per_playlist=5)
                self.assertNotIn(S.IS_CONTROL, loaded.columns)
                ids = set(loaded[S.PLAYLIST_ID]) if S.PLAYLIST_ID in loaded else set()
                self.assertFalse(ids & set(control[S.PLAYLIST_ID]))

            # The real file at the real name IS picked up, so the check above
            # is about the filename and not about load_zodiac being broken.
            real = sc.synthetic_scrape(signs=["Aries"], n_playlists_per_sign=2,
                                       catalog=catalog, contaminate=False, seed=1)
            real.to_csv(raw / "zodiac.csv", index=False)
            with mock.patch.object(D_data, "RAW", raw):
                self.assertTrue(D_data.using_real_data()["zodiac"])
                self.assertEqual(len(D_data.load_zodiac(verbose=False)), len(real))


# ══ the labelling aid ═══════════════════════════════════════════════════════

class TestLabelPlaylists(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lp = _load_script("label_playlists")

    def frame(self):
        spec = ([(f"g{i}", "Virgo", "virgo season", 30) for i in range(40)]
                + [(f"b{i}", "Cancer", "Cancer fighting songs", 30) for i in range(6)]
                + [("p0", "Aries", "aries moon", 30)]
                + [("f0", "Leo", "leo 528hz", 30)])
        rows = [{S.PLAYLIST_ID: pid, S.STAR_SIGN: sign, S.PLAYLIST_NAME: name,
                 S.PLAYLIST_OWNER: f"own-{pid}", S.TRACK_ID: f"{pid}t{i}"}
                for pid, sign, name, n in spec for i in range(n)]
        return pd.DataFrame(rows)

    def test_sheet_has_the_review_columns_and_an_empty_human_column(self):
        sheet = self.lp.sample_for_review(self.frame(), n=20, seed=0)
        self.assertEqual(list(sheet.columns), self.lp.REVIEW_COLUMNS)
        self.assertTrue((sheet["human_class"] == "").all())
        self.assertEqual(sheet[S.PLAYLIST_ID].nunique(), len(sheet))

    def test_sheet_includes_rejected_playlists_or_recall_is_unmeasurable(self):
        sheet = self.lp.sample_for_review(self.frame(), n=20, seed=0)
        self.assertIn(F.REJECT, set(sheet["auto_class"]))
        self.assertIn(F.PLACEMENT, set(sheet["auto_class"]))
        # Every rule that fired is represented, including the ones that fired
        # once -- those are the likeliest to be wrong.
        self.assertEqual(set(sheet["auto_reason"]),
                         {"astro_context", "disease_context", "placement:moon",
                          "frequency_healing"})

    def test_stratum_weights_add_back_up_to_the_population(self):
        """Equal allocation over-samples rare rules on purpose, so the weights
        are what make the scored rates population estimates."""
        df = self.frame()
        sheet = self.lp.sample_for_review(df, n=20, seed=0)
        self.assertAlmostEqual(sheet["stratum_weight"].sum(),
                               df[S.PLAYLIST_ID].nunique(), places=4)

    def test_score_is_the_hand_computed_one(self):
        """4 kept-and-right, 1 kept-but-wrong, 2 rejected-but-genuine,
        3 rejected-and-right  ->  precision 4/5, recall 4/6."""
        sheet = pd.DataFrame([
            ("keep", "k1", "keep"), ("keep", "k2", "keep"),
            ("placement", "k3", "placement"), ("placement", "k4", "keep"),
            ("keep", "k5", "reject"),
            ("reject", "r1", "keep"), ("reject", "r2", "placement"),
            ("reject", "r3", "reject"), ("reject", "r4", "reject"),
            ("reject", "r5", "reject"),
        ], columns=["auto_class", S.PLAYLIST_NAME, "human_class"])
        sheet["auto_reason"] = sheet["auto_class"]
        score = self.lp.score_filter(sheet)
        self.assertEqual(score["reviewed"], 10)
        self.assertAlmostEqual(score["unweighted"]["precision"], 4 / 5)
        self.assertAlmostEqual(score["unweighted"]["recall"], 4 / 6)
        self.assertAlmostEqual(score["unweighted"]["f1"],
                               2 * (4 / 5) * (4 / 6) / ((4 / 5) + (4 / 6)))
        # Exact 3-class agreement is stricter than the binary rates: k4
        # (placement called keep) and r2 (reject called placement) are
        # counted right by the binary collapse and wrong here. 6 of 10.
        self.assertAlmostEqual(score["exact_agreement"], 0.6)

    def test_placement_counts_as_a_positive_decision(self):
        """A placement playlist is genuine astrology, just a different
        question -- scoring it as a rejection would understate recall."""
        sheet = pd.DataFrame([("placement", "x", "keep"), ("placement", "y", "placement")],
                             columns=["auto_class", S.PLAYLIST_NAME, "human_class"])
        sheet["auto_reason"] = "placement:moon"
        score = self.lp.score_filter(sheet)
        self.assertAlmostEqual(score["unweighted"]["precision"], 1.0)
        self.assertAlmostEqual(score["unweighted"]["recall"], 1.0)

    def test_weights_change_the_estimate(self):
        """One over-sampled rare stratum must not dominate the population
        estimate -- which is the whole reason the column exists."""
        sheet = pd.DataFrame([
            ("keep", "a", "keep", 100.0), ("keep", "b", "keep", 100.0),
            ("keep", "c", "reject", 1.0),
        ], columns=["auto_class", S.PLAYLIST_NAME, "human_class", "stratum_weight"])
        sheet["auto_reason"] = "r"
        score = self.lp.score_filter(sheet)
        self.assertAlmostEqual(score["unweighted"]["precision"], 2 / 3)
        self.assertAlmostEqual(score["weighted"]["precision"], 200 / 201)

    def test_unfilled_sheets_are_reported_not_scored(self):
        sheet = pd.DataFrame({"auto_class": ["keep"], "auto_reason": ["x"],
                              S.PLAYLIST_NAME: ["n"], "human_class": [""]})
        self.assertEqual(self.lp.score_filter(sheet), {"reviewed": 0})

    def test_per_reason_errors_name_the_rule_to_edit(self):
        sheet = pd.DataFrame([
            ("reject", "disease_context", "Cancer Survivor Fight Songs", "reject"),
            ("reject", "disease_context", "healing cancer season", "keep"),
            ("keep", "sign_in_title", "virgo", "keep"),
        ], columns=["auto_class", "auto_reason", S.PLAYLIST_NAME, "human_class"])
        by_reason = self.lp.score_filter(sheet)["by_reason"].set_index("auto_reason")
        self.assertEqual(int(by_reason.loc["disease_context", "disagreements"]), 1)
        self.assertEqual(int(by_reason.loc["sign_in_title", "disagreements"]), 0)

    def test_round_trip_through_a_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            sheet = self.lp.sample_for_review(self.frame(), n=12, seed=1)
            sheet.to_csv(path, index=False)
            back = pd.read_csv(path).fillna({"human_class": ""})
            back["human_class"] = back["auto_class"]        # a perfect reviewer
            score = self.lp.score_filter(back)
            self.assertEqual(score["exact_agreement"], 1.0)
            self.assertAlmostEqual(score["weighted"]["precision"], 1.0)


# ══ the second measurement: 85-131 real titles per sign ═════════════════════
# Everything in this section is verbatim from the live API. The lenient
# default was CONFIRMED by it (strict mode cuts keeps ~40% per sign and the
# titles it discards are genuine), and "virgo is 9 of 9 clean" was RETRACTED
# by it -- nine titles was too small a sample.

MEASURED_COLLISIONS = {
    "Libra": ("Grupo Libra – Puras Romanticas",
              "GRUPO LIBRA RANCHERAS",
              "El Grupo Libra-Éxitos Inmortales",
              "Sergio Mendivil y Grupo Libra-Puras Romanticas-",
              "Ladrón,grupo libra,Mandingo,los rehenes,bronco"),
    "Virgo": ("Best Of Romain Virgo", "VIRGO’S GROOVE"),
    "Gemini": ("Gemini Home Entertainment",),
}

MEASURED_GENUINE = {
    "Taurus": ("Taurus Playlist (Made by a Taurus)", "✨Taurus Shit✨",
               "2026 TAURUS"),
    "Scorpio": ("she's a scorpio",
                "scorpio woman tries to enter her dark feminine main character"),
    "Pisces": ("Can you tell I'm a Pisces? \U0001f90d\U0001f972",
               "the real pisces playlist", "Pisces core\U0001f49e"),
    "Libra": ("songs for a libra",),
    "Gemini": ("gemini radio",),
    "Virgo": ("virgo",),
}


class TestMeasuredNameCollisions(unittest.TestCase):
    def test_every_measured_collision_is_rejected(self):
        for sign, titles in MEASURED_COLLISIONS.items():
            for title in titles:
                cls, reason = F.classify_playlist_title(title, sign)
                self.assertEqual(cls, F.REJECT, f"{sign}: {title!r} -> {reason}")
                self.assertEqual(reason, "name_collision", title)

    def test_grupo_libra_variants(self):
        """5 of the 11 at-risk Libra titles are this band, and the real ones
        use every separator going -- plus the frontman is often credited in
        front of the band name."""
        for title in ("grupo libra", "GRUPO-LIBRA exitos", "grupolibra",
                      "El Grupo Libra-Exitos", "x,grupo libra,y",
                      "Sergio Mendivil y Grupo Libra"):
            self.assertEqual(F.classify_playlist_title(title, "Libra")[1],
                             "name_collision", title)

    def test_virgos_groove_in_both_apostrophe_spellings(self):
        """A Beyonce track: playlists named for it are about the song. The
        curly apostrophe is the one the API actually returned."""
        for title in ("VIRGO'S GROOVE", "VIRGO’S GROOVE", "virgos groove",
                      "virgo’s groove playlist"):
            self.assertEqual(F.classify_playlist_title(title, "Virgo")[1],
                             "name_collision", title)

    def test_curly_apostrophes_fold_to_ascii(self):
        self.assertEqual(F.normalise_title("VIRGO’S GROOVE"), "virgo's groove")

    def test_a_collision_only_affects_its_own_sign(self):
        """'Grupo Libra' is not evidence about Virgo, and the rule tables are
        per sign so it cannot become so."""
        self.assertEqual(F.classify_playlist_title("Grupo Libra", "Virgo")[1],
                         "sign_not_in_title")
        self.assertEqual(F.classify_playlist_title("Best Of Romain Virgo", "Libra")[1],
                         "sign_not_in_title")


class TestMeasuredGenuineTitlesSurvive(unittest.TestCase):
    def test_every_measured_genuine_title_is_kept(self):
        for sign, titles in MEASURED_GENUINE.items():
            for title in titles:
                cls, reason = F.classify_playlist_title(title, sign)
                self.assertNotEqual(cls, F.REJECT, f"{sign}: {title!r} -> {reason}")

    def test_strict_mode_would_eat_all_of_them(self):
        """The measurement that settled the default: strict mode cuts keeps by
        ~40% per sign (Virgo 85->57, Pisces 83->48, Aquarius 78->48) and these
        are the titles it takes. Hence lenient is the default."""
        eaten = [t for sign, titles in MEASURED_GENUINE.items() for t in titles
                 if F.classify_playlist_title(t, sign, strict=True)[0] == F.REJECT]
        self.assertEqual(len(eaten), sum(len(v) for v in MEASURED_GENUINE.values()))

    def test_gemini_radio_is_not_blocked(self):
        """A sign radio playlist is an ordinary genuine title; blocking it to
        catch Gemini Home Entertainment would cost more than it saves."""
        self.assertEqual(F.classify_playlist_title("gemini radio", "Gemini")[0], F.KEEP)

    def test_ambiguous_signs_stays_at_two(self):
        """A guard, not a tautology: the fix for Libra/Virgo/Gemini is a
        targeted blocklist row. Adding them here instead would reject
        'songs for a libra' and ~40% of each sign's genuine playlists."""
        self.assertEqual(set(F.AMBIGUOUS_SIGNS), {"Cancer", "Leo"})
        for sign in ("Libra", "Virgo", "Gemini"):
            self.assertIn(sign, F.SIGN_REJECT)


class TestDiseaseRuleNarrowed(unittest.TestCase):
    def test_bare_healing_no_longer_rejects(self):
        """'healing' was the one disease term with real false-positive risk,
        and it was doing little work: the actual tone playlists are caught by
        frequency_healing first."""
        self.assertNotEqual(
            F.classify_playlist_title("cancer season ♋ healing energy",
                                      "Cancer")[0], F.REJECT)
        self.assertNotEqual(
            F.classify_playlist_title("cancer season heals me", "Cancer")[0], F.REJECT)

    def test_the_actual_tone_playlist_is_still_rejected(self):
        self.assertEqual(F.classify_playlist_title("Healing Cancer Frequencies",
                                                   "Cancer"),
                         (F.REJECT, "frequency_healing"))

    def test_disease_titles_without_an_astrology_word_still_fail(self):
        """Why narrowing is safe: Cancer is in AMBIGUOUS_SIGNS, so a
        disease-shaped title with no astrology word is rejected anyway."""
        self.assertEqual(F.classify_playlist_title("healing after cancer", "Cancer"),
                         (F.REJECT, "no_astro_context"))

    def test_the_rest_of_the_disease_table_still_fires(self):
        for title in ("Cancer Survivor Fight Songs", "cancer season chemo playlist",
                      "pov: my mom has cancer", "cancer awareness vibes"):
            self.assertEqual(F.classify_playlist_title(title, "Cancer")[1],
                             "disease_context", title)


class TestPlantedTitlesInTheDryRun(unittest.TestCase):
    """The dry run must exercise the filter in BOTH directions."""

    @classmethod
    def setUpClass(cls):
        cls.df = sc.synthetic_scrape(n_playlists_per_sign=6, n_catalog=900, seed=11)
        cls.kept, cls.verdicts = F.apply_filter(cls.df)

    def test_all_planted_collisions_are_filtered_out(self):
        planted = {t for titles in sc.CONTAMINATED_TITLES.values() for t in titles}
        survivors = planted & set(self.kept[S.PLAYLIST_NAME])
        self.assertFalse(survivors, f"contaminated titles survived: {survivors}")

    def test_all_planted_genuine_titles_survive(self):
        planted = {t for titles in sc.GENUINE_TITLES.values() for t in titles}
        present = planted & set(self.df[S.PLAYLIST_NAME])
        self.assertTrue(present, "no genuine titles were planted")
        self.assertEqual(present - set(self.kept[S.PLAYLIST_NAME]), set())

    def test_the_new_rules_remove_exactly_the_new_collisions(self):
        """Before/after on the dry-run corpus: switching the three new
        blocklist rows off puts 8 playlists back, and they are the 8 planted
        ones -- no collateral."""
        saved = {k: F.SIGN_REJECT[k] for k in ("Libra", "Virgo", "Gemini")}
        try:
            for k in saved:
                F.SIGN_REJECT[k] = ()
            lenient_kept, _ = F.apply_filter(self.df)
        finally:
            for k, v in saved.items():
                F.SIGN_REJECT[k] = v
        extra = set(lenient_kept[S.PLAYLIST_NAME]) - set(self.kept[S.PLAYLIST_NAME])
        expected = {t for s_ in saved for t in sc.CONTAMINATED_TITLES.get(s_, ())}
        self.assertEqual(extra, expected)
