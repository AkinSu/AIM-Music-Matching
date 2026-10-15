"""The library fetch, against a fake HTTP layer.

This covers the real SpotifyClient, not the test double, because the fetch
strategy is the part of this work most likely to be wrong in a way nothing
else catches.

Why it matters: top tracks alone return roughly 110 unique ids across the
three time ranges, and only about 60% of those resolve against the feature
store -- around 66 usable tracks, under the 100 the scorer requires before
it will report anything. Both people would land in "insufficient history"
and the product would refuse almost every real user. Saved songs are what
carry an ordinary listener over the line, so "we also read saved songs and
recent plays" is a correctness property, not an optimisation.

Not tested here, deliberately: anything that touches the network.
"""

from __future__ import annotations

import pytest

from spotify_client import NOT_ALLOWLISTED, SpotifyClient, SpotifyError


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeHTTP:
    """Just enough of requests.Session to drive the client offline."""

    def __init__(self, top=(), saved=(), recent=(), status=200):
        # `top` may be a flat list (same for every time range, which is the
        # worst case for overlap) or a dict keyed by time_range.
        self.top = top if isinstance(top, dict) else {
            tr: list(top) for tr in ("short_term", "medium_term", "long_term")}
        self.saved, self.recent = list(saved), list(recent)
        self.status = status
        self.requests = []

    def get(self, url, headers=None, params=None, timeout=None):
        params = params or {}
        self.requests.append((url, params))
        if self.status != 200:
            return FakeResponse({}, self.status)
        offset, limit = params.get("offset", 0), params.get("limit", 50)
        if url.endswith("/me/top/tracks"):
            page = self.top.get(params["time_range"], [])[offset:offset + limit]
            return FakeResponse({"items": [{"id": t} for t in page]})
        if url.endswith("/me/tracks"):
            page = self.saved[offset:offset + limit]
            return FakeResponse({"items": [{"track": {"id": t}} for t in page]})
        if url.endswith("/me/player/recently-played"):
            page = self.recent[:limit]
            return FakeResponse({"items": [{"track": {"id": t}} for t in page]})
        raise AssertionError(f"unexpected call to {url}")


def client(http):
    return SpotifyClient("id", "secret", "http://localhost/callback", session=http)


def test_saved_songs_are_what_get_a_normal_listener_over_the_line():
    """The measured shortfall, reproduced and then fixed.

    Three top-track time ranges overlap heavily -- here they share 40 of
    their 50 tracks, which is typical -- so they yield about 110 unique ids
    between them. At ~60% feature-store coverage that is roughly 66 usable
    tracks, below the 100 the scorer requires. Saved songs are the only
    source with the volume to close that gap.
    """
    shared = [f"t{i}" for i in range(40)]
    top = {"short_term": shared + [f"short{i}" for i in range(10)],
           "medium_term": shared + [f"med{i}" for i in range(10)],
           "long_term": shared + [f"long{i}" for i in range(10)]}
    http = FakeHTTP(top=top, saved=[f"s{i}" for i in range(400)])

    tops_only = sum(client(http).collect_library("token", budget=400)["sources"][k]
                    for k in ("top_short_term", "top_medium_term", "top_long_term"))
    assert tops_only == 70, "top tracks alone are nowhere near 100 after coverage"

    library = client(FakeHTTP(top=top, saved=[f"s{i}" for i in range(400)])
                     ).collect_library("token", budget=400)
    assert library["sources"]["saved"] == 330
    assert len(library["track_ids"]) == 400, "enough to survive a 41% join"


def test_saved_songs_are_paged_rather_than_capped_at_one_request():
    http = FakeHTTP(saved=[f"s{i}" for i in range(300)])
    ids = client(http).saved_track_ids("token", limit=250)

    assert len(ids) == 250
    offsets = [p.get("offset") for u, p in http.requests if u.endswith("/me/tracks")]
    assert offsets == [0, 50, 100, 150, 200], "must page, not stop at 50"


def test_recently_played_is_not_paged_because_the_endpoint_has_no_offset():
    http = FakeHTTP(recent=[f"r{i}" for i in range(50)])
    client(http).recently_played_ids("token", limit=200)

    url, params = http.requests[0]
    assert "offset" not in params
    assert len(http.requests) == 1


def test_ids_are_deduplicated_across_sources():
    """A song can be a top track, a saved track and a recent play at once."""
    http = FakeHTTP(top=["x", "y"], saved=["y", "z"], recent=["z", "x"])
    library = client(http).collect_library("token", budget=100)

    assert library["track_ids"] == ["x", "y", "z"]
    assert library["sources"] == {"top_short_term": 2, "top_medium_term": 0,
                                  "top_long_term": 0, "saved": 1,
                                  "recently_played": 0}


def test_the_budget_is_respected_and_the_best_signal_survives_it():
    """Top tracks come first, so truncation drops the weakest signal."""
    http = FakeHTTP(top=[f"t{i}" for i in range(50)],
                    saved=[f"s{i}" for i in range(500)])
    library = client(http).collect_library("token", budget=120)

    assert len(library["track_ids"]) == 120
    assert library["track_ids"][:50] == [f"t{i}" for i in range(50)]


def test_one_empty_source_does_not_sink_the_whole_fetch():
    """A new account with no saved songs should still get as far as it can."""
    http = FakeHTTP(top=["a", "b", "c"], saved=[], recent=["d"])
    library = client(http).collect_library("token", budget=400)

    assert library["track_ids"] == ["a", "b", "c", "d"]
    assert library["sources"]["saved"] == 0


def test_the_allowlist_failure_is_never_swallowed():
    """Tolerating a dead source must not hide 'you are not allowed in'."""
    http = FakeHTTP(status=403)
    with pytest.raises(SpotifyError) as caught:
        client(http).collect_library("token")

    assert caught.value.allowlist is True
    assert caught.value.args[0] == NOT_ALLOWLISTED


def test_the_account_chooser_is_forced_on_the_authorize_url():
    """Without it, the second person on a shared laptop silently reuses the
    first person's Spotify session and is matched against themselves."""
    url = client(FakeHTTP()).authorize_url("some-state")
    assert "show_dialog=true" in url
    assert "user-library-read" in url and "user-read-recently-played" in url


def test_the_default_budget_is_sized_against_measured_coverage():
    """Why 400 and not 150.

    Resolution against the feature store is 41% overall and ~60% for the
    popular tracks that dominate a personal library. 400 fetched therefore
    yields 164 usable in the pessimistic case -- still clear of the 100 the
    scorer needs. The honest fix for a shortfall is a bigger fetch, never a
    smaller MIN_LIBRARY_TRACKS: that threshold is 100 because below it the
    percentile is biased by tens of points.
    """
    import lovify
    from scoring import MIN_LIBRARY_TRACKS

    worst_case_coverage = 0.41
    assert lovify.LIBRARY_BUDGET * worst_case_coverage >= MIN_LIBRARY_TRACKS


# ── deployment wiring ────────────────────────────────────────────────────────

def test_a_hosting_provider_database_url_is_made_openable():
    """Both corrections fail only in deployment, so they are tested here."""
    from models import normalise_database_url

    assert normalise_database_url("sqlite:///x.db") == "sqlite:///x.db"
    # Railway's scheme, which SQLAlchemy 2 does not recognise.
    assert normalise_database_url("postgres://u:p@h/db").startswith("postgresql")
    # An explicit driver is never second-guessed.
    assert (normalise_database_url("postgresql+psycopg2://u@h/db")
            == "postgresql+psycopg2://u@h/db")
