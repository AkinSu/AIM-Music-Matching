"""Test fixtures: a real app, a fake Spotify, and no network.

Two rules shape everything here.

**No real Spotify calls.** The project's credentials have already earned a
20-hour rate-limit ban once. Every test runs against FakeSpotifyClient, which
is injected into the app the same way the real client is.

**Two browsers means two cookie jars.** Flask's test client keeps its own
cookies, so `app.test_client()` twice is genuinely two devices -- which is
the only honest way to test the bug this work exists to fix.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

# These must be set before `import lovify`: the module reads them at import
# time, and deliberately refuses to start without SECRET_KEY.
os.environ["SECRET_KEY"] = "test-key-not-a-real-secret-0123456789abcdef"
# A local BASE_URL keeps SESSION_COOKIE_SECURE off, so the test client (which
# speaks plain http) actually keeps its cookies.
os.environ["BASE_URL"] = "http://127.0.0.1:8080"
os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(tempfile.mkdtemp()) / "test.db")
os.environ.setdefault("SPOTIFY_CLIENT_ID", "test-client")
os.environ.setdefault("SPOTIFY_CLIENT_SECRET", "test-client-secret")

import pytest  # noqa: E402

import lovify  # noqa: E402
import models  # noqa: E402
from models import Base, TrackFeature  # noqa: E402
from spotify_client import FakeSpotifyClient, SpotifyProfile  # noqa: E402

MIN_TRACKS = 100


class StubScorer:
    """Stands in for LovifyModels with the same refusal contract.

    The real scorer needs a joblib artifact that is gitignored, so it cannot
    be loaded in CI. What matters for the app is the *contract*: it returns a
    status, and it refuses to produce a percentile below MIN_LIBRARY_TRACKS.
    That refusal is reproduced here so the app's handling of it is exercised;
    the correctness of the refusal itself belongs to ml/tests.
    """

    unavailable = None

    def __init__(self, min_tracks: int = MIN_TRACKS, ready: bool = True):
        self.min_tracks = min_tracks
        self.ready = ready
        self.calls = 0

    def compatibility(self, tracks_a, tracks_b) -> dict:
        self.calls += 1
        n_a, n_b = len(tracks_a), len(tracks_b)
        if not n_a or not n_b:
            return {"status": "no_resolvable_tracks", "percentile": None,
                    "tracks": {"a": n_a, "b": n_b},
                    "interpretation": "Nothing resolvable to compare."}
        if min(n_a, n_b) < self.min_tracks:
            return {"status": "insufficient_library", "percentile": None,
                    "tracks": {"a": n_a, "b": n_b},
                    "min_required": self.min_tracks,
                    "shared_artists": ["Fleetwood Mac"],
                    "interpretation": (
                        f"Not enough listening history to score this fairly "
                        f"yet ({min(n_a, n_b)} tracks; {self.min_tracks} needed).")}
        return {"status": "ok", "percentile": 82.4,
                "raw_score": 0.71,
                "tracks": {"a": n_a, "b": n_b},
                "components": {"audio_cosine": 0.68, "genre_overlap": 0.42},
                "shared_artists": ["Fleetwood Mac", "Sade"],
                "n_shared_artists": 7,
                "top_shared_genres": ["soul", "indie"],
                "interpretation": "Clearly more alike than most pairs."}


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def scorer():
    return StubScorer()


@pytest.fixture
def spotify():
    return FakeSpotifyClient()


@pytest.fixture
def app(spotify, scorer):
    """A clean database and an injected Spotify for every test."""
    Base.metadata.drop_all(lovify.engine)
    models.init_db(lovify.engine)
    lovify.app.config.update(TESTING=True)
    lovify.app.spotify = spotify
    lovify.app.scorer = scorer
    return lovify.app


@pytest.fixture
def db():
    session = lovify.Session()
    yield session
    session.close()


# ── helpers ──────────────────────────────────────────────────────────────────

def track_ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(n)]


def seed_features(db, ids) -> None:
    """Put audio features in the store for these track ids.

    Tests choose coverage explicitly by seeding a subset: in production only
    ~41% of a real library resolves, and pretending otherwise would hide the
    outcomes that matter most.
    """
    for i, tid in enumerate(ids):
        db.merge(TrackFeature(
            track_id=tid,
            danceability=0.5 + (i % 10) / 40, energy=0.4 + (i % 7) / 30,
            speechiness=0.05, acousticness=0.2, instrumentalness=0.01,
            liveness=0.15, valence=0.3 + (i % 5) / 20, tempo=100 + i % 40,
            loudness=-8.0, duration_ms=210000, genre="soul",
            artist=f"artist-{i % 12}", popularity=60, year=2015))
    db.commit()


def give_library(spotify, code: str, name: str, top=(), saved=(), recent=()):
    """Register one fake Spotify user and what we would read from them."""
    spotify.profiles[code] = SpotifyProfile(
        spotify_id=f"sp-{code}", display_name=name,
        photo_url=f"https://img.test/{code}.jpg")
    spotify.tracks[code] = list(top)
    spotify.saved[code] = list(saved)
    spotify.recent[code] = list(recent)


def oauth_roundtrip(client, start_url: str, code: str):
    """Do what a browser does: start the login, come back from Spotify.

    Reads the signed `state` out of the redirect rather than minting one, so
    the test exercises the real state the app issued, in the browser session
    the app issued it to.
    """
    started = client.get(start_url)
    assert started.status_code == 302, started.data[:300]
    state = parse_qs(urlparse(started.headers["Location"]).query)["state"][0]
    return client.get(f"/callback?code={code}&state={state}"), state


def match_id_from(response) -> str:
    """Pull the match id out of the redirect to /m/<id>."""
    assert response.status_code == 302, response.data[:400]
    return urlparse(response.headers["Location"]).path.rsplit("/", 1)[-1]
