"""The endings that are not a score, and the ones that are our fault.

Feature-store coverage against real Spotify libraries is about 41%, so "we
could not resolve enough of your music" is not an edge case -- it is a
routine outcome for ordinary users. These tests pin down that each such
ending reaches a page that says what happened, rather than a stack trace or
a confidently invented number.
"""

from __future__ import annotations

import json

from conftest import (
    StubScorer, give_library, match_id_from, oauth_roundtrip, seed_features,
    track_ids,
)
from models import Match
from spotify_client import NOT_ALLOWLISTED, SpotifyError


def _pair(app, spotify, db, a_tracks, b_tracks, seeded):
    seed_features(db, seeded)
    give_library(spotify, "a-code", "Ada", saved=a_tracks)
    give_library(spotify, "b-code", "Bo", saved=b_tracks)
    first, second = app.test_client(), app.test_client()
    match_id = match_id_from(oauth_roundtrip(first, "/login", "a-code")[0])
    oauth_roundtrip(second, f"/m/{match_id}/join", "b-code")
    return match_id, first, second


def test_too_little_resolvable_history_refuses_to_guess(app, spotify, db):
    """Below MIN_LIBRARY_TRACKS the scorer declines, and we say so.

    The refusal is the feature. A score built on 40 tracks is not noisier,
    it is biased downward by tens of percentile points in a direction the
    user cannot detect.
    """
    a, b = track_ids("a", 120), track_ids("b", 120)
    # Only 40 of Bo's tracks exist in the feature store.
    match_id, first, _ = _pair(app, spotify, db, a, b, a + b[:40])

    db.expire_all()
    assert db.get(Match, match_id).status == "insufficient"

    page = first.get(f"/m/{match_id}")
    assert page.status_code == 200
    text = page.get_data(as_text=True)
    assert "Not enough listening history" in text
    assert "100 needed" in text
    assert "Fleetwood Mac" in text, "exact facts are still worth showing"


def test_music_we_have_no_audio_data_for_is_its_own_outcome(app, spotify, db):
    a, b = track_ids("a", 120), track_ids("b", 120)
    match_id, first, _ = _pair(app, spotify, db, a, b, a)   # none of Bo's seeded

    db.expire_all()
    assert db.get(Match, match_id).status == "unresolved"

    text = first.get(f"/m/{match_id}").get_data(as_text=True)
    assert "could not" in text.lower()
    assert "Bo" in text, "name who we could not resolve, not 'user two'"
    assert "November 2024" in text, "explain that this is our limit, not theirs"


def test_a_missing_model_artifact_degrades_instead_of_crashing(app, spotify, db):
    """`ml/artifacts/` is gitignored; a fresh clone has no trained model.

    The match flow, the OAuth round trip and the database are all useful
    without it, so the site stays up and says scoring is not configured.
    """
    app.scorer = StubScorer(ready=False)
    a, b = track_ids("a", 120), track_ids("b", 120)
    match_id, first, _ = _pair(app, spotify, db, a, b, a + b)

    db.expire_all()
    assert db.get(Match, match_id).status == "failed"

    page = first.get(f"/m/{match_id}")
    assert page.status_code == 200
    assert "not available" in page.get_data(as_text=True)


def test_the_allowlist_refusal_tells_people_what_to_do(app, spotify, db):
    """The likeliest failure on demo day: Spotify's FIVE-user dev-mode cap.

    Spotify's own wording for it ("User not registered in the Developer
    Dashboard") tells the person nothing they can act on, so we replace it.

    The number was 25 everywhere in this project until it was checked against
    the documentation on 2026-10-13. It is five, both people in a match count
    against it, and extended quota now needs 250k monthly active users, so
    there is no version of this app that more than five people can log into.
    https://developer.spotify.com/documentation/web-api/concepts/quota-modes
    """
    spotify.fail_with = SpotifyError(NOT_ALLOWLISTED, allowlist=True)

    client = app.test_client()
    state = client.get("/login").headers["Location"].split("state=")[1]
    refused = client.get(f"/callback?code=x&state={state}")

    assert refused.status_code == 403
    text = refused.get_data(as_text=True)
    assert "allowlist" in text and "5 people" in text
    assert "25" not in text, "the cap is five; 25 was wrong for the life of the project"


def test_a_spotify_outage_is_a_bad_gateway_not_a_crash(app, spotify, db):
    spotify.fail_with = SpotifyError("Spotify is not responding.")
    client = app.test_client()
    state = client.get("/login").headers["Location"].split("state=")[1]
    broken = client.get(f"/callback?code=x&state={state}")
    assert broken.status_code == 502


def test_the_counts_nobody_has_measured_are_recorded(app, spotify, db):
    """Fetched, resolved, coverage and per-source counts, on the row.

    Whether a real Spotify library clears MIN_LIBRARY_TRACKS after a ~41%
    feature-store join is the biggest open question in this design. The first
    real match should answer it, which means the numbers have to be written
    down rather than inferred later from a score that may not exist.
    """
    a = track_ids("a", 150)
    b = track_ids("b", 120)
    seed_features(db, a[:90] + b)
    give_library(spotify, "a-code", "Ada", top=a[:50], saved=a[50:130],
                 recent=a[130:])
    give_library(spotify, "b-code", "Bo", saved=b)

    first, second = app.test_client(), app.test_client()
    match_id = match_id_from(oauth_roundtrip(first, "/login", "a-code")[0])
    oauth_roundtrip(second, f"/m/{match_id}/join", "b-code")

    db.expire_all()
    ada = db.get(Match, match_id).inviter
    assert ada.n_tracks == 150
    assert ada.n_resolved == 90
    assert ada.coverage == 0.6
    assert json.loads(ada.sources_json) == {
        "top_short_term": 50, "saved": 80, "recently_played": 20}
