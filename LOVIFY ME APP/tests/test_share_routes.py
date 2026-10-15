"""The two links, exercised through real HTTP in separate browsers.

The point of the split is only provable at the route layer: a stranger
holding the invite link must get nothing, and the same stranger holding a
public link must get a score and no way to join.
"""

from __future__ import annotations

import secrets

import pytest

import security
from conftest import (
    give_library, match_id_from, oauth_roundtrip, seed_features, track_ids,
)

ENOUGH = 140


def csrf(client) -> str:
    """The token this browser would have had rendered into its form.

    Planted in the session rather than scraped out of the HTML, so these
    tests stay about sharing. That a missing or mismatched token is actually
    refused is asserted directly, below.
    """
    with client.session_transaction() as sess:
        existing = sess.get(security.CSRF_SESSION_KEY)
        if not existing:
            existing = secrets.token_urlsafe(24)
            sess[security.CSRF_SESSION_KEY] = existing
        return existing


@pytest.fixture
def finished(app, db, spotify):
    """Bob and Alice, matched and scored, in two separate browsers."""
    bob_tracks = track_ids("b", ENOUGH)
    alice_tracks = track_ids("b", 40) + track_ids("a", ENOUGH - 40)
    seed_features(db, set(bob_tracks) | set(alice_tracks))

    give_library(spotify, "bob", "Bob", top=bob_tracks)
    give_library(spotify, "alice", "Alice", top=alice_tracks)

    bob = app.test_client()
    alice = app.test_client()

    created, _ = oauth_roundtrip(bob, "/login", "bob")
    match_id = match_id_from(created)
    oauth_roundtrip(alice, f"/m/{match_id}/join", "alice")

    return {"id": match_id, "bob": bob, "alice": alice,
            "stranger": app.test_client()}


# ── the invite link stops being a public result page ─────────────────────────

def test_a_stranger_with_the_invite_link_is_refused(finished):
    page = finished["stranger"].get(f"/m/{finished['id']}")
    assert page.status_code == 403
    body = page.get_data(as_text=True)
    assert "percentile" not in body.lower()
    assert "Fleetwood Mac" not in body


def test_both_participants_still_see_their_result(finished):
    for who in ("bob", "alice"):
        page = finished[who].get(f"/m/{finished['id']}")
        assert page.status_code == 200
        assert "82.4" in page.get_data(as_text=True)


def test_a_participant_who_cleared_their_cookies_can_get_back(app, finished):
    """The recovery path, which is the reason identity is not per-match.

    Bob clears his browser. The old role cookie is gone, so the invite link
    alone no longer proves anything -- and it should not, that is the whole
    fix. What gets him back in is signing in again: the Spotify account is
    the identity, so his history finds the match and his result opens.
    """
    bob2 = app.test_client()
    assert bob2.get(f"/m/{finished['id']}").status_code == 403

    oauth_roundtrip(bob2, "/login", "bob")          # signs in, new match of his own

    body = bob2.get("/me/matches").get_data(as_text=True)
    assert finished["id"] in body, "his old match is missing from his history"

    page = bob2.get(f"/m/{finished['id']}")
    assert page.status_code == 200
    assert "82.4" in page.get_data(as_text=True)


def test_rejoining_your_own_match_as_the_other_role_is_still_refused(finished, app):
    """Recovery must not become a way to be both people."""
    bob2 = app.test_client()
    landed, _ = oauth_roundtrip(bob2, f"/m/{finished['id']}/join", "bob")
    assert landed.status_code == 409


# ── publishing ───────────────────────────────────────────────────────────────

def _publish(client, match_id, details=False):
    data = {"csrf_token": csrf(client)}
    if details:
        data["include_details"] = "on"
    posted = client.post(f"/m/{match_id}/share", data=data)
    assert posted.status_code == 302
    page = client.get(f"/m/{match_id}")
    body = page.get_data(as_text=True)
    assert "/s/" in body, body[:500]
    token = body.split("/s/")[1].split('"')[0].split("<")[0].strip()
    return token


def test_publishing_gives_a_link_a_stranger_can_open(finished):
    token = _publish(finished["bob"], finished["id"])
    page = finished["stranger"].get(f"/s/{token}")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "82.4" in body and "Bob" in body and "Alice" in body


def test_the_public_page_does_not_leak_the_invite_link(finished):
    token = _publish(finished["bob"], finished["id"], details=True)
    body = finished["stranger"].get(f"/s/{token}").get_data(as_text=True)
    assert finished["id"] not in body


def test_details_are_off_unless_asked_for(finished):
    token = _publish(finished["bob"], finished["id"])
    body = finished["stranger"].get(f"/s/{token}").get_data(as_text=True)
    assert "Fleetwood Mac" not in body

    token2 = _publish(finished["bob"], finished["id"], details=True)
    body2 = finished["stranger"].get(f"/s/{token2}").get_data(as_text=True)
    assert "Fleetwood Mac" in body2


def test_a_stranger_cannot_publish(finished):
    refused = finished["stranger"].post(f"/m/{finished['id']}/share")
    # Not signed in at all: bounced to login rather than 403.
    assert refused.status_code == 302
    assert "/login" in refused.headers["Location"]


def test_a_signed_in_outsider_cannot_publish(app, finished, spotify, db):
    """Mallory has a Spotify account, just not a seat in this match."""
    give_library(spotify, "mal", "Mallory", top=track_ids("m", 5))
    mallory = app.test_client()
    oauth_roundtrip(mallory, "/login", "mal")      # her own, separate match

    refused = mallory.post(f"/m/{finished['id']}/share",
                           data={"csrf_token": csrf(mallory)})
    assert refused.status_code == 403


# ── rotation and takedown, over HTTP ─────────────────────────────────────────

def test_a_new_link_breaks_the_old_one(finished):
    first = _publish(finished["bob"], finished["id"])
    second = _publish(finished["bob"], finished["id"])
    assert first != second

    assert finished["stranger"].get(f"/s/{first}").status_code == 404
    assert finished["stranger"].get(f"/s/{second}").status_code == 200


def test_the_other_person_can_take_it_down(finished):
    token = _publish(finished["bob"], finished["id"])
    assert finished["stranger"].get(f"/s/{token}").status_code == 200

    finished["alice"].post(f"/m/{finished['id']}/unshare",
                           data={"scope": "all",
                                 "csrf_token": csrf(finished["alice"])})
    assert finished["stranger"].get(f"/s/{token}").status_code == 404


def test_taking_mine_down_does_not_touch_theirs(finished):
    bobs = _publish(finished["bob"], finished["id"])
    alices = _publish(finished["alice"], finished["id"])

    finished["bob"].post(f"/m/{finished['id']}/unshare",
                         data={"csrf_token": csrf(finished["bob"])})

    assert finished["stranger"].get(f"/s/{bobs}").status_code == 404
    assert finished["stranger"].get(f"/s/{alices}").status_code == 200


def test_an_unknown_token_is_a_clean_404(finished):
    page = finished["stranger"].get("/s/not-a-real-token")
    assert page.status_code == 404
    assert "no longer active" in page.get_data(as_text=True)


# ── history ──────────────────────────────────────────────────────────────────

def test_my_matches_lists_the_match_and_the_other_person(finished):
    page = finished["bob"].get("/me/matches")
    assert page.status_code == 200
    body = page.get_data(as_text=True)
    assert "Alice" in body and "82.4" in body


def test_my_matches_requires_signing_in(finished):
    bounced = finished["stranger"].get("/me/matches")
    assert bounced.status_code == 302
    assert "/login" in bounced.headers["Location"]


def test_my_matches_shows_only_mine(app, finished, spotify):
    give_library(spotify, "mal", "Mallory", top=track_ids("m", 5))
    mallory = app.test_client()
    oauth_roundtrip(mallory, "/login", "mal")

    body = mallory.get("/me/matches").get_data(as_text=True)
    assert "Alice" not in body and "Bob" not in body


def test_the_public_link_appears_on_the_history_page(finished):
    token = _publish(finished["bob"], finished["id"])
    body = finished["bob"].get("/me/matches").get_data(as_text=True)
    assert token in body


# ── CSRF ─────────────────────────────────────────────────────────────────────

class TestCrossSiteRequests:
    """A page on another domain must not be able to drive these.

    SameSite=Lax already blocks the cross-site form post in any browser that
    honours it. The token is the layer that does not depend on the browser
    getting that right.
    """

    def test_publishing_without_a_token_is_refused(self, finished):
        refused = finished["bob"].post(f"/m/{finished['id']}/share")
        assert refused.status_code == 400
        assert "could not be verified" in refused.get_data(as_text=True)

    def test_publishing_with_someone_elses_token_is_refused(self, finished):
        stolen = csrf(finished["alice"])
        refused = finished["bob"].post(f"/m/{finished['id']}/share",
                                       data={"csrf_token": stolen})
        assert refused.status_code == 400

    def test_a_takedown_cannot_be_forged(self, finished):
        """Otherwise any page Bob visits could silently unpublish his match."""
        token = _publish(finished["bob"], finished["id"])
        forged = finished["bob"].post(f"/m/{finished['id']}/unshare",
                                      data={"scope": "all"})
        assert forged.status_code == 400
        assert finished["stranger"].get(f"/s/{token}").status_code == 200
