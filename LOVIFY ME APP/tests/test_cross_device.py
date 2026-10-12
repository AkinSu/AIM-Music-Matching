"""The test this work exists for: two devices, one match.

The old code kept both participants in `session[...]`. That is one browser's
cookie, so the share link only ever worked if the "second person" was sitting
at the first person's keyboard. A real invitee on a real phone arrived with
an empty session and the results page had half the data.

Every test below uses two separate `test_client()` instances. Flask gives
each its own cookie jar, so they are two browsers in the only sense that
matters here: nothing one of them stores is visible to the other.
"""

from __future__ import annotations

from conftest import (
    give_library, match_id_from, oauth_roundtrip, seed_features, track_ids,
)
from models import Match, Role


def test_two_separate_browsers_complete_a_match(app, spotify, db, scorer):
    akin_tracks = track_ids("akin", 160)
    sam_tracks = track_ids("sam", 140)
    seed_features(db, akin_tracks + sam_tracks)
    give_library(spotify, "akin-code", "Akin", top=akin_tracks[:50],
                 saved=akin_tracks[50:])
    give_library(spotify, "sam-code", "Sam", top=sam_tracks[:50],
                 saved=sam_tracks[50:])

    akin = app.test_client()      # device 1
    sam = app.test_client()       # device 2, has never seen this server

    # --- A signs in and gets a link -----------------------------------------
    done, _ = oauth_roundtrip(akin, "/login", "akin-code")
    match_id = match_id_from(done)

    waiting = akin.get(f"/m/{match_id}")
    assert waiting.status_code == 200
    body = waiting.get_data(as_text=True)
    assert f"/m/{match_id}" in body, "the share link must be on the page to share"
    assert "Akin" in body

    # --- B opens the link on their own device -------------------------------
    # This is the exact moment the old code failed: B's session is empty.
    with sam.session_transaction() as s:
        assert dict(s) == {}, "the invitee starts with no session at all"

    invite = sam.get(f"/m/{match_id}")
    assert invite.status_code == 200
    invite_body = invite.get_data(as_text=True)
    assert "Akin" in invite_body, "B should be told who is asking"
    assert "Spotify" in invite_body

    done_b, _ = oauth_roundtrip(sam, f"/m/{match_id}/join", "sam-code")
    assert match_id_from(done_b) == match_id

    # --- both see the result ------------------------------------------------
    for who, client in (("inviter", akin), ("invited", sam)):
        page = client.get(f"/m/{match_id}")
        assert page.status_code == 200, who
        text = page.get_data(as_text=True)
        assert "82.4" in text, f"{who} should see the score"
        assert "Akin" in text and "Sam" in text, f"{who} should see both people"

    db.expire_all()
    match = db.get(Match, match_id)
    assert match.status == "complete"
    assert {p.role for p in match.participants} == {Role.INVITER, Role.INVITED}
    assert scorer.calls == 1, "scored once, when the second person joined"


def test_the_invitee_session_holds_nothing_about_the_inviter(app, spotify, db):
    """The cookie's only job is 'which participant is this browser'.

    If the invitee's session ever carried the inviter's data again, the
    cross-device bug would be back in a new costume. Asserting the exact key
    set makes that regression loud.
    """
    seed_features(db, track_ids("x", 120) + track_ids("y", 120))
    give_library(spotify, "a-code", "Ada", saved=track_ids("x", 120))
    give_library(spotify, "b-code", "Bo", saved=track_ids("y", 120))

    first, second = app.test_client(), app.test_client()
    match_id = match_id_from(oauth_roundtrip(first, "/login", "a-code")[0])
    oauth_roundtrip(second, f"/m/{match_id}/join", "b-code")

    for client, role in ((first, Role.INVITER), (second, Role.INVITED)):
        with client.session_transaction() as s:
            assert dict(s) == {f"role:{match_id}": role}


def test_a_third_person_with_the_link_sees_the_result_but_is_not_in_it(app, spotify, db):
    seed_features(db, track_ids("x", 120) + track_ids("y", 120))
    give_library(spotify, "a-code", "Ada", saved=track_ids("x", 120))
    give_library(spotify, "b-code", "Bo", saved=track_ids("y", 120))

    first, second, bystander = (app.test_client() for _ in range(3))
    match_id = match_id_from(oauth_roundtrip(first, "/login", "a-code")[0])
    oauth_roundtrip(second, f"/m/{match_id}/join", "b-code")

    page = bystander.get(f"/m/{match_id}")
    assert page.status_code == 200
    assert "82.4" in page.get_data(as_text=True)
    with bystander.session_transaction() as s:
        assert dict(s) == {}, "looking at a link must not enrol you in the match"


def test_the_same_spotify_account_cannot_match_itself(app, spotify, db):
    """A shared laptop, or someone testing their own link, is not a match."""
    seed_features(db, track_ids("x", 120))
    give_library(spotify, "a-code", "Ada", saved=track_ids("x", 120))
    # A second login code belonging to the same Spotify account.
    give_library(spotify, "a-again", "Ada", saved=track_ids("x", 120))
    spotify.profiles["a-again"].spotify_id = spotify.profiles["a-code"].spotify_id

    first, second = app.test_client(), app.test_client()
    match_id = match_id_from(oauth_roundtrip(first, "/login", "a-code")[0])

    refused, _ = oauth_roundtrip(second, f"/m/{match_id}/join", "a-again")
    assert refused.status_code == 409
    assert "same Spotify account" in refused.get_data(as_text=True)


def test_rejoining_after_a_dropped_redirect_is_not_an_error(app, spotify, db):
    """The invitee is on a phone, on a text-message link, on bad signal.

    They tap it, authorise, lose the redirect, and tap it again. Treating the
    second attempt as a conflict would strand them with no way forward, so a
    repeat join in the same role overwrites rather than failing.
    """
    seed_features(db, track_ids("x", 120) + track_ids("y", 120))
    give_library(spotify, "a-code", "Ada", saved=track_ids("x", 120))
    give_library(spotify, "b-code", "Bo", saved=track_ids("y", 120))

    first, second = app.test_client(), app.test_client()
    match_id = match_id_from(oauth_roundtrip(first, "/login", "a-code")[0])

    oauth_roundtrip(second, f"/m/{match_id}/join", "b-code")
    retry, _ = oauth_roundtrip(second, f"/m/{match_id}/join", "b-code")
    assert match_id_from(retry) == match_id

    db.expire_all()
    match = db.get(Match, match_id)
    assert len(match.participants) == 2, "a retry must not create a third row"
    assert match.status == "complete"


def test_the_old_cross_device_link_explains_itself(app):
    gone = app.test_client().get("/second/user/login")
    assert gone.status_code == 410
    assert "second device" in gone.get_data(as_text=True)


def test_an_unknown_or_expired_link_is_a_dead_end(app, db):
    client = app.test_client()
    assert client.get("/m/0123456789abcdef0123456789abcdef").status_code == 404

    from datetime import timedelta
    from models import utcnow
    match = Match()
    match.expires_at = utcnow() - timedelta(days=1)
    db.add(match)
    db.commit()

    stale = client.get(f"/m/{match.id}")
    assert stale.status_code == 404
    assert "expired" in stale.get_data(as_text=True).lower()


def test_the_pages_that_already_existed_still_render(app):
    """Old links are in people's messages and in the team's slides."""
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/what-about-us").status_code == 200
    # /dashboard used to be the post-login page; with no match in the session
    # it should start one rather than 404.
    assert client.get("/dashboard").headers["Location"].endswith("/login")
    assert client.get("/find-compatibility").status_code == 302


def test_logout_forgets_which_participant_this_browser_is(app, spotify, db):
    seed_features(db, track_ids("x", 120))
    give_library(spotify, "a-code", "Ada", saved=track_ids("x", 120))

    client = app.test_client()
    match_id = match_id_from(oauth_roundtrip(client, "/login", "a-code")[0])
    client.get("/logout")

    with client.session_transaction() as s:
        assert dict(s) == {}
    # The match itself survives: the other person may still be mid-join.
    db.expire_all()
    assert db.get(Match, match_id) is not None
