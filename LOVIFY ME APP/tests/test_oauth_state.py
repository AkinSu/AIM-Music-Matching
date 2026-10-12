"""The signed `state`, at both levels: the scheme, and the route using it.

Carrying the match id through Spotify is what makes the cross-device flow
possible, and it is also the part that could quietly become a vulnerability.
An attacker who could choose which match a victim's login lands in would be
able to read the victim's listening data off their own results page. The
tests below are written against that attack, not against the happy path.
"""

from __future__ import annotations

import pytest

import oauth_state
from conftest import give_library, match_id_from, oauth_roundtrip, seed_features, track_ids
from models import Role

KEY = "a-test-signing-key"
OTHER_KEY = "a-different-signing-key"


# ── the scheme ───────────────────────────────────────────────────────────────

def test_a_state_this_server_minted_round_trips():
    nonce = oauth_state.new_nonce()
    state = oauth_state.build(KEY, nonce, "match-abc", Role.INVITED)
    assert oauth_state.parse(KEY, state, nonce) == ("match-abc", Role.INVITED)


def test_a_tampered_match_id_is_rejected():
    """The attack this scheme exists to stop."""
    nonce = oauth_state.new_nonce()
    state = oauth_state.build(KEY, nonce, "victims-match", Role.INVITED)
    forged = state.replace("victims-match", "attackers-match")

    with pytest.raises(oauth_state.StateError, match="signature"):
        oauth_state.parse(KEY, forged, nonce)


def test_a_tampered_role_is_rejected():
    nonce = oauth_state.new_nonce()
    state = oauth_state.build(KEY, nonce, "match-abc", Role.INVITED)
    forged = state.replace(Role.INVITED, Role.INVITER)

    with pytest.raises(oauth_state.StateError, match="signature"):
        oauth_state.parse(KEY, forged, nonce)


def test_a_state_signed_with_another_key_is_rejected():
    nonce = oauth_state.new_nonce()
    state = oauth_state.build(OTHER_KEY, nonce, "match-abc", Role.INVITED)

    with pytest.raises(oauth_state.StateError, match="signature"):
        oauth_state.parse(KEY, state, nonce)


def test_a_valid_state_is_useless_in_another_browser():
    """The original CSRF property, preserved rather than traded away."""
    nonce = oauth_state.new_nonce()
    state = oauth_state.build(KEY, nonce, "match-abc", Role.INVITED)

    with pytest.raises(oauth_state.StateError, match="browser session"):
        oauth_state.parse(KEY, state, oauth_state.new_nonce())


def test_a_session_with_no_nonce_cannot_complete_a_login():
    nonce = oauth_state.new_nonce()
    state = oauth_state.build(KEY, nonce, "match-abc", Role.INVITED)

    with pytest.raises(oauth_state.StateError, match="not started here"):
        oauth_state.parse(KEY, state, None)


@pytest.mark.parametrize("state", ["", None, "garbage", "a.b.c", "a.b.c.d.e"])
def test_malformed_states_are_rejected_rather_than_parsed(state):
    with pytest.raises(oauth_state.StateError):
        oauth_state.parse(KEY, state, "whatever")


def test_a_separator_in_a_field_is_refused_at_minting_time():
    """Better to fail loudly here than to mint an ambiguous state."""
    with pytest.raises(ValueError):
        oauth_state.build(KEY, "nonce", "match.with.dots", Role.INVITED)


# ── the route ────────────────────────────────────────────────────────────────

def test_the_callback_refuses_a_state_it_did_not_mint(app, spotify, db):
    """End to end: an attacker-crafted state does not reach Spotify data."""
    seed_features(db, track_ids("x", 120))
    give_library(spotify, "victim", "Victim", saved=track_ids("x", 120))

    victim = app.test_client()
    # The victim legitimately starts a login, so their session has a nonce.
    started = victim.get("/login")
    state = started.headers["Location"].split("state=")[1]
    nonce, match_id, role, _sig = state.split(".")

    # The attacker swaps in a match they control, and re-signs with a key
    # they do not have -- which is the whole point.
    forged = oauth_state.build("guessed-key", nonce, "attacker-match", role)
    blocked = victim.get(f"/callback?code=victim&state={forged}")

    assert blocked.status_code == 400
    assert "could not be verified" in blocked.get_data(as_text=True)
    assert spotify.calls == [], "no token was exchanged for an unverified state"


def test_a_captured_callback_url_cannot_be_replayed(app, spotify, db):
    seed_features(db, track_ids("x", 120))
    give_library(spotify, "a-code", "Ada", saved=track_ids("x", 120))

    client = app.test_client()
    done, state = oauth_roundtrip(client, "/login", "a-code")
    match_id_from(done)

    replayed = client.get(f"/callback?code=a-code&state={state}")
    assert replayed.status_code == 400, "the nonce is single use"


def test_declining_the_spotify_permission_says_so(app):
    refused = app.test_client().get("/callback?error=access_denied&state=x")
    assert refused.status_code == 400
    assert "declined" in refused.get_data(as_text=True)
