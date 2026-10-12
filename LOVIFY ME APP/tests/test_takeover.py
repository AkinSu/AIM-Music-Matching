"""A link-holder must not be able to take someone else's seat.

Regression test for the finding that blocked PR #4. The original code
overwrote a participant unconditionally, so anyone with the share link could
evict a legitimate participant -- even after the match completed -- and
re-score in their place, as often as they liked.

That made the results page an oracle: each takeover asks "does the other
person listen to X?" and the shared-artists list answers. A single reused
Spotify account recovered a victim's library exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matching  # noqa: E402
from models import Role, init_db, make_engine, make_session_factory  # noqa: E402


def person(spotify_id, name):
    return SimpleNamespace(spotify_id=spotify_id, display_name=name,
                           photo_url="https://example.invalid/i.png")


@pytest.fixture
def db():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    return make_session_factory(engine)()


@pytest.fixture
def completed(db):
    """Bob invites, Alice joins, match is scored."""
    match = matching.create_match(db)
    matching.join_match(db, match, Role.INVITER, person("bob", "Bob"),
                        ["t_duster", "t_nbt", "t_lenderman", "t_alvvays"])
    matching.join_match(db, match, Role.INVITED, person("alice", "Alice"),
                        ["t_alvvays", "t_other"])
    return match


def test_a_stranger_cannot_take_an_occupied_seat(db, completed):
    with pytest.raises(matching.MatchError) as exc:
        matching.join_match(db, completed, Role.INVITED,
                            person("mallory", "Mallory"), ["t_probe"])
    assert "already joined" in str(exc.value).lower()

    db.refresh(completed)
    assert completed.invited.spotify_id == "alice", "Alice was evicted"
    assert completed.invited.display_name == "Alice"


def test_the_inviter_seat_is_protected_too(db, completed):
    with pytest.raises(matching.MatchError):
        matching.join_match(db, completed, Role.INVITER,
                            person("mallory", "Mallory"), ["t_probe"])
    db.refresh(completed)
    assert completed.inviter.spotify_id == "bob"


def test_the_same_person_may_still_retry(db, completed):
    """The legitimate case the old behaviour was protecting.

    Dropped redirect, refresh, flaky signal: the same account re-joining
    must still work and overwrite its own row.
    """
    matching.join_match(db, completed, Role.INVITED, person("alice", "Alice"),
                        ["t_alvvays", "t_other", "t_new"])
    db.refresh(completed)
    assert completed.invited.spotify_id == "alice"
    assert "t_new" in completed.invited.tracks


def test_the_library_oracle_is_closed(db, completed):
    """The attack itself: probing the victim's library one track at a time."""
    bob = set(completed.inviter.tracks)
    recovered = []
    for track in ["t_duster", "t_nbt", "t_taylor", "t_alvvays"]:
        try:
            matching.join_match(db, completed, Role.INVITED,
                                person("mallory", "Mallory"), [track])
        except matching.MatchError:
            continue          # every probe must be refused
        db.refresh(completed)
        if track in bob:
            recovered.append(track)

    assert recovered == [], (
        f"the oracle still leaks: recovered {recovered} from the victim's library")
