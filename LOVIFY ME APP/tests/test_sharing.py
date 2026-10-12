"""Two links, two jobs.

    /m/<match_id>   private. A capability to JOIN.
    /s/<token>      public.  A capability to READ one finished result.

These tests exist mostly to keep those two from collapsing back into one,
which is what they were before, and which is what made the invite link
dangerous to post.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matching  # noqa: E402
import sharing  # noqa: E402
from models import (  # noqa: E402
    MatchStatus, Role, ShareCard, init_db, make_engine, make_session_factory,
)


def person(spotify_id, name):
    return SimpleNamespace(spotify_id=spotify_id, display_name=name,
                           photo_url="https://example.invalid/i.png")


@pytest.fixture
def db():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    return make_session_factory(engine)()


@pytest.fixture
def match(db):
    """A finished match between Bob (inviter) and Alice (invited)."""
    m = matching.create_match(db)
    matching.join_match(db, m, Role.INVITER, person("bob", "Bob"), ["t1", "t2"])
    matching.join_match(db, m, Role.INVITED, person("alice", "Alice"), ["t2", "t3"])
    m.status = MatchStatus.COMPLETE
    m.result_json = json.dumps({
        "status": "ok",
        "percentile": 84.0,
        "interpretation": "Unusually close taste.",
        "shared_artists": ["Duster", "Alvvays"],
        "top_shared_genres": ["slowcore", "dream pop"],
        "components": {"genre_overlap": 0.71, "audio_distance": 0.44},
    })
    db.commit()
    return m


# ── who may publish ──────────────────────────────────────────────────────────

class TestWhoMayShare:
    def test_a_participant_may_publish(self, db, match):
        card = sharing.create_card(db, match, "bob")
        assert card.active
        assert sharing.load_card(db, card.token).match_id == match.id

    def test_a_stranger_may_not(self, db, match):
        with pytest.raises(sharing.ShareError):
            sharing.create_card(db, match, "mallory")

    def test_an_unfinished_match_has_nothing_to_publish(self, db):
        m = matching.create_match(db)
        matching.join_match(db, m, Role.INVITER, person("bob", "Bob"), ["t1"])
        with pytest.raises(sharing.ShareError):
            sharing.create_card(db, m, "bob")


# ── rotation and revocation ──────────────────────────────────────────────────

class TestRotation:
    def test_a_new_link_kills_the_old_one(self, db, match):
        """The only recall a bearer token can have."""
        first = sharing.create_card(db, match, "bob")
        second = sharing.create_card(db, match, "bob")

        assert first.token != second.token
        with pytest.raises(sharing.ShareError):
            sharing.load_card(db, first.token)
        assert sharing.load_card(db, second.token).token == second.token

    def test_each_person_gets_their_own_link(self, db, match):
        bobs = sharing.create_card(db, match, "bob")
        alices = sharing.create_card(db, match, "alice")
        assert bobs.token != alices.token
        # Alice publishing must not have rotated Bob's out from under him.
        assert sharing.load_card(db, bobs.token).token == bobs.token

    def test_revoking_mine_leaves_theirs_alone(self, db, match):
        bobs = sharing.create_card(db, match, "bob")
        alices = sharing.create_card(db, match, "alice")

        sharing.revoke_mine(db, match, "bob")

        with pytest.raises(sharing.ShareError):
            sharing.load_card(db, bobs.token)
        assert sharing.load_card(db, alices.token).active

    def test_either_person_can_take_the_whole_thing_down(self, db, match):
        """Alice's veto over a result that is half hers."""
        bobs = sharing.create_card(db, match, "bob")
        alices = sharing.create_card(db, match, "alice")

        killed = sharing.revoke_all(db, match, "alice")

        assert killed == 2
        for token in (bobs.token, alices.token):
            with pytest.raises(sharing.ShareError):
                sharing.load_card(db, token)

    def test_a_stranger_cannot_take_it_down(self, db, match):
        sharing.create_card(db, match, "bob")
        with pytest.raises(sharing.ShareError):
            sharing.revoke_all(db, match, "mallory")


# ── what the public sees ─────────────────────────────────────────────────────

class TestPublicView:
    def test_the_default_card_is_names_and_a_score(self, db, match):
        view = sharing.public_view(sharing.create_card(db, match, "bob"))
        assert sorted(view["names"]) == ["Alice", "Bob"]
        assert view["percentile"] == 84.0
        assert view["shared_artists"] == []
        assert view["top_shared_genres"] == []

    def test_details_appear_only_when_opted_in(self, db, match):
        card = sharing.create_card(db, match, "bob", include_details=True)
        view = sharing.public_view(card)
        assert "Duster" in view["shared_artists"]
        assert "slowcore" in view["top_shared_genres"]

    def test_the_public_view_never_carries_the_match_id(self, db, match):
        """The match id is the JOIN capability. It must not ride along.

        Checked over the serialised view rather than field by field, so a
        field added to the scorer later cannot smuggle it through.
        """
        for details in (False, True):
            card = sharing.create_card(db, match, "bob", include_details=details)
            blob = json.dumps(sharing.public_view(card), default=str)
            assert match.id not in blob

    def test_the_public_view_never_carries_spotify_ids(self, db, match):
        card = sharing.create_card(db, match, "bob", include_details=True)
        blob = json.dumps(sharing.public_view(card), default=str)
        for secret in ("bob", "alice"):
            assert f'"{secret}"' not in blob

    def test_components_and_track_counts_stay_private(self, db, match):
        card = sharing.create_card(db, match, "bob", include_details=True)
        view = sharing.public_view(card)
        assert "components" not in view
        assert "n_tracks" not in view

    def test_share_text_reads_like_a_person_wrote_it(self, db, match):
        text = sharing.share_text(sharing.public_view(
            sharing.create_card(db, match, "bob")))
        assert "84th percentile" in text
        assert len(text) < 200


# ── history ──────────────────────────────────────────────────────────────────

class TestHistory:
    def test_it_lists_my_matches_and_names_the_other_person(self, db, match):
        rows = sharing.history(db, "bob")
        assert len(rows) == 1
        assert rows[0].them.display_name == "Alice"
        assert rows[0].shareable
        assert not rows[0].shared

    def test_it_knows_which_of_mine_are_public(self, db, match):
        sharing.create_card(db, match, "bob")
        assert sharing.history(db, "bob")[0].shared
        # Bob's card is Bob's; it must not show as Alice's.
        assert not sharing.history(db, "alice")[0].shared

    def test_a_stranger_has_no_history(self, db, match):
        assert sharing.history(db, "mallory") == []

    def test_newest_first(self, db, match):
        second = matching.create_match(db)
        matching.join_match(db, second, Role.INVITER, person("bob", "Bob"), ["t9"])
        rows = sharing.history(db, "bob")
        assert [r.match.id for r in rows] == [second.id, match.id]
