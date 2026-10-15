"""The cheapest security property available is not having the data."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matching  # noqa: E402
import retention  # noqa: E402
import sharing  # noqa: E402
from models import (  # noqa: E402
    MATCH_TTL, Match, MatchStatus, Role, ShareCard, init_db, make_engine,
    make_session_factory, utcnow,
)


def person(sid, name):
    return SimpleNamespace(spotify_id=sid, display_name=name, photo_url="")


@pytest.fixture
def db():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    return make_session_factory(engine)()


def aged(db, match, days):
    """Backdate a match as if it had been sitting there."""
    match.created_at = utcnow() - timedelta(days=days)
    if match.completed_at:
        match.completed_at = utcnow() - timedelta(days=days)
    match.expires_at = match.created_at + MATCH_TTL
    db.commit()


class TestAbandonedMatches:
    def test_an_abandoned_invitation_is_marked_expired(self, db):
        m = matching.create_match(db)
        matching.join_match(db, m, Role.INVITER, person("bob", "Bob"), ["t1", "t2"])
        aged(db, m, 10)

        report = retention.sweep(db)

        assert report.expired == 1
        assert m.status == MatchStatus.EXPIRED

    def test_it_loses_the_one_library_it_was_holding(self, db):
        """The whole point. Nobody joined, so this is one person's listening
        history sitting in a database for no reason at all."""
        m = matching.create_match(db)
        matching.join_match(db, m, Role.INVITER, person("bob", "Bob"), ["t1", "t2"])
        aged(db, m, 10)

        retention.sweep(db)

        assert m.inviter.tracks == []
        assert m.inviter.n_tracks == 2, "the coverage measurement survives"

    def test_a_fresh_invitation_is_left_alone(self, db):
        m = matching.create_match(db)
        matching.join_match(db, m, Role.INVITER, person("bob", "Bob"), ["t1"])

        retention.sweep(db)

        assert m.status == MatchStatus.WAITING
        assert m.inviter.tracks == ["t1"]


class TestScoredMatches:
    @pytest.fixture
    def scored(self, db):
        m = matching.create_match(db)
        matching.join_match(db, m, Role.INVITER, person("bob", "Bob"), ["t1", "t2"])
        matching.join_match(db, m, Role.INVITED, person("alice", "Alice"), ["t2"])
        m.status = MatchStatus.COMPLETE
        m.result_json = '{"percentile": 70.0, "shared_artists": ["Duster"]}'
        m.completed_at = utcnow()
        db.commit()
        return m

    def test_the_raw_libraries_go_once_the_result_exists(self, db, scored):
        aged(db, scored, 3)
        retention.sweep(db)
        assert scored.inviter.tracks == []
        assert scored.invited.tracks == []

    def test_the_result_itself_survives(self, db, scored):
        aged(db, scored, 3)
        retention.sweep(db)
        assert "70.0" in scored.result_json
        assert scored.status == MatchStatus.COMPLETE

    def test_a_just_scored_match_keeps_its_tracks_briefly(self, db, scored):
        """A day's grace, so a scoring incident can be re-run."""
        retention.sweep(db)
        assert scored.inviter.tracks == ["t1", "t2"]

    def test_a_public_card_still_works_after_the_libraries_go(self, db, scored):
        card = sharing.create_card(db, scored, "bob")
        aged(db, scored, 3)
        retention.sweep(db)

        view = sharing.public_view(sharing.load_card(db, card.token))
        assert view["percentile"] == 70.0


class TestDeletion:
    def test_an_ancient_match_is_removed_entirely(self, db):
        m = matching.create_match(db)
        matching.join_match(db, m, Role.INVITER, person("bob", "Bob"), ["t1"])
        m.status = MatchStatus.COMPLETE
        m.result_json = "{}"
        db.commit()
        match_id = m.id
        aged(db, m, 400)

        report = retention.sweep(db)

        assert report.deleted == 1
        assert db.get(Match, match_id) is None

    def test_its_share_cards_go_with_it(self, db):
        m = matching.create_match(db)
        matching.join_match(db, m, Role.INVITER, person("bob", "Bob"), ["t1"])
        matching.join_match(db, m, Role.INVITED, person("alice", "Alice"), ["t2"])
        m.status = MatchStatus.COMPLETE
        m.result_json = "{}"
        m.completed_at = utcnow()
        db.commit()
        token = sharing.create_card(db, m, "bob").token
        aged(db, m, 400)

        retention.sweep(db)

        assert db.get(ShareCard, token) is None

    def test_a_six_month_old_match_is_still_there(self, db):
        m = matching.create_match(db)
        matching.join_match(db, m, Role.INVITER, person("bob", "Bob"), ["t1"])
        aged(db, m, 100)
        retention.sweep(db)
        assert db.get(Match, m.id) is not None


def test_the_sweep_is_safe_to_run_twice(db):
    m = matching.create_match(db)
    matching.join_match(db, m, Role.INVITER, person("bob", "Bob"), ["t1"])
    aged(db, m, 10)

    first = retention.sweep(db)
    second = retention.sweep(db)

    assert first.tracks_cleared == 1
    assert second.tracks_cleared == 0, "a second pass must find nothing left to do"


def test_the_sweep_is_fine_on_an_empty_database(db):
    assert retention.sweep(db).deleted == 0
