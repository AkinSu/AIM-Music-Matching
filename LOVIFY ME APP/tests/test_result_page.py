"""What the results page actually tells people.

Three things landed together here, all asked for directly:

  - the zodiac work, which was fully built in ml/ and shown on no page
  - a thin library getting an answer instead of a refusal
  - matching yourself, which the five-user Spotify cap makes necessary

The framing assertions matter as much as the layout ones. This project
measured zodiac prediction at 1.39x chance and pre-registered that as dead,
so the page must never read as a guess at someone's birthday.
"""

from __future__ import annotations

import pytest

from conftest import (
    give_library, match_id_from, oauth_roundtrip, seed_features, track_ids,
)

BIG, THIN = 140, 20


def _pair(app, spotify, db, a_tracks, b_tracks, a="ada", b="bo"):
    seed_features(db, sorted(set(a_tracks) | set(b_tracks)))
    give_library(spotify, a, "Ada", saved=a_tracks)
    give_library(spotify, b, "Bo", saved=b_tracks)
    one, two = app.test_client(), app.test_client()
    match_id = match_id_from(oauth_roundtrip(one, "/login", a)[0])
    oauth_roundtrip(two, f"/m/{match_id}/join", b)
    return one, two, match_id


# ── zodiac ───────────────────────────────────────────────────────────────────

class TestZodiacOnThePage:
    @pytest.fixture
    def page(self, app, spotify, db):
        one, _, mid = _pair(app, spotify, db,
                            track_ids("x", BIG), track_ids("y", BIG))
        return one.get(f"/m/{mid}").get_data(as_text=True)

    def test_both_people_get_a_summary(self, page):
        assert page.count("listening leans") == 2

    def test_the_headline_sign_is_shown(self, page):
        assert "Scorpio" in page

    def test_lifts_are_printed_as_numbers(self, page):
        """A sign with no number beside it is a horoscope."""
        assert "1.69" in page
        assert "&times;" in page or "×" in page

    def test_it_says_this_is_about_music_not_birthdays(self, page):
        """The measured result is 1.39x chance. The page must not imply more."""
        low = page.lower()
        assert "not a guess at a birthday" in low or "guess at a birthday" in low
        assert "average listener" in low

    def test_it_never_claims_to_know_your_sign(self, page):
        low = page.lower()
        for claim in ("your star sign is", "you are a scorpio", "your sign:"):
            assert claim not in low


class TestWeakSignalIsShownAsWeak:
    def test_a_thin_library_gets_hedged_wording(self, app, spotify, db):
        one, _, mid = _pair(app, spotify, db,
                            track_ids("x", THIN), track_ids("x", THIN) + track_ids("y", 5))
        body = one.get(f"/m/{mid}").get_data(as_text=True)
        assert "but only just" in body
        assert "invent a personality" in body


# ── thin libraries get an answer, not a refusal ──────────────────────────────

class TestSmallLibrary:
    @pytest.fixture
    def body(self, app, spotify, db):
        shared = track_ids("x", 15)
        one, _, mid = _pair(app, spotify, db, shared, shared + track_ids("y", 5))
        return one.get(f"/m/{mid}").get_data(as_text=True)

    def test_it_does_not_lead_with_a_refusal(self, body):
        assert "Here is what we can tell you" in body
        assert "Not enough listening history yet" not in body

    def test_the_exact_parts_are_still_shown(self, body):
        assert "Artists you both listen to" in body
        assert "Fleetwood Mac" in body

    def test_it_still_withholds_the_percentile(self, body):
        """The one number that genuinely needs 100+ tracks."""
        assert "The one thing we are leaving out" in body
        assert "82.4" not in body

    def test_it_explains_why_in_plain_words(self, body):
        assert "biased" in body.lower()

    def test_the_zodiac_summary_still_appears(self, body):
        """A taste summary needs one library, not a calibrated pair."""
        assert "listening leans" in body


# ── matching yourself ────────────────────────────────────────────────────────

class TestSelfMatch:
    @pytest.fixture
    def setup(self, app, spotify, db):
        tracks = track_ids("x", BIG)
        seed_features(db, tracks)
        give_library(spotify, "ada", "Ada", saved=tracks)
        give_library(spotify, "ada2", "Ada", saved=tracks)
        spotify.profiles["ada2"].spotify_id = spotify.profiles["ada"].spotify_id
        one, two = app.test_client(), app.test_client()
        mid = match_id_from(oauth_roundtrip(one, "/login", "ada")[0])
        oauth_roundtrip(two, f"/m/{mid}/join", "ada2")
        return one, mid

    def test_it_completes(self, setup):
        one, mid = setup
        assert one.get(f"/m/{mid}").status_code == 200

    def test_it_is_labelled_so_nobody_misreads_it(self, setup):
        one, mid = setup
        # Collapse whitespace: the template wraps this sentence across lines.
        body = " ".join(one.get(f"/m/{mid}").get_data(as_text=True).split())
        assert "compared with yourself" in body
        assert "not a comparison with another person" in body

    def test_the_flag_is_stored_with_the_result(self, setup):
        """Stored, not derived at render time: the track lists are deleted a
        day after scoring, so this cannot be recomputed later."""
        import lovify
        import matching
        from models import Match

        _, mid = setup
        with lovify.Session() as s:
            assert matching.result_for(s.get(Match, mid))["self_match"] is True


# ── resilience ───────────────────────────────────────────────────────────────

def test_a_scorer_without_taste_still_renders_a_score(app, spotify, db):
    """The summary is a bonus. It must never take down a real result."""
    delattr(type(app.scorer), "taste_profile") if hasattr(
        type(app.scorer), "taste_profile") else None
    app.scorer.taste_profile = None
    object.__setattr__(app.scorer, "taste_profile", None)

    one, _, mid = _pair(app, spotify, db, track_ids("x", BIG), track_ids("y", BIG))
    page = one.get(f"/m/{mid}")
    assert page.status_code == 200
    assert "82.4" in page.get_data(as_text=True)
