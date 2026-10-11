"""Tests for the survey pipeline.

The load-bearing pair is at the bottom: `TestPipelineCanDetectAnEffect`.
A null result is only meaningful if the machinery that produced it would
have found an effect had one been there. So we plant a correlation and
require the pipeline to find it, then plant none and require it to report
nothing. Without both halves, "we found no effect" is indistinguishable
from "our code does not work".
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lovify_ml import schema as S     # noqa: E402
from lovify_ml import survey as SV    # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "analyse_survey", ROOT / "scripts" / "analyse_survey.py")
analyse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyse)


# ══ sign boundaries ══════════════════════════════════════════════════════════

class TestSignFromBirthday(unittest.TestCase):
    """Cusps are where this goes wrong, so test only cusps."""

    EXPECTED = [
        ((1, 19), "Capricorn"), ((1, 20), "Aquarius"),
        ((2, 18), "Aquarius"),  ((2, 19), "Pisces"),
        ((3, 20), "Pisces"),    ((3, 21), "Aries"),
        ((4, 19), "Aries"),     ((4, 20), "Taurus"),
        ((5, 20), "Taurus"),    ((5, 21), "Gemini"),
        ((6, 20), "Gemini"),    ((6, 21), "Cancer"),
        ((7, 22), "Cancer"),    ((7, 23), "Leo"),
        ((8, 22), "Leo"),       ((8, 23), "Virgo"),
        ((9, 22), "Virgo"),     ((9, 23), "Libra"),
        ((10, 22), "Libra"),    ((10, 23), "Scorpio"),
        ((11, 21), "Scorpio"),  ((11, 22), "Sagittarius"),
        ((12, 21), "Sagittarius"), ((12, 22), "Capricorn"),
    ]

    def test_every_cusp_both_sides(self):
        for (month, day), expected in self.EXPECTED:
            with self.subTest(month=month, day=day):
                self.assertEqual(SV.sign_from_birthday(month, day), expected)

    def test_year_boundary_is_capricorn(self):
        self.assertEqual(SV.sign_from_birthday(12, 31), "Capricorn")
        self.assertEqual(SV.sign_from_birthday(1, 1), "Capricorn")

    def test_every_day_of_the_year_maps_to_a_real_sign(self):
        days = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
        for month in range(1, 13):
            for day in range(1, [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1] + 1):
                self.assertIn(SV.sign_from_birthday(month, day), S.ZODIAC_SIGNS)

    def test_all_twelve_signs_are_reachable(self):
        got = {SV.sign_from_birthday(m, d)
               for m in range(1, 13) for d in range(1, 29)}
        self.assertEqual(got, set(S.ZODIAC_SIGNS))

    def test_rejects_impossible_dates(self):
        for bad in [(0, 5), (13, 5), (5, 0), (5, 32)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                SV.sign_from_birthday(*bad)


# ══ parsing what people actually paste ═══════════════════════════════════════

class TestParseArtists(unittest.TestCase):

    def test_numbered_list_with_play_counts(self):
        text = "1. Tyler, The Creator — 1,234 plays\n2. SZA — 988 plays\n3. Frank Ocean"
        self.assertEqual(SV.parse_artists(text),
                         ["Tyler, The Creator", "SZA", "Frank Ocean"])

    def test_comma_separated_run_is_split(self):
        self.assertEqual(SV.parse_artists("Drake, Future, Travis Scott"),
                         ["Drake", "Future", "Travis Scott"])

    def test_comma_inside_a_single_name_is_preserved(self):
        """The case that makes naive comma-splitting wrong."""
        self.assertEqual(SV.parse_artists("Tyler, The Creator"), ["Tyler, The Creator"])

    def test_comma_plus_and_is_treated_as_a_list(self):
        self.assertEqual(SV.parse_artists("Drake, Future and Travis Scott"),
                         ["Drake", "Future", "Travis Scott"])

    def test_bullets_and_emoji_are_stripped_but_scripts_survive(self):
        got = SV.parse_artists("• BTS 🎵\n- NewJeans\n* 아이유")
        self.assertEqual(got, ["BTS", "NewJeans", "아이유"])

    def test_duplicates_collapse_and_order_is_kept(self):
        self.assertEqual(SV.parse_artists("SZA\nDrake\nSZA"), ["SZA", "Drake"])

    def test_junk_answers_yield_nothing(self):
        for junk in ["", "   ", "n/a", "idk", None, float("nan")]:
            with self.subTest(junk=junk):
                self.assertEqual(SV.parse_artists(junk), [])

    def test_respects_max_artists(self):
        text = "\n".join(f"Artist {i}" for i in range(200))
        self.assertEqual(len(SV.parse_artists(text, max_artists=25)), 25)

    def test_semicolons_split_too(self):
        self.assertEqual(SV.parse_artists("Radiohead; Portishead; Massive Attack"),
                         ["Radiohead", "Portishead", "Massive Attack"])


class TestNormaliseArtist(unittest.TestCase):

    def test_matches_across_case_accents_and_leading_the(self):
        keys = {SV.normalise_artist(n) for n in
                ["Beyoncé", "BEYONCE", "beyonce"]}
        self.assertEqual(len(keys), 1)
        self.assertEqual(SV.normalise_artist("The Weeknd"),
                         SV.normalise_artist("weeknd"))

    def test_distinct_artists_stay_distinct(self):
        self.assertNotEqual(SV.normalise_artist("Drake"), SV.normalise_artist("Future"))


# ══ Google Forms ingest ══════════════════════════════════════════════════════

def _forms_csv(rows: list[dict], path: Path) -> Path:
    """Write a CSV with Google Forms' full-question-text headers."""
    cols = {
        "Timestamp": "ts",
        "What month were you born?": "month",
        "What day of the month were you born?": "day",
        "What's your star sign?": "self_sign",
        "Paste your top artists": "artists",
        "How much do you believe in astrology?": "belief",
        "Where did you find this form?": "source",
    }
    df = pd.DataFrame([{header: r.get(key, "") for header, key in cols.items()}
                       for r in rows])
    df.to_csv(path, index=False)
    return path


class TestLoadResponses(unittest.TestCase):

    def test_maps_verbose_headers_and_computes_the_sign(self):
        with tempfile.TemporaryDirectory() as d:
            p = _forms_csv([
                {"ts": "2026/10/11", "month": "March", "day": "25",
                 "self_sign": "aries", "artists": "SZA\nDrake", "belief": "4",
                 "source": "Reddit"},
                {"ts": "2026/10/11", "month": "11", "day": "30",
                 "self_sign": "Sagittarius", "artists": "Metallica, Slayer, Tool",
                 "belief": "1 - not at all", "source": "Discord"},
            ], Path(d) / "r.csv")
            df = SV.load_responses(p, verbose=False)

        self.assertEqual(len(df), 2)
        self.assertEqual(list(df[S.STAR_SIGN]), ["Aries", "Sagittarius"])
        self.assertEqual(df.loc[0, "artists"], ["SZA", "Drake"])
        self.assertEqual(df.loc[1, "n_artists"], 3)
        self.assertEqual(list(df["belief"]), [4, 1])
        self.assertEqual(df.loc[0, "self_sign"], "Aries")

    def test_drops_unusable_rows_and_keeps_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            p = _forms_csv([
                {"month": "March", "day": "25", "artists": "SZA"},
                {"month": "", "day": "", "artists": "Drake"},          # no date
                {"month": "May", "day": "2", "artists": "   "},        # no artists
            ], Path(d) / "r.csv")
            df = SV.load_responses(p, verbose=False)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, S.STAR_SIGN], "Aries")

    def test_raises_a_useful_error_on_the_wrong_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "wrong.csv"
            pd.DataFrame({"a": [1], "b": [2]}).to_csv(p, index=False)
            with self.assertRaises(ValueError) as ctx:
                SV.load_responses(p, verbose=False)
            self.assertIn("birth_month", str(ctx.exception))


# ══ resolver ═════════════════════════════════════════════════════════════════

class FakeSpotify:
    """Minimal stand-in; records calls so we can prove the cache works."""

    def __init__(self, table: dict[str, str]):
        self.table = table
        self.calls = 0

    def get(self, path, params=None):
        self.calls += 1
        name = (params or {}).get("q", "")
        genre = self.table.get(name)
        if genre is None:
            return {"artists": {"items": []}}
        return {"artists": {"items": [
            {"id": f"id-{name}", "name": name, "genres": [genre], "popularity": 50}]}}


class TestArtistResolver(unittest.TestCase):

    def test_does_not_touch_the_network_unless_allowed(self):
        client = FakeSpotify({"Drake": "hip hop"})
        r = SV.ArtistResolver(client=client, cache_path=None, snapshot_path=None)
        self.assertIsNone(r.resolve_one("Drake"))
        self.assertEqual(client.calls, 0, "resolver called the API with allow_network off")

    def test_network_lookup_when_explicitly_enabled(self):
        client = FakeSpotify({"Drake": "hip hop"})
        r = SV.ArtistResolver(client=client, cache_path=None, snapshot_path=None,
                              allow_network=True)
        self.assertEqual(r.resolve_one("Drake")["genre"], "hip hop")

    def test_second_lookup_is_served_from_cache(self):
        client = FakeSpotify({"Drake": "hip hop"})
        r = SV.ArtistResolver(client=client, cache_path=None, snapshot_path=None,
                              allow_network=True)
        r.resolve_one("Drake")
        r.resolve_one("drake")          # different spelling, same key
        self.assertEqual(client.calls, 1, "cache did not prevent a second request")
        self.assertEqual(r.stats["cache"], 1)

    def test_snapshot_is_preferred_over_the_network(self):
        with tempfile.TemporaryDirectory() as d:
            snap = Path(d) / "snap.csv"
            pd.DataFrame({"id": ["x"], "name": ["Drake"], "genre": ["rap"],
                          "genres": ["['rap']"], "followers": [1],
                          "popularity": [9]}).to_csv(snap, index=False)
            client = FakeSpotify({"Drake": "hip hop"})
            r = SV.ArtistResolver(client=client, cache_path=None,
                                  snapshot_path=snap, allow_network=True)
            self.assertEqual(r.resolve_one("Drake")["genre"], "rap")
            self.assertEqual(client.calls, 0, "went to the network despite a snapshot hit")


# ══ the pair that makes a null meaningful ════════════════════════════════════

def _synthetic_responses(n_per_sign: int, effect: float, seed: int = 0):
    """Build responses where sign predicts genre with strength `effect`.

    effect = 0.0  -> genre is independent of sign (true null)
    effect = 1.0  -> each sign listens only to its own genre (perfect signal)
    """
    rng = np.random.default_rng(seed)
    genres = [f"genre_{i}" for i in range(12)]
    rows = []
    for si, sign in enumerate(S.ZODIAC_SIGNS):
        for k in range(n_per_sign):
            uid = f"{sign[:3]}{k:03d}"
            for _ in range(12):
                g = genres[si] if rng.random() < effect else genres[rng.integers(12)]
                rows.append({S.USER_ID: uid, S.ARTIST: f"a{rng.integers(400)}",
                             S.GENRE: g, S.POPULARITY: float(rng.integers(10, 90))})
    long = pd.DataFrame(rows)
    truth = {f"{s[:3]}{k:03d}": s for s in S.ZODIAC_SIGNS for k in range(n_per_sign)}
    return long, truth


class TestPipelineCanDetectAnEffect(unittest.TestCase):
    """Positive and negative control for the analysis itself.

    Without these two, a null result from `analyse_survey.py` proves nothing:
    broken code also returns chance.
    """

    @staticmethod
    def _run(long, truth):
        from lovify_ml import features as Fm

        vectors = Fm.genre_profile(long)
        users = list(vectors.index)
        X = vectors.loc[users].to_numpy()
        y = np.array([truth[u] for u in users])
        real = analyse._classify(X, y)
        rng = np.random.default_rng(0)
        shuffled = np.mean([analyse._classify(X, rng.permutation(y), seed=i)["balanced"]
                            for i in range(3)])
        return real["balanced"], float(shuffled)

    def test_planted_effect_is_found(self):
        long, truth = _synthetic_responses(n_per_sign=12, effect=0.75, seed=1)
        real, shuffled = self._run(long, truth)
        self.assertGreater(real, 0.5,
                           f"planted a strong effect; pipeline only reached {real:.3f}")
        self.assertGreater(real - shuffled, analyse.CONTROL_MARGIN,
                           "control gap did not open on a planted effect")

    def test_no_effect_reports_chance(self):
        long, truth = _synthetic_responses(n_per_sign=12, effect=0.0, seed=2)
        real, shuffled = self._run(long, truth)
        self.assertLess(real, 0.20,
                        f"found {real:.3f} where nothing was planted -- false positive")
        self.assertLessEqual(real - shuffled, analyse.CONTROL_MARGIN + 0.05,
                             "control gap opened on pure noise")

    def test_control_margin_is_a_sane_threshold(self):
        self.assertGreater(analyse.CONTROL_MARGIN, 0.0)
        self.assertLess(analyse.CONTROL_MARGIN, 0.10)


class TestPowerCalculation(unittest.TestCase):

    def test_power_rises_with_n_and_with_effect(self):
        self.assertLess(analyse._power_for(50, 0.12), analyse._power_for(500, 0.12))
        self.assertLess(analyse._power_for(200, 0.10), analyse._power_for(200, 0.25))

    def test_matches_the_preregistered_table(self):
        """~148 responses for 80% power at a 15% effect."""
        self.assertGreater(analyse._power_for(148, 0.15), 0.72)
        self.assertLess(analyse._power_for(50, 0.15), 0.55)

    def test_no_power_at_the_chance_rate_beyond_alpha(self):
        self.assertLess(analyse._power_for(200, S.CHANCE_ACCURACY), 0.10)


class TestSurveyScorer(unittest.TestCase):

    def setUp(self):
        self.long = pd.DataFrame({
            S.USER_ID: ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            S.ARTIST: ["x", "y", "z", "x", "y", "w", "p", "q", "r"],
            S.GENRE: ["pop", "pop", "rock", "pop", "pop", "rock", "jazz", "jazz", "folk"],
            S.POPULARITY: [70, 60, 50, 70, 60, 40, 20, 30, 10],
        })

    def test_similar_users_score_higher(self):
        s = SV.SurveyScorer(self.long)
        self.assertGreater(s.raw_score("A", "B"), s.raw_score("A", "C"))

    def test_weights_are_renormalised_over_what_is_computable(self):
        s = SV.SurveyScorer(self.long)
        self.assertAlmostEqual(sum(s.weights.values()), 1.0, places=9)
        self.assertNotIn("feature_wasserstein", s.weights)
        self.assertNotIn("era_overlap", s.weights)

    def test_works_without_audio_features_present(self):
        """The whole reason this class exists instead of CompatibilityScorer."""
        for feature in S.AUDIO_FEATURES:
            self.assertNotIn(feature, self.long.columns)
        SV.SurveyScorer(self.long).raw_score("A", "B")  # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
