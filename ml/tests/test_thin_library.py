"""What the service says to someone who has barely listened to anything.

Both features on the results page were only ever exercised on full libraries,
and both behaved badly on thin ones in opposite directions:

  * `taste_profile()` always names a headline sign. Measured on 98 real
    listeners from the scrape, split into disjoint halves, that sign changes
    between two independent samples of the SAME person 68% of the time at 10
    tracks and 59% at 20. It was stating a coin flip as a result.
  * `compatibility()` withheld almost everything below MIN_LIBRARY_TRACKS,
    including the shared artists and genres that are exact at any size. A
    dormant account got a page that read as "nothing in common" when the
    truth was "we declined to say".

These tests pin both fixes. The shared property worth naming: the payload
shape must not change with library size, because the web app renders one
layout and a missing key is a crash rather than a smaller answer.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from lovify_ml import schema as S  # noqa: E402
from lovify_ml import service as SV  # noqa: E402
from lovify_ml import synthetic as syn  # noqa: E402

SEED = 7


def _fitted():
    catalog = syn.make_catalog(2000, seed=SEED)
    zodiac = syn.make_zodiac_playlists(catalog, seed=11)
    listens, _ = syn.make_users(catalog, n_users=20, tracks_per_user=160, seed=23)
    models = SV.LovifyModels().fit(catalog, zodiac, listens)
    return models, catalog, listens


class TasteSummaryBands(unittest.TestCase):
    """taste_summary() returns less as the library shrinks, never nothing."""

    @classmethod
    def setUpClass(cls):
        cls.models, cls.catalog, cls.listens = _fitted()
        users = sorted(cls.listens[S.USER_ID].unique())
        cls.lib = cls.listens[cls.listens[S.USER_ID] == users[0]]

    # ── the bands ────────────────────────────────────────────────────────────

    def test_empty_frame_is_answered_not_raised(self):
        """A user whose tracks all fail the feature join arrives with 0 rows.

        Reachable in production: feature-store coverage is ~41%.
        """
        out = self.models.taste_summary(self.lib.head(0))
        self.assertEqual(out["n_tracks"], 0)
        self.assertEqual(out["confidence"], "none")
        self.assertFalse(out["show_headline"])
        self.assertFalse(out["show_mix"])
        self.assertIsNone(out["headline_sign"])
        self.assertEqual(out["mix"], {})
        self.assertTrue(out["caveat"], "an empty library still needs a sentence")

    def test_taste_profile_says_what_to_call_instead_on_an_empty_frame(self):
        with self.assertRaises(ValueError) as ctx:
            self.models.taste_profile(self.lib.head(0))
        self.assertIn("taste_summary", str(ctx.exception))

    def test_a_very_thin_library_gets_no_sign_and_no_mix(self):
        out = self.models.taste_summary(self.lib.head(20))
        self.assertEqual(out["confidence"], "none")
        self.assertFalse(out["show_headline"])
        self.assertFalse(out["show_mix"])
        self.assertIsNone(out["headline_sign"])
        self.assertIsNone(out["headline_lift"])
        self.assertEqual(out["top_signs"], [])

    def test_a_middling_library_gets_the_blend_but_no_headline(self):
        out = self.models.taste_summary(self.lib.head(50))
        self.assertEqual(out["confidence"], "low")
        self.assertTrue(out["show_mix"])
        self.assertFalse(out["show_headline"],
                         "50 tracks is below the measured 60%-stability floor")
        self.assertIsNone(out["headline_sign"])
        self.assertEqual(len(out["mix"]), S.N_SIGNS)
        self.assertAlmostEqual(sum(out["mix"].values()), 1.0, places=2)
        self.assertEqual(len(out["top_signs"]), 3)

    def test_a_full_library_gets_the_headline(self):
        out = self.models.taste_summary(self.lib)
        self.assertEqual(out["confidence"], "ok")
        self.assertTrue(out["show_headline"])
        self.assertTrue(out["show_mix"])
        self.assertIn(out["headline_sign"], out["mix"])
        self.assertIsInstance(out["headline_lift"], float)
        # The summary must agree with the full-detail method it wraps.
        self.assertEqual(out["headline_sign"],
                         self.models.taste_profile(self.lib)["headline_sign"])

    # ── properties that must hold at every size ──────────────────────────────

    def test_the_payload_shape_never_changes_with_library_size(self):
        keys = None
        for n in (0, 1, 5, 20, 29, 30, 50, 74, 75, 100, len(self.lib)):
            out = self.models.taste_summary(self.lib.head(n))
            if keys is None:
                keys = set(out)
            self.assertEqual(set(out), keys,
                             f"the key set changed at {n} tracks -- the UI "
                             "would have to special-case it")
            self.assertTrue(out["caveat"], f"no caveat sentence at {n} tracks")
            self.assertEqual(out["n_tracks"], n)

    def test_a_sign_is_never_named_without_the_flag_that_allows_it(self):
        """The payload itself must make over-claiming impossible.

        If `show_headline` is False there is no sign in the payload to
        render, so a caller that ignores the flag still cannot print one.
        """
        for n in (0, 5, 20, 40, 74):
            out = self.models.taste_summary(self.lib.head(n))
            self.assertFalse(out["show_headline"])
            self.assertIsNone(out["headline_sign"])
            self.assertFalse(out["is_distinctive"],
                             "the distinctive badge fires for 80% of 10-track "
                             "libraries against 53% of 100-track ones; below "
                             "the headline floor it is a small-sample artefact")

    def test_the_floor_is_where_the_measurement_put_it(self):
        """MIN_TRACKS_FOR_HEADLINE must stay the first >=60%-stable rung.

        This is the test that fails if someone lowers the floor to make more
        users see a sign. The table it reads is the measured one in
        service.py; moving the floor without re-measuring breaks it.
        """
        M = SV.LovifyModels
        stable = [n for n, s in sorted(M.HEADLINE_STABILITY.items()) if s >= 0.60]
        self.assertTrue(stable, "no measured rung reaches 60% stability")
        self.assertLessEqual(M.MIN_TRACKS_FOR_HEADLINE, 100)
        self.assertGreaterEqual(
            M.HEADLINE_STABILITY[
                max(n for n in M.HEADLINE_STABILITY if n <= M.MIN_TRACKS_FOR_HEADLINE)],
            0.60,
            "the headline floor sits below the measured 60% stability point")
        self.assertLess(M.MIN_TRACKS_FOR_MIX, M.MIN_TRACKS_FOR_HEADLINE)

    def test_quoted_stability_is_never_better_than_a_measured_rung(self):
        M = self.models
        self.assertIsNone(M.stability_at(0))
        self.assertIsNone(M.stability_at(9))
        self.assertEqual(M.stability_at(10), M.HEADLINE_STABILITY[10])
        self.assertEqual(M.stability_at(49), M.HEADLINE_STABILITY[40],
                         "stability must step DOWN to the nearest measured rung")
        self.assertEqual(M.stability_at(10_000), M.HEADLINE_STABILITY[150],
                         "no extrapolation past the largest rung we measured")


class ThinLibraryCompatibilityPayload(unittest.TestCase):
    """The insufficient branch is a smaller answer now, not a refusal."""

    @classmethod
    def setUpClass(cls):
        cls.models, cls.catalog, cls.listens = _fitted()
        users = sorted(cls.listens[S.USER_ID].unique())
        cls.a = cls.listens[cls.listens[S.USER_ID] == users[0]]
        cls.b = cls.listens[cls.listens[S.USER_ID] == users[1]]
        # Two dormant accounts built from the SAME 40 tracks, so the shared
        # artist count is known to exceed the display cap.
        same = cls.a.head(40)
        cls.thin_a, cls.thin_b = same, same.copy()

    def _thin(self):
        return self.models.compatibility(self.thin_a, self.thin_b)

    def test_percentile_is_the_only_thing_withheld(self):
        out = self._thin()
        self.assertEqual(out["status"], "insufficient_library")
        self.assertIsNone(out["percentile"])
        for k in ("raw_score", "components", "shared_artists",
                  "n_shared_artists", "top_shared_genres"):
            self.assertIn(k, out, f"{k} is exact at any library size")
        self.assertTrue(out["components"], "components are computable here")
        self.assertTrue(out["top_shared_genres"],
                        "two identical libraries share every genre they have")
        self.assertIsNone(out["null_size_used"],
                          "no rung was consulted, so none should be named")

    def test_shared_artist_count_is_the_total_not_the_shown_list(self):
        out = self._thin()
        true_total = self.thin_a[S.ARTIST].nunique()
        self.assertEqual(out["n_shared_artists"], true_total)
        self.assertLessEqual(len(out["shared_artists"]),
                             SV.LovifyModels.SHARED_ARTISTS_SHOWN)
        self.assertGreater(out["n_shared_artists"], len(out["shared_artists"]),
                           "fixture should have more shared artists than the cap")

    def test_the_two_branches_show_the_same_number_of_artists(self):
        thin = self._thin()
        full = self.models.compatibility(self.a, self.b)
        cap = SV.LovifyModels.SHARED_ARTISTS_SHOWN
        self.assertLessEqual(len(thin["shared_artists"]), cap)
        self.assertLessEqual(len(full["shared_artists"]), cap)
        self.assertEqual(len(thin["shared_artists"]), cap,
                         "the thin branch should fill the cap on this fixture")

    def test_every_branch_returns_the_same_keys(self):
        ok = self.models.compatibility(self.a, self.b)
        thin = self._thin()
        none = self.models.compatibility(self.a.head(0), self.b)
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(none["status"], "no_resolvable_tracks")
        self.assertEqual(set(thin), set(ok))
        self.assertEqual(set(none), set(ok))

    def test_the_sentence_leads_with_what_we_found(self):
        out = self._thin()
        text = out["interpretation"]
        self.assertFalse(text.lower().startswith("not enough"),
                         "lead with what we can tell them, not what we cannot")
        first = text.split(".")[0].lower()
        self.assertTrue(
            any(w in first for w in ("in common", "share", "count")),
            f"the opening sentence should be about the overlap: {first!r}")
        self.assertIn("percentage", text.lower(),
                      "the missing number still has to be explained, second")

    def test_a_pair_with_nothing_in_common_still_gets_a_sentence(self):
        """No shared artist and no shared genre must not produce a dangling lead."""
        thin_a = self.a.head(30)
        other = self.b[~self.b[S.ARTIST].isin(set(thin_a[S.ARTIST]))].head(30)
        out = self.models.compatibility(thin_a, other)
        self.assertEqual(out["status"], "insufficient_library")
        self.assertTrue(out["interpretation"].strip())
        self.assertTrue(out["interpretation"].endswith("."))

    def test_an_empty_library_still_renders(self):
        out = self.models.compatibility(self.a.head(0), self.b)
        self.assertIsNone(out["percentile"])
        self.assertEqual(out["components"], {})
        self.assertEqual(out["top_shared_genres"], [])
        self.assertEqual(out["n_shared_artists"], 0)
        self.assertTrue(out["interpretation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
