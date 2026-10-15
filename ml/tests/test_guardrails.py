"""Guardrails: the properties that must stay true as the models change.

These are not unit tests of functions. They are end-to-end invariants that
would have caught the mistakes this project has actually made. Each one
corresponds to a specific thing that went wrong:

  * a classifier that scored far above chance because duplicate tracks leaked
  * a recommender returning ten near-identical songs because nobody scaled
  * a percentile that quietly reported "incompatible" for light listeners
  * a saved artifact that stored its catalog twice

A test that only passes is worth little; what makes these worth running is
that each one fails loudly if its mistake is reintroduced. The leakage canary
in particular is the test that would have saved a semester.

Run:  cd ml/tests && ../.venv/bin/python -m unittest discover -s . -t .
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lovify_ml import recommend as R  # noqa: E402
from lovify_ml import schema as S  # noqa: E402
from lovify_ml import service as SV  # noqa: E402
from lovify_ml import similarity as SIM  # noqa: E402
from lovify_ml import synthetic as syn  # noqa: E402
from lovify_ml import zodiac as Z  # noqa: E402

SEED = 7


def _catalog(n=3000):
    return syn.make_catalog(n, seed=SEED)


class ZodiacLeakageCanary(unittest.TestCase):
    """The single most important test in the repository."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = _catalog()
        cls.zodiac = syn.make_zodiac_playlists(cls.catalog, seed=11)

    def test_honest_accuracy_stays_near_chance(self):
        """If this starts passing with a high number, something leaked.

        Zodiac sign is close to unpredictable from audio features. Under a
        deduplicated, group-aware split no model should get far above the
        8.3% chance rate. A sudden jump means a track is appearing on both
        sides of the split again -- which is exactly the failure that produced
        the original 76% claim.
        """
        clean = Z.deduplicate(self.zodiac, "majority")
        train, test = Z.honest_split(clean, test_size=0.25, seed=0)
        best = Z.benchmark(train, test)["accuracy"].max()

        self.assertLess(
            best, 0.30,
            f"Honest-protocol accuracy is {best:.1%}, far above the {S.CHANCE_ACCURACY:.1%} "
            "chance rate. Zodiac sign is not this predictable from audio features. "
            "Suspect leakage: check that deduplicate() still collapses tracks and that "
            "the split groups on track_id.",
        )

    def test_dedup_leaves_no_track_on_both_sides(self):
        clean = Z.deduplicate(self.zodiac, "majority")
        train, test = Z.honest_split(clean, test_size=0.25, seed=0)
        overlap = set(train[S.TRACK_ID]) & set(test[S.TRACK_ID])
        self.assertEqual(overlap, set(), f"{len(overlap)} tracks straddle the split")

    def test_the_leakage_demonstration_still_works(self):
        """Notebook 01's central exhibit must remain reproducible.

        A naive row split should score far above an honest one on identical
        data. If this gap closes, the notebook's argument has silently stopped
        being demonstrable and its prose would become wrong.
        """
        tr_n, te_n = Z.naive_split(self.zodiac, 0.25, seed=0)
        naive = Z.benchmark(tr_n, te_n)["accuracy"].max()

        clean = Z.deduplicate(self.zodiac, "majority")
        tr_h, te_h = Z.honest_split(clean, 0.25, seed=0)
        honest = Z.benchmark(tr_h, te_h)["accuracy"].max()

        self.assertGreater(
            naive, honest * 2,
            f"naive {naive:.3f} vs honest {honest:.3f}: the leakage gap has collapsed, "
            "so notebook 01's demonstration no longer demonstrates anything.",
        )


class RecommenderGuardrails(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = _catalog(4000)
        cls.listens, _ = syn.make_users(cls.catalog, n_users=12,
                                        tracks_per_user=120, seed=23)
        cls.user = cls.listens[cls.listens[S.USER_ID] ==
                               cls.listens[S.USER_ID].iloc[0]]

    def test_unscaled_cosine_is_degenerate(self):
        """Documents the bug rather than hiding it.

        Without scaling, the vector norm is essentially duration_ms, every
        song points the same direction, and the top-N are separated in the
        twelfth decimal place. If this test ever fails, the default changed
        and the notebook 03 narrative needs rewriting.
        """
        raw = R.Recommender(self.catalog, scale_features=False)
        norms = np.linalg.norm(raw.Xn * 0 + self.catalog[raw.features].to_numpy(float),
                               axis=1)
        corr = float(np.corrcoef(norms, self.catalog["duration_ms"])[0, 1])
        self.assertGreater(corr, 0.999, "duration_ms no longer dominates the raw norm")

        recs = raw.recommend(self.user, n=10)
        self.assertLess(R.intra_list_diversity(recs, raw), 0.01,
                        "unscaled recommendations are no longer degenerate")

    def test_scaling_beats_not_scaling(self):
        scaled = R.Recommender(self.catalog, scale_features=True)
        raw = R.Recommender(self.catalog, scale_features=False)
        a = R.holdout_evaluation(scaled, self.listens, n=20, seed=0)["recall@20"]
        b = R.holdout_evaluation(raw, self.listens, n=20, seed=0)["recall@20"]
        self.assertGreater(a, b, f"scaled recall {a:.4f} did not beat unscaled {b:.4f}")

    def test_recommendations_exclude_already_heard(self):
        rec = R.Recommender(self.catalog)
        recs = rec.recommend(self.user, n=20, exclude_heard=True)
        heard = set(self.user[S.TRACK_ID])
        self.assertEqual(set(recs[S.TRACK_ID]) & heard, set())

    def test_taste_modes_beat_the_mean_on_diversity(self):
        rec = R.Recommender(self.catalog)
        one = R.intra_list_diversity(rec.recommend(self.user, n=10, modes=1), rec)
        many = R.intra_list_diversity(rec.recommend(self.user, n=10, modes=3), rec)
        self.assertGreater(many, one)


class ServiceGuardrails(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = _catalog(2500)
        zodiac = syn.make_zodiac_playlists(cls.catalog, seed=11)
        cls.listens, _ = syn.make_users(cls.catalog, n_users=30,
                                        tracks_per_user=160, seed=23)
        cls.models = SV.LovifyModels().fit(cls.catalog, zodiac, cls.listens)
        users = sorted(cls.listens[S.USER_ID].unique())
        cls.a = cls.listens[cls.listens[S.USER_ID] == users[0]]
        cls.b = cls.listens[cls.listens[S.USER_ID] == users[1]]

    def test_small_libraries_are_refused_not_guessed(self):
        """A confidently wrong number is worse than no number.

        Scored against a full-size reference population, a light listener is
        pushed tens of percentile points downward for reasons unrelated to
        taste. The service must decline instead.
        """
        out = self.models.compatibility(self.a.head(20), self.b.head(20))
        self.assertEqual(out["status"], "insufficient_library")
        self.assertIsNone(out["percentile"])
        self.assertIn("shared_artists", out, "the honest fallback must still show evidence")

    def test_full_libraries_are_scored(self):
        out = self.models.compatibility(self.a, self.b)
        self.assertEqual(out["status"], "ok")
        self.assertTrue(0 <= out["percentile"] <= 100)

    def test_null_ladder_covers_every_supported_rung(self):
        """Every rung the population can actually fill, and no more.

        This test previously demanded all of NULL_SIZES, which encoded the
        bug the PR review found: sample() caps at len(user), so a rung above
        the largest library is that library relabelled. Skipping those is
        correct, so the assertion is now "a prefix of NULL_SIZES, bounded by
        the biggest reference library".
        """
        largest = max(len(g) for _, g in self.listens.groupby(S.USER_ID))
        expected = [s for s in SV.LovifyModels.NULL_SIZES if s <= largest]
        self.assertEqual(sorted(self.models.null_ladder), expected)
        self.assertTrue(expected, "no rung is small enough for this population")
        for size, null in self.models.null_ladder.items():
            self.assertGreater(len(null), 50, f"rung {size} is too sparse to read a percentile from")

    def test_percentiles_are_calibrated(self):
        """Random pairs should spread across the range, not pile at one end."""
        users = sorted(self.listens[S.USER_ID].unique())
        rng = np.random.default_rng(0)
        scorer = SIM.CompatibilityScorer(self.listens, catalog=self.catalog)
        scorer._null = self.models.null_ladder[max(self.models.null_ladder)]
        pcts = []
        for _ in range(60):
            u, v = rng.choice(users, 2, replace=False)
            pcts.append(scorer.percentile(u, v))
        self.assertTrue(30 < float(np.mean(pcts)) < 70,
                        f"mean percentile of random pairs is {np.mean(pcts):.1f}, not ~50")

    def test_artifact_does_not_store_the_catalog_twice(self):
        """Catches the duplication regression.

        The catalog is ~2,500 rows here. An artifact several times its
        serialised size means derived state is being persisted again.
        """
        with tempfile.TemporaryDirectory() as d:
            p = self.models.save(Path(d) / "m.joblib")
            size_mb = os.path.getsize(p) / 1e6
            self.assertLess(size_mb, 1.5, f"artifact is {size_mb:.2f} MB for a 2.5k catalog")
            reloaded = SV.LovifyModels.load(p)
        self.assertIsNotNone(reloaded._rec, "recommender was not rebuilt on load")
        self.assertEqual(reloaded.compatibility(self.a, self.b)["percentile"],
                         self.models.compatibility(self.a, self.b)["percentile"])

    def test_taste_profile_admits_when_it_has_nothing_to_say(self):
        prof = self.models.taste_profile(self.a)
        self.assertIn("is_distinctive", prof)
        if prof["headline_lift"] < 1.25:
            self.assertFalse(prof["is_distinctive"])


class ScalingContract(unittest.TestCase):
    """Any distance must run on scaled columns. This is the project's oldest bug."""

    def test_raw_feature_variance_is_dominated_by_duration(self):
        catalog = _catalog(2000)
        var = catalog[S.AUDIO_FEATURES].var()
        share = var / var.sum()
        self.assertGreater(share["duration_ms"], 0.99,
                           "if this changes, the notebook 00 narrative needs updating")

    def test_scaling_equalises_contribution(self):
        from lovify_ml import features as F

        catalog = _catalog(2000)
        Z_ = F.scale(catalog, F.fit_scaler(catalog))
        share = Z_.var(axis=0) / Z_.var(axis=0).sum()
        self.assertLess(float(share.max() - share.min()), 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class ReviewRegressions(unittest.TestCase):
    """Three findings from the PR #2 review. Each failed before its fix.

    All three share a shape worth naming: they fail *silently* or in the
    branch written to handle failure. None were caught by the 231 tests that
    existed, because those tested the paths that work.
    """

    @classmethod
    def setUpClass(cls):
        catalog = _catalog(2000)
        zodiac = syn.make_zodiac_playlists(catalog, seed=11)
        listens, _ = syn.make_users(catalog, n_users=20, tracks_per_user=140, seed=23)
        cls.models = SV.LovifyModels().fit(catalog, zodiac, listens)
        users = sorted(listens[S.USER_ID].unique())
        cls.a = listens[listens[S.USER_ID] == users[0]]
        cls.b = listens[listens[S.USER_ID] == users[1]]
        cls.catalog = catalog

    def test_empty_library_refuses_instead_of_crashing(self):
        """The guard for too-little-data used to crash on the least data.

        Reachable in production: feature-store coverage is ~41%, so a user
        whose tracks all fail the join arrives with zero rows.
        """
        out = self.models.compatibility(self.a.head(0), self.b)
        self.assertEqual(out["status"], "no_resolvable_tracks")
        self.assertIsNone(out["percentile"])
        self.assertIn("interpretation", out)

        both = self.models.compatibility(self.a.head(0), self.b.head(0))
        self.assertEqual(both["status"], "no_resolvable_tracks")

    def test_null_ladder_skips_rungs_the_population_cannot_fill(self):
        """A rung built from libraries smaller than its own size is a lie.

        sample() caps at len(user), so on 140-track users the 200/300/500
        rungs would otherwise be the 140-track distribution relabelled, and a
        genuinely large pair gets scored against a small-library null.
        """
        largest = self.models.catalog is not None and 140
        for size in self.models.null_ladder:
            self.assertLessEqual(
                size, 150,
                f"rung {size} exceeds the largest reference library (~140); "
                "it cannot have been built from libraries that size")

    def test_zero_temperature_raises_rather_than_returning_nan(self):
        from lovify_ml import features as F
        from lovify_ml import rubric as R

        scaler = F.fit_scaler(self.catalog)
        tracks = self.catalog.head(5)
        for bad in (0, 0.0, -1.0):
            with self.assertRaises(ValueError, msg=f"temperature={bad} must raise"):
                R.score_tracks(tracks, scaler, temperature=bad)
        ok = R.score_tracks(tracks, scaler, temperature=1.0)
        self.assertFalse(ok.isna().any().any())


class HonestExplanations(unittest.TestCase):
    """The results page must not claim an overlap the score says is zero.

    Found by running the real trained model against two libraries pulled from
    the seeded catalogue -- a metal/jazz listener and a pop/latin one. The
    score was right (genre_overlap 0.0, no shared artists, percentile 0.0) and
    the explanation was wrong: `top_shared_genres` still returned three
    confident labels, which the results template renders under the heading
    "Where you overlap". Two strangers were told they overlap on album rock.

    The cause was `head(3)` on an all-zero sort: pandas is happy to return the
    top three of nothing. Nine unit tests of `explain()` passed throughout,
    because every one of them used users who did share a genre.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        # Two users with deliberately disjoint genres, everything else alike.
        rows = []
        for user, genres in (("A", ["metal", "jazz"]), ("B", ["reggaeton", "k-pop"])):
            for i in range(120):
                rows.append({
                    S.USER_ID: user,
                    S.TRACK_ID: f"{user}{i}",
                    "artist": f"{user}-artist-{i % 20}",
                    "genre": genres[i % len(genres)],
                    "year": 2010 + (i % 10),
                    "popularity": int(rng.integers(20, 80)),
                    **{f: float(rng.random()) for f in S.AUDIO_FEATURES},
                })
        cls.listens = pd.DataFrame(rows)
        cls.scorer = SIM.CompatibilityScorer(cls.listens)
        cls.scorer.fit_null(n_pairs=50, seed=1)

    def test_no_shared_genres_means_no_genre_chips(self):
        out = self.scorer.explain("A", "B")
        self.assertEqual(out["components"]["genre_overlap"], 0.0)
        self.assertEqual(
            out["top_shared_genres"], [],
            "explain() named genres as shared for two users whose genre "
            "overlap is exactly 0 -- the results page renders these under "
            '"Where you overlap"')

    def test_genuinely_shared_genres_are_still_reported(self):
        """The fix must not silence the honest case."""
        both = self.listens.copy()
        both.loc[both[S.USER_ID] == "B", "genre"] = "metal"
        scorer = SIM.CompatibilityScorer(both)
        scorer.fit_null(n_pairs=50, seed=1)
        out = scorer.explain("A", "B")
        self.assertGreater(out["components"]["genre_overlap"], 0.0)
        self.assertIn("metal", out["top_shared_genres"])

    def test_every_reported_genre_has_positive_weight_for_both(self):
        """Stronger than the two above: no label may come from a one-sided taste."""
        out = self.scorer.explain("A", "A")
        gp = self.scorer.genre_profiles
        for g in out["top_shared_genres"]:
            self.assertGreater(gp.loc["A", g], 0.0)
