"""Tests for the unsupervised taste archetypes.

The properties worth pinning here are not "does k-means run" but the ones
that would silently break the product: scaling actually applied, the sign
mapping being a one-to-one assignment, soft membership behaving like a
distribution, and lift meaning what the UI claims it means.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lovify_ml import archetypes as A  # noqa: E402
from lovify_ml import schema as S  # noqa: E402
from lovify_ml import synthetic as syn  # noqa: E402


def _catalog(n=4000):
    return syn.make_catalog(n, seed=7)


class SignAssignment(unittest.TestCase):
    def test_every_sign_used_exactly_once(self):
        """Greedy naming would reuse signs; the assignment must not."""
        dev = pd.DataFrame(
            np.random.default_rng(0).normal(size=(12, len(S.AUDIO_FEATURES))),
            columns=S.AUDIO_FEATURES)
        names = A.assign_sign_names(dev)
        self.assertEqual(sorted(names), sorted(S.ZODIAC_SIGNS))

    def test_assignment_is_order_independent(self):
        """Shuffling the clusters must not change which cluster gets which sign.

        This is the property greedy assignment fails and the reason the
        Hungarian algorithm is used.
        """
        rng = np.random.default_rng(1)
        dev = pd.DataFrame(rng.normal(size=(12, len(S.AUDIO_FEATURES))),
                           columns=S.AUDIO_FEATURES)
        names = A.assign_sign_names(dev)

        order = rng.permutation(12)
        shuffled = dev.iloc[order].reset_index(drop=True)
        shuffled_names = A.assign_sign_names(shuffled)
        self.assertEqual([shuffled_names[list(order).index(i)] for i in range(12)],
                         names)

    def test_a_cluster_matching_a_template_gets_that_sign(self):
        """Sanity: an unambiguous cluster should land on the obvious sign."""
        dev = pd.DataFrame(np.zeros((12, len(S.AUDIO_FEATURES))),
                           columns=S.AUDIO_FEATURES)
        # Make cluster 0 a textbook Pisces and everything else mild noise.
        for f, w in A.SIGN_TEMPLATES["Pisces"].items():
            dev.loc[0, f] = w * 3
        rng = np.random.default_rng(2)
        dev.iloc[1:] = rng.normal(scale=0.05, size=(11, len(S.AUDIO_FEATURES)))
        self.assertEqual(A.assign_sign_names(dev)[0], "Pisces")


class Fitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = _catalog()
        cls.model = A.fit_archetypes(cls.catalog, k=12, seed=0, sample=None)

    def test_shapes_and_names(self):
        self.assertEqual(self.model.k, 12)
        self.assertEqual(sorted(self.model.names), sorted(S.ZODIAC_SIGNS))
        self.assertEqual(self.model.deviations.shape, (12, len(S.AUDIO_FEATURES)))

    def test_shares_sum_to_one(self):
        self.assertAlmostEqual(float(self.model.shares.sum()), 1.0, places=6)

    def test_clustering_is_not_just_song_length(self):
        """The failure mode if scaling is ever dropped.

        Unscaled, duration_ms carries >99.99% of the variance and the
        clusters become length bands. Check the centroids differ on
        something else too.
        """
        dev = self.model.deviations
        spread_duration = float(dev["duration_ms"].max() - dev["duration_ms"].min())
        spread_other = float(max(dev[f].max() - dev[f].min()
                                 for f in ["energy", "valence", "acousticness"]))
        self.assertGreater(spread_other, spread_duration * 0.25,
                           "clusters look like duration bands -- is scaling applied?")


class SoftAssignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = _catalog()
        cls.model = A.fit_archetypes(cls.catalog, k=12, seed=0, sample=None)

    def test_responsibilities_are_a_distribution(self):
        R = A._responsibilities(self.model, self.catalog.head(200))
        self.assertEqual(R.shape, (200, 12))
        np.testing.assert_allclose(R.sum(axis=1), 1.0, atol=1e-9)
        self.assertTrue((R >= 0).all())

    def test_mix_sums_to_one(self):
        mix = A.taste_mix(self.model, self.catalog.head(300))
        self.assertAlmostEqual(float(mix.sum()), 1.0, places=6)
        self.assertEqual(sorted(mix.index), sorted(self.model.names))

    def test_soft_assignment_keeps_more_than_the_argmax(self):
        """A hard assignment would give a track all-or-nothing membership."""
        R = A._responsibilities(self.model, self.catalog.head(500))
        self.assertLess(float(R.max(axis=1).mean()), 0.999,
                        "assignment is effectively hard; temperature is wrong")

    def test_mix_of_the_whole_catalog_matches_the_base_rate(self):
        """Soft shares over everything should track the hard cluster shares.

        Not an identity -- soft and hard assignment differ at the margins --
        but a large gap means the temperature is mis-scaled.
        """
        mix = A.taste_mix(self.model, self.catalog)
        base = A.base_rate(self.model)
        self.assertLess(float((mix - base.reindex(mix.index)).abs().max()), 0.25)


class LiftAndProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = _catalog()
        cls.model = A.fit_archetypes(cls.catalog, k=12, seed=0, sample=None)

    def test_an_average_listener_has_lift_near_one(self):
        """The whole point of lift: the catalog itself is not distinctive."""
        mix = A.taste_mix(self.model, self.catalog)
        lf = A.lift(mix, A.base_rate(self.model))
        self.assertLess(abs(float(lf.iloc[0]) - 1.0), 0.6,
                        "the catalog looks distinctive against its own base rate")

    def test_a_skewed_listener_is_distinctive(self):
        """Someone listening to one corner of feature space must show lift."""
        corner = self.catalog.nlargest(300, "energy")
        prof = A.profile(self.model, corner)
        self.assertGreater(prof["headline_lift"], 1.25)
        self.assertTrue(prof["is_distinctive"])

    def test_profile_payload_shape(self):
        prof = A.profile(self.model, self.catalog.head(200))
        for key in ("mix", "lift", "headline_sign", "headline_lift", "is_distinctive"):
            self.assertIn(key, prof)
        self.assertIn(prof["headline_sign"], self.model.names)

    def test_describe_covers_every_archetype(self):
        d = A.describe(self.model)
        self.assertEqual(len(d), 12)
        self.assertEqual(sorted(d["archetype"]), sorted(S.ZODIAC_SIGNS))
        self.assertAlmostEqual(float(d["catalog_share"].sum()), 1.0, places=3)


class Determinism(unittest.TestCase):
    def test_same_seed_gives_the_same_model(self):
        cat = _catalog(3000)
        a = A.fit_archetypes(cat, k=12, seed=3, sample=None)
        b = A.fit_archetypes(cat, k=12, seed=3, sample=None)
        self.assertEqual(a.names, b.names)
        np.testing.assert_allclose(a.deviations.to_numpy(), b.deviations.to_numpy())

    def test_a_frozen_model_scores_a_user_identically_every_time(self):
        """What actually ships: fit once, then every call must agree.

        Cluster membership is unstable across REFITS (ARI ~0.72), which is
        why the model is frozen in production. This pins the property that
        makes freezing sufficient.
        """
        cat = _catalog(3000)
        m = A.fit_archetypes(cat, k=12, seed=0, sample=None)
        user = cat.sample(150, random_state=5)
        first = A.taste_mix(m, user)
        for _ in range(3):
            pd.testing.assert_series_equal(A.taste_mix(m, user), first)


if __name__ == "__main__":
    unittest.main(verbosity=2)
