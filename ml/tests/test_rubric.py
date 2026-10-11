"""Tests for the designed zodiac rubric.

The most important tests here are not about numerical correctness. They are
structural: they assert that the rubric *stays* a composition of seven
profiles and never acquires per-sign special cases. That property is the only
reason to prefer this to twelve hand-written templates, and it is exactly the
kind of thing that erodes quietly under later edits.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lovify_ml import features as Fm  # noqa: E402
from lovify_ml import rubric as R  # noqa: E402
from lovify_ml import schema as S  # noqa: E402
from lovify_ml import synthetic as syn  # noqa: E402


def _catalog(n=3000):
    return syn.make_catalog(n, seed=7)


class TheStructuralClaim(unittest.TestCase):
    """element x modality is a bijection onto the twelve signs (ref. 1.4)."""

    def test_every_sign_has_both_axes(self):
        for sign in S.ZODIAC_SIGNS:
            self.assertIn(sign, R.ELEMENT_OF)
            self.assertIn(sign, R.MODALITY_OF)
        self.assertEqual(set(R.ELEMENT_OF), set(S.ZODIAC_SIGNS))
        self.assertEqual(set(R.MODALITY_OF), set(S.ZODIAC_SIGNS))

    def test_twelve_distinct_pairs_no_collisions_no_gaps(self):
        pairs = [(R.ELEMENT_OF[s], R.MODALITY_OF[s]) for s in S.ZODIAC_SIGNS]
        self.assertEqual(len(set(pairs)), 12, "two signs share an (element, modality) cell")
        self.assertEqual(
            set(pairs),
            {(e, m) for e in ("Fire", "Earth", "Air", "Water")
             for m in ("Cardinal", "Fixed", "Mutable")},
            "the 4x3 grid is not fully covered",
        )

    def test_three_signs_per_element_four_per_modality(self):
        for element in ("Fire", "Earth", "Air", "Water"):
            n = sum(1 for s in S.ZODIAC_SIGNS if R.ELEMENT_OF[s] == element)
            self.assertEqual(n, 3, f"{element} should hold 3 signs, holds {n}")
        for modality in ("Cardinal", "Fixed", "Mutable"):
            n = sum(1 for s in S.ZODIAC_SIGNS if R.MODALITY_OF[s] == modality)
            self.assertEqual(n, 4, f"{modality} should hold 4 signs, holds {n}")

    def test_zodiacal_order_cycles_4_and_3(self):
        """The bijection is forced by arithmetic, not chosen (ref. 1.4.1)."""
        elements = ["Fire", "Earth", "Air", "Water"]
        modalities = ["Cardinal", "Fixed", "Mutable"]
        for i, sign in enumerate(S.ZODIAC_SIGNS):
            self.assertEqual(R.ELEMENT_OF[sign], elements[i % 4])
            self.assertEqual(R.MODALITY_OF[sign], modalities[i % 3])


class SevenDecisionsNotTwelve(unittest.TestCase):
    """The property that makes the rubric auditable. Guard it."""

    def test_only_seven_profiles_are_specified(self):
        self.assertEqual(len(R.ELEMENT_PROFILE), 4)
        self.assertEqual(len(R.MODALITY_PROFILE), 3)

    def test_no_per_sign_table_exists(self):
        """A SIGN_PROFILE dict appearing here would defeat the whole design.

        archetypes.py legitimately hand-writes twelve templates. This module
        must not, or it is just that module with extra steps.
        """
        for name in dir(R):
            self.assertNotIn(
                name,
                {"SIGN_PROFILE", "SIGN_PROFILES", "SIGN_TEMPLATES"},
                "a per-sign profile table has appeared; the twelve must stay derived",
            )

    def test_every_sign_is_exactly_its_two_axes_summed(self):
        """No residue. Reconstruct all twelve from the seven and compare."""
        for sign in S.ZODIAC_SIGNS:
            expected = pd.Series(0.0, index=R.RUBRIC_FEATURES)
            for feat, d in R.ELEMENT_PROFILE[R.ELEMENT_OF[sign]].items():
                expected[feat] += d
            for feat, d in R.MODALITY_PROFILE[R.MODALITY_OF[sign]].items():
                expected[feat] += d
            pd.testing.assert_series_equal(
                R.sign_profile(sign), expected, check_names=False,
                obj=f"{sign} is not the sum of its axes",
            )

    def test_signs_sharing_an_element_differ_only_by_modality(self):
        fire = ["Aries", "Leo", "Sagittarius"]
        for a, b in zip(fire, fire[1:]):
            diff = R.sign_profile(a) - R.sign_profile(b)
            mod_a = pd.Series(R.MODALITY_PROFILE[R.MODALITY_OF[a]]).reindex(
                R.RUBRIC_FEATURES).fillna(0.0)
            mod_b = pd.Series(R.MODALITY_PROFILE[R.MODALITY_OF[b]]).reindex(
                R.RUBRIC_FEATURES).fillna(0.0)
            pd.testing.assert_series_equal(diff, mod_a - mod_b, check_names=False)

    def test_polarity_is_not_an_input(self):
        """ref. 1.6.1 -- determined by element, so it would add nothing."""
        for name in dir(R):
            self.assertNotIn(name, {"POLARITY_PROFILE", "POLARITY_OF"},
                             "polarity is fully determined by element; do not add it")

    def test_modalities_do_not_overwhelm_elements(self):
        """If they did, the 4x3 grid would collapse toward 3 clusters."""
        el = max(sum(abs(v) for v in p.values()) for p in R.ELEMENT_PROFILE.values())
        mod = max(sum(abs(v) for v in p.values()) for p in R.MODALITY_PROFILE.values())
        self.assertLess(mod, el, "modality offsets should modulate, not dominate")

    def test_cyclic_and_categorical_features_are_excluded(self):
        for f in ("key", "mode", "time_signature"):
            self.assertNotIn(f, R.RUBRIC_FEATURES)


class ContestedCells(unittest.TestCase):
    """ref. 1.7.1(a) -- the decisions must be recorded, not just made."""

    def test_all_three_are_documented(self):
        self.assertEqual(set(R.CONTESTED), {"Libra", "Aquarius", "Cancer"})
        for sign, note in R.CONTESTED.items():
            self.assertGreater(len(note), 40, f"{sign}'s rationale is too thin to audit")

    def test_structure_was_followed_in_all_three(self):
        """Each contested sign is still pure composition -- no override."""
        for sign in R.CONTESTED:
            expected = pd.Series(0.0, index=R.RUBRIC_FEATURES)
            for feat, d in R.ELEMENT_PROFILE[R.ELEMENT_OF[sign]].items():
                expected[feat] += d
            for feat, d in R.MODALITY_PROFILE[R.MODALITY_OF[sign]].items():
                expected[feat] += d
            pd.testing.assert_series_equal(R.sign_profile(sign), expected,
                                           check_names=False)

    def test_cancer_resolves_inward_as_claimed(self):
        """The docstring claims Water outweighs Cardinal here. Check it."""
        cancer = R.sign_profile("Cancer")
        self.assertLess(cancer["valence"], -0.5, "Cancer should read as emotionally heavy")
        self.assertLess(cancer["energy"], 0.0, "Water's pull inward should survive Cardinal")

    def test_explain_surfaces_the_contested_note(self):
        for sign in S.ZODIAC_SIGNS:
            exp = R.explain(sign)
            if sign in R.CONTESTED:
                self.assertIsNotNone(exp["contested"])
            else:
                self.assertIsNone(exp["contested"])


class Scoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = _catalog()
        cls.scaler = Fm.fit_scaler(cls.catalog)

    def test_affinities_are_a_distribution(self):
        s = R.score_tracks(self.catalog.head(200), self.scaler)
        self.assertEqual(list(s.columns), S.ZODIAC_SIGNS)
        np.testing.assert_allclose(s.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
        self.assertTrue((s.to_numpy() >= 0).all())

    def test_soft_not_hard(self):
        """A track between two signs must not be rounded to a vote."""
        s = R.score_tracks(self.catalog.head(500), self.scaler)
        self.assertLess(s.max(axis=1).max(), 0.999,
                        "assignment looks hard; the in-between information is lost")

    def test_a_track_built_to_be_fiery_scores_fiery(self):
        """Sanity: the rubric points the direction it says it does."""
        row = self.catalog.iloc[0].copy()
        mean, std = self.catalog[R.RUBRIC_FEATURES].mean(), self.catalog[R.RUBRIC_FEATURES].std()
        for f in R.RUBRIC_FEATURES:
            row[f] = mean[f]
        for f, d in R.ELEMENT_PROFILE["Fire"].items():
            row[f] = mean[f] + 2.0 * d * std[f]
        scores = R.score_track(row, self.scaler)
        fire = {"Aries", "Leo", "Sagittarius"}
        self.assertIn(scores.idxmax(), fire,
                      f"a fire-shaped track scored {scores.idxmax()}")

    def test_temperature_controls_sharpness(self):
        sharp = R.score_tracks(self.catalog.head(300), self.scaler, temperature=0.2)
        soft = R.score_tracks(self.catalog.head(300), self.scaler, temperature=5.0)
        self.assertGreater(sharp.max(axis=1).mean(), soft.max(axis=1).mean())

    def test_taste_mix_matches_archetypes_shape(self):
        mix = R.taste_mix(self.catalog.head(400), self.scaler)
        self.assertIsInstance(mix, pd.Series)
        self.assertEqual(set(mix.index), set(S.ZODIAC_SIGNS))
        self.assertAlmostEqual(float(mix.sum()), 1.0, places=9)
        self.assertTrue(mix.is_monotonic_decreasing, "mix should come back sorted")

    def test_lift_is_relative_to_the_catalog(self):
        base = R.base_rate(self.catalog, self.scaler, sample=1500)
        mix = R.taste_mix(self.catalog.head(300), self.scaler)
        lf = R.lift(mix, base)
        self.assertEqual(set(lf.index), set(S.ZODIAC_SIGNS))
        self.assertTrue((lf > 0).all())

    def test_a_listener_identical_to_the_catalog_has_no_lift(self):
        """The guard against reporting the prior back as a personality."""
        base = R.base_rate(self.catalog, self.scaler, sample=None)
        lf = R.lift(R.taste_mix(self.catalog, self.scaler), base)
        np.testing.assert_allclose(lf.to_numpy(), 1.0, atol=1e-6)

    def test_profile_payload_matches_the_service_shape(self):
        base = R.base_rate(self.catalog, self.scaler, sample=1500)
        p = R.profile(self.catalog.head(300), self.scaler, base)
        self.assertEqual(set(p), {"mix", "lift", "headline_sign",
                                  "headline_lift", "is_distinctive"})
        self.assertIn(p["headline_sign"], S.ZODIAC_SIGNS)
        self.assertIsInstance(p["is_distinctive"], bool)


class Auditability(unittest.TestCase):
    def test_explain_traces_every_sign_to_its_sources(self):
        for sign in S.ZODIAC_SIGNS:
            exp = R.explain(sign)
            self.assertEqual(exp["element"], R.ELEMENT_OF[sign])
            self.assertEqual(exp["modality"], R.MODALITY_OF[sign])
            self.assertEqual(set(exp["sources"]), {"element", "modality", "composition"})
            for cite in exp["sources"].values():
                self.assertIn("ASTROLOGY_REFERENCE.md", cite)

    def test_explained_components_reconstruct_the_composed_profile(self):
        for sign in S.ZODIAC_SIGNS:
            exp = R.explain(sign)
            total = pd.Series(0.0, index=R.RUBRIC_FEATURES)
            for part in ("element_contribution", "modality_contribution"):
                for f, v in exp[part].items():
                    total[f] += v
            for f, v in exp["composed"].items():
                self.assertAlmostEqual(total[f], v, places=3,
                                       msg=f"{sign}/{f} does not reconstruct")

    def test_describe_reports_distribution_and_no_accuracy(self):
        catalog = _catalog(2000)
        d = R.describe(catalog, Fm.fit_scaler(catalog), sample=None)
        self.assertEqual(len(d), 12)
        for banned in ("accuracy", "score", "correct", "precision", "recall", "auc"):
            self.assertNotIn(banned, [c.lower() for c in d.columns],
                             "the rubric must not report an accuracy-shaped number")
        np.testing.assert_allclose(d["soft_share"].sum(), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
