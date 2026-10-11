"""The integration surface: one object the Flask app can import and call.

Notebooks are for deciding what the models should be. This is what the
decision compiles down to -- fit once offline, save, then load inside
lovify.py and call three methods. Nothing here re-derives anything; it only
assembles the pieces the notebooks validated.

    # offline, once (see notebooks/04_end_to_end.ipynb)
    models = LovifyModels().fit(catalog, zodiac_df, listens)
    models.save("artifacts/lovify_models.joblib")

    # inside the Flask app
    models = LovifyModels.load("artifacts/lovify_models.joblib")
    models.taste_profile(user_tracks)
    models.compatibility(user_a_tracks, user_b_tracks)
    models.recommend(user_tracks, n=10)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import features as Fm
from . import recommend as Rm
from . import schema as S
from . import similarity as SIMm
from . import zodiac as Zm


class LovifyModels:
    """Zodiac taste mix + pairwise compatibility + recommendations."""

    # Library sizes at which a null distribution is pre-computed. A pair is
    # scored against the rung nearest its own size -- see fit_null_ladder().
    NULL_SIZES = (25, 50, 75, 100, 150, 200, 300, 500)

    # Below this many tracks the percentile is not reportable.
    #
    # Measured against a single full-size null, a pair is understated by ~41
    # points at 10 tracks, ~19 at 50 and ~8 at 100. The matched ladder below,
    # indexed by pair_size(), cuts the worst case to about 6 points across
    # both symmetric and asymmetric pairs -- which would justify a threshold
    # as low as 50.
    #
    # It stays at 100 deliberately. Those figures come from synthetic users,
    # and the recurring lesson of this project is that the generator flatters
    # whoever wrote it. Lower it once real libraries have been measured.
    MIN_LIBRARY_TRACKS = 100

    # A user's top sign counts as "distinctive" if its lift lands in the top
    # DISTINCTIVE_QUANTILE of the reference population. This used to be a
    # hardcoded 1.25x, which is the kind of number that looks principled and
    # is not: measured on synthetic listeners it passed 100% of them, and a
    # flag that fires for everyone carries exactly as much information as one
    # that never fires.
    #
    # Fitting it to the population means the share of users who see a
    # "distinctive" badge stays roughly fixed no matter how concentrated the
    # population's listening is, which is the property the product wants.
    DISTINCTIVE_QUANTILE = 0.75

    def __init__(self, zodiac_model=None, blend_weights: dict | None = None,
                 n_modes: int = 3, mmr_lambda: float = 1.0,
                 min_library_tracks: int | None = None):
        self.zodiac_model = zodiac_model
        self.blend_weights = blend_weights or dict(SIMm.DEFAULT_WEIGHTS)
        self.n_modes = n_modes
        # 1.0 = pure relevance, i.e. MMR off. Notebook 03 measured MMR as
        # buying nothing once taste modes are on (diversity 0.914 -> 0.935,
        # inside the error bars) while costing recall, so off is the default.
        self.mmr_lambda = mmr_lambda
        self.min_library_tracks = (min_library_tracks if min_library_tracks is not None
                                   else self.MIN_LIBRARY_TRACKS)
        self.scaler = None
        self.catalog: pd.DataFrame | None = None
        self.zodiac_base_rate: pd.Series | None = None
        self.null_ladder: dict[int, np.ndarray] = {}
        # Falls back to the old constant until fit() sees a population.
        self.distinctive_threshold: float = 1.25
        self._rec: Rm.Recommender | None = None

    @property
    def null_distribution(self) -> np.ndarray | None:
        """The full-size null, kept for callers that predate the ladder."""
        return self.null_ladder.get(max(self.null_ladder)) if self.null_ladder else None

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, catalog: pd.DataFrame, zodiac_df: pd.DataFrame,
            listens: pd.DataFrame | None = None, dedup: str = "majority") -> "LovifyModels":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.catalog = catalog.reset_index(drop=True)
        self.scaler = Fm.fit_scaler(self.catalog)

        # Deduplicate before fitting -- see zodiac.py for why this is not optional.
        clean = Zm.deduplicate(zodiac_df, dedup)
        if self.zodiac_model is None:
            # Logistic regression, not the deck's decision tree: on the honest
            # split it scored highest AND its probabilities are calibrated,
            # which the taste mix depends on.
            self.zodiac_model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
        self.zodiac_model.fit(clean[S.AUDIO_FEATURES].to_numpy(float),
                              clean[S.STAR_SIGN].to_numpy())

        # The base rate: what the model says about the catalog at large. Every
        # user's mix is reported relative to this.
        self.zodiac_base_rate = Zm.base_rate(self.zodiac_model, self.catalog)

        self._rec = Rm.Recommender(self.catalog, scale_features=True)

        if listens is not None:
            self.fit_null_ladder(listens)
            self.fit_distinctive_threshold(listens)
        return self

    def fit_distinctive_threshold(self, listens: pd.DataFrame) -> float:
        """Set the "distinctive" bar from the population, not from a guess.

        Computes every reference listener's top-sign lift and takes the
        DISTINCTIVE_QUANTILE of that distribution. On a population where
        everyone is concentrated the bar rises; where everyone is eclectic it
        falls. Either way roughly the top quarter of users get the badge,
        which is what makes it mean something when they do.
        """
        tops = []
        for _, g in listens.groupby(S.USER_ID):
            mix = Zm.taste_mix(self.zodiac_model, g)
            tops.append(float(Zm.lift(mix, self.zodiac_base_rate).iloc[0]))
        if tops:
            self.distinctive_threshold = float(np.quantile(tops, self.DISTINCTIVE_QUANTILE))
        return self.distinctive_threshold

    def fit_null_ladder(self, listens: pd.DataFrame, n_pairs: int = 400,
                        seed: int = 0) -> dict[int, np.ndarray]:
        """Build one null distribution per library size, not a single global one.

        A percentile only means something against a reference population that
        resembles what is being scored. A user with 30 tracks scores lower on
        nearly every component for reasons unrelated to taste -- artist overlap
        collapses toward zero, genre histograms turn sparse -- so comparing
        them against a null built from 150-track libraries puts them near the
        bottom of it no matter who they are paired with. Measured, that is a
        systematic -41 percentile points at 10 tracks and -8 at 100: every
        light listener gets told they are incompatible with everyone.

        Subsampling the reference pairs to the query's own size largely
        removes it. Over 25 pairs, bias against the pair's own full-library
        score:

            tracks      25     50    100    150
            global   -11.1  -11.0   -8.4   -6.2
            matched   +7.7   +2.8   +0.8   +1.0

        It is a reduction, not an elimination -- 25-track libraries are still
        off by several points, which is why MIN_LIBRARY_TRACKS exists on top
        of this. The cost is this loop at fit time, O(len(NULL_SIZES) *
        n_pairs) scorer constructions; a few seconds, paid once.
        """
        rng = np.random.default_rng(seed)
        by_user = {u: g for u, g in listens.groupby(S.USER_ID)}
        users = list(by_user)
        if len(users) < 2:
            raise ValueError("need at least 2 users to build null distributions")

        self.null_ladder = {}
        for size in self.NULL_SIZES:
            scores = []
            for _ in range(n_pairs):
                u, v = rng.choice(users, size=2, replace=False)
                a = by_user[u].sample(n=min(size, len(by_user[u])),
                                      random_state=int(rng.integers(1_000_000_000)))
                b = by_user[v].sample(n=min(size, len(by_user[v])),
                                      random_state=int(rng.integers(1_000_000_000)))
                sc = SIMm.CompatibilityScorer(pd.concat([a, b], ignore_index=True),
                                              self.blend_weights,
                                              scaler=self.scaler, catalog=self.catalog)
                scores.append(sc.raw_score(u, v))
            self.null_ladder[size] = np.sort(np.array(scores))
        return self.null_ladder

    @staticmethod
    def pair_size(n_a: int, n_b: int) -> int:
        """The effective library size of a pair, as a harmonic mean.

        An earlier version used min(n_a, n_b), which is right when both
        libraries are small and wrong when only one is: a 25-vs-300 pair was
        scored against a null built from 25-vs-25 pairs, which share less than
        the real pair does, placing it ~11 percentile points too high.

        Measured over 20 asymmetric pairs, mean bias by choice of statistic:

            small side      25     50    100
            min          +10.0  +12.6   +6.7
            harmonic      -3.5   +6.2   -2.1
            arithmetic   -24.3  -11.2   -5.5

        Harmonic is the conventional way to collapse two unequal sample sizes
        to one effective size, and it degenerates to n for symmetric pairs --
        so this changes nothing for the common case and only repairs the
        asymmetric one. Residual error of a few points remains; removing it
        properly needs a null indexed on (n_a, n_b), at several times the fit
        cost. Worth revisiting if asymmetric pairs turn out to be common --
        "new user meets established user" is exactly that shape.
        """
        if n_a <= 0 or n_b <= 0:
            return 0
        return int(round(2 * n_a * n_b / (n_a + n_b)))

    def _null_for(self, n_tracks: int) -> np.ndarray | None:
        """The rung of the ladder closest to this pair's effective size."""
        if not self.null_ladder:
            return None
        return self.null_ladder[min(self.null_ladder, key=lambda s: abs(s - n_tracks))]

    # ── inference ────────────────────────────────────────────────────────────
    def taste_profile(self, tracks: pd.DataFrame) -> dict:
        """The zodiac feature, reported honestly."""
        mix = Zm.taste_mix(self.zodiac_model, tracks)
        lift = Zm.lift(mix, self.zodiac_base_rate)
        top = lift.index[0]
        return {
            "mix": {k: round(float(v), 4) for k, v in mix.items()},
            "lift": {k: round(float(v), 3) for k, v in lift.items()},
            "headline_sign": top,
            "headline_lift": round(float(lift.iloc[0]), 2),
            # Say so when the signal is thin, rather than dressing noise as insight.
            "is_distinctive": bool(lift.iloc[0] >= self.distinctive_threshold),
            "distinctive_threshold": round(float(self.distinctive_threshold), 3),
        }

    def compatibility(self, tracks_a: pd.DataFrame, tracks_b: pd.DataFrame) -> dict:
        """Score two users against each other, calibrated against random pairs.

        Refuses to return a percentile when either library is too small to
        support one. That refusal is the feature: the alternative is not a
        slightly noisier number, it is a confidently wrong one, biased
        downward by tens of points in a direction the user cannot detect.
        """
        a = tracks_a.copy(); a[S.USER_ID] = "A"
        b = tracks_b.copy(); b[S.USER_ID] = "B"
        n_a, n_b = len(a), len(b)
        smaller = min(n_a, n_b)

        pair = pd.concat([a, b], ignore_index=True)
        scorer = SIMm.CompatibilityScorer(pair, self.blend_weights,
                                          scaler=self.scaler, catalog=self.catalog)

        if smaller < self.min_library_tracks:
            comp = scorer.components("A", "B")
            return {
                "percentile": None,
                "status": "insufficient_library",
                "tracks": {"a": n_a, "b": n_b},
                "min_required": self.min_library_tracks,
                "components": {k: round(v, 4) for k, v in comp.items()},
                "shared_artists": sorted(scorer.artist_sets["A"] & scorer.artist_sets["B"])[:5],
                "interpretation": (
                    f"Not enough listening history to score this fairly yet "
                    f"({smaller} tracks; {self.min_library_tracks} needed). "
                    "Shared artists are still shown below -- those are exact."
                ),
            }

        null = self._null_for(self.pair_size(n_a, n_b))
        if null is None:
            raise RuntimeError("no null distribution fitted; call fit() with listens")
        scorer._null = null

        out = scorer.explain("A", "B")
        out["status"] = "ok"
        out["tracks"] = {"a": n_a, "b": n_b}
        eff = self.pair_size(n_a, n_b)
        out["effective_pair_size"] = eff
        out["null_size_used"] = min(self.null_ladder, key=lambda s: abs(s - eff))
        out["interpretation"] = _phrase(out["percentile"])
        return out

    def recommend(self, tracks: pd.DataFrame, n: int = 10,
                  popularity_penalty: float = 0.0) -> pd.DataFrame:
        return self._rec.recommend(tracks, n=n, modes=self.n_modes,
                                   mmr_lambda=self.mmr_lambda,
                                   popularity_penalty=popularity_penalty)

    def blend_recommend(self, tracks_a: pd.DataFrame, tracks_b: pd.DataFrame,
                        n: int = 10) -> pd.DataFrame:
        """A playlist for the pair: half from each person's modes, deduplicated.

        Drawing from both users' modes rather than from their combined average
        is the same argument as everywhere else here -- the average of two
        people is a third person who does not exist.
        """
        ra = self.recommend(tracks_a, n=n)
        rb = self.recommend(tracks_b, n=n)
        heard = set(tracks_a[S.TRACK_ID]) | set(tracks_b[S.TRACK_ID])
        both = pd.concat([ra, rb], ignore_index=True)
        both = both[~both[S.TRACK_ID].isin(heard)].drop_duplicates(S.TRACK_ID)
        return both.sort_values("score", ascending=False).head(n).reset_index(drop=True)

    # ── persistence ──────────────────────────────────────────────────────────
    # The Recommender holds its own reset-index copy of the catalog, so a naive
    # pickle stores the catalog twice. Drop the whole Recommender on save and
    # rebuild it from self.catalog on load -- it is derived state, and on a
    # real 600k catalog the duplication is the difference between a ~65 MB
    # artifact and a ~200 MB one.

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_rec"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self.catalog is not None:
            self._rec = Rm.Recommender(self.catalog, scale_features=True)

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: str | Path) -> "LovifyModels":
        import joblib

        return joblib.load(path)


def _phrase(pct: float) -> str:
    if pct >= 90:
        return "Unusually aligned -- higher than 9 in 10 random pairs."
    if pct >= 70:
        return "Clearly more alike than most pairs."
    if pct >= 40:
        return "About as alike as two random listeners."
    if pct >= 15:
        return "Less overlap than most pairs -- different corners of music."
    return "Very little common ground, which is its own kind of interesting."
