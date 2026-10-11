"""How alike are two listeners -- and what number do you show them?

Two problems the deck left open.

PROBLEM 1: which metric. The deck used KNN over one mean vector per user.
But for a two-person compatibility score, KNN is the wrong shape of tool
entirely -- there is no neighbourhood to search, just a pair. What you need
is a distance function, and the interesting question is which one actually
tracks real similarity. This module implements six and scores them against
ground truth.

PROBLEM 2: calibration. A raw cosine of 0.92 is meaningless to a user --
cosines between any two music listeners run high, because all humans listen
to human music. Shown as "92% compatible", every pair looks like soulmates
and the number stops carrying information. The fix is a null distribution:
score thousands of RANDOM pairs, then report where this pair falls against
them. "More compatible than 87% of random pairs" is a claim that can be
wrong, which is what makes it worth showing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from . import schema as S


# ── Pairwise metrics ──────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def profile_cosine(profiles: pd.DataFrame, u: str, v: str) -> float:
    """Cosine between two fixed-length profile vectors (mean / mean+std / quantile)."""
    return cosine(profiles.loc[u].to_numpy(float), profiles.loc[v].to_numpy(float))


def profile_euclid(profiles: pd.DataFrame, u: str, v: str) -> float:
    """Negated L2, so that bigger is always more similar."""
    return -float(np.linalg.norm(profiles.loc[u].to_numpy(float) - profiles.loc[v].to_numpy(float)))


def genre_overlap(genre_profiles: pd.DataFrame, u: str, v: str) -> float:
    """Bhattacharyya coefficient of two genre distributions, in [0, 1].

    sum(sqrt(p*q)) -- 1.0 when the two listen to the identical genre mix, 0
    when they share no genre at all. Unlike cosine it is a real overlap of
    distributions, so the value is interpretable on its own.
    """
    p = genre_profiles.loc[u].to_numpy(float)
    q = genre_profiles.loc[v].to_numpy(float)
    return float(np.sqrt(p * q).sum())


def artist_jaccard(artist_sets: dict[str, set], u: str, v: str) -> float:
    """|A∩B| / |A∪B| over artists. The most legible signal there is: you can
    show the user the intersection, and they will immediately agree or not."""
    a, b = artist_sets[u], artist_sets[v]
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def feature_wasserstein(listens_by_user: dict[str, pd.DataFrame], u: str, v: str,
                        features: list[str] | None = None) -> float:
    """Mean earth-mover distance per feature, negated.

    This is the metric that respects SPREAD. Two users whose energy
    histograms are both centred at 0.5 -- one tightly, one bimodally at the
    extremes -- have identical means and a large Wasserstein distance. The
    mean-vector approach calls them identical; this calls them different, and
    it is right.
    """
    features = features or S.AUDIO_FEATURES
    A, B = listens_by_user[u], listens_by_user[v]
    ds = [wasserstein_distance(A[f].to_numpy(float), B[f].to_numpy(float)) for f in features]
    return -float(np.mean(ds))


# NOTE: pass SCALED frames to feature_wasserstein. On raw columns the mean is
# dominated by duration_ms (range ~500,000) and the result is meaningless --
# CompatibilityScorer uses self.by_user_scaled for exactly this reason.


def era_overlap(listens_by_user: dict[str, pd.DataFrame], u: str, v: str) -> float:
    """Bhattacharyya overlap of release-decade distributions.

    Cheap, and it separates pairs that audio features cannot: two people who
    both like loud guitars, one exclusively from 1974 and one from 2019.
    """
    def decades(df):
        d = (df[S.YEAR].to_numpy(float) // 10 * 10).astype(int)
        vc = pd.Series(d).value_counts(normalize=True)
        return vc
    p, q = decades(listens_by_user[u]), decades(listens_by_user[v])
    idx = p.index.union(q.index)
    p, q = p.reindex(idx, fill_value=0.0).to_numpy(), q.reindex(idx, fill_value=0.0).to_numpy()
    return float(np.sqrt(p * q).sum())


def obscurity_match(listens_by_user: dict[str, pd.DataFrame], u: str, v: str) -> float:
    """1 - |difference in median popularity|, normalised.

    Two people who both live off deep cuts have something in common that no
    audio feature captures -- and a crate-digger paired with a top-40 listener
    will not enjoy each other's playlists however well their valence matches.
    """
    pu = listens_by_user[u][S.POPULARITY].median()
    pv = listens_by_user[v][S.POPULARITY].median()
    return 1.0 - abs(float(pu) - float(pv)) / 100.0


# ── The blended score ─────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "genre_overlap": 0.30,
    "artist_jaccard": 0.25,
    "feature_wasserstein": 0.25,
    "era_overlap": 0.10,
    "obscurity_match": 0.10,
}


class CompatibilityScorer:
    """Blends the signals above into one score, then calibrates it.

    Why a blend rather than one metric: the signals fail in different places.
    Artist overlap is sharp but sparse -- two people with no shared artist get
    0.0 even if they listen to the same genres. Genre overlap is dense but
    coarse. Audio distance works when both libraries are large. Blending keeps
    a usable score when any one of them is uninformative.
    """

    def __init__(self, listens: pd.DataFrame, weights: dict[str, float] | None = None,
                 features: list[str] | None = None, scaler=None, catalog: pd.DataFrame | None = None):
        from . import features as Fm

        self.features = features or S.AUDIO_FEATURES
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        # Fit the scaler on the catalog when we have it, else on the pooled
        # listens. Either way the audio distances below MUST run on scaled
        # columns: raw duration_ms spans ~500,000 and raw valence spans 1, so
        # an unscaled distance is a measurement of song length and nothing else.
        self.scaler = scaler or Fm.fit_scaler(catalog if catalog is not None else listens,
                                              self.features)
        self.by_user = {u: g for u, g in listens.groupby(S.USER_ID)}
        self.by_user_scaled = {
            u: pd.DataFrame(Fm.scale(g, self.scaler, self.features), columns=self.features)
            for u, g in self.by_user.items()
        }
        self.users = list(self.by_user)
        self.genre_profiles = Fm.genre_profile(listens)
        self.artist_sets = Fm.artist_sets(listens)
        self._null: np.ndarray | None = None

    def components(self, u: str, v: str) -> dict[str, float]:
        return {
            "genre_overlap": genre_overlap(self.genre_profiles, u, v),
            "artist_jaccard": artist_jaccard(self.artist_sets, u, v),
            # Wasserstein is negative and unbounded; squash to (0,1] so it can
            # share a weighted sum with the overlap measures.
            "feature_wasserstein": float(np.exp(feature_wasserstein(self.by_user_scaled, u, v, self.features))),
            "era_overlap": era_overlap(self.by_user, u, v),
            "obscurity_match": obscurity_match(self.by_user, u, v),
        }

    def raw_score(self, u: str, v: str) -> float:
        c = self.components(u, v)
        return float(sum(self.weights[k] * c[k] for k in self.weights))

    def fit_null(self, n_pairs: int = 4000, seed: int = 0) -> np.ndarray:
        """Score random pairs to learn what 'unremarkable' looks like."""
        rng = np.random.default_rng(seed)
        if len(self.users) < 2:
            raise ValueError("need at least 2 users to build a null distribution")
        scores = []
        for _ in range(n_pairs):
            u, v = rng.choice(self.users, size=2, replace=False)
            scores.append(self.raw_score(u, v))
        self._null = np.sort(np.array(scores))
        return self._null

    def percentile(self, u: str, v: str) -> float:
        """The shippable number: 0-100, where 50 means 'a typical pair'."""
        if self._null is None:
            self.fit_null()
        return float(np.searchsorted(self._null, self.raw_score(u, v)) / len(self._null) * 100)

    def explain(self, u: str, v: str, top_artists: int = 5) -> dict:
        """Everything the UI needs: the score, why, and proof."""
        c = self.components(u, v)
        shared = sorted(self.artist_sets[u] & self.artist_sets[v])
        gu, gv = self.genre_profiles.loc[u], self.genre_profiles.loc[v]
        shared_genres = (np.sqrt(gu * gv)).sort_values(ascending=False)
        return {
            "percentile": round(self.percentile(u, v), 1),
            "raw_score": round(self.raw_score(u, v), 4),
            "components": {k: round(v_, 4) for k, v_ in c.items()},
            "shared_artists": shared[:top_artists],
            "n_shared_artists": len(shared),
            "top_shared_genres": list(shared_genres.head(3).index),
        }


# ── Evaluating the metrics ────────────────────────────────────────────────────

def score_matrix(fn, users: list[str]) -> pd.DataFrame:
    n = len(users)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = fn(users[i], users[j])
    np.fill_diagonal(M, np.nan)
    return pd.DataFrame(M, index=users, columns=users)


def rank_agreement(pred: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    """Spearman correlation between a metric's pair rankings and the truth.

    This is the number the deck never had. 'Results seem to be quite well' is
    not a result; a Spearman rho against known-similar users is.
    """
    from scipy.stats import spearmanr

    users = [u for u in pred.index if u in truth.index]
    P = pred.loc[users, users].to_numpy()
    T = truth.loc[users, users].to_numpy()
    iu = np.triu_indices(len(users), k=1)
    rho, p = spearmanr(P[iu], T[iu])
    return {"spearman": float(rho), "p_value": float(p), "n_pairs": int(len(iu[0]))}


def top1_accuracy(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
    """How often the metric's #1 match is also the true #1 match."""
    users = [u for u in pred.index if u in truth.index]
    hits = 0
    for u in users:
        p = pred.loc[u, users].drop(u)
        t = truth.loc[u, users].drop(u)
        hits += int(p.idxmax() == t.idxmax())
    return hits / len(users)


def tune_weights(scorer: "CompatibilityScorer", truth: pd.DataFrame,
                 users: list[str] | None = None, n_trials: int = 400,
                 seed: int = 0) -> tuple[dict[str, float], float, pd.DataFrame]:
    """Search blend weights that maximise rank agreement with the truth.

    Hand-picked weights are a guess. This is a random search over the simplex
    (Dirichlet draws), scoring each weight vector by Spearman rho. Cheap
    because the expensive part -- the per-pair component values -- is computed
    once and reused for every trial.

    Returns (best_weights, best_rho, trial_log). With real users there is no
    truth matrix; pass a proxy built by similarity_proxy_from_holdout().
    """
    from scipy.stats import spearmanr

    users = users or [u for u in scorer.users if u in truth.index]
    keys = list(scorer.weights)
    pairs = [(i, j) for i in range(len(users)) for j in range(i + 1, len(users))]

    # Compute each component once per pair: an n_pairs x n_components matrix.
    C = np.array([[scorer.components(users[i], users[j])[k] for k in keys] for i, j in pairs])
    T = np.array([truth.loc[users[i], users[j]] for i, j in pairs])

    rng = np.random.default_rng(seed)
    log = []
    best_w, best_rho = None, -np.inf
    for _ in range(n_trials):
        w = rng.dirichlet(np.ones(len(keys)))
        rho = spearmanr(C @ w, T).statistic
        log.append({**dict(zip(keys, w.round(4))), "spearman": float(rho)})
        if rho > best_rho:
            best_rho, best_w = float(rho), dict(zip(keys, w))
    return best_w, best_rho, pd.DataFrame(log).sort_values("spearman", ascending=False)


def similarity_proxy_from_holdout(listens: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ground truth for REAL users, who come with no taste matrix.

    Split every user's library in half and treat the halves as two people.
    A good metric must rank a user's own other half above every stranger --
    that is a verifiable claim, and it needs no labels at all.

    Returns (listens_with_split_ids, truth) where truth[u_a, u_b] = 1 iff the
    two halves came from the same person.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for u, g in listens.groupby(S.USER_ID):
        idx = rng.permutation(len(g))
        half = len(g) // 2
        a, b = g.iloc[idx[:half]].copy(), g.iloc[idx[half:]].copy()
        a[S.USER_ID] = f"{u}_a"
        b[S.USER_ID] = f"{u}_b"
        parts += [a, b]
    split = pd.concat(parts, ignore_index=True)

    ids = sorted(split[S.USER_ID].unique())
    base = pd.Series([i.rsplit("_", 1)[0] for i in ids], index=ids)
    T = (base.to_numpy()[:, None] == base.to_numpy()[None, :]).astype(float)
    np.fill_diagonal(T, np.nan)
    truth = pd.DataFrame(T, index=ids, columns=ids)
    return split, truth


def self_match_rate(pred: pd.DataFrame) -> float:
    """Fraction of half-libraries whose nearest neighbour is their own other half.

    The headline number for the held-out-half protocol. Chance is ~1/(n-1).
    """
    users = list(pred.index)
    hits = 0
    for u in users:
        row = pred.loc[u].drop(u)
        mate = u.rsplit("_", 1)[0] + ("_b" if u.endswith("_a") else "_a")
        if mate in row.index and row.idxmax() == mate:
            hits += 1
    return hits / len(users)
