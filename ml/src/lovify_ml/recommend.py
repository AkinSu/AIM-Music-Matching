"""Song recommendation -- and the three ways the original version misleads.

The deck's recommender: average a user's ~150 songs into one summary vector,
cosine it against 600k catalog songs, return the top 10. Three failures are
baked into that, and each has a cheap fix:

  1. NO SCALING. Cosine over raw columns is dominated by duration_ms (range
     ~500,000) and tempo (~200); danceability (~1) contributes nothing. The
     "most similar songs" are mostly songs of a similar LENGTH.

  2. THE MEAN IS NOWHERE. The average of your ambient and your metal is
     mid-tempo rock -- a point in feature space where you have never actually
     listened to anything. Recommending its neighbours recommends music
     nobody asked for. Fix: cluster the user's own library into taste modes
     and recommend around each.

  3. NEAREST IS NOT BEST. The top 10 by cosine are 10 near-identical songs,
     and typically the 10 most generic songs in the catalog, since the dense
     middle of feature space is where the mass is. Fix: MMR for diversity,
     and a popularity penalty.

Everything here is measured by leave-out-the-tail: hide part of a user's
library, recommend from the rest, and check whether the hidden songs come
back. That is the metric the deck did not have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as Fm
from . import schema as S


def _unit(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=-1, keepdims=True)
    return M / np.where(n == 0, 1, n)


class Recommender:
    """Cosine retrieval over a catalog, with the fixes switchable on and off.

    Keeping the broken behaviour available as a flag is deliberate: the
    notebook turns each fix on one at a time and measures what it bought.
    """

    def __init__(self, catalog: pd.DataFrame, features: list[str] | None = None,
                 scale_features: bool = True):
        self.catalog = catalog.reset_index(drop=True)
        self.features = features or S.AUDIO_FEATURES
        self.scale_features = scale_features
        if scale_features:
            self.scaler = Fm.fit_scaler(self.catalog, self.features)
            X = Fm.scale(self.catalog, self.scaler, self.features)
        else:
            self.scaler = None  # the deck's behaviour: raw columns
            X = self.catalog[self.features].to_numpy(float)
        # Only the unit-normalised matrix is ever used -- cosine cares about
        # direction alone. Keeping the unnormalised X as well cost ~0.6 MB on
        # an 8k catalog and would be ~45 MB on the real 600k one, for nothing.
        self.Xn = _unit(X)
        self.pop = self.catalog[S.POPULARITY].to_numpy(float)

    # ── pickling ─────────────────────────────────────────────────────────────
    # Xn and pop are pure functions of the catalog, so persisting them triples
    # the artifact for no information. Rebuild on load instead: cheap, and it
    # keeps the saved model from drifting out of sync with its own catalog.

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for derived in ("Xn", "pop"):
            state.pop(derived, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self.scale_features:
            X = Fm.scale(self.catalog, self.scaler, self.features)
        else:
            X = self.catalog[self.features].to_numpy(float)
        self.Xn = _unit(X)
        self.pop = self.catalog[S.POPULARITY].to_numpy(float)

    # ── profiles ─────────────────────────────────────────────────────────────
    def _embed(self, tracks: pd.DataFrame) -> np.ndarray:
        if self.scale_features:
            return Fm.scale(tracks, self.scaler, self.features)
        return tracks[self.features].to_numpy(float)

    def mean_vector(self, tracks: pd.DataFrame) -> np.ndarray:
        return self._embed(tracks).mean(axis=0)

    def mode_vectors(self, tracks: pd.DataFrame, k: int = 3, seed: int = 0) -> np.ndarray:
        """k centroids of the user's OWN library -- their distinct listening modes."""
        from sklearn.cluster import KMeans

        Z = self._embed(tracks)
        kk = int(min(k, max(1, len(Z) // 10), len(Z)))
        if kk <= 1:
            return Z.mean(axis=0, keepdims=True)
        return KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(Z).cluster_centers_

    # ── retrieval ────────────────────────────────────────────────────────────
    def _scores(self, query: np.ndarray) -> np.ndarray:
        return self.Xn @ _unit(query)

    def recommend(self, tracks: pd.DataFrame, n: int = 10, *,
                  modes: int = 1, exclude_heard: bool = True,
                  popularity_penalty: float = 0.0, mmr_lambda: float = 1.0,
                  candidate_pool: int = 400, seed: int = 0) -> pd.DataFrame:
        """
        modes=1               -- the deck's single mean vector
        modes>1               -- split the user's taste and serve each mode
        exclude_heard         -- never recommend something already in the library
        popularity_penalty    -- subtract p * normalised_popularity; >0 digs deeper
        mmr_lambda            -- 1.0 pure relevance, lower trades it for variety
        """
        queries = (self.mean_vector(tracks)[None, :] if modes <= 1
                   else self.mode_vectors(tracks, k=modes, seed=seed))

        heard = set(tracks[S.TRACK_ID]) if exclude_heard else set()
        mask = ~self.catalog[S.TRACK_ID].isin(heard).to_numpy()

        per_mode = max(1, int(np.ceil(n / len(queries))))
        picks: list[int] = []
        for q in queries:
            s = self._scores(q)
            if popularity_penalty:
                s = s - popularity_penalty * (self.pop / 100.0)
            s = np.where(mask, s, -np.inf)
            for i in picks:
                s[i] = -np.inf
            pool = np.argsort(s)[::-1][:candidate_pool]
            pool = pool[np.isfinite(s[pool])]
            chosen = (list(pool[:per_mode]) if mmr_lambda >= 1.0
                      else self._mmr(pool, s, per_mode, mmr_lambda))
            picks.extend(chosen)

        picks = list(dict.fromkeys(picks))[:n]
        out = self.catalog.iloc[picks].copy()
        out["score"] = [float(self._scores(queries.mean(0))[i]) for i in picks]
        return out.reset_index(drop=True)

    def _mmr(self, pool: np.ndarray, scores: np.ndarray, n: int, lam: float) -> list[int]:
        """Maximal Marginal Relevance: each pick trades relevance against
        how close it already is to what has been picked. This is the
        difference between ten songs and ten copies of one song."""
        selected: list[int] = []
        P = self.Xn[pool]
        while len(selected) < min(n, len(pool)):
            if not selected:
                best = int(np.argmax(scores[pool]))
            else:
                sim_to_sel = (P @ self.Xn[selected].T).max(axis=1)
                mmr = lam * scores[pool] - (1 - lam) * sim_to_sel
                mmr[[list(pool).index(i) for i in selected if i in pool]] = -np.inf
                best = int(np.argmax(mmr))
            selected.append(int(pool[best]))
        return selected


# ── Metrics ───────────────────────────────────────────────────────────────────

def intra_list_diversity(recs: pd.DataFrame, rec_obj: Recommender) -> float:
    """Mean pairwise cosine DISTANCE within one recommendation list.

    Near 0 means you returned ten versions of the same song."""
    idx = rec_obj.catalog.index[rec_obj.catalog[S.TRACK_ID].isin(recs[S.TRACK_ID])]
    V = rec_obj.Xn[idx]
    if len(V) < 2:
        return 0.0
    Smat = V @ V.T
    iu = np.triu_indices(len(V), k=1)
    return float(1 - Smat[iu].mean())


def holdout_evaluation(rec: Recommender, listens: pd.DataFrame, n: int = 20,
                       holdout_frac: float = 0.25, seed: int = 0, **kw) -> dict:
    """Hide part of each user's library, recommend from the rest, count hits.

    recall@n  -- fraction of a user's hidden tracks that were recommended
    hit_rate  -- fraction of users who got at least one hidden track back
    These are small numbers by nature (n out of a whole catalog), so read them
    against the random baseline reported alongside, not against 100%.
    """
    rng = np.random.default_rng(seed)
    recalls, hits, divs, pops = [], [], [], []

    for u, g in listens.groupby(S.USER_ID):
        if len(g) < 20:
            continue
        idx = rng.permutation(len(g))
        cut = int(len(g) * (1 - holdout_frac))
        seen, hidden = g.iloc[idx[:cut]], g.iloc[idx[cut:]]
        hidden_ids = set(hidden[S.TRACK_ID])

        recs = rec.recommend(seen, n=n, **kw)
        got = len(set(recs[S.TRACK_ID]) & hidden_ids)
        recalls.append(got / max(1, len(hidden_ids)))
        hits.append(int(got > 0))
        divs.append(intra_list_diversity(recs, rec))
        pops.append(float(recs[S.POPULARITY].mean()))

    n_users = len(recalls)
    random_recall = n / max(1, len(rec.catalog))
    return {
        "users_evaluated": n_users,
        f"recall@{n}": float(np.mean(recalls)) if n_users else 0.0,
        f"hit_rate@{n}": float(np.mean(hits)) if n_users else 0.0,
        "lift_vs_random": (float(np.mean(recalls)) / random_recall) if n_users else 0.0,
        "intra_list_diversity": float(np.mean(divs)) if n_users else 0.0,
        "mean_rec_popularity": float(np.mean(pops)) if n_users else 0.0,
        "catalog_mean_popularity": float(rec.catalog[S.POPULARITY].mean()),
    }
