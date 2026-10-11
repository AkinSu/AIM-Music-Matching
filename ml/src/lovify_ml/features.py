"""Turning a pile of songs into something you can measure distance between.

The original design collapsed each user to ONE mean audio-feature vector.
That is the single most damaging choice in the pipeline: it throws away the
spread of someone's taste, which is most of what taste *is*. Someone who
splits their listening between ambient and metal averages out to "mid-tempo
rock", matches strangers who actually listen to mid-tempo rock, and gets
recommended mid-tempo rock they have never wanted.

This module keeps the mean profile (as the baseline to beat) and adds three
alternatives that survive that collapse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import schema as S


# ── Scaling ───────────────────────────────────────────────────────────────────

def fit_scaler(df: pd.DataFrame, features: list[str] | None = None) -> StandardScaler:
    """Fit a standardiser on the CATALOG, not on any one user.

    Everything -- user profiles, candidate songs, zodiac tracks -- must be put
    into the same space by this one scaler. Without it, tempo (0-200) and
    duration_ms (0-600,000) swamp danceability (0-1) and every cosine
    similarity silently becomes 'songs of about the same length'.
    """
    features = features or S.AUDIO_FEATURES
    return StandardScaler().fit(df[features].to_numpy(float))


def scale(df: pd.DataFrame, scaler: StandardScaler, features: list[str] | None = None) -> np.ndarray:
    features = features or S.AUDIO_FEATURES
    return scaler.transform(df[features].to_numpy(float))


def encode_key(key: pd.Series) -> pd.DataFrame:
    """Pitch class is cyclic: key 11 (B) is adjacent to key 0 (C).

    Feeding it as the integer 11 tells a linear model that B is eleven times
    something that C is. Sin/cos puts it on a circle where the adjacency is true.
    """
    theta = 2 * np.pi * key.to_numpy(float) / 12.0
    return pd.DataFrame({"key_sin": np.sin(theta), "key_cos": np.cos(theta)}, index=key.index)


# ── User profiles ─────────────────────────────────────────────────────────────

def mean_profile(listens: pd.DataFrame, scaler: StandardScaler,
                 features: list[str] | None = None) -> pd.DataFrame:
    """BASELINE (the original approach): one averaged vector per user."""
    features = features or S.AUDIO_FEATURES
    Z = pd.DataFrame(scale(listens, scaler, features), columns=features, index=listens.index)
    Z[S.USER_ID] = listens[S.USER_ID].to_numpy()
    return Z.groupby(S.USER_ID).mean()


def quantile_profile(listens: pd.DataFrame, scaler: StandardScaler,
                     quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
                     features: list[str] | None = None) -> pd.DataFrame:
    """Per feature, the shape of the user's distribution rather than its centre.

    The 10th/90th percentile gap is exactly the 'how wide is your taste'
    information the mean destroys. Cheap: still one fixed-length vector per
    user, so every downstream distance still works unchanged.
    """
    features = features or S.AUDIO_FEATURES
    Z = pd.DataFrame(scale(listens, scaler, features), columns=features, index=listens.index)
    Z[S.USER_ID] = listens[S.USER_ID].to_numpy()
    out = Z.groupby(S.USER_ID).quantile(list(quantiles))
    out = out.unstack(level=-1)
    out.columns = [f"{f}_q{int(q * 100)}" for f, q in out.columns]
    return out


def dispersion_profile(listens: pd.DataFrame, scaler: StandardScaler,
                       features: list[str] | None = None) -> pd.DataFrame:
    """Mean AND standard deviation -- the cheapest fix to the collapse problem."""
    features = features or S.AUDIO_FEATURES
    Z = pd.DataFrame(scale(listens, scaler, features), columns=features, index=listens.index)
    Z[S.USER_ID] = listens[S.USER_ID].to_numpy()
    g = Z.groupby(S.USER_ID)
    m, s = g.mean(), g.std().fillna(0.0)
    m.columns = [f"{c}_mean" for c in m.columns]
    s.columns = [f"{c}_std" for c in s.columns]
    return m.join(s)


def genre_profile(listens: pd.DataFrame, genres: list[str] | None = None) -> pd.DataFrame:
    """Normalised genre histogram per user -- a taste distribution, not a point."""
    tab = pd.crosstab(listens[S.USER_ID], listens[S.GENRE]).astype(float)
    if genres is not None:
        tab = tab.reindex(columns=genres, fill_value=0.0)
    return tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def artist_sets(listens: pd.DataFrame) -> dict[str, set]:
    """Who each user actually listens to. The single most legible match signal."""
    return {u: set(g) for u, g in listens.groupby(S.USER_ID)[S.ARTIST]}


def taste_modes(listens: pd.DataFrame, scaler: StandardScaler, k: int = 3,
                features: list[str] | None = None, seed: int = 0) -> dict[str, np.ndarray]:
    """Cluster each user's own library into k 'modes' (study music, gym music...).

    Returns {user_id: (k, n_features) centroids}. This is the profile that
    keeps a bimodal listener bimodal, and it is what the recommender should
    query -- once per mode -- instead of once from the average.
    """
    from sklearn.cluster import KMeans

    features = features or S.AUDIO_FEATURES
    out = {}
    for u, g in listens.groupby(S.USER_ID):
        Z = scale(g, scaler, features)
        kk = int(min(k, max(1, len(Z) // 10), len(Z)))
        if kk <= 1:
            out[u] = Z.mean(0, keepdims=True)
            continue
        km = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(Z)
        out[u] = km.cluster_centers_
    return out


PROFILE_BUILDERS = {
    "mean": mean_profile,
    "mean+std": dispersion_profile,
    "quantile": quantile_profile,
}
