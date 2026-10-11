"""Synthetic data that behaves like the real thing.

Why this exists: the notebooks must run today, with no Spotify credentials and
no 600k-row Kaggle download. But a toy dataset of pure noise would make every
notebook lie -- models would look uniformly useless and we'd learn nothing.

So this generator reproduces the *structural* properties that actually matter
for the Lovify models:

  1. Genre-conditioned audio features, so genre is genuinely learnable.
  2. Zodiac playlists built by simulated curators who (a) have only a weak,
     noisy notion of what an "Aries song" is, and (b) all gravitate to popular
     tracks -- which makes the SAME TRACK appear under several signs. That
     duplication is the exact mechanism behind the original 76% vs 14% result.
  3. Users as mixtures over genres, each with a known latent taste vector --
     which gives us ground truth to score similarity metrics against.

Swap in real data via data.load_catalog(); every downstream function is
indifferent to which source it got.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S

# ── Genre feature profiles ────────────────────────────────────────────────────
# (mean, std) per feature. Hand-set so genres are separable but overlapping,
# the way real audio features are.
GENRE_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "hip hop":    dict(danceability=(0.78, .10), energy=(0.66, .13), speechiness=(0.24, .12),
                       acousticness=(0.14, .13), instrumentalness=(0.01, .04), liveness=(0.17, .11),
                       valence=(0.53, .20), tempo=(93, 18), loudness=(-6.5, 2.2), duration_ms=(201_000, 48_000)),
    "rnb":        dict(danceability=(0.66, .12), energy=(0.52, .14), speechiness=(0.10, .07),
                       acousticness=(0.27, .19), instrumentalness=(0.02, .07), liveness=(0.14, .09),
                       valence=(0.47, .19), tempo=(98, 20), loudness=(-8.0, 2.4), duration_ms=(222_000, 55_000)),
    "pop":        dict(danceability=(0.68, .11), energy=(0.70, .13), speechiness=(0.07, .05),
                       acousticness=(0.18, .16), instrumentalness=(0.00, .02), liveness=(0.16, .11),
                       valence=(0.60, .20), tempo=(118, 22), loudness=(-5.5, 2.0), duration_ms=(199_000, 40_000)),
    "indie":      dict(danceability=(0.55, .13), energy=(0.57, .16), speechiness=(0.05, .04),
                       acousticness=(0.33, .24), instrumentalness=(0.08, .17), liveness=(0.19, .13),
                       valence=(0.44, .21), tempo=(116, 25), loudness=(-8.5, 2.8), duration_ms=(232_000, 62_000)),
    "rock":       dict(danceability=(0.47, .13), energy=(0.79, .13), speechiness=(0.06, .05),
                       acousticness=(0.10, .13), instrumentalness=(0.05, .14), liveness=(0.24, .17),
                       valence=(0.52, .22), tempo=(126, 26), loudness=(-6.0, 2.5), duration_ms=(247_000, 72_000)),
    "metal":      dict(danceability=(0.38, .11), energy=(0.93, .06), speechiness=(0.09, .07),
                       acousticness=(0.02, .05), instrumentalness=(0.10, .21), liveness=(0.25, .18),
                       valence=(0.33, .19), tempo=(138, 28), loudness=(-4.2, 1.8), duration_ms=(272_000, 90_000)),
    "edm":        dict(danceability=(0.70, .10), energy=(0.86, .09), speechiness=(0.06, .05),
                       acousticness=(0.05, .08), instrumentalness=(0.33, .31), liveness=(0.19, .13),
                       valence=(0.47, .21), tempo=(126, 8), loudness=(-5.0, 2.0), duration_ms=(221_000, 58_000)),
    "ambient":    dict(danceability=(0.30, .12), energy=(0.18, .13), speechiness=(0.04, .03),
                       acousticness=(0.74, .22), instrumentalness=(0.82, .22), liveness=(0.11, .07),
                       valence=(0.22, .16), tempo=(92, 30), loudness=(-19.0, 5.0), duration_ms=(298_000, 140_000)),
    "jazz":       dict(danceability=(0.53, .14), energy=(0.37, .17), speechiness=(0.06, .05),
                       acousticness=(0.61, .24), instrumentalness=(0.47, .35), liveness=(0.20, .14),
                       valence=(0.51, .22), tempo=(111, 32), loudness=(-13.0, 4.2), duration_ms=(305_000, 130_000)),
    "country":    dict(danceability=(0.57, .11), energy=(0.63, .14), speechiness=(0.04, .03),
                       acousticness=(0.29, .21), instrumentalness=(0.01, .04), liveness=(0.18, .12),
                       valence=(0.59, .20), tempo=(123, 22), loudness=(-6.8, 2.2), duration_ms=(214_000, 42_000)),
    "classical":  dict(danceability=(0.26, .11), energy=(0.17, .14), speechiness=(0.04, .03),
                       acousticness=(0.90, .13), instrumentalness=(0.86, .19), liveness=(0.14, .11),
                       valence=(0.27, .19), tempo=(108, 35), loudness=(-21.0, 5.5), duration_ms=(355_000, 210_000)),
    "latin":      dict(danceability=(0.76, .09), energy=(0.78, .11), speechiness=(0.09, .07),
                       acousticness=(0.21, .17), instrumentalness=(0.00, .02), liveness=(0.18, .12),
                       valence=(0.74, .17), tempo=(108, 20), loudness=(-4.8, 1.9), duration_ms=(208_000, 40_000)),
}

GENRES = list(GENRE_PROFILES)


def _clip(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in S.feature_bounds().items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def make_catalog(n_tracks: int = 6000, seed: int = 7) -> pd.DataFrame:
    """A song catalog standing in for the 600k-row Kaggle dump."""
    rng = np.random.default_rng(seed)

    # Zipf-ish genre popularity, so the catalog isn't uniformly balanced.
    genre_w = rng.dirichlet(np.full(len(GENRES), 2.0))
    genres = rng.choice(GENRES, size=n_tracks, p=genre_w)

    rows: dict[str, np.ndarray] = {}
    for feat in S.AUDIO_FEATURES:
        mu = np.array([GENRE_PROFILES[g][feat][0] for g in genres])
        sd = np.array([GENRE_PROFILES[g][feat][1] for g in genres])
        rows[feat] = rng.normal(mu, sd)

    df = pd.DataFrame(rows)
    df[S.GENRE] = genres
    df[S.TRACK_ID] = [f"T{i:06d}" for i in range(n_tracks)]
    df[S.TRACK_NAME] = [f"{_word(rng)} {_word(rng)}".title() for _ in range(n_tracks)]
    # A pool of artists much smaller than the track count, so artists recur --
    # which is what makes artist-overlap a usable similarity signal.
    n_artists = max(40, n_tracks // 12)
    artist_pool = np.array([f"{_word(rng).title()} {_word(rng).title()}" for _ in range(n_artists)])
    artist_genre = rng.choice(GENRES, size=n_artists)
    # Artists mostly stay in their genre.
    art_idx = np.empty(n_tracks, dtype=int)
    for g in GENRES:
        mask = genres == g
        same = np.flatnonzero(artist_genre == g)
        pool = same if len(same) else np.arange(n_artists)
        art_idx[mask] = rng.choice(pool, size=int(mask.sum()))
    df[S.ARTIST] = artist_pool[art_idx]
    df[S.ALBUM] = [f"{_word(rng).title()}" for _ in range(n_tracks)]

    # Popularity: heavily right-skewed, as in the real catalogue.
    df[S.POPULARITY] = np.clip(rng.beta(1.6, 5.0, n_tracks) * 100, 0, 100).round().astype(int)
    df[S.YEAR] = np.clip(rng.normal(2012, 11, n_tracks), 1955, 2024).round().astype(int)

    df["key"] = rng.integers(0, 12, n_tracks)
    df["mode"] = rng.integers(0, 2, n_tracks)
    df["time_signature"] = rng.choice([3, 4, 5], n_tracks, p=[.10, .86, .04])

    df = _clip(df)
    return df[S.META_COLUMNS + S.AUDIO_FEATURES + ["key", "mode", "time_signature"]]


_SYLL = ["lo", "vi", "fy", "mar", "ta", "ven", "sol", "nox", "ria", "ken",
         "dro", "mel", "aza", "rin", "quo", "sev", "ila", "tor", "nym", "vex"]


def _word(rng: np.random.Generator) -> str:
    return "".join(rng.choice(_SYLL, size=rng.integers(2, 4)))


# ── Zodiac playlists ──────────────────────────────────────────────────────────

def make_zodiac_playlists(
    catalog: pd.DataFrame,
    n_playlists_per_sign: int = 10,
    tracks_per_playlist: int = 50,
    pool_size: int = 220,
    signal_strength: float = 0.45,
    popularity_pull: float = 1.6,
    seed: int = 11,
) -> pd.DataFrame:
    """Simulate the scrape: top-N playlists per sign, labelled by the curator.

    Two layers, because both matter:

    1. Each sign gets a random weak direction in standardised feature space.
       A *candidate pool* of `pool_size` tracks is drawn for that sign with
       probability proportional to

           exp(signal_strength * <z_track, direction_sign> + popularity_pull * pop)

       `signal_strength` is how much real signal exists -- set it to 0 for a
       pure-noise control. `popularity_pull` makes every sign reach for the
       same hits, so pools overlap and one track picks up contradictory
       labels across signs. That is the LABEL NOISE.

    2. All `n_playlists_per_sign` playlists for a sign then draw from that one
       small pool. Ten "Aries vibes" playlists therefore share most of their
       tracks, exactly as real ones do -- curators copy each other. That is
       the WITHIN-SIGN DUPLICATION, and it is the leakage engine: split those
       rows at random and the same song, with the same label, lands in both
       train and test.
    """
    rng = np.random.default_rng(seed)
    feats = S.AUDIO_FEATURES
    X = catalog[feats].to_numpy(float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)

    directions = rng.normal(size=(S.N_SIGNS, len(feats)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    pop = catalog[S.POPULARITY].to_numpy(float) / 100.0
    pool_size = min(pool_size, len(catalog))

    frames = []
    for si, sign in enumerate(S.ZODIAC_SIGNS):
        logits = signal_strength * (Z @ directions[si]) + popularity_pull * pop
        p = np.exp(logits - logits.max())
        p /= p.sum()
        pool = rng.choice(len(catalog), size=pool_size, replace=False, p=p)

        # Within the pool, popular tracks recur across that sign's playlists.
        w = pop[pool] + 0.25
        w /= w.sum()
        for pl in range(n_playlists_per_sign):
            take = rng.choice(pool, size=min(tracks_per_playlist, pool_size),
                              replace=False, p=w)
            sub = catalog.iloc[take].copy()
            sub[S.STAR_SIGN] = sign
            sub[S.PLAYLIST_NAME] = f"{sign} vibes #{pl + 1}"
            frames.append(sub)

    out = pd.concat(frames, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ── Users ─────────────────────────────────────────────────────────────────────

def make_users(
    catalog: pd.DataFrame,
    n_users: int = 300,
    tracks_per_user: int = 150,
    concentration: float = 0.45,
    seed: int = 23,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate listeners with a KNOWN latent taste vector.

    Each user is a Dirichlet mixture over genres (low `concentration` => a few
    dominant genres, i.e. a distinctive listener; high => everyone likes
    everything). Their library is sampled from the catalog under that mixture.

    Returns (listens, taste):
      listens -- long frame, one row per (user, track), joined to features
      taste   -- n_users x n_genres matrix, the ground truth. Two users'
                 true similarity is the overlap of their taste rows, and that
                 is what similarity.py's metrics get scored against.
    """
    rng = np.random.default_rng(seed)
    taste = rng.dirichlet(np.full(len(GENRES), concentration), size=n_users)

    by_genre = {g: catalog.index[catalog[S.GENRE] == g].to_numpy() for g in GENRES}
    # Within a genre, popular tracks are likelier to be heard.
    weights = {g: (catalog.loc[idx, S.POPULARITY].to_numpy(float) + 5) for g, idx in by_genre.items()}
    weights = {g: w / w.sum() for g, w in weights.items()}

    rows = []
    for u in range(n_users):
        picks = rng.multinomial(tracks_per_user, taste[u])
        chosen = []
        for gi, k in enumerate(picks):
            if k == 0:
                continue
            g = GENRES[gi]
            pool = by_genre[g]
            if len(pool) == 0:
                continue
            k = min(k, len(pool))
            chosen.append(rng.choice(pool, size=k, replace=False, p=weights[g]))
        if not chosen:
            continue
        idx = np.concatenate(chosen)
        sub = catalog.loc[idx, [S.TRACK_ID, S.ARTIST, S.GENRE, S.POPULARITY, S.YEAR] + S.AUDIO_FEATURES].copy()
        sub.insert(0, S.USER_ID, f"U{u:04d}")
        rows.append(sub)

    listens = pd.concat(rows, ignore_index=True)
    taste_df = pd.DataFrame(taste, columns=GENRES, index=[f"U{u:04d}" for u in range(n_users)])
    taste_df = taste_df.loc[listens[S.USER_ID].unique()]
    return listens, taste_df


def true_similarity(taste: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth user-user similarity: 1 - Jensen-Shannon distance of taste.

    JS distance is a proper metric on probability vectors and is bounded [0,1],
    so 1-JSD is a clean 'how alike are these two listeners really' target.
    """
    from scipy.spatial.distance import jensenshannon

    P = taste.to_numpy(float)
    n = len(P)
    out = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jensenshannon(P[i], P[j], base=2)
            d = 0.0 if np.isnan(d) else float(d)
            out[i, j] = out[j, i] = 1.0 - d
    return pd.DataFrame(out, index=taste.index, columns=taste.index)
