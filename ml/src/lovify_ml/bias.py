"""Bias measurements for the Lovify models.

Not a compliance checklist. Two of the items here -- feature-store coverage
and library-size instability -- are engineering problems that should change
what the product does, because in both cases the system currently returns a
confident number in situations where it has no business being confident.
The rest are distributional facts worth knowing and stating.

Every function returns a tidy DataFrame or a plain dict so it can run in CI
and be diffed between releases, rather than living only inside a notebook.

The ordering below follows severity, highest first:

  1. coverage          -- audio features now come from a local store, and the
                          join fails non-randomly (see spotify.py)
  2. catalog           -- you can only recommend what you hold
  3. curator           -- who wrote the zodiac labels
  4. popularity        -- recommenders concentrate exposure on hits
  5. (cultural)        -- stated in BIAS.md, not measurable from here
  6. library size      -- small libraries give noisy percentiles
  7. (social framing)  -- a product decision, handled in BIAS.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as Fm
from . import schema as S
from . import similarity as SIMm


# ── Shared helpers ────────────────────────────────────────────────────────────

def gini(x: np.ndarray | pd.Series) -> float:
    """Gini coefficient: 0 = perfectly even, 1 = one item takes everything.

    Used here for exposure concentration -- how unequally recommendation
    slots are spread across the catalog.
    """
    a = np.sort(np.asarray(x, dtype=float))
    if a.size == 0 or a.sum() == 0:
        return 0.0
    n = a.size
    index = np.arange(1, n + 1)
    return float((2 * (index * a).sum() - (n + 1) * a.sum()) / (n * a.sum()))


def total_variation(p: pd.Series, q: pd.Series) -> float:
    """Half the L1 distance between two distributions, in [0, 1].

    Reads directly as "the share of mass that would have to move for these
    two distributions to match", which makes it easy to set a threshold on.
    """
    idx = p.index.union(q.index)
    p = p.reindex(idx, fill_value=0.0)
    q = q.reindex(idx, fill_value=0.0)
    return float(0.5 * np.abs(p.to_numpy() - q.to_numpy()).sum())


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature-store coverage
# ─────────────────────────────────────────────────────────────────────────────

def simulate_store_coverage(catalog: pd.DataFrame, strength: float = 12.0,
                            midpoint: float = 0.12, ceiling: float = 0.98,
                            seed: int = 0) -> set[str]:
    """Which tracks a local feature store would actually resolve.

    Spotify's /v1/audio-features died on 2024-11-27, so features now come
    from a static dump (the Kaggle 600k set) joined on track_id. That dump
    was assembled from what was popular when it was built, so the join
    succeeds for hits and fails for everything else.

    Modelled as P(resolved) = ceiling * sigmoid(strength * (pop - midpoint)).
    `midpoint` is the popularity at which a track is a coin flip; it defaults
    low because catalog popularity is heavily right-skewed, and a midpoint of
    0.5 would mean the store resolved almost nothing.

    `strength=0` makes coverage independent of popularity -- the control
    case. Always run it: it separates "the store is incomplete" (tolerable)
    from "the store is incomplete in a patterned way" (the actual problem).
    """
    rng = np.random.default_rng(seed)
    pop = catalog[S.POPULARITY].to_numpy(float) / 100.0
    if strength == 0:
        # Control: the same overall coverage RATE, but allocated at random.
        # Note the parenthesis placement -- the mean must be taken over the
        # whole probability, not over the denominator, or Jensen's inequality
        # leaves the control at a visibly different rate from the real thing
        # and the comparison stops being like-for-like.
        matched_rate = float((ceiling / (1.0 + np.exp(-12.0 * (pop - midpoint)))).mean())
        p_resolved = np.full(len(catalog), matched_rate)
    else:
        p_resolved = ceiling / (1.0 + np.exp(-strength * (pop - midpoint)))
    resolved = rng.random(len(catalog)) < p_resolved
    return set(catalog.loc[resolved, S.TRACK_ID])


def make_obscurity_varied_users(catalog: pd.DataFrame, n_users: int = 120,
                                tracks_per_user: int = 150,
                                concentration: float = 0.45,
                                seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Listeners who differ in how deep they dig, not only in which genres.

    `synthetic.make_users` varies users by genre mixture but samples every
    one of them by popularity in the same way, so every synthetic listener
    ends up about equally mainstream. That is fine for the similarity work,
    and useless for measuring coverage bias: if nobody has obscure taste,
    a store that fails on obscure music harms nobody.

    Real listener populations are not like that. Here each user also draws
    an `obscurity` preference, and their sampling weight within a genre is
    popularity tilted by it -- so the population spans chart-followers and
    crate-diggers. Returns (listens, profile) where profile carries each
    user's true obscurity preference for auditing.
    """
    from . import synthetic as syn

    rng = np.random.default_rng(seed)
    genres = [g for g in syn.GENRES if (catalog[S.GENRE] == g).any()]
    taste = rng.dirichlet(np.full(len(genres), concentration), size=n_users)
    obscurity = rng.uniform(-1.5, 2.5, size=n_users)

    by_genre = {g: catalog.index[catalog[S.GENRE] == g].to_numpy() for g in genres}
    pop_z = {}
    for g, idx in by_genre.items():
        p = catalog.loc[idx, S.POPULARITY].to_numpy(float)
        pop_z[g] = (p - p.mean()) / (p.std() + 1e-9)

    rows = []
    for u in range(n_users):
        picks = rng.multinomial(tracks_per_user, taste[u])
        chosen = []
        for gi, k in enumerate(picks):
            if k == 0:
                continue
            g = genres[gi]
            idx = by_genre[g]
            if len(idx) == 0:
                continue
            # obscurity > 0 tilts the draw towards low-popularity tracks
            w = np.exp(-obscurity[u] * pop_z[g])
            w = w / w.sum()
            k = min(k, len(idx))
            chosen.append(rng.choice(idx, size=k, replace=False, p=w))
        if not chosen:
            continue
        sel = np.concatenate(chosen)
        sub = catalog.loc[sel, [S.TRACK_ID, S.ARTIST, S.GENRE, S.POPULARITY, S.YEAR]
                          + S.AUDIO_FEATURES].copy()
        sub.insert(0, S.USER_ID, f"U{u:04d}")
        rows.append(sub)

    listens = pd.concat(rows, ignore_index=True)
    profile = pd.DataFrame({"obscurity_preference": obscurity},
                           index=[f"U{u:04d}" for u in range(n_users)])
    return listens, profile.loc[listens[S.USER_ID].unique()]


def coverage_by_user(listens: pd.DataFrame, store_ids: set[str]) -> pd.DataFrame:
    """Per-user: what fraction of their library the store can resolve.

    The columns beside `coverage` are there to answer the question that
    matters -- is low coverage randomly distributed across users, or does it
    land on a particular kind of listener?
    """
    rows = []
    for u, g in listens.groupby(S.USER_ID):
        resolved = g[S.TRACK_ID].isin(store_ids)
        rows.append({
            S.USER_ID: u,
            "n_tracks": len(g),
            "coverage": float(resolved.mean()),
            "median_popularity": float(g[S.POPULARITY].median()),
            "n_genres": int(g[S.GENRE].nunique()),
        })
    return pd.DataFrame(rows)


def coverage_skew(cov: pd.DataFrame) -> dict:
    """Is coverage correlated with how mainstream a listener is?

    A strong positive correlation is the finding: it means the people the
    product serves worst are systematically the ones with the most distinctive
    taste, which is both an accuracy problem and a fairness one.
    """
    from scipy.stats import spearmanr

    r = spearmanr(cov["coverage"], cov["median_popularity"])
    return {
        "spearman_coverage_vs_popularity": float(r.statistic),
        "p_value": float(r.pvalue),
        "mean_coverage": float(cov["coverage"].mean()),
        "worst_decile_coverage": float(cov["coverage"].quantile(0.10)),
        "best_decile_coverage": float(cov["coverage"].quantile(0.90)),
    }


class PartialCoverageScorer:
    """Compatibility scoring when only some tracks have audio features.

    This is the shape the real service has to take. Of the five blend
    components, only `feature_wasserstein` needs audio features; genre,
    artist, era and obscurity are all computed from metadata that Spotify
    still serves. So partial coverage does not break the score -- it drops
    one component, and the remaining weights are renormalised over what
    survived.

    `min_resolved` is the guard: below that many resolved tracks the audio
    component is dropped rather than computed from a handful of songs, since
    a Wasserstein distance over 3 tracks is noise with a decimal point.
    """

    def __init__(self, listens: pd.DataFrame, store_ids: set[str] | None = None,
                 weights: dict[str, float] | None = None, scaler=None,
                 catalog: pd.DataFrame | None = None, min_resolved: int = 10):
        self.features = S.AUDIO_FEATURES
        self.weights = dict(weights or SIMm.DEFAULT_WEIGHTS)
        self.min_resolved = min_resolved
        self.scaler = scaler or Fm.fit_scaler(
            catalog if catalog is not None else listens, self.features)

        self.by_user = {u: g for u, g in listens.groupby(S.USER_ID)}
        self.genre_profiles = Fm.genre_profile(listens)
        self.artist_sets = Fm.artist_sets(listens)

        # Audio-feature frames built ONLY from tracks the store resolved.
        self.resolved_scaled: dict[str, pd.DataFrame] = {}
        for u, g in self.by_user.items():
            sub = g if store_ids is None else g[g[S.TRACK_ID].isin(store_ids)]
            self.resolved_scaled[u] = pd.DataFrame(
                Fm.scale(sub, self.scaler, self.features), columns=self.features
            ) if len(sub) else pd.DataFrame(columns=self.features)

        self._null: np.ndarray | None = None

    def components(self, u: str, v: str) -> dict[str, float]:
        """Components, with the audio one set to NaN when unusable."""
        audio = np.nan
        if (len(self.resolved_scaled[u]) >= self.min_resolved
                and len(self.resolved_scaled[v]) >= self.min_resolved):
            audio = float(np.exp(SIMm.feature_wasserstein(
                self.resolved_scaled, u, v, self.features)))
        return {
            "genre_overlap": SIMm.genre_overlap(self.genre_profiles, u, v),
            "artist_jaccard": SIMm.artist_jaccard(self.artist_sets, u, v),
            "feature_wasserstein": audio,
            "era_overlap": SIMm.era_overlap(self.by_user, u, v),
            "obscurity_match": SIMm.obscurity_match(self.by_user, u, v),
        }

    def raw_score(self, u: str, v: str) -> float:
        c = self.components(u, v)
        usable = {k: w for k, w in self.weights.items() if not np.isnan(c[k])}
        total = sum(usable.values())
        if total == 0:
            return float("nan")
        return float(sum(w / total * c[k] for k, w in usable.items()))

    def set_null(self, null: np.ndarray) -> None:
        """Reuse a population null distribution rather than refitting one."""
        self._null = null

    def fit_null(self, n_pairs: int = 2000, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        users = list(self.by_user)
        scores = [self.raw_score(*rng.choice(users, size=2, replace=False))
                  for _ in range(n_pairs)]
        self._null = np.sort(np.array([s for s in scores if not np.isnan(s)]))
        return self._null

    def percentile(self, u: str, v: str) -> float:
        if self._null is None:
            self.fit_null()
        s = self.raw_score(u, v)
        if np.isnan(s):
            return float("nan")
        return float(np.searchsorted(self._null, s) / len(self._null) * 100)


def coverage_error_report(listens: pd.DataFrame, catalog: pd.DataFrame,
                          store_ids: set[str], n_pairs: int = 400,
                          min_resolved: int = 10, seed: int = 0) -> pd.DataFrame:
    """How wrong is the score when the store cannot resolve part of a library?

    Compares the percentile a pair receives under full coverage against the
    one they receive under simulated store coverage, and reports the error
    alongside the weaker of the two users' coverage. If the error grows as
    coverage falls, the product is quietly less accurate for exactly the
    users whose libraries the store handles worst.
    """
    rng = np.random.default_rng(seed)
    scaler = Fm.fit_scaler(catalog)

    full = PartialCoverageScorer(listens, None, scaler=scaler, min_resolved=min_resolved)
    part = PartialCoverageScorer(listens, store_ids, scaler=scaler, min_resolved=min_resolved)
    full.fit_null(n_pairs=1500, seed=seed)
    part.fit_null(n_pairs=1500, seed=seed)

    cov = coverage_by_user(listens, store_ids).set_index(S.USER_ID)["coverage"]
    users = list(full.by_user)

    rows = []
    for _ in range(n_pairs):
        u, v = rng.choice(users, size=2, replace=False)
        p_full, p_part = full.percentile(u, v), part.percentile(u, v)
        rows.append({
            "user_a": u, "user_b": v,
            "pct_full": p_full,
            "pct_partial": p_part,
            "abs_error": abs(p_full - p_part) if not np.isnan(p_part) else np.nan,
            "min_coverage": float(min(cov[u], cov[v])),
            "audio_component_dropped": bool(np.isnan(part.components(u, v)["feature_wasserstein"])),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Catalog bias
# ─────────────────────────────────────────────────────────────────────────────

def distribution_shift(catalog: pd.DataFrame, surfaced: pd.DataFrame,
                       column: str, bins: list | None = None,
                       top_n: int = 12) -> pd.DataFrame:
    """Share of the catalog vs share of what actually gets recommended.

    A recommender cannot surface what the catalog does not hold, but it can
    still distort what it does hold. This separates the two: the catalog
    column is the ceiling, the surfaced column is the behaviour.
    """
    if bins is not None:
        cat_vals = pd.cut(catalog[column], bins=bins)
        sur_vals = pd.cut(surfaced[column], bins=bins)
    else:
        cat_vals, sur_vals = catalog[column], surfaced[column]

    cat_share = cat_vals.value_counts(normalize=True)
    sur_share = sur_vals.value_counts(normalize=True)
    keep = cat_share.sort_values(ascending=False).head(top_n).index

    out = pd.DataFrame({
        "catalog_share": cat_share.reindex(keep, fill_value=0.0),
        "recommended_share": sur_share.reindex(keep, fill_value=0.0),
    })
    out["ratio"] = out["recommended_share"] / out["catalog_share"].replace(0, np.nan)
    return out.sort_values("recommended_share", ascending=False)


def catalog_bias_report(catalog: pd.DataFrame, surfaced: pd.DataFrame) -> dict:
    """Genre / era / popularity shift between the catalog and what is served."""
    era_bins = [1950, 1980, 1990, 2000, 2010, 2015, 2020, 2030]
    genre = distribution_shift(catalog, surfaced, S.GENRE)
    era = distribution_shift(catalog, surfaced, S.YEAR, bins=era_bins, top_n=20)
    return {
        "genre": genre,
        "era": era,
        "genre_total_variation": total_variation(genre["catalog_share"],
                                                 genre["recommended_share"]),
        "era_total_variation": total_variation(era["catalog_share"],
                                               era["recommended_share"]),
        "catalog_mean_popularity": float(catalog[S.POPULARITY].mean()),
        "recommended_mean_popularity": float(surfaced[S.POPULARITY].mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Curator provenance (for the zodiac scrape)
# ─────────────────────────────────────────────────────────────────────────────

PROVENANCE_COLUMNS = ["playlist_id", "playlist_owner", "playlist_followers",
                      "playlist_description", "playlist_created"]


def provenance_report(zodiac: pd.DataFrame) -> dict:
    """Who actually wrote these labels.

    The zodiac training labels are one curator's opinion each. That is fine
    if the curators are many and varied, and worthless if a handful of
    high-follower accounts wrote most of the data -- in which case the model
    learns one person's idea of an Aries song and calls it a measurement.

    Expects the provenance columns the scraper attaches. Returns NaN-free
    partial results if some columns are absent, so this can run early.
    """
    out: dict = {"rows": len(zodiac)}

    if "playlist_owner" in zodiac.columns:
        per_owner = zodiac.groupby("playlist_owner").size().sort_values(ascending=False)
        out["n_owners"] = int(len(per_owner))
        out["owner_gini"] = gini(per_owner.to_numpy())
        out["top5_owner_share"] = float(per_owner.head(5).sum() / per_owner.sum())
        out["owners_per_sign"] = (
            zodiac.groupby(S.STAR_SIGN)["playlist_owner"].nunique().to_dict()
            if S.STAR_SIGN in zodiac.columns else {}
        )

    if "playlist_followers" in zodiac.columns:
        f = zodiac.groupby("playlist_id")["playlist_followers"].first()
        out["follower_median"] = float(f.median())
        out["follower_gini"] = gini(f.to_numpy())
        # Rows contributed by the single most-followed playlist: if one
        # viral playlist supplies a large share of a sign's tracks, that
        # sign's label is effectively one opinion.
        top_pl = f.idxmax()
        out["top_playlist_row_share"] = float(
            (zodiac["playlist_id"] == top_pl).mean())

    if "playlist_created" in zodiac.columns:
        years = pd.to_datetime(zodiac["playlist_created"], errors="coerce").dt.year
        out["created_year_median"] = float(years.median())
        out["created_year_iqr"] = float(years.quantile(.75) - years.quantile(.25))

    return out


def synthetic_provenance(zodiac: pd.DataFrame, n_owners: int = 60,
                         concentration: float = 0.6, seed: int = 0) -> pd.DataFrame:
    """Attach plausible provenance so the report above can be demonstrated.

    Deliberately skewed: a few prolific accounts make many zodiac playlists
    and attract most followers, which is what the real distribution looks
    like. Replace this with the scraper's real columns the moment they exist.
    """
    rng = np.random.default_rng(seed)
    df = zodiac.copy()

    if S.PLAYLIST_NAME in df.columns:
        playlists = df[S.PLAYLIST_NAME].astype(str)
    else:
        playlists = pd.Series(["pl"] * len(df), index=df.index)
    uniq = playlists.unique()

    owner_w = rng.dirichlet(np.full(n_owners, concentration))
    owner_of = {p: f"curator_{rng.choice(n_owners, p=owner_w):03d}" for p in uniq}
    follow_of = {p: int(np.exp(rng.normal(7.0, 2.2))) for p in uniq}
    year_of = {p: int(np.clip(rng.normal(2021, 2.0), 2014, 2025)) for p in uniq}

    df["playlist_id"] = playlists.to_numpy()
    df["playlist_owner"] = playlists.map(owner_of).to_numpy()
    df["playlist_followers"] = playlists.map(follow_of).to_numpy()
    df["playlist_created"] = playlists.map(lambda p: f"{year_of[p]}-06-01").to_numpy()
    df["playlist_description"] = ""
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Popularity bias in recommendations
# ─────────────────────────────────────────────────────────────────────────────

def recommendation_exposure(rec, listens: pd.DataFrame, n: int = 20,
                            max_users: int = 150, seed: int = 0, **kw) -> pd.Series:
    """Count how often each catalog track is recommended across all users.

    Exposure, not accuracy. A recommender can score well on recall while
    serving the same 200 songs to everybody, and this is the measurement
    that catches it.
    """
    rng = np.random.default_rng(seed)
    users = list(listens[S.USER_ID].unique())
    if len(users) > max_users:
        users = list(rng.choice(users, size=max_users, replace=False))

    counts: dict[str, int] = {}
    for u in users:
        recs = rec.recommend(listens[listens[S.USER_ID] == u], n=n, **kw)
        for t in recs[S.TRACK_ID]:
            counts[t] = counts.get(t, 0) + 1
    return pd.Series(counts, dtype=float).sort_values(ascending=False)


def popularity_bias_report(catalog: pd.DataFrame, exposure: pd.Series,
                           n_users: int | None = None) -> dict:
    """Concentration and long-tail reach of the recommendations.

    Both headline numbers need a baseline or they mislead:

    * Raw catalog coverage looks terrible for any catalog larger than the
      number of recommendation slots, because you cannot surface 600k tracks
      in 1,200 slots. So we also report what a RANDOM recommender would
      cover with the same number of slots, and the ratio between them. Below
      1.0 means genuinely concentrated; near 1.0 means the apparent
      concentration was just arithmetic.
    * Gini over the whole catalog is likewise dragged towards 1 by the
      unavoidable zeros, so `exposure_gini_recommended` restricts it to
      tracks that were surfaced at least once.
    """
    full = exposure.reindex(catalog[S.TRACK_ID], fill_value=0.0)
    surfaced = catalog[catalog[S.TRACK_ID].isin(exposure.index)]
    n_slots = float(full.sum())
    n_cat = len(catalog)

    # Expected distinct tracks if the same number of slots were filled at random.
    expected_random = n_cat * (1.0 - (1.0 - 1.0 / n_cat) ** n_slots) if n_slots else 0.0
    observed = float((full > 0).sum())

    return {
        "catalog_size": int(n_cat),
        "recommendation_slots": int(n_slots),
        "distinct_tracks_recommended": int(observed),
        "catalog_coverage": float((full > 0).mean()),
        "random_baseline_coverage": float(expected_random / n_cat),
        "coverage_vs_random": float(observed / expected_random) if expected_random else np.nan,
        "exposure_gini_full_catalog": gini(full.to_numpy()),
        "exposure_gini_recommended": gini(full[full > 0].to_numpy()),
        "top1pct_exposure_share": float(
            full.sort_values(ascending=False).head(max(1, n_cat // 100)).sum()
            / max(n_slots, 1e-9)),
        "catalog_mean_popularity": float(catalog[S.POPULARITY].mean()),
        "recommended_mean_popularity": float(surfaced[S.POPULARITY].mean()) if len(surfaced) else np.nan,
        "n_users": n_users,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Library-size instability
# ─────────────────────────────────────────────────────────────────────────────

def percentile_stability(listens: pd.DataFrame, catalog: pd.DataFrame,
                         sizes: list[int] | None = None, n_pairs: int = 5,
                         n_boot: int = 40, seed: int = 0) -> pd.DataFrame:
    """How much does a pair's percentile move if you resample their libraries?

    The compatibility percentile is reported to one decimal place, which
    implies a precision it does not have when a user has 20 tracks. This
    bootstraps the number: subsample both libraries to `size`, rescore, and
    measure the spread across resamples.

    The spread IS the uncertainty. If it is 30 points wide at 20 tracks, then
    a "73rd percentile" badge on a 20-track user is decoration.
    """
    sizes = sizes or [10, 20, 30, 50, 75, 100, 150, 200]
    rng = np.random.default_rng(seed)
    scaler = Fm.fit_scaler(catalog)

    population = SIMm.CompatibilityScorer(listens, catalog=catalog, scaler=scaler)
    null = population.fit_null(n_pairs=2000, seed=seed)

    by_user = {u: g for u, g in listens.groupby(S.USER_ID)}
    eligible = [u for u, g in by_user.items() if len(g) >= max(sizes)]
    if len(eligible) < 2:
        raise ValueError("need at least two users with enough tracks for the largest size")

    pairs = [tuple(rng.choice(eligible, size=2, replace=False)) for _ in range(n_pairs)]

    rows = []
    for u, v in pairs:
        # The target: the percentile this pair gets with their libraries intact.
        truth = population.percentile(u, v)
        for size in sizes:
            scores = []
            for _ in range(n_boot):
                a_sub = by_user[u].sample(n=min(size, len(by_user[u])),
                                          random_state=int(rng.integers(1e9)))
                b_sub = by_user[v].sample(n=min(size, len(by_user[v])),
                                          random_state=int(rng.integers(1e9)))
                pair_df = pd.concat([a_sub, b_sub], ignore_index=True)
                sc = SIMm.CompatibilityScorer(pair_df, scaler=scaler, catalog=catalog)
                sc._null = null
                scores.append(sc.percentile(u, v))
            arr = np.array(scores)
            rows.append({
                "pair": f"{u}|{v}",
                "library_size": size,
                "true_percentile": float(truth),
                "mean_percentile": float(arr.mean()),
                # Spread alone UNDERSTATES the damage at small sizes: with few
                # tracks the components degenerate (artist overlap goes to
                # zero, genre profiles get sparse) and the score pins against
                # one end of the null, so it is consistently wrong but
                # consistent. Error against the full-library percentile is the
                # quantity the product actually cares about.
                "sd": float(arr.std()),
                "ci95_width": float(np.percentile(arr, 97.5) - np.percentile(arr, 2.5)),
                "bias": float(arr.mean() - truth),
                "rmse": float(np.sqrt(np.mean((arr - truth) ** 2))),
                "p95_abs_error": float(np.percentile(np.abs(arr - truth), 95)),
            })
    return pd.DataFrame(rows)


def recommend_min_library(stability: pd.DataFrame, max_error: float = 10.0,
                          metric: str = "rmse") -> dict:
    """Smallest library size at which the displayed percentile is trustworthy.

    Defaults to RMSE against the full-library percentile rather than the
    bootstrap interval width, because interval width is NOT monotonic in
    library size: at very small sizes the score degenerates and pins against
    one end of the null, so the spread narrows while the answer gets worse.
    Selecting on spread would therefore hand back an absurdly small
    threshold. Error against the truth is monotonic and is what the user
    experiences.

    `max_error` is a product decision: it states how many percentile points
    of error the badge is allowed to carry.
    """
    summary = stability.groupby("library_size")[metric].mean()
    passing = summary[summary <= max_error]
    return {
        "metric": metric,
        "max_error": max_error,
        "min_library_size": int(passing.index.min()) if len(passing) else None,
        f"{metric}_by_size": summary.round(2).to_dict(),
    }


def fit_size_matched_null(listens: pd.DataFrame, catalog: pd.DataFrame, size: int,
                          n_pairs: int = 600, scaler=None, seed: int = 0) -> np.ndarray:
    """A null distribution built from libraries of the SAME size as the query.

    This is the fix for the systematic bias that `percentile_stability`
    exposes. The default null is fitted on full-size libraries, but a user
    with 20 tracks scores lower on almost every component -- artist overlap
    collapses towards zero, genre profiles turn sparse -- so comparing their
    raw score against a full-library null places them near the bottom of it
    regardless of who they are paired with.

    The percentile is only meaningful against a reference population that
    resembles the thing being scored. Subsample the random pairs to the same
    library size and the bias disappears.
    """
    rng = np.random.default_rng(seed)
    scaler = scaler if scaler is not None else Fm.fit_scaler(catalog)
    by_user = {u: g for u, g in listens.groupby(S.USER_ID)}
    users = list(by_user)

    scores = []
    for _ in range(n_pairs):
        u, v = rng.choice(users, size=2, replace=False)
        a = by_user[u].sample(n=min(size, len(by_user[u])), random_state=int(rng.integers(1e9)))
        b = by_user[v].sample(n=min(size, len(by_user[v])), random_state=int(rng.integers(1e9)))
        sc = SIMm.CompatibilityScorer(pd.concat([a, b], ignore_index=True),
                                      scaler=scaler, catalog=catalog)
        scores.append(sc.raw_score(u, v))
    return np.sort(np.array(scores))


def stability_with_matched_null(listens: pd.DataFrame, catalog: pd.DataFrame,
                                sizes: list[int] | None = None, n_pairs: int = 6,
                                n_boot: int = 25, null_pairs: int = 400,
                                seed: int = 0) -> pd.DataFrame:
    """Re-run the stability sweep using a size-matched null at each size.

    Produces the same columns as `percentile_stability` so the two can be
    compared directly -- which is the point: it demonstrates that the bias
    was an artefact of the reference population, not an irreducible property
    of small libraries.
    """
    sizes = sizes or [10, 20, 30, 50, 75, 100, 150, 200]
    rng = np.random.default_rng(seed)
    scaler = Fm.fit_scaler(catalog)

    population = SIMm.CompatibilityScorer(listens, catalog=catalog, scaler=scaler)
    population.fit_null(n_pairs=2000, seed=seed)

    by_user = {u: g for u, g in listens.groupby(S.USER_ID)}
    eligible = [u for u, g in by_user.items() if len(g) >= max(sizes)]
    pairs = [tuple(rng.choice(eligible, size=2, replace=False)) for _ in range(n_pairs)]

    rows = []
    for size in sizes:
        null = fit_size_matched_null(listens, catalog, size, n_pairs=null_pairs,
                                     scaler=scaler, seed=seed)
        for u, v in pairs:
            truth = population.percentile(u, v)
            scores = []
            for _ in range(n_boot):
                a = by_user[u].sample(n=min(size, len(by_user[u])),
                                      random_state=int(rng.integers(1e9)))
                b = by_user[v].sample(n=min(size, len(by_user[v])),
                                      random_state=int(rng.integers(1e9)))
                sc = SIMm.CompatibilityScorer(pd.concat([a, b], ignore_index=True),
                                              scaler=scaler, catalog=catalog)
                sc._null = null
                scores.append(sc.percentile(u, v))
            arr = np.array(scores)
            rows.append({
                "pair": f"{u}|{v}", "library_size": size,
                "true_percentile": float(truth), "mean_percentile": float(arr.mean()),
                "sd": float(arr.std()),
                "ci95_width": float(np.percentile(arr, 97.5) - np.percentile(arr, 2.5)),
                "bias": float(arr.mean() - truth),
                "rmse": float(np.sqrt(np.mean((arr - truth) ** 2))),
                "p95_abs_error": float(np.percentile(np.abs(arr - truth), 95)),
            })
    return pd.DataFrame(rows)
