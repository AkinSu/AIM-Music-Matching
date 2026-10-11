"""Taste archetypes: the zodiac feature without the zodiac labels.

WHY THIS EXISTS

The supervised zodiac classifier does not work. Measured on the real scrape
with cluster-grouped CV, balanced accuracy lands at 0.109-0.121 against a
0.083 chance rate -- between 1.31x and 1.45x, depending on the feature set.
That is a real edge and far too small to present as a prediction about a
person.

The reason is the labels, not the model or the features. On identical
features and identical methodology, predicting *genre* reaches ~7.3x chance
while predicting zodiac reaches ~1.75x. The audio tells you plenty; it just
does not tell you what a stranger decided an "Aries song" was.

So: stop predicting the label. Cluster the catalog into twelve audio
archetypes and attach the sign names through a documented mapping. The
user-facing output is unchanged -- the same mix, the same pie chart, the
same lift framing from zodiac.py -- but the thing underneath is now
well-defined. There are no labels, so nothing can leak, no curator can
disagree, and there is no accuracy claim to defend.

WHAT THIS IS AND IS NOT

It is a taste fingerprint with playful names. Honest by construction: every
number is a statement about where a user's music sits in audio-feature
space, which is a fact, and the sign names are an interface decision we
wrote down.

It is NOT evidence that music taste relates to birth signs. Nothing here
tests that, and SURVEY.md describes the experiment that would.

THE STABILITY CAVEAT, WHICH YOU MUST READ BEFORE SHIPPING

Refit this on a different random seed and the twelve clusters do not come
back the same. Measured across five seeds on a common probe set: track-level
adjusted Rand index 0.72, a user's mix ranking correlates 0.64 with itself,
and the single headline archetype survives a refit only 43% of the time.

That does NOT make the feature unusable, but it does constrain how it may be
described. Two things follow.

First, **fit once and freeze it**. Every user is then scored against the
same persisted centroids, so each person's answer is perfectly reproducible
and the instability never reaches them. Retraining silently reshuffles
everyone's sign, so a retrain is a product decision, not a maintenance task.

Second, **do not claim these are discovered natural kinds**. Twelve is a
product requirement inherited from the zodiac, not a number the data asked
for. The clusters are a convenient partition of a continuous space, and a
different seed draws the boundaries differently. "Here is where your taste
sits" is supportable; "we discovered twelve types of listener" is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

from . import features as Fm
from . import schema as S

# ── The sign mapping: a DESIGN CHOICE, not a finding ─────────────────────────
#
# Each sign is described as a direction in standardised audio-feature space,
# drawn from the conventional personality associations -- fire signs loud and
# fast, water signs soft and melancholy, earth signs acoustic and steady, air
# signs wordy and danceable.
#
# Nothing validates this table and nothing could: there is no ground truth for
# what an Aries sounds like. It exists so the naming is reproducible and
# auditable rather than someone eyeballing twelve clusters and picking names
# that sound good. Edit it freely -- doing so changes the labels on the output
# and nothing about the clustering underneath.
#
# Weights are relative; only direction and rough magnitude matter.
SIGN_TEMPLATES: dict[str, dict[str, float]] = {
    "Aries":       {"energy": 1.0, "tempo": 0.8, "loudness": 0.6, "valence": 0.3,
                    "acousticness": -0.6},
    "Taurus":      {"acousticness": 0.9, "valence": 0.5, "energy": -0.6,
                    "tempo": -0.4, "instrumentalness": 0.2},
    "Gemini":      {"speechiness": 1.0, "danceability": 0.7, "tempo": 0.4,
                    "instrumentalness": -0.5},
    "Cancer":      {"acousticness": 0.8, "valence": -0.7, "energy": -0.6,
                    "loudness": -0.5},
    "Leo":         {"valence": 1.0, "energy": 0.8, "danceability": 0.6,
                    "loudness": 0.5},
    "Virgo":       {"instrumentalness": 0.9, "acousticness": 0.6, "energy": -0.4,
                    "speechiness": -0.5},
    "Libra":       {"danceability": 0.8, "valence": 0.7, "acousticness": 0.3,
                    "liveness": -0.3},
    "Scorpio":     {"energy": 0.8, "valence": -0.9, "loudness": 0.5,
                    "danceability": 0.2},
    "Sagittarius": {"liveness": 1.0, "tempo": 0.6, "valence": 0.4, "energy": 0.4},
    "Capricorn":   {"tempo": -0.8, "valence": -0.5, "instrumentalness": 0.5,
                    "duration_ms": 0.4},
    "Aquarius":    {"instrumentalness": 0.8, "energy": 0.5, "speechiness": 0.3,
                    "acousticness": -0.7},
    "Pisces":      {"acousticness": 0.8, "energy": -0.9, "tempo": -0.5,
                    "valence": -0.4, "instrumentalness": 0.4},
}


@dataclass
class ArchetypeModel:
    """A fitted clustering plus the sign names assigned to its clusters."""

    kmeans: object
    scaler: object
    features: list[str]
    names: list[str]                      # names[i] is the sign for cluster i
    deviations: pd.DataFrame              # cluster x feature, in catalog SDs
    shares: pd.Series                     # fraction of the catalog per cluster
    temperature: float                    # softness of the soft assignment
    examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.names)


# ── Fitting ──────────────────────────────────────────────────────────────────

def fit_archetypes(catalog: pd.DataFrame, k: int = 12, seed: int = 0,
                   features: list[str] | None = None,
                   minibatch: bool = False, sample: int | None = 120_000,
                   assign_names: bool = True) -> ArchetypeModel:
    """Cluster the catalog into k audio archetypes and name them.

    Scaling is not optional. On the real catalog `duration_ms` outweighs the
    bounded features by roughly 38 billion to one, so k-means on raw columns
    partitions the catalog by song length and nothing else.

    `sample` fits on a subset and then labels the full catalog, which is how
    586k rows stay tractable. Default 120k, which takes a few seconds.

    `minibatch` defaults to FALSE, which is the opposite of what you would
    guess at this scale, and the reason is stability rather than speed.
    Measured by adjusted Rand index across seeds on a common probe set:

        MiniBatchKMeans   ARI 0.363
        full KMeans       ARI 0.811

    MiniBatch's per-batch updates leave it sensitive to initialisation, and
    an archetype that moves when you change the random seed is not a taste
    profile -- it is noise with a zodiac sign printed on it. Fitting full
    k-means on a sample is both more stable and fast enough, so it wins on
    every axis that matters here.
    """
    features = features or S.AUDIO_FEATURES
    scaler = Fm.fit_scaler(catalog, features)

    fit_on = catalog.sample(n=min(sample, len(catalog)), random_state=seed) \
        if sample else catalog
    Z_fit = Fm.scale(fit_on, scaler, features)

    if minibatch:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10,
                             batch_size=4096, max_iter=300).fit(Z_fit)
    else:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z_fit)

    Z_all = Fm.scale(catalog, scaler, features)
    labels = km.predict(Z_all)

    # Centroids are already in standardised space, so they ARE the deviation
    # from the catalog mean, in standard deviations. That is what makes the
    # naming legible: "+1.4 SD energy" rather than "0.73 energy".
    deviations = pd.DataFrame(km.cluster_centers_, columns=features)
    shares = pd.Series(np.bincount(labels, minlength=k) / len(labels))

    # Temperature for the soft assignment: the typical squared distance from a
    # track to its own centroid. Scaling by this makes the softness adapt to
    # how tight the clusters actually are rather than being a magic number.
    d2 = ((Z_all - km.cluster_centers_[labels]) ** 2).sum(axis=1)
    temperature = float(np.mean(d2)) or 1.0

    names = (assign_sign_names(deviations) if assign_names
             else [f"Cluster {i}" for i in range(k)])

    model = ArchetypeModel(kmeans=km, scaler=scaler, features=list(features),
                           names=names, deviations=deviations, shares=shares,
                           temperature=temperature)
    model.examples = _examples(catalog, labels, names)
    return model


def assign_sign_names(deviations: pd.DataFrame,
                      templates: dict[str, dict[str, float]] | None = None) -> list[str]:
    """Match clusters to signs one-to-one, optimally.

    Score every (cluster, sign) pair by how well the cluster's feature
    deviations line up with the sign's template, then solve the assignment
    problem so each sign is used exactly once.

    Greedy assignment would be easier and worse: the first cluster would take
    the sign it fits best and later clusters would be left with whatever
    remained, making the result depend on cluster ordering. The Hungarian
    algorithm has no such dependence, which is what makes a re-fit comparable
    to the previous one.
    """
    from scipy.optimize import linear_sum_assignment

    templates = templates or SIGN_TEMPLATES
    signs = list(templates)
    k = len(deviations)

    score = np.zeros((k, len(signs)))
    for si, sign in enumerate(signs):
        w = np.array([templates[sign].get(f, 0.0) for f in deviations.columns])
        nw = np.linalg.norm(w)
        if nw == 0:
            continue
        for ci in range(k):
            v = deviations.iloc[ci].to_numpy(float)
            nv = np.linalg.norm(v)
            # Cosine: direction in feature space is what the template encodes,
            # not magnitude.
            score[ci, si] = float(v @ w / (nv * nw)) if nv else 0.0

    rows, cols = linear_sum_assignment(-score)       # maximise total fit
    names = [""] * k
    for ci, si in zip(rows, cols):
        names[ci] = signs[si]
    for ci in range(k):                               # k > len(signs)
        if not names[ci]:
            names[ci] = f"Cluster {ci}"
    return names


def _examples(catalog: pd.DataFrame, labels: np.ndarray, names: list[str],
              n: int = 5) -> dict[str, list[str]]:
    """The most popular real tracks in each cluster, as a sanity check.

    If an archetype's examples look nothing like its name, the mapping table
    is wrong -- and that is a judgement a person has to make, which is why
    these are surfaced rather than buried.
    """
    out: dict[str, list[str]] = {}
    df = catalog.assign(_c=labels)
    for ci, name in enumerate(names):
        sub = df[df["_c"] == ci]
        if not len(sub):
            out[name] = []
            continue
        top = sub.nlargest(n, S.POPULARITY) if S.POPULARITY in sub.columns else sub.head(n)
        out[name] = [f"{r[S.TRACK_NAME]} — {r[S.ARTIST]}"[:60]
                     for _, r in top.iterrows()]
    return out


# ── Inference ────────────────────────────────────────────────────────────────

def _responsibilities(model: ArchetypeModel, tracks: pd.DataFrame) -> np.ndarray:
    """Soft cluster membership per track, shape (n_tracks, k).

    Hard nearest-centroid assignment throws away exactly what notebook 01
    argued for keeping: a track sitting halfway between two archetypes is
    real information, and rounding it to a vote discards it.
    """
    Z = Fm.scale(tracks, model.scaler, model.features)
    d2 = ((Z[:, None, :] - model.kmeans.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
    logits = -d2 / (2.0 * model.temperature)
    logits -= logits.max(axis=1, keepdims=True)       # stabilise the exponential
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def taste_mix(model: ArchetypeModel, tracks: pd.DataFrame) -> pd.Series:
    """A user's distribution over archetypes -- the pie chart."""
    R = _responsibilities(model, tracks)
    return pd.Series(R.mean(axis=0), index=model.names).sort_values(ascending=False)


def base_rate(model: ArchetypeModel, reference: pd.DataFrame | None = None) -> pd.Series:
    """What an average listener looks like.

    With no reference tracks this is the catalog share of each cluster, which
    is the prior a user's mix should be read against.
    """
    if reference is None:
        return pd.Series(model.shares.to_numpy(), index=model.names) \
            .sort_values(ascending=False)
    return taste_mix(model, reference)


def lift(mix: pd.Series, base: pd.Series) -> pd.Series:
    """How much more X this listener is than an average one.

    Identical in spirit to zodiac.lift, and for the same reason: the raw
    percentages are mostly the cluster prior, and the ratio is the part that
    is actually about the person.
    """
    return (mix / base.reindex(mix.index)).sort_values(ascending=False)


def profile(model: ArchetypeModel, tracks: pd.DataFrame,
            distinctive_at: float = 1.25) -> dict:
    """The shippable payload, matching service.taste_profile's shape."""
    mix = taste_mix(model, tracks)
    lf = lift(mix, base_rate(model))
    return {
        "mix": {k: round(float(v), 4) for k, v in mix.items()},
        "lift": {k: round(float(v), 3) for k, v in lf.items()},
        "headline_sign": lf.index[0],
        "headline_lift": round(float(lf.iloc[0]), 2),
        # Say nothing rather than invent a personality out of the prior.
        "is_distinctive": bool(lf.iloc[0] >= distinctive_at),
    }


# ── Inspection ───────────────────────────────────────────────────────────────

def describe(model: ArchetypeModel, top_features: int = 3) -> pd.DataFrame:
    """One row per archetype: name, defining features, share, examples."""
    rows = []
    for ci, name in enumerate(model.names):
        dev = model.deviations.iloc[ci]
        strongest = dev.reindex(dev.abs().sort_values(ascending=False).index)[:top_features]
        rows.append({
            "archetype": name,
            "defining_features": ", ".join(
                f"{f} {v:+.2f}" for f, v in strongest.items()),
            "catalog_share": round(float(model.shares.iloc[ci]), 4),
            "examples": "; ".join(model.examples.get(name, [])[:3]),
        })
    return pd.DataFrame(rows).sort_values("catalog_share", ascending=False) \
        .reset_index(drop=True)


def stability(catalog: pd.DataFrame, k: int = 12, seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
              sample: int = 60_000, features: list[str] | None = None) -> dict:
    """Do the same tracks land in the same archetype when you re-fit?

    This is the validity question for the whole idea. An archetype assignment
    that moves when you change the random seed is not a taste profile, it is
    noise with a zodiac sign printed on it.

    Returns pairwise adjusted Rand index across seeds, measured on a common
    held-out sample so the comparison is like-for-like.
    """
    from itertools import combinations

    from sklearn.metrics import adjusted_rand_score

    features = features or S.AUDIO_FEATURES
    probe = catalog.sample(n=min(sample, len(catalog)), random_state=999)

    labelings = {}
    for s in seeds:
        m = fit_archetypes(catalog, k=k, seed=s, features=features,
                           sample=sample, assign_names=False)
        labelings[s] = m.kmeans.predict(Fm.scale(probe, m.scaler, features))

    pairs = {f"{a}-{b}": float(adjusted_rand_score(labelings[a], labelings[b]))
             for a, b in combinations(seeds, 2)}
    vals = np.array(list(pairs.values()))
    return {"pairwise_ari": pairs, "mean_ari": float(vals.mean()),
            "min_ari": float(vals.min()), "n_probe": len(probe)}
