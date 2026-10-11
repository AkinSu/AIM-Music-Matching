"""The zodiac rubric: a designed mapping, deliberately not a learned one.

WHAT THIS IS NOT
----------------
This module carries no accuracy figure and never will, because there is
nothing to be accurate *about*. We wrote the mapping; a model that recovered
it would only be reciting us back to ourselves. That circularity already cost
this project one headline finding (notebook 02, where truth was defined in
genre space so the genre metric could not lose), and reintroducing it behind
a zodiac mask would be the same mistake in a costume.

Any number in this module describes how the catalog distributes across the
twelve buckets. None of it describes how *correct* the buckets are.

WHY A RUBRIC AT ALL
-------------------
We tried learning the mapping from 1,325 scraped zodiac playlists. It reached
11.6% balanced accuracy against an 8.33% chance rate -- 1.39x -- while a
control of unrelated playlists given random zodiac labels scored 0.94x. So
the signal is real and far too small to ship as a prediction (README.md).

ASTROLOGY_REFERENCE.md explains why, and the explanation is not "we needed a
better model". The product assumes a chain, sun sign -> personality -> music
taste. The first link is null at N > 20M and across 2,011 time-twin sets
(ref. 2.6.2-2.6.3); the second is |r| .10-.21, "a poor predictor" (Schaefer &
Mehlhorn 2017 meta-analysis of 28 studies, ref. 2.7). The composition is
approximately zero. 1.39x is what the literature predicted in advance.

THE DESIGN
----------
Seven decisions, not twelve. Elements x modalities produces exactly twelve
(element, modality) pairs with no collisions and no gaps -- verified sign by
sign in ref. 1.4, and forced by arithmetic rather than coincidence: element
cycles with period 4, modality with period 3, gcd(3,4)=1, so the joint cycle
is lcm(3,4)=12 (ref. 1.4.1).

So we specify four element profiles and three modality profiles and compose.
Every sign is reconstructible from its two axes, which means anything that
cannot be reconstructed is an unsourced addition and is visible as one in
review. That is the auditability property, and it is the whole reason for
preferring this to twelve hand-written profiles.

For contrast, `archetypes.SIGN_TEMPLATES` hand-writes all twelve
independently. This module exists partly to do that properly.

WHY THESE FEATURES
------------------
The tradition says nothing about audio features -- there is no peer-reviewed
work mapping signs to musical attributes at all (ref. 1.8). The element and
modality profiles below are OUR translation of documented temperament terms
into Spotify's feature space. The temperament terms are sourced; the
translation is a design decision, and is labelled as such on every line.

One encouraging check: the main folk mapping in circulation (Bustle, ref.
1.8.1) turns out to be approximately element x modality already, without
appearing to intend it -- cardinal fire reads as "high-energy anthems", fixed
fire as "theatrical and dramatic", mutable fire as "eclectic across genres".
Where this rubric agrees with audience expectation, that is corroboration
from an independent construction, not evidence of having fitted to it. We
never saw per-sign expectations while choosing the seven profiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as Fm
from . import schema as S

# ── The two axes ─────────────────────────────────────────────────────────────
# ref. 1.4: every sign is exactly one (element, modality) pair.

ELEMENT_OF = {
    "Aries": "Fire", "Leo": "Fire", "Sagittarius": "Fire",
    "Taurus": "Earth", "Virgo": "Earth", "Capricorn": "Earth",
    "Gemini": "Air", "Libra": "Air", "Aquarius": "Air",
    "Cancer": "Water", "Scorpio": "Water", "Pisces": "Water",
}

MODALITY_OF = {
    "Aries": "Cardinal", "Cancer": "Cardinal", "Libra": "Cardinal",
    "Capricorn": "Cardinal",
    "Taurus": "Fixed", "Leo": "Fixed", "Scorpio": "Fixed", "Aquarius": "Fixed",
    "Gemini": "Mutable", "Virgo": "Mutable", "Sagittarius": "Mutable",
    "Pisces": "Mutable",
}

# Polarity is deliberately absent. It is fully determined by element -- Fire
# and Air positive, Earth and Water negative -- so including it would add a
# third axis carrying zero information beyond the first. ref. 1.6.1,
# Appendix B item 6.


# ── Element profiles: 4 of the 7 decisions ───────────────────────────────────
# Offsets in standardised units (catalog standard deviations). Each line cites
# the sourced temperament term it translates; the translation is ours.

ELEMENT_PROFILE: dict[str, dict[str, float]] = {
    # "hot, dry, ardent" (Manilius); "spontaneous, reactive, assertive";
    # "passion, courage, impulse". ref. 1.2
    "Fire": {
        "energy": 0.85,         # ardent, assertive
        "loudness": 0.55,       # assertive presence
        "tempo": 0.40,          # reactive, quick
        "acousticness": -0.55,  # the opposite pole of Earth's organic quality
    },
    # "heavy, cold, dry" (Manilius); "reliable, practical, sensual"; they
    # "collect and build", seek solid results. ref. 1.2
    "Earth": {
        "acousticness": 0.80,   # organic, sensual, material
        "energy": -0.45,        # heavy, unhurried
        "tempo": -0.40,         # unhurried
        "liveness": -0.25,      # studio-solid rather than in-the-moment
    },
    # "light, hot, wet" (Manilius); "intellectual, detached, communicative";
    # "humanitarian, social". ref. 1.2
    "Air": {
        "speechiness": 0.75,       # communicative, verbal
        "danceability": 0.45,      # social
        "instrumentalness": -0.45, # words over wordlessness
        "acousticness": -0.30,     # detached rather than earthy
    },
    # "cold, wet, soft" (Manilius); "fluid, emotional, intuitive"; "rely more
    # on their senses than logic"; music "they can lose themselves in"
    # (ref. 1.8 folk gloss, consistent with the classical terms). ref. 1.2
    "Water": {
        "valence": -0.70,        # emotional weight rather than brightness
        "acousticness": 0.45,    # soft
        "energy": -0.45,         # cold, yielding
        "instrumentalness": 0.35,  # immersive, losing oneself
    },
}


# ── Modality profiles: the other 3 ───────────────────────────────────────────
# Modalities modulate rather than define. Their magnitudes are deliberately
# smaller than the element offsets: the element says what kind of music, the
# modality says how it behaves. Keeping them smaller also prevents a modality
# from overwhelming its element and collapsing the 4x3 grid into 3 clusters.

MODALITY_PROFILE: dict[str, dict[str, float]] = {
    # "initiating, active, involved, ambitious and opportunistic". ref. 1.3
    #
    # Expect this to be the weakest of the three, and know why in advance:
    # modern "cardinal" merges two separate Ptolemaic categories, the
    # equinoctial (Aries, Libra) and the solstitial (Cancer, Capricorn), which
    # were not one thing for him. ref. 1.3.2, Appendix B item 4. If cardinal
    # signs come out mushier than fixed or mutable ones, that is the cause.
    "Cardinal": {
        "loudness": 0.30,      # declarative openings
        "tempo": 0.20,         # forward motion
        "duration_ms": -0.20,  # a statement, not a dwelling
    },
    # "consolidating, stabilising, resistant to change". ref. 1.3
    "Fixed": {
        "duration_ms": 0.35,   # sustained
        "danceability": 0.25,  # steady, repeating pulse
        "liveness": -0.25,     # controlled rather than contingent
    },
    # "adapting, transitional, dispersing". ref. 1.3
    "Mutable": {
        "liveness": 0.40,      # contingent, in-the-moment
        "speechiness": 0.20,   # shifting registers
        "duration_ms": -0.20,  # restless, moving on
    },
}


# Features the rubric is allowed to touch. `key`, `mode` and `time_signature`
# are excluded: `key` is cyclic (schema.py), and the tradition's claims about
# signs corresponding to musical keys are explicitly unfalsifiable as stated
# (ref. 1.7.2) -- usable as decoration, not as a premise.
RUBRIC_FEATURES = list(S.AUDIO_FEATURES)


# ── The three contested cells ────────────────────────────────────────────────
# ref. 1.7.1(a), Appendix B item 3. In three signs the folk personality fights
# the sign's own structural position. We were told to decide deliberately and
# write the decision down, so:
#
#   Libra     Cardinal (initiating) vs folk "indecisive".
#             -> FOLLOW THE STRUCTURE. "Indecisive" has no sourced route into
#                audio features, and bending Libra would break the one
#                property that makes this rubric auditable.
#
#   Aquarius  Fixed (stable, resistant) vs folk "inventive, revolutionary".
#             -> FOLLOW THE STRUCTURE. The folk reading is usually credited to
#                Uranus, a modern rulership we were told not to mix in
#                (ref. 1.5.2, Appendix B item 5). Honouring it would mean
#                importing exactly the layer we excluded.
#
#   Cancer    Cardinal (initiating) vs folk "retreating, shell-dwelling".
#             -> FOLLOW THE STRUCTURE, with a caveat recorded. Water's
#                strongly negative valence and energy already pull Cancer
#                inward, so the composed profile lands near the folk
#                expectation anyway: cardinal's small positive loudness is
#                outweighed. The tension resolves itself here rather than
#                needing an override.
#
# One rule, applied three times: when the two disagree, the sourced axes win.
# An exception would be an unsourced addition, which is the thing ref. 1.4.1
# says should be visible in review -- so there are none.
CONTESTED = {
    "Libra": "Cardinal 'initiating' vs folk 'indecisive' -- followed structure; "
             "'indecisive' has no sourced route into audio features.",
    "Aquarius": "Fixed 'stable' vs folk 'revolutionary' -- followed structure; "
                "the folk reading rests on Uranus, a modern rulership excluded "
                "by ref. 1.5.2.",
    "Cancer": "Cardinal 'initiating' vs folk 'retreating' -- followed structure; "
              "Water's negative valence and energy dominate, so the composed "
              "profile lands near the folk expectation regardless.",
}


def sign_profile(sign: str, features: list[str] | None = None) -> pd.Series:
    """Compose a sign's target point from its element and modality.

    This is the whole design in one function: nothing is specified per sign,
    everything is composed from the seven profiles above.
    """
    if sign not in ELEMENT_OF:
        raise KeyError(f"{sign!r} is not one of the twelve signs")
    features = features or RUBRIC_FEATURES

    out = pd.Series(0.0, index=features, dtype=float)
    for table, axis in ((ELEMENT_PROFILE, ELEMENT_OF[sign]),
                        (MODALITY_PROFILE, MODALITY_OF[sign])):
        for feat, delta in table[axis].items():
            if feat in out.index:
                out[feat] += delta
    return out


def all_profiles(features: list[str] | None = None) -> pd.DataFrame:
    """The twelve composed targets, signs as rows."""
    features = features or RUBRIC_FEATURES
    return pd.DataFrame(
        {s: sign_profile(s, features) for s in S.ZODIAC_SIGNS}).T.loc[S.ZODIAC_SIGNS]


# ── Scoring ──────────────────────────────────────────────────────────────────

def _affinity(Z: np.ndarray, targets: np.ndarray, temperature: float) -> np.ndarray:
    """Soft affinity of each row to each sign, rows summing to 1.

    Soft, not nearest-sign: a track sitting between two signs is real
    information and rounding it to a vote discards it. Same argument notebook
    01 made for averaging probabilities instead of counting hard labels.

    `temperature` must be positive. Zero divides by zero and returns all-NaN
    rather than raising, which then propagates silently through taste_mix and
    lift so that `headline_sign` comes out as garbage instead of an error. A
    negative value is worse: it flips the sign of every logit and quietly
    selects the sign a track is *least* like.
    """
    if not temperature > 0:
        raise ValueError(
            f"temperature must be positive, got {temperature!r}. Zero yields "
            "all-NaN affinities and a negative value inverts the ranking, "
            "selecting the least-matching sign -- both fail silently."
        )
    d2 = ((Z[:, None, :] - targets[None, :, :]) ** 2).sum(axis=2)
    logits = -d2 / (2.0 * temperature)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def score_tracks(tracks: pd.DataFrame, scaler, features: list[str] | None = None,
                 temperature: float = 1.0) -> pd.DataFrame:
    """Per-track affinity to each of the twelve signs.

    `scaler` must be fitted on the catalog, never on the tracks being scored
    -- the rubric's offsets are in catalog standard deviations, so scoring
    against a locally-fitted scaler would silently change what they mean.
    """
    features = features or RUBRIC_FEATURES
    Z = Fm.scale(tracks, scaler, features)
    targets = all_profiles(features).to_numpy(float)
    return pd.DataFrame(_affinity(Z, targets, temperature),
                        index=tracks.index, columns=S.ZODIAC_SIGNS)


def score_track(track: pd.Series | pd.DataFrame, scaler,
                features: list[str] | None = None,
                temperature: float = 1.0) -> pd.Series:
    """Affinity of a single track to each sign."""
    frame = track.to_frame().T if isinstance(track, pd.Series) else track
    return score_tracks(frame, scaler, features, temperature).iloc[0]


def taste_mix(tracks: pd.DataFrame, scaler, features: list[str] | None = None,
              temperature: float = 1.0) -> pd.Series:
    """A listener's distribution over the twelve signs -- the pie chart.

    Same shape as archetypes.taste_mix, so the two are swappable behind the
    service layer.
    """
    return score_tracks(tracks, scaler, features, temperature) \
        .mean(axis=0).sort_values(ascending=False)


def base_rate(catalog: pd.DataFrame, scaler, features: list[str] | None = None,
              temperature: float = 1.0, sample: int | None = 100_000,
              seed: int = 0) -> pd.Series:
    """What the rubric says about the catalog at large.

    The number a user's mix has to be read against. Without it, "you are 14%
    Pisces" is mostly a statement about how much of the catalog is Pisces.
    """
    ref = catalog
    if sample is not None and len(catalog) > sample:
        ref = catalog.sample(n=sample, random_state=seed)
    return taste_mix(ref, scaler, features, temperature)


def lift(mix: pd.Series, base: pd.Series) -> pd.Series:
    """How much more X this listener is than an average one.

    Identical in spirit to zodiac.lift and archetypes.lift. The raw shares are
    mostly the prior; the ratio is the part that is about the person.
    """
    return (mix / base.reindex(mix.index)).sort_values(ascending=False)


def profile(tracks: pd.DataFrame, scaler, catalog_base: pd.Series,
            distinctive_at: float = 1.25,
            features: list[str] | None = None,
            temperature: float = 1.0) -> dict:
    """The shippable payload, matching service.taste_profile's shape.

    On `distinctive_at`: a hardcoded 1.25 is a guess, and on this project that
    guess fired for 100% of listeners under one model and 52% under another --
    a badge that fires for everyone carries as much information as one that
    never fires. `service.LovifyModels` now fits the threshold to the
    reference population's own top-lift distribution instead. Pass that fitted
    value in rather than relying on this default.
    """
    mix = taste_mix(tracks, scaler, features, temperature)
    lf = lift(mix, catalog_base)
    return {
        "mix": {k: round(float(v), 4) for k, v in mix.items()},
        "lift": {k: round(float(v), 3) for k, v in lf.items()},
        "headline_sign": lf.index[0],
        "headline_lift": round(float(lf.iloc[0]), 2),
        "is_distinctive": bool(lf.iloc[0] >= distinctive_at),
    }


# ── Auditability ─────────────────────────────────────────────────────────────

def explain(sign: str, features: list[str] | None = None) -> dict:
    """Why this sign looks the way it does, traced to its two axes.

    The point of the rubric is that this function can exist. Every component
    of every sign resolves to one of the seven profiles and to the reference
    section that profile came from -- there is no per-sign residue to
    explain away.
    """
    element, modality = ELEMENT_OF[sign], MODALITY_OF[sign]
    composed = sign_profile(sign, features)
    strongest = composed.reindex(composed.abs().sort_values(ascending=False).index)

    return {
        "sign": sign,
        "element": element,
        "modality": modality,
        "element_contribution": dict(ELEMENT_PROFILE[element]),
        "modality_contribution": dict(MODALITY_PROFILE[modality]),
        "composed": {k: round(float(v), 3) for k, v in composed.items()},
        "defining_features": [f"{f} {v:+.2f}" for f, v in strongest.head(3).items()],
        "sources": {
            "element": "ASTROLOGY_REFERENCE.md 1.2 (Manilius qualities, Ptolemy I.18)",
            "modality": "ASTROLOGY_REFERENCE.md 1.3 (modern convention; cf. 1.3.2)",
            "composition": "ASTROLOGY_REFERENCE.md 1.4.1 (element x modality is a bijection)",
        },
        "contested": CONTESTED.get(sign),
        # There is no fourth key here on purpose. If a sign ever needs one,
        # that is an unsourced addition and should fail review.
    }


def describe(catalog: pd.DataFrame, scaler, features: list[str] | None = None,
             temperature: float = 1.0, sample: int | None = 100_000,
             seed: int = 0, n_examples: int = 3) -> pd.DataFrame:
    """One row per sign: axes, defining features, catalog share, examples.

    `catalog_share` is how the catalog distributes, not how well the rubric
    works. There is no second number here, and that is deliberate.
    """
    features = features or RUBRIC_FEATURES
    ref = catalog
    if sample is not None and len(catalog) > sample:
        ref = catalog.sample(n=sample, random_state=seed)

    scores = score_tracks(ref, scaler, features, temperature)
    winner = scores.idxmax(axis=1)
    share = scores.mean(axis=0)

    rows = []
    for sign in S.ZODIAC_SIGNS:
        exp = explain(sign, features)
        hits = ref.loc[winner == sign]
        if len(hits) and S.POPULARITY in hits.columns:
            hits = hits.nlargest(n_examples, S.POPULARITY)
        examples = [f"{r[S.TRACK_NAME]} - {r[S.ARTIST]}"
                    for _, r in hits.head(n_examples).iterrows()] \
            if S.TRACK_NAME in ref.columns else []
        rows.append({
            "sign": sign,
            "element": exp["element"],
            "modality": exp["modality"],
            "defining_features": ", ".join(exp["defining_features"]),
            "soft_share": round(float(share[sign]), 4),
            "hard_share": round(float((winner == sign).mean()), 4),
            "examples": "; ".join(examples),
        })
    return pd.DataFrame(rows)
